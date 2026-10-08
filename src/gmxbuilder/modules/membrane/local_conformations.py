"""Chemistry-defined internal coordinates for single-lipid relaxation evidence."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

METHOD = "single-lipid-internal-coordinates-2"


@lru_cache(maxsize=256)
def descriptor_definition(smiles, names, elements, bonds, polar, tails, anchor):
    """Map reference rotatable bonds to the exact force-field coordinate order."""
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    from gmxbuilder.geometry.molecular_identity import reference_mappings

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid lipid reference")
    mappings = reference_mappings(smiles, elements, bonds)
    # Equivalent mappings are resolved by atom names, independently of coordinates.
    mapping = min(mappings, key=lambda row: tuple(names[i] for i in row))
    adjacency = [set() for _ in names]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    heavy = tuple(i for i, element in enumerate(elements) if element != "H")
    endpoints = tuple(
        sorted(
            (i for i in tails if len(adjacency[i] & set(heavy)) == 1),
            key=lambda i: names[i],
        )
    )
    torsions = []
    for left, right in molecule.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        a, b = mapping[left], mapping[right]
        if names[a] > names[b]:
            a, b = b, a
        outer_a = sorted((adjacency[a] & set(heavy)) - {b}, key=lambda i: names[i])
        outer_b = sorted((adjacency[b] & set(heavy)) - {a}, key=lambda i: names[i])
        if outer_a and outer_b:
            torsions.append((outer_a[0], a, b, outer_b[0]))
    torsions = sorted(set(torsions), key=lambda row: tuple(names[i] for i in row))
    from itertools import combinations

    heavy_set = set(heavy)
    heavy_bonds = sorted(
        [
            tuple(sorted((a, b), key=lambda i: names[i]))
            for a, b in bonds
            if a in heavy_set and b in heavy_set
        ],
        key=lambda row: tuple(names[i] for i in row),
    )
    heavy_angles = sorted(
        [
            (a, center, b)
            for center in heavy
            for a, b in combinations(
                sorted(adjacency[center] & heavy_set, key=lambda i: names[i]), 2
            )
        ],
        key=lambda row: tuple(names[i] for i in row),
    )
    record = {
        "method": METHOD,
        "smiles": smiles,
        "atom_names": list(names),
        "elements": list(elements),
        "bonds": sorted([sorted([int(a), int(b)]) for a, b in bonds]),
        "heavy": list(heavy),
        "polar": list(polar),
        "tails": list(tails),
        "anchor": int(anchor),
        "endpoints": list(endpoints),
        "torsions": [list(row) for row in torsions],
        "heavy_bonds": [list(row) for row in heavy_bonds],
        "heavy_angles": [list(row) for row in heavy_angles],
    }
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def definition_for(group):
    return descriptor_definition(
        group["smiles"],
        tuple(group["names"]),
        tuple(group["elements"]),
        tuple(group["bonds"]),
        tuple(map(int, group["polar"])),
        tuple(map(int, group["tails"])),
        int(group["anchor"]),
    )


def validate_definition(record):
    expected = descriptor_definition(
        record["smiles"],
        tuple(record["atom_names"]),
        tuple(record["elements"]),
        tuple(map(tuple, record["bonds"])),
        tuple(record["polar"]),
        tuple(record["tails"]),
        record["anchor"],
    )
    if record != expected:
        raise ValueError("Changed local-conformation atom definitions")


def dihedrals(xyz, indices):
    """Signed dihedrals in radians, for a batch of whole molecules."""
    if not len(indices):
        return np.empty((len(xyz), 0))
    a, b, c, d = np.asarray(indices, dtype=int).T
    axis = xyz[:, c] - xyz[:, b]
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    first, last = xyz[:, a] - xyz[:, b], xyz[:, d] - xyz[:, c]
    first -= (first * axis).sum(axis=-1, keepdims=True) * axis
    last -= (last * axis).sum(axis=-1, keepdims=True) * axis
    return np.arctan2((np.cross(axis, first) * last).sum(axis=-1), (first * last).sum(axis=-1))


def internal_descriptors(xyz, definition):
    """Return per-molecule shape values and three-sector torsion occupancies."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 3 or not np.isfinite(xyz).all():
        raise ValueError("Invalid local-conformation coordinates")
    heavy = xyz[:, definition["heavy"]]
    centered = heavy - heavy.mean(axis=1, keepdims=True)
    values = {
        "shape:rg_nm": np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1)),
        "shape:head_tail_nm": np.linalg.norm(
            xyz[:, definition["polar"]].mean(axis=1) - xyz[:, definition["tails"]].mean(axis=1),
            axis=1,
        ),
    }
    for endpoint in definition["endpoints"]:
        values[f"shape:end_{definition['atom_names'][endpoint]}_nm"] = np.linalg.norm(
            xyz[:, endpoint] - xyz[:, definition["anchor"]], axis=1
        )
    bonds = definition.get("heavy_bonds", [])
    if bonds:
        a, b = np.asarray(bonds, dtype=int).T
        distances = np.linalg.norm(xyz[:, a] - xyz[:, b], axis=2)
        for index, (a, b) in enumerate(bonds):
            values[f"bond:{definition['atom_names'][a]}-{definition['atom_names'][b]}"] = distances[
                :, index
            ]
    triples = definition.get("heavy_angles", [])
    if triples:
        a, b, c = np.asarray(triples, dtype=int).T
        first, last = xyz[:, a] - xyz[:, b], xyz[:, c] - xyz[:, b]
        denominator = np.maximum(
            np.linalg.norm(first, axis=2) * np.linalg.norm(last, axis=2), 1e-15
        )
        measured = np.degrees(np.arccos(np.clip(np.sum(first * last, axis=2) / denominator, -1, 1)))
        for index, triple in enumerate(triples):
            key = "angle:" + "-".join(definition["atom_names"][i] for i in triple)
            values[key] = measured[:, index]
    angles = dihedrals(xyz, definition["torsions"])
    # Sectors centred on -60, +60 and 180 degrees. These are geometric sectors,
    # not a claim that every chemically different bond has alkane energy minima.
    for index, (_, a, b, _) in enumerate(definition["torsions"]):
        label = f"{definition['atom_names'][a]}-{definition['atom_names'][b]}"
        angle = angles[:, index]
        values[f"torsion:{label}:minus"] = ((angle < 0) & (angle > -2 * np.pi / 3)).astype(float)
        values[f"torsion:{label}:plus"] = ((angle >= 0) & (angle < 2 * np.pi / 3)).astype(float)
        values[f"torsion:{label}:trans"] = (np.abs(angle) >= 2 * np.pi / 3).astype(float)
    return values


def population_descriptors(xyz, definition, flags, hydration=None):
    molecular = internal_descriptors(xyz, definition)
    if hydration is not None:
        molecular["hydration:contacts"] = np.asarray(hydration, dtype=float)
    result = {}
    for label, flag in (("upper", True), ("lower", False)):
        for key, values in molecular.items():
            selected = values[np.asarray(flags) == flag]
            if not len(selected):
                raise ValueError("Local relaxation requires both source leaflets")
            if key.startswith("shape:"):
                for q, value in zip(
                    (10, 50, 90), np.percentile(selected, [10, 50, 90]), strict=True
                ):
                    result[f"{key}:{label}:q{q}"] = float(value)
            else:
                result[f"{key}:{label}"] = float(selected.mean())
    return result


@lru_cache(maxsize=256)
def _intrinsic_bounds(elements, bonds):
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    covalent = np.array([table.GetRcovalent(element) / 10 for element in elements])
    vdw = np.array([table.GetRvdw(element) / 10 for element in elements])
    adjacency = [set() for _ in elements]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    nonbonded = []
    for a, element in enumerate(elements):
        if element == "H":
            continue
        reached, frontier = {a}, {a}
        for _ in range(3):
            frontier = {c for b in frontier for c in adjacency[b]} - reached
            reached |= frontier
        nonbonded.extend(
            (a, b) for b in range(a + 1, len(elements)) if elements[b] != "H" and b not in reached
        )
    return covalent, vdw, tuple(nonbonded)


def validate_intrinsic_geometry(xyz, elements, bonds):
    """Broad covalent/overlap guards; not a zero-force or energy-minimum test."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (len(elements), 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid intrinsic molecular coordinates")
    covalent, vdw, pairs = _intrinsic_bounds(tuple(elements), tuple(bonds))
    a, b = np.asarray(bonds, dtype=int).T
    ratio = np.linalg.norm(xyz[a] - xyz[b], axis=1) / (covalent[a] + covalent[b])
    if np.any((ratio < 0.65) | (ratio > 1.45)):
        raise ValueError("Extracted lipid has an implausible covalent distance")
    if pairs:
        a, b = np.asarray(pairs, dtype=int).T
        distance = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        if np.any(distance < 0.55 * (vdw[a] + vdw[b])):
            raise ValueError("Extracted lipid has a severe internal heavy-atom overlap")


def source_orientation(xyz, selection):
    """Describe source geometry without rejecting an individual folded molecule.

    The population source gate decides acceptable outliers. The builder decides
    whether a particular conformer supplies a usable placement direction.
    """
    from gmxbuilder.modules.membrane.lipid_orientation import LipidOrientation

    xyz = np.asarray(xyz, dtype=float)
    polar, tails = np.asarray(selection["polar"]), np.asarray(selection["tails"])
    head, tail = xyz[polar].mean(axis=0), xyz[tails].mean(axis=0)
    vector = head - tail
    return LipidOrientation(
        polar, tails, head, tail, vector, max(float(np.linalg.norm(vector)), 1e-12)
    )


def canonical_coordinates(xyz, definition):
    """Set a reference frame without discarding a valid folded conformation."""
    xyz = np.asarray(xyz, dtype=float).copy()
    xyz -= xyz[definition["anchor"]]
    axis = xyz[definition["polar"]].mean(axis=0) - xyz[definition["tails"]].mean(axis=0)
    if np.linalg.norm(axis) < 1e-8:
        axis = xyz[definition["heavy"]][np.argmax(np.linalg.norm(xyz[definition["heavy"]], axis=1))]
    axis /= np.linalg.norm(axis)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(axis, target)
    cosine = float(np.dot(axis, target))
    if cosine < -1 + 1e-12:
        return xyz @ np.diag([1.0, -1.0, -1.0])
    if np.linalg.norm(cross) < 1e-12:
        return xyz
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3) + skew + skew @ skew / (1 + cosine)
    return xyz @ rotation.T

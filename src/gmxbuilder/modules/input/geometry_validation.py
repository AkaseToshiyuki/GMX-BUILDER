"""Conservative geometry checks against pinned standard-residue templates.

These are rejection guards, not refinement restraints or quality scores.
The reference coordinates come from the installed, pinned PDBFixer package.
No OpenMM context, minimization or dynamics is created here.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import distribution
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from gmxbuilder.core.chemistry import is_hydrogen

# Close ring bonds absent from the repair eligibility parent trees.
_CLOSURES = {
    "PRO": [("CD", "N")],
    "HIS": [("NE2", "CD2")],
    "PHE": [("CZ", "CE2")],
    "TYR": [("CZ", "CE2")],
    "TRP": [("CE2", "CD2"), ("CH2", "CZ3")],
}
# Explicitly ordered neighbours: signs are compared to reference geometry,
# rather than equating all L residues with one CIP R/S label (CYS differs).
_CHIRAL = {"CA": ("N", "C", "CB")}
_PLANAR = {
    "ARG": ("NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "GLN": ("CG", "CD", "OE1", "NE2"),
    "GLU": ("CG", "CD", "OE1", "OE2"),
    "HIS": ("CG", "ND1", "CE1", "NE2", "CD2"),
    "PHE": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TYR": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TRP": ("CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"),
}


@lru_cache(maxsize=32)
def reference(resname):
    path = distribution("pdbfixer").locate_file(f"pdbfixer/templates/{resname}.pdb")
    coords = {}
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coords[line[12:16].strip()] = (
                np.array([float(line[i : i + 8]) for i in (30, 38, 46)]) / 10
            )
    return coords


def _angle(a, b, c):
    left, right = a - b, c - b
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right) / norm, -1, 1)))) if norm else 0.0


def protein_geometry_issues(structure):
    from gmxbuilder.modules.input.protein_repair import _SIDECHAIN_PARENTS, _protein_residue_groups

    issues = []
    for key, indices in _protein_residue_groups(structure).items():
        chain, resid, resname = key
        observed = {structure.atom_names[i].strip(): i for i in indices}
        ref = reference(resname)
        bonds = [("N", "CA"), ("CA", "C"), ("C", "O")]
        bonds += list(_SIDECHAIN_PARENTS[resname].items()) + _CLOSURES.get(resname, [])

        def issue(detail):
            issues.append(
                {
                    "code": "protein_geometry",
                    "chain": chain,
                    "resid": resid,
                    "message": f"{chain or '<unnamed>'}:{resid} {resname}: {detail}",
                }
            )

        for atom, index in observed.items():
            if atom in ref and structure.elements[index].upper() != atom[0]:
                issue(f"atom {atom} has element {structure.elements[index]!r}; expected {atom[0]}")
        neighbours = {}
        for a, b in bonds:
            if not all(x in observed and x in ref for x in (a, b)):
                continue
            expected = float(np.linalg.norm(ref[a] - ref[b]))
            actual = float(
                np.linalg.norm(
                    structure.coordinates[observed[a]] - structure.coordinates[observed[b]]
                )
            )
            # Deliberately broad relative limits identify gross damage, rather
            # than reject experimental deviations from an ideal structure.
            if not 0.6 * expected <= actual <= 1.5 * expected:
                issue(f"invalid {a}-{b} distance {actual:.4f} nm (reference {expected:.4f} nm)")
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
        for centre, adjacent in neighbours.items():
            for a, c in combinations(adjacent, 2):
                expected = _angle(ref[a], ref[centre], ref[c])
                actual = _angle(*(structure.coordinates[observed[x]] for x in (a, centre, c)))
                if abs(expected - actual) > 35:
                    issue(
                        f"invalid {a}-{centre}-{c} angle {actual:.1f} degrees "
                        f"(reference {expected:.1f})"
                    )
        planar = _PLANAR.get(resname, ())
        if planar and all(atom in observed for atom in planar):
            xyz = np.array([structure.coordinates[observed[atom]] for atom in planar])
            centred = xyz - xyz.mean(axis=0)
            normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
            # A broad 0.4 Angstrom deviation rejects gross sp2/ring damage.
            if np.max(np.abs(centred @ normal)) > 0.04:
                issue("non-planar aromatic or conjugated group; review its geometry")
        chiral = dict(_CHIRAL)
        if resname == "ILE":
            chiral["CB"] = ("CA", "CG1", "CG2")
        if resname == "THR":
            chiral["CB"] = ("CA", "OG1", "CG2")
        for centre, adjacent in chiral.items():
            if not all(x in observed and x in ref for x in (centre, *adjacent)):
                continue
            expected = float(np.linalg.det([ref[x] - ref[centre] for x in adjacent]))
            actual = float(
                np.linalg.det(
                    [
                        structure.coordinates[observed[x]] - structure.coordinates[observed[centre]]
                        for x in adjacent
                    ]
                )
            )
            if expected * actual <= 0 or abs(actual) < 0.2 * abs(expected):
                issue(
                    f"inverted or degenerate stereochemistry at {centre}; confirm residue identity"
                )
    return issues


def coincident_atom_issues(structure):
    heavy = [
        i
        for i in range(structure.num_atoms)
        if not is_hydrogen(structure.atom_names[i], structure.elements[i])
    ]
    if not heavy:
        return []
    # 0.06 nm is below ordinary heavy-atom covalent distances; this catches
    # impossible overlaps without treating close contacts as a force field.
    distances, neighbours = cKDTree(structure.coordinates[heavy]).query(
        structure.coordinates[heavy], k=2, distance_upper_bound=0.06
    )
    pairs = sorted(
        {
            tuple(sorted((i, int(j))))
            for i, (d, j) in enumerate(zip(distances[:, 1], neighbours[:, 1], strict=True))
            if np.isfinite(d)
        }
    )
    return [
        {
            "code": "coincident_atoms",
            "message": f"Overlapping heavy atoms: {structure.chain_ids[heavy[a]]}:"
            f"{structure.resids[heavy[a]]} "
            f"{structure.atom_names[heavy[a]]} and {structure.chain_ids[heavy[b]]}:"
            f"{structure.resids[heavy[b]]} {structure.atom_names[heavy[b]]} "
            "are less than 0.06 nm apart",
        }
        for a, b in pairs[:50]
    ]


def added_atom_clashes(structure, added):
    """Check new-new and new-old severe nonbonded overlaps with graph exclusions."""
    from rdkit import Chem

    from gmxbuilder.modules.input.protein_repair import _SIDECHAIN_PARENTS, _protein_residue_groups

    adjacency = {i: set() for i in range(structure.num_atoms)}
    groups = _protein_residue_groups(structure)
    previous = {}
    for (chain, _resid, resname), indices in groups.items():
        atoms = {structure.atom_names[i]: i for i in indices}
        bonds = [("N", "CA"), ("CA", "C"), ("C", "O")]
        bonds += list(_SIDECHAIN_PARENTS[resname].items()) + _CLOSURES.get(resname, [])
        for a, b in bonds:
            if a in atoms and b in atoms:
                adjacency[atoms[a]].add(atoms[b])
                adjacency[atoms[b]].add(atoms[a])
        if chain in previous and "N" in atoms:
            a, b = previous[chain], atoms["N"]
            if np.linalg.norm(structure.coordinates[a] - structure.coordinates[b]) <= 0.2:
                adjacency[a].add(b)
                adjacency[b].add(a)
        if "C" in atoms:
            previous[chain] = atoms["C"]
    table = Chem.GetPeriodicTable()
    radii = []
    for element in structure.elements:
        try:
            radii.append(table.GetRvdw(element.title()) / 10)
        except RuntimeError:
            radii.append(0.0)
    tree = cKDTree(structure.coordinates)
    max_radius = max(radii, default=0)
    problems = []
    for i in added:
        excluded = {i}
        boundary = {i}
        # Exclude 1-2, 1-3 and 1-4 pairs from this nonbonded diagnostic.
        for _ in range(3):
            boundary = (
                set().union(*(adjacency[j] for j in boundary)) - excluded if boundary else set()
            )
            excluded.update(boundary)
        for j in tree.query_ball_point(structure.coordinates[i], radii[i] + max_radius):
            if j in excluded or is_hydrogen(structure.atom_names[j], structure.elements[j]):
                continue
            # Metals require a coordination model rather than organic vdW rejection.
            if structure.elements[j].upper() not in {"C", "N", "O", "S", "P", "F", "CL", "BR", "I"}:
                continue
            distance = float(np.linalg.norm(structure.coordinates[i] - structure.coordinates[j]))
            # 1 Angstrom of vdW overlap is a gross-clash guard, not an energy model.
            if distance < radii[i] + radii[j] - 0.10:
                problems.append(
                    f"new {structure.atom_names[i]} at {structure.chain_ids[i]}:"
                    f"{structure.resids[i]} clashes with {structure.atom_names[j]} at "
                    f"{structure.chain_ids[j]}:{structure.resids[j]} ({distance:.3f} nm)"
                )
                if len(problems) >= 20:
                    return problems
    return problems

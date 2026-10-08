"""Improve RTP bootstrap tail directions without distorting covalent geometry.

An arbitrary rotation about a branch root preserves distances *within* that
branch, but changes bond angles at its attachment. Rotate about the attachment
bond instead; exact graph mapping establishes that this is a non-ring C–C
single bond, and cutting it must isolate the complete moving component.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation

from gmxbuilder.geometry.molecular_identity import reference_mappings


def _single_carbon_bonds(smiles, names, rtp):
    from rdkit import Chem

    from gmxbuilder.modules.membrane.lipid_orientation import atom_element

    positions = {name: i for i, name in enumerate(names)}
    edges = tuple((positions[a], positions[b]) for a, b in rtp["bonds"])
    elements = tuple(atom_element(name) for name in names)
    molecule = Chem.MolFromSmiles(smiles)
    mappings = reference_mappings(smiles, elements, edges)
    permitted = []
    for mapping in mappings:
        permitted.append(
            {
                frozenset((names[mapping[b.GetBeginAtomIdx()]], names[mapping[b.GetEndAtomIdx()]]))
                for b in molecule.GetBonds()
                if b.GetBondType() == Chem.BondType.SINGLE
                and not b.IsInRing()
                and b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 6
            }
        )
    # A symmetric graph must not let the chosen mapping change which bonds
    # are allowed to rotate. Ambiguous permissions conservatively intersect.
    return set.intersection(*permitted)


def _moving_component(adjacency, root, first):
    moving, stack = set(), [first]
    while stack:
        atom = stack.pop()
        if atom in moving:
            continue
        moving.add(atom)
        for neighbour in adjacency[atom]:
            if atom == first and neighbour == root:
                continue
            if neighbour == root:
                return None  # Another path closes a ring across the cut.
            stack.append(neighbour)
    return moving


def align_tail_torsions(coords, names, rtp, *, smiles):
    """Choose bounded single-bond torsions; a downward tail is a preference.

    Packing must accommodate the resulting molecule. We never bend its
    attachment angle merely to achieve an idealized parallel-tail picture.
    The caller independently checks identity and steric validity before use.
    """
    positions = {name: i for i, name in enumerate(names)}
    adjacency = {name: set() for name in names}
    for a, b in rtp["bonds"]:
        adjacency[a].add(b)
        adjacency[b].add(a)
    permitted = _single_carbon_bonds(smiles, names, rtp)
    for root, first, pattern, spread in (
        ("C21", "C22", r"C2(\d+)", 1),
        ("C31", "C32", r"C3(\d+)", -1),
        ("C1F", "C2F", r"C(\d+)F", 1),
        ("C3S", "C4S", r"C(\d+)S", -1),
    ):
        if frozenset((root, first)) not in permitted:
            continue
        moving = _moving_component(adjacency, root, first)
        if not moving:
            continue
        terminals = [(int(m[1]), n) for n in moving if (m := re.fullmatch(pattern, n))]
        if not terminals:
            continue
        terminal = positions[max(terminals)[1]]
        indices = np.array(sorted(positions[n] for n in moving))
        fixed = np.array([i for i in range(len(names)) if i not in set(indices)])
        origin = coords[positions[root]]
        axis = coords[positions[first]] - origin
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        axis = axis / length
        # Small opposite XY biases separate the preferred tail directions during packing.
        target = np.array([0.12 * spread, 0.10 * spread, -1.0])
        target /= np.linalg.norm(target)
        best, best_score = coords[indices].copy(), -np.inf
        # Include the original torsion and deterministic alternatives. Both
        # atoms of the connecting bond lie on the rotation axis, so every
        # covalent bond length and bond angle is preserved, including C3S.
        for degrees in (0, 60, -60, 120, -120, 180):
            candidate = coords.copy()
            candidate[indices] = (
                Rotation.from_rotvec(axis * np.deg2rad(degrees)).apply(coords[indices] - origin)
                + origin
            )
            clearance = np.linalg.norm(
                candidate[indices, None, :] - candidate[None, fixed, :], axis=2
            ).min()
            # 0.05 nm screens near-coincident atoms; full steric validation remains separate.
            if clearance < 0.05:
                continue
            direction = candidate[terminal] - origin
            score = float(np.dot(direction, target) / max(np.linalg.norm(direction), 1e-12))
            if score > best_score:
                best, best_score = candidate[indices], score
        coords[indices] = best
    return coords

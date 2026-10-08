"""Read chain positions and geometry from a molecular graph, never a tail summary."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=512)
def _chain_records(smiles: str) -> tuple:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ()

    def carbon_path(root, blocked=()):
        path = []
        current = root
        while True:
            if current.IsInRing() or current.GetIdx() in path:
                return []
            path.append(current.GetIdx())
            following = [
                atom
                for atom in current.GetNeighbors()
                if atom.GetAtomicNum() == 6 and atom.GetIdx() not in (*blocked, *path)
            ]
            if not following:
                return path
            if len(following) != 1:
                return []
            current = following[0]

    carbonyls = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            for bond in atom.GetBonds()
        )
    }
    candidates = []
    for index in sorted(carbonyls):
        root = molecule.GetAtomWithIdx(index)
        links = [
            bond.GetOtherAtom(root)
            for bond in root.GetBonds()
            if bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetOtherAtom(root).GetAtomicNum() in (7, 8)
        ]
        if not links:
            continue
        linkage = "amide" if any(a.GetAtomicNum() == 7 for a in links) else "ester/acyl"
        path = carbon_path(root)
        # Amino-acid substituents (e.g. lysyl-PG) are polar headgroup chemistry,
        # not an extra fatty-acyl tail. The carbonyl root itself is allowed.
        if path and all(
            neighbour.GetAtomicNum() in (1, 6)
            for index in path[1:]
            for neighbour in molecule.GetAtomWithIdx(index).GetNeighbors()
        ):
            candidates.append((linkage, path))
        for nitrogen in (a for a in links if a.GetAtomicNum() == 7):
            # The sphingoid C1 hydroxymethyl is adjacent to the N-bearing C2.
            for c2 in nitrogen.GetNeighbors():
                if c2.GetAtomicNum() != 6 or c2.GetIdx() in carbonyls:
                    continue
                for c1 in c2.GetNeighbors():
                    if c1.GetAtomicNum() != 6:
                        continue
                    if sum(a.GetAtomicNum() == 6 for a in c1.GetNeighbors()) == 1 and any(
                        a.GetAtomicNum() == 8 for a in c1.GetNeighbors()
                    ):
                        candidates.append(("sphingoid", carbon_path(c1)))
    for oxygen in molecule.GetAtoms():
        if oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2:
            continue
        neighbours = list(oxygen.GetNeighbors())
        if not all(a.GetAtomicNum() == 6 and a.GetIdx() not in carbonyls for a in neighbours):
            continue
        for root in neighbours:
            path = carbon_path(root)
            # Ether alkyl chains contain no extra heteroatom substituents;
            # exclude sugar and glycerol fragments traversed from the other side.
            if path and all(
                a.GetAtomicNum() in (1, 6) or a.GetIdx() == oxygen.GetIdx()
                for index in path
                for a in molecule.GetAtomWithIdx(index).GetNeighbors()
            ):
                candidates.append(("ether", path))
    result = []
    seen = set()
    for linkage, path in candidates:
        if len(path) < 6 or tuple(path) in seen:
            continue
        seen.add(tuple(path))
        double_bonds = []
        for position, (first, second) in enumerate(zip(path, path[1:]), 1):
            bond = molecule.GetBondBetweenAtoms(first, second)
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = str(bond.GetStereo()).removeprefix("STEREO")
                double_bonds.append((position, stereo if stereo in ("E", "Z") else "?"))
        if linkage == "ether" and any(position == 1 for position, _ in double_bonds):
            linkage = "vinyl-ether"
        result.append((linkage, len(path), tuple(double_bonds)))
    return tuple(sorted(result))


def chain_identity(smiles: str) -> dict:
    """Return explicit chain annotations without inventing sn assignments.

    Acyl numbering includes the carbonyl C1; ether/sphingoid numbering starts
    at the attachment/backbone C1. Unrecognized, branched and cyclic structures
    retain the complete SMILES and are not guessed from nominal tail tuples.
    """
    chains = []
    for linkage, length, bonds in _chain_records(str(smiles)):
        label = f"{length}:{len(bonds)}"
        if bonds:
            label += "(" + ",".join(f"{position}{geometry}" for position, geometry in bonds) + ")"
        chains.append(
            {
                "linkage": linkage,
                "length": length,
                "label": label,
                "double_bonds": [{"position": p, "geometry": g} for p, g in bonds],
            }
        )
    return {
        "chains": chains,
        "chain_label": " + ".join(f"{c['linkage']} {c['label']}" for c in chains)
        or "See full molecular structure",
        "chain_annotation_scope": "Detected linear chains; no sn assignment or full-identity claim",
    }

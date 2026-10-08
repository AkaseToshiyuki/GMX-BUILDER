"""Coordinate checks against a declared molecular graph, independent of names."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def itp_graph(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Read one explicit-atom ITP; retain atom order and all covalent edges."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    names, elements, edges = [], [], []
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[] ").strip()
            continue
        fields = line.split()
        if section == "atoms":
            if len(fields) < 8 or int(fields[0]) != len(names) + 1:
                raise ValueError("ITP requires contiguous atoms with explicit masses")
            mass = float(fields[7])
            # Restrict the mass-based fallback to H through I, with a 0.5 Da match tolerance.
            atomic_number = min(range(1, 54), key=lambda z: abs(table.GetAtomicWeight(z) - mass))
            if abs(table.GetAtomicWeight(atomic_number) - mass) > 0.5:
                raise ValueError(f"Cannot identify element from ITP mass {mass}")
            names.append(fields[4])
            elements.append(table.GetElementSymbol(atomic_number))
        elif section in {"bonds", "constraints"}:
            edges.append((int(fields[0]) - 1, int(fields[1]) - 1))
    if not names or not edges or len(set(names)) != len(names):
        raise ValueError("ITP lacks a unique explicit molecular graph")
    if any(i == j or min(i, j) < 0 or max(i, j) >= len(names) for i, j in edges):
        raise ValueError("ITP contains an invalid covalent edge")
    return tuple(names), tuple(elements), tuple(sorted(set(edges)))


def whole_molecule(coordinates, bonds, box) -> np.ndarray:
    """Unwrap along bonds, never about a global atom-wise periodic cut."""
    xyz = np.asarray(coordinates, dtype=float)
    lattice = np.asarray(box, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Molecular coordinates are not finite Nx3 values")
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("Invalid periodic box")
    if abs(float(np.linalg.det(lattice))) < 1e-12:
        raise ValueError("Singular periodic box")
    inverse = np.linalg.inv(lattice)
    adjacency = [[] for _ in xyz]
    for i, j in bonds:
        if i == j or min(i, j) < 0 or max(i, j) >= len(xyz):
            raise ValueError("Invalid molecular bond index")
        adjacency[i].append(j)
        adjacency[j].append(i)
    result = xyz.copy()
    visited = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j in adjacency[i]:
            delta = (xyz[j] - xyz[i]) @ inverse
            delta -= np.rint(delta)
            proposed = result[i] + delta @ lattice
            if j in visited:
                # A closed bond cycle must return to the same image within 1e-6 nm.
                if not np.allclose(result[j], proposed, atol=1e-6, rtol=0):
                    raise ValueError("Periodic molecular graph has an inconsistent cycle")
                continue
            result[j] = proposed
            visited.add(j)
            pending.append(j)
    if len(visited) != len(xyz):
        raise ValueError("Molecular graph is disconnected")
    return result


@lru_cache(maxsize=512)
def reference_mappings(smiles: str, elements: tuple[str, ...], bonds: tuple):
    """Map reference heavy atoms by connectivity and bonded hydrogen counts."""
    from rdkit import Chem

    reference = Chem.MolFromSmiles(smiles)
    if reference is None:
        raise ValueError("Invalid reference SMILES")
    heavy = [i for i, symbol in enumerate(elements) if symbol != "H"]
    if len(heavy) != reference.GetNumAtoms():
        raise ValueError("Reference and coordinate heavy-atom counts disagree")
    index = {atom: i for i, atom in enumerate(heavy)}
    actual = Chem.RWMol()
    for i in heavy:
        atom = Chem.Atom(elements[i])
        atom.SetNoImplicit(True)
        actual.AddAtom(atom)
    hydrogen_counts = [0 for _ in elements]
    for i, j in bonds:
        if i in index and j in index:
            actual.AddBond(index[i], index[j], Chem.BondType.SINGLE)
        elif elements[j] == "H":
            hydrogen_counts[i] += 1
        elif elements[i] == "H":
            hydrogen_counts[j] += 1
    query = Chem.RWMol(reference)
    for atom in query.GetAtoms():
        # Encode bonded-H count as a temporary graph colour before searching.
        # Filtering it only after enumeration exhausts the symmetry budget on
        # equivalent phosphate oxygens in PIP3, despite an exact valid graph.
        # These isotope fields exist only in the matching graphs, never in the
        # chemical reference or saved topology/coordinates.
        atom.SetIsotope(reference.GetAtomWithIdx(atom.GetIdx()).GetTotalNumHs() + 1)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for i, atom in enumerate(actual.GetAtoms()):
        atom.SetIsotope(hydrogen_counts[heavy[i]] + 1)
    for bond in query.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    source, target = query.GetMol(), actual.GetMol()
    source.UpdatePropertyCache(strict=False)
    target.UpdatePropertyCache(strict=False)
    matches = target.GetSubstructMatches(source, uniquify=False, maxMatches=512)
    if len(matches) == 512:
        raise ValueError("Molecular graph mapping exceeds the exhaustive symmetry limit")
    valid = tuple(
        tuple(heavy[j] for j in match)
        for match in matches
        if all(
            reference.GetAtomWithIdx(i).GetTotalNumHs() == hydrogen_counts[heavy[j]]
            for i, j in enumerate(match)
        )
    )
    if not valid:
        raise ValueError("Reference cannot be mapped onto the coordinate bond graph")
    return valid


def validate_stereochemistry(smiles, elements, bonds, coordinates) -> dict:
    """Reject inverted/undefined specified carbon centres and E/Z geometry.

    This validates a declared identity, not its external scientific provenance
    or the force-field coefficients. Unspecified phosphorus centres must not
    invent stereochemical priorities at remote, specified carbon centres.
    """
    from rdkit import Chem

    reference = Chem.MolFromSmiles(smiles)
    if reference is None:
        raise ValueError("Invalid reference SMILES")
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (len(elements), 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid molecular coordinate array")
    centres = {
        a.GetIdx(): a.GetProp("_CIPCode")
        for a in reference.GetAtoms()
        if a.GetAtomicNum() == 6 and a.HasProp("_CIPCode")
    }
    double_bonds = {
        b.GetIdx(): b.GetStereo()
        for b in reference.GetBonds()
        if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
    }
    failures = []
    for mapping in reference_mappings(str(smiles), tuple(elements), tuple(bonds)):
        molecule = Chem.Mol(reference)
        Chem.RemoveStereochemistry(molecule)
        conformer = Chem.Conformer(reference.GetNumAtoms())
        conformer.Set3D(True)
        points = xyz[list(mapping)]
        for i, point in enumerate(points):
            conformer.SetAtomPosition(i, tuple(point * 10.0))
        molecule.AddConformer(conformer)
        Chem.AssignStereochemistryFrom3D(molecule, replaceExistingTags=True)
        for atom in molecule.GetAtoms():
            if atom.GetIdx() not in centres:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
        for bond in molecule.GetBonds():
            if bond.GetIdx() not in double_bonds:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        problems = []
        for i, expected in centres.items():
            atom = molecule.GetAtomWithIdx(i)
            actual = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "undefined"
            neighbours = sorted(n.GetIdx() for n in atom.GetNeighbors())
            arms = points[neighbours]
            vectors = arms - points[i] if len(arms) == 3 else arms[:3] - arms[3]
            norms = np.linalg.norm(vectors, axis=1)
            volume = abs(float(np.linalg.det(vectors / np.maximum(norms[:, None], 1e-12))))
            # A numerical degeneracy guard, not a target tetrahedral angle.
            if actual != expected or volume < 0.05:
                problems.append(
                    f"atom {mapping[i]}: expected {expected}, got {actual}; volume={volume:.4g}"
                )
        for i, expected in double_bonds.items():
            if molecule.GetBondWithIdx(i).GetStereo() != expected:
                problems.append(f"reference bond {i}: E/Z configuration differs")
        if not problems:
            return {
                "passed": True,
                "carbon_centres": len(centres),
                "double_bonds": len(double_bonds),
                "mapping": list(mapping),
            }
        failures.append(problems)
    detail = min(failures, key=len)
    raise ValueError("Coordinate stereochemistry mismatch: " + "; ".join(detail))

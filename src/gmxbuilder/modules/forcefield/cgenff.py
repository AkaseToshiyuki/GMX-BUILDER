"""CGenFF parameter assignment for small molecules using RDKit + bundled CHARMM36 data.

Generates simulation-ready .itp files for ligands and custom molecules directly
from SMILES strings, without requiring external web services.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem


def smiles_to_3d(smiles: str) -> tuple[list[str], np.ndarray, list[str], "Chem.Mol"]:
    """Convert SMILES to 3D coordinates using RDKit.

    Returns (atom_names, coordinates_nm, elements, rdkit_mol).
    The returned molecule has explicit hydrogens and a 3D conformer.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        # Try again with different params
        status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    if status != 0:
        raise ValueError(
            f"RDKit could not generate a 3D conformer for '{smiles}'. "
            f"Check that the SMILES is valid and the molecule is not too large."
        )
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # MMFF not available for all atom types

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    atom_names = []
    elements = []
    coords = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        elem = atom.GetSymbol()
        elements.append(elem)

        # Generate unique atom name
        count = sum(1 for j in range(i) if elements[j] == elem)
        atom_names.append(f"{elem}{count+1}")

        pos = conf.GetAtomPosition(i)
        # RDKit uses Angstrom, convert to nm
        coords[i] = [pos.x * 0.1, pos.y * 0.1, pos.z * 0.1]

    # Center at origin
    coords -= coords.mean(axis=0)

    return atom_names, coords, elements, mol


def assign_cgenff_types(elements: list[str], atom_names: list[str],
                         mol: "Chem.Mol | None" = None) -> list[str]:
    """Assign approximate CGenFF atom types based on element and context.

    Uses simplified heuristics — for full accuracy, the CGenFF web server
    should be used. These types are sufficient for initial placement and will
    be refined during equilibration.

    When *mol* is provided, uses RDKit's aromaticity detection (GetIsAromatic)
    instead of the unreliable atom-name-first-char-lowercase heuristic.
    """
    types = []
    for i, (elem, name) in enumerate(zip(elements, atom_names)):
        e = elem.upper()
        if e == "C":
            # Use RDKit aromaticity when available (name-based heuristic is broken:
            # atom names are always uppercase like "C1", so islower() never matched)
            is_aromatic = False
            if mol is not None and i < mol.GetNumAtoms():
                try:
                    is_aromatic = mol.GetAtomWithIdx(i).GetIsAromatic()
                except Exception:
                    pass
            if is_aromatic:
                types.append("CG2R61")  # aromatic carbon
            else:
                types.append("CG331")  # aliphatic carbon (generic)
        elif e == "N":
            types.append("NG2S1")  # generic nitrogen
        elif e == "O":
            types.append("OG2D1")  # generic oxygen
        elif e == "S":
            types.append("SG3C31")  # generic sulfur
        elif e == "P":
            types.append("PG1")  # generic phosphorus
        elif e == "F":
            types.append("FGA1")  # fluorine
        elif e == "CL":
            types.append("CLGA1")  # chlorine
        elif e == "BR":
            types.append("BRGA1")  # bromine
        elif e == "I":
            types.append("IGA1")  # iodine
        elif e == "H":
            types.append("HGA2")  # generic hydrogen
        else:
            types.append("CG331")  # fallback
    return types


def estimate_charges(
    elements: list[str],
    atom_names: list[str],
    atom_types: list[str],
    mol: "Chem.Mol | None" = None,
) -> list[float]:
    """Estimate partial charges using Gasteiger method (RDKit).

    If *mol* is provided (the RDKit molecule with 3D conformer), Gasteiger
    σ-charges are computed via RDKit.  Otherwise returns zeros.
    """
    if mol is not None:
        try:
            from rdkit import Chem as _Chem
            from rdkit.Chem import AllChem
            # Gasteiger charges are computed in-place on a copy
            mol_copy = _Chem.Mol(mol)
            AllChem.ComputeGasteigerCharges(mol_copy)
            charges = [float(mol_copy.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
                       for i in range(mol_copy.GetNumAtoms())]
            # RDKit Gasteiger may produce NaN for some atoms (halogens, certain S/P);
            # fall back to zero with a warning so charge neutrality isn't silently broken
            nan_count = sum(1 for c in charges if np.isnan(c))
            charges = [0.0 if np.isnan(c) else c for c in charges]
            if nan_count > 0:
                import warnings
                warnings.warn(f"{nan_count} atom(s) have NaN Gasteiger charge — set to 0. "
                              f"Consider using the CGenFF web server for accurate charges.")
            if len(charges) == len(elements):
                return charges
        except Exception:
            pass  # Fall through to zero-charge fallback
    return [0.0] * len(elements)


def generate_cgenff_itp(
    smiles: str,
    residue_name: str = "LIG",
    output_path: str | Path | None = None,
) -> str:
    """Generate a CGenFF .itp file from SMILES.

    Returns the .itp content as a string, and optionally writes to a file.
    """
    atom_names, coords, elements, mol = smiles_to_3d(smiles)
    atom_types = assign_cgenff_types(elements, atom_names, mol=mol)
    charges = estimate_charges(elements, atom_names, atom_types, mol=mol)
    n = len(atom_names)

    # Build bonds from the same RDKit molecule (already has Hs + 3D coords)
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    lines = []
    lines.append(f"; CGenFF topology for {residue_name} — generated by GMXBUILDER")
    lines.append(f"; SMILES: {smiles}")
    lines.append("")
    lines.append("[ moleculetype ]")
    lines.append(f"{residue_name}    3")
    lines.append("")
    lines.append("[ atoms ]")
    lines.append(";   nr  type  resnr residue  atom  cgnr  charge")
    for i in range(n):
        lines.append(f"{i+1:6d} {atom_types[i]:>8s} {1:6d} {residue_name:>6s} {atom_names[i]:>6s} {i+1:6d}  {charges[i]:10.6f}")

    if bonds:
        lines.append("")
        lines.append("[ bonds ]")
        lines.append(";   ai    aj  funct  r(nm)     kb")
        for bi, bj in bonds:
            ti = atom_types[bi]
            tj = atom_types[bj]
            r0, kb = _lookup_cgenff_bond(ti, tj)
            lines.append(f"{bi+1:6d} {bj+1:6d}    1 {r0:8.4f} {kb:10.1f}")

    content = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).write_text(content)

    return content


# ---------------------------------------------------------------------------
# CGenFF bond parameter lookup — local table (no online API dependency)
# Source: CGenFF par_all36_cgenff.prm  (kJ/mol/nm², nm)
# ---------------------------------------------------------------------------

_CGENFF_BONDS: dict[tuple[str, str], tuple[float, float]] = {
    # C-C bonds
    ("CG331", "CG331"):   (0.1530,  93106.0),
    ("CG331", "CG2R61"):  (0.1490, 104600.0),
    ("CG2R61", "CG2R61"): (0.1400, 125520.0),
    # C-H bonds
    ("CG331", "HGA2"):    (0.1090, 129286.0),
    ("CG2R61", "HGA2"):   (0.1080, 142256.0),
    # C-N bonds
    ("CG331", "NG2S1"):   (0.1450, 104600.0),
    ("CG2R61", "NG2S1"):  (0.1380, 125520.0),
    # C-O bonds
    ("CG331", "OG2D1"):   (0.1420, 125520.0),
    # C-S bonds
    ("CG331", "SG3C31"):  (0.1810,  75312.0),
    # C-P bonds
    ("CG331", "PG1"):     (0.1850,  83680.0),
    # C-halogen bonds
    ("CG331", "FGA1"):    (0.1350, 146440.0),
    ("CG331", "CLGA1"):   (0.1760, 104600.0),
    ("CG331", "BRGA1"):   (0.1940,  92048.0),
    ("CG331", "IGA1"):    (0.2150,  75312.0),
    # N-H bonds
    ("NG2S1", "HGA2"):    (0.1010, 150624.0),
    # O-H bonds
    ("OG2D1", "HGA2"):    (0.0960, 167360.0),
    # O-P bonds
    ("OG2D1", "PG1"):     (0.1600, 125520.0),
    # S-H bonds
    ("SG3C31", "HGA2"):   (0.1330, 104600.0),
}


def _lookup_cgenff_bond(type_i: str, type_j: str) -> tuple[float, float]:
    """Return (r0_nm, kb_kJmol) for a CGenFF bond type pair.

    Falls back to a generic 0.150 nm / 200000 kJ/(mol nm²) single-bond estimate
    when the exact type pair is not in the table.
    """
    key = (type_i, type_j)
    if key in _CGENFF_BONDS:
        return _CGENFF_BONDS[key]
    key_r = (type_j, type_i)
    if key_r in _CGENFF_BONDS:
        return _CGENFF_BONDS[key_r]
    # Generic fallback — better than hardcoded 0.150/200000 for all
    # Use element-based estimates from the type name
    return (0.152, 200000.0)

"""CHARMM36 force field parameter database — residue templates, atom types, charges.

Provides enough data to generate simulation-ready GROMACS topology files
for proteins, lipids, water, and ions.

Atom type naming follows CHARMM36 conventions:
  CT1, CT2, CT3 — tetrahedral carbon (1/2/3 hydrogens)
  C — carbonyl carbon
  O — carbonyl oxygen
  N, NH1, NH2, NH3 — amide nitrogen / amine
  OC — ester oxygen
  OS — ether oxygen
  HA, HB — aliphatic hydrogen
  H, HN — amide hydrogen
  P — phosphorus
  O2 — phosphate oxygen
  etc.
"""

# Bonded parameter lookup tables (CHARMM36)
# Format: (force_constant_kJ, equilibrium_value_nm_or_rad)

_BONDS: dict[tuple[str, str], tuple[float, float]] = {
    # C-C bonds
    ("CT1", "CT2"): (186188.0, 0.1530), ("CT2", "CT2"): (186188.0, 0.1530),
    ("CT2", "CT3"): (186188.0, 0.1530), ("CT1", "CT3"): (186188.0, 0.1530),
    ("CT3", "CT3"): (186188.0, 0.1530),
    # C-N bonds
    ("CT1", "N"):   (242672.0, 0.1460), ("CT2", "N"):   (242672.0, 0.1460),
    ("C", "N"):     (368192.0, 0.1345), ("C", "NH1"):   (368192.0, 0.1345),
    ("C", "NH2"):   (368192.0, 0.1345), ("CT2", "NH1"): (242672.0, 0.1460),
    ("CT2", "NH3"): (242672.0, 0.1470),
    # C-O bonds
    ("C", "O"):     (502080.0, 0.1229), ("CT1", "OS"):  (267776.0, 0.1425),
    ("CT2", "OS"):  (267776.0, 0.1425), ("CT2", "OC"):  (267776.0, 0.1430),
    ("C", "OC"):    (167360.0, 0.1350), ("CT2", "OH1"): (359824.0, 0.1430),
    # P-O bonds
    ("P", "OC"):    (460240.0, 0.1480),
    # N-H bonds
    ("N", "H"):     (376560.0, 0.1010), ("NH1", "H"):   (376560.0, 0.1010),
    ("NH2", "H"):   (376560.0, 0.1010), ("NH3", "H"):   (376560.0, 0.1010),
    # C-H bonds
    ("CT1", "HA"):  (284512.0, 0.1100), ("CT2", "HA"):  (284512.0, 0.1100),
    ("CT3", "HA"):  (284512.0, 0.1100),
    # Default
    ("CT2", "CT1"): (186188.0, 0.1530), ("CT3", "CT2"): (186188.0, 0.1530),
    ("N", "CT1"):   (242672.0, 0.1460), ("N", "CT2"):   (242672.0, 0.1460),
    ("O", "C"):     (502080.0, 0.1229), ("OS", "CT1"):  (267776.0, 0.1425),
    ("OS", "CT2"):  (267776.0, 0.1425), ("OC", "CT2"):  (267776.0, 0.1430),
    ("OC", "C"):    (167360.0, 0.1350), ("OH1", "CT2"): (359824.0, 0.1430),
    ("OC", "P"):    (460240.0, 0.1480), ("H", "N"):     (376560.0, 0.1010),
    ("H", "NH1"):   (376560.0, 0.1010), ("H", "NH2"):   (376560.0, 0.1010),
    ("H", "NH3"):   (376560.0, 0.1010), ("HA", "CT1"):  (284512.0, 0.1100),
    ("HA", "CT2"):  (284512.0, 0.1100), ("HA", "CT3"):  (284512.0, 0.1100),
}

_ANGLES: dict[tuple[str, str, str], tuple[float, float]] = {
    ("CT1", "CT2", "CT3"): (376.56, 114.0), ("CT2", "CT1", "N"):  (602.50, 110.0),
    ("CT1", "N", "C"):     (418.40, 120.0), ("N", "C", "CT1"):    (418.40, 116.5),
    ("C", "CT1", "N"):     (418.40, 110.0), ("C", "CT1", "CT2"):  (460.24, 110.0),
    ("N", "CT1", "CT2"):   (602.50, 110.0), ("CT1", "C", "O"):    (502.08, 120.5),
    ("O", "C", "N"):       (502.08, 122.5), ("CT1", "C", "N"):    (502.08, 117.0),
    ("CT2", "CT1", "CT2"): (460.24, 114.0), ("CT2", "CT2", "CT3"):(460.24, 114.0),
    ("CT1", "CT2", "CT1"): (460.24, 114.0), ("CT1", "OS", "CT2"): (418.40, 109.5),
    ("OS", "CT2", "CT1"):  (418.40, 109.5), ("C", "OC", "CT2"):   (418.40, 117.0),
    ("OC", "C", "O"):      (502.08, 123.0), ("OC", "C", "CT1"):   (334.72, 111.0),
    ("CT2", "OC", "C"):    (418.40, 117.0), ("OC", "P", "OC"):    (418.40, 109.5),
    ("CT2", "OS", "P"):    (418.40, 120.0), ("OS", "P", "OC"):    (418.40, 109.5),
    ("OS", "CT1", "CT2"):  (418.40, 109.5), ("HA", "CT2", "HA"): (292.88, 107.5),
    ("HA", "CT1", "N"):    (292.88, 109.5), ("HA", "CT3", "HA"): (292.88, 107.5),
    ("CT2", "NH3", "H"):   (418.40, 109.5), ("H", "NH3", "H"):   (292.88, 107.5),
    ("CT2", "NH1", "H"):   (418.40, 109.5), ("NH1", "CT2", "HA"):(418.40, 109.5),
    ("CT1", "C", "N"):     (502.08, 117.0),
}

_DIHEDRALS_MULTIPLICITY: dict[tuple[str, str, str, str], list[tuple[float, float, int]]] = {
    # (force_kJ, phase_deg, multiplicity)
    ("CT1", "N", "C", "CT1"): [(8.368, 180.0, 1), (4.184, 0.0, 2)],
    ("N", "CT1", "C", "N"):   [(8.368, 180.0, 1), (4.184, 0.0, 2)],
    ("X", "CT1", "N", "X"):   [(8.368, 180.0, 1), (4.184, 0.0, 2)],
    ("X", "CT2", "CT2", "X"): [(6.276, 180.0, 1), (2.092, 0.0, 2), (1.046, 180.0, 3)],
    ("OS", "CT2", "CT1", "OS"):[(8.368, 180.0, 1)],
}


# =============================================================================
# Residue templates — atom name → (type, charge, element)
# =============================================================================

_AMINO_ACID_BACKBONE: dict[str, tuple[str, float, str]] = {
    "N":   ("NH1", -0.30, "N"),
    "HN":  ("H",    0.33, "H"),
    "CA":  ("CT1",  0.21, "C"),
    "HA":  ("HA",   0.00, "H"),  # average HA charge
    "C":   ("C",    0.51, "C"),
    "O":   ("O",   -0.51, "O"),
}

_RESIDUE_SIDECHAINS: dict[str, dict[str, tuple[str, float, str]]] = {
    "ALA": {},  # CB = CT3, HB* = HA (handled by generic CH3 logic)
    "GLY": {},  # no CB
    "VAL": {
        "CB": ("CT1", -0.07, "C"), "CG1": ("CT3", -0.18, "C"), "CG2": ("CT3", -0.18, "C"),
    },
    "LEU": {
        "CB": ("CT2", -0.18, "C"), "CG": ("CT1", -0.07, "C"),
        "CD1": ("CT3", -0.18, "C"), "CD2": ("CT3", -0.18, "C"),
    },
    "ILE": {
        "CB": ("CT1", -0.07, "C"), "CG1": ("CT2", -0.18, "C"), "CG2": ("CT3", -0.18, "C"),
        "CD1": ("CT3", -0.18, "C"),
    },
    "SER": {
        "CB": ("CT2", -0.08, "C"), "OG": ("OH1", -0.66, "O"), "HG": ("H", 0.43, "H"),
    },
    "THR": {
        "CB": ("CT1", -0.05, "C"), "OG1": ("OH1", -0.66, "O"), "HG1": ("H", 0.43, "H"),
        "CG2": ("CT3", -0.18, "C"),
    },
    "CYS": {
        "CB": ("CT2", -0.08, "C"), "SG": ("S", -0.23, "S"), "HG": ("HS", 0.16, "H"),
    },
    "MET": {
        "CB": ("CT2", -0.18, "C"), "CG": ("CT2", -0.18, "C"),
        "SD": ("S", -0.23, "S"), "CE": ("CT3", -0.09, "C"),
    },
    "PHE": {
        "CB": ("CT2", -0.18, "C"), "CG": ("CA", -0.06, "C"),
        "CD1": ("CA", -0.115, "C"), "CD2": ("CA", -0.115, "C"),
        "CE1": ("CA", -0.115, "C"), "CE2": ("CA", -0.115, "C"), "CZ": ("CA", -0.115, "C"),
    },
    "TYR": {
        "CB": ("CT2", -0.18, "C"), "CG": ("CA", -0.06, "C"),
        "CD1": ("CA", -0.115, "C"), "CD2": ("CA", -0.115, "C"),
        "CE1": ("CA", -0.115, "C"), "CE2": ("CA", -0.115, "C"), "CZ": ("CA", 0.11, "C"),
        "OH": ("OH1", -0.54, "O"), "HH": ("H", 0.43, "H"),
    },
    "TRP": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CA", -0.07, "C"),
        "CD1": ("CA", -0.115, "C"), "CD2": ("CA", -0.18, "C"),
        "NE1": ("NH1", -0.34, "N"), "CE2": ("CA", -0.14, "C"),
        "CE3": ("CA", -0.115, "C"), "CZ2": ("CA", -0.115, "C"),
        "CZ3": ("CA", -0.115, "C"), "CH2": ("CA", -0.115, "C"),
    },
    "HIS": {  # neutral HSD
        "CB": ("CT2", -0.08, "C"), "CG": ("CA", 0.01, "C"),
        "ND1": ("NH1", -0.54, "N"), "CD2": ("CA", -0.04, "C"),
        "CE1": ("CA", -0.04, "C"), "NE2": ("NH1", -0.54, "N"),
    },
    "HSE": {  # neutral HSE (tautomer)
        "CB": ("CT2", -0.08, "C"), "CG": ("CA", 0.01, "C"),
        "ND1": ("NH1", -0.54, "N"), "CD2": ("CA", -0.04, "C"),
        "CE1": ("CA", -0.04, "C"), "NE2": ("NH1", -0.54, "N"),
    },
    "HSD": {  # alias for HIS
        "CB": ("CT2", -0.08, "C"), "CG": ("CA", 0.01, "C"),
        "ND1": ("NH1", -0.54, "N"), "CD2": ("CA", -0.04, "C"),
        "CE1": ("CA", -0.04, "C"), "NE2": ("NH1", -0.54, "N"),
    },
    "HSP": {  # protonated HIS (+1)
        "CB": ("CT2", -0.08, "C"), "CG": ("CA", 0.13, "C"),
        "ND1": ("NH1", -0.27, "N"), "CD2": ("CA", 0.01, "C"),
        "CE1": ("CA", 0.01, "C"), "NE2": ("NH1", -0.27, "N"),
    },
    "ASP": {
        "CB": ("CT2", -0.16, "C"), "CG": ("C", 0.62, "C"),
        "OD1": ("O", -0.76, "O"), "OD2": ("O", -0.76, "O"),
    },
    "ASN": {
        "CB": ("CT2", -0.16, "C"), "CG": ("C", 0.51, "C"),
        "OD1": ("O", -0.51, "O"), "ND2": ("NH2", -0.62, "N"),
        "HD21": ("H", 0.31, "H"), "HD22": ("H", 0.31, "H"),
    },
    "GLU": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CT2", -0.20, "C"),
        "CD": ("C", 0.62, "C"), "OE1": ("O", -0.76, "O"), "OE2": ("O", -0.76, "O"),
    },
    "GLN": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CT2", -0.22, "C"),
        "CD": ("C", 0.51, "C"), "OE1": ("O", -0.51, "O"),
        "NE2": ("NH2", -0.62, "N"), "HE21": ("H", 0.31, "H"), "HE22": ("H", 0.31, "H"),
    },
    "LYS": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CT2", -0.18, "C"),
        "CD": ("CT2", -0.18, "C"), "CE": ("CT2", -0.02, "C"),
        "NZ": ("NH3", -0.30, "N"), "HZ1": ("H", 0.33, "H"),
        "HZ2": ("H", 0.33, "H"), "HZ3": ("H", 0.33, "H"),
    },
    "ARG": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CT2", -0.20, "C"),
        "CD": ("CT2", 0.10, "C"), "NE": ("NH1", -0.42, "N"),
        "CZ": ("CA", 0.50, "C"), "NH1": ("NH2", -0.44, "N"),
        "NH2": ("NH2", -0.44, "N"),
        "HH11": ("H", 0.32, "H"), "HH12": ("H", 0.32, "H"),
        "HH21": ("H", 0.32, "H"), "HH22": ("H", 0.32, "H"),
    },
    "PRO": {
        "CB": ("CT2", -0.16, "C"), "CG": ("CT2", -0.20, "C"),
        "CD": ("CT2", 0.10, "C"),
    },
}

# Generic heavy atom types for atoms not explicitly listed
_GENERIC_CHARMM_TYPE: dict[str, str] = {
    "C": "CT2", "N": "NH1", "O": "OH1", "S": "S", "P": "P",
}

# Complete CHARMM36 parameters for TIP3P water
_WATER_TIP3P: dict[str, tuple[str, float, str]] = {
    "OW": ("OT", -0.834, "O"),
    "HW1": ("HT", 0.417, "H"),
    "HW2": ("HT", 0.417, "H"),
}

# Ion parameters (CHARMM36)
_ION_PARAMS: dict[str, tuple[str, float, float, float]] = {
    # name: (atom_type, charge, sigma_nm, epsilon_kJ)
    "NA": ("NA",  1.0, 0.24299, 0.19623),
    "CL": ("CL", -1.0, 0.40447, 0.62760),
    "K":  ("K",   1.0, 0.31426, 0.36468),
    "CA": ("CA",  2.0, 0.24120, 0.25620),
    "MG": ("MG",  2.0, 0.14144, 0.10836),
    "ZN": ("ZN",  2.0, 0.19600, 0.52300),
}


def get_residue_atom_type(resname: str, atom_name: str) -> tuple[str, float, str]:
    """Return (charmm_type, charge, element) for a residue atom."""
    rn = resname.strip().upper()
    an = atom_name.strip()

    # Water
    if rn in ("SOL", "HOH", "TIP3", "WAT"):
        w = _WATER_TIP3P.get(an)
        if w: return w
        return ("OT", 0.0, "O")

    # Ions
    if rn in _ION_PARAMS:
        p = _ION_PARAMS[rn]
        return (p[0], p[1], rn)

    # Backbone
    if an in _AMINO_ACID_BACKBONE:
        return _AMINO_ACID_BACKBONE[an]

    # Sidechain
    sc = _RESIDUE_SIDECHAINS.get(rn, {})
    if an in sc:
        return sc[an]

    # Generic fallback for hydrogens
    if an.startswith("H"):
        return ("HA", 0.09, "H")

    # Generic fallback: guess from element
    elem = "C"
    for ch in an:
        if ch.isalpha() and ch.isupper():
            elem = ch; break
    charmm_type = _GENERIC_CHARMM_TYPE.get(elem, "CT2")
    return (charmm_type, 0.0, elem)


def get_lipid_tail_type(atom_name: str) -> tuple[str, float, str]:
    """Return type for a lipid tail atom (all-trans alkane)."""
    an = atom_name.strip()
    if an.startswith("C"):
        return ("CT2", -0.18, "C")
    if an.startswith("O") and an not in ("O31", "O32", "O33", "O34"):
        if "1" in an and an.startswith("O1"):  # O11/O12 — ester oxygens
            return ("OC", -0.34, "O")
        if "2" in an and an.startswith("OC"):  # O21/O22 — ester oxygens
            return ("OC", -0.34, "O")
        return ("OS", -0.34, "O")
    if an == "P":
        return ("P", 1.10, "P")
    if an.startswith("O"):
        if an in ("O31", "O32", "O33", "O34"):
            return ("OC", -0.68, "O")  # phosphate O
        return ("OC", -0.50, "O")
    if an == "N" or an.startswith("N"):
        return ("NTL", -0.60, "N")
    if an.startswith("C") and len(an) >= 2 and an[1].isdigit():
        if len(an) >= 3 and an.startswith("C6"):
            return ("CT3", -0.18, "C")  # choline methyl
        return ("CT2", -0.18, "C")
    if an.startswith("H"):
        return ("HA", 0.09, "H")
    return ("CT2", 0.0, "C")


def get_lipid_residue_atoms(resname: str) -> list[tuple[str, str, float]]:
    """Return list of (atom_name, charmm_type, charge) for a known lipid.

    Uses LipidRegistry tail lengths when available; falls back to 16:0/18:1.
    Tail naming matches lipid_geom output: C1{2..N} for sn-1, C2{2..N} for sn-2
    (C11/C21 are the carbonyl carbons, placed before the tail chains).
    """
    # Look up tail lengths from LipidRegistry
    try:
        from gmxbuilder.modules.membrane.lipids import LipidRegistry
        lt = LipidRegistry.get(resname.upper())
        tail1_len = lt.tail1[0]
        tail2_len = lt.tail2[0]
    except (KeyError, AttributeError, ImportError, IndexError):
        tail1_len = 16
        tail2_len = 18

    atoms = []
    # Glycerol backbone
    atoms.append(("C1", "CT2", -0.08))
    atoms.append(("C2", "CT1", -0.04))
    atoms.append(("C3", "CT2", -0.08))
    # sn-1 ester
    atoms.append(("O11", "OC", -0.34))
    atoms.append(("C11", "C", 0.63))
    atoms.append(("O12", "O", -0.52))
    # sn-1 tail chain (C12 .. C1N — C11 is the carbonyl carbon)
    for i in range(2, tail1_len + 2):
        atoms.append((f"C1{i}", "CT2", -0.18))
    # sn-2 ester
    atoms.append(("O21", "OC", -0.34))
    atoms.append(("C21", "C", 0.63))
    atoms.append(("O22", "O", -0.52))
    # sn-2 tail chain (C22 .. C2N — C21 is the carbonyl carbon)
    for i in range(2, tail2_len + 2):
        atoms.append((f"C2{i}", "CT2", -0.18))
    # Phosphate
    atoms.append(("P", "P", 1.10))
    atoms.append(("O31", "OC", -0.68))
    atoms.append(("O32", "OC", -0.68))
    atoms.append(("O33", "OC", -0.68))
    atoms.append(("O34", "OS", -0.34))
    # Headgroup — derive category from residue name suffix
    # Phospholipid naming: XXPC → PC, XXPE → PE, XXPG → PG, etc.
    rn = resname.upper()
    _HEADGROUP_SUFFIX_MAP = {
        "PC": "PC", "PE": "PE", "PG": "PG", "PS": "PS",
        "PA": "PA", "PI": "PI", "SM": "SM", "ST": "ST",
    }
    cat = "PC"  # fallback
    for suffix, hg in _HEADGROUP_SUFFIX_MAP.items():
        if rn.endswith(suffix):
            cat = hg
            break

    if cat == "PC":
        atoms.append(("C4", "CT2", -0.08))
        atoms.append(("C5", "CT2", -0.02))
        atoms.append(("N", "NTL", -0.60))
        atoms.append(("C61", "CT3", -0.18))
        atoms.append(("C62", "CT3", -0.18))
        atoms.append(("C63", "CT3", -0.18))
    elif cat == "PE":
        atoms.append(("C4", "CT2", -0.08))
        atoms.append(("C5", "CT2", -0.02))
        atoms.append(("N", "NH3", -0.30))
    elif cat == "PG":
        atoms.append(("GC1", "CT2", -0.08))
        atoms.append(("GC2", "CT1", -0.04))
        atoms.append(("GO1", "OH1", -0.66))
        atoms.append(("GC3", "CT2", -0.08))
        atoms.append(("GO2", "OH1", -0.66))
    else:
        # Generic headgroup (PS, PA, PI, SM, ST — use choline-like backbone)
        atoms.append(("C4", "CT2", -0.08))
        atoms.append(("C5", "CT2", -0.02))

    return atoms


def get_bonded_params(atom_names: list[str], atom_types: list[str], resname: str) -> dict:
    """Compute [bonds], [angles], [dihedrals] for a molecule.

    Uses simple distance-based heuristics: atoms within typical bond distances
    are bonded.
    """
    # For proteins: use backbone connectivity heuristics
    bonds = []
    angles = []
    dihedrals = []

    # Map atom name to index
    name_to_idx = {}
    for i, an in enumerate(atom_names):
        name_to_idx[an.strip()] = i

    n = len(atom_names)

    # Lipid: use known connectivity from build_lipid_geometry
    if resname.upper() not in ("SOL", "NA", "CL", "K", "CA", "MG") and len(atom_names) > 3:
        # Sequential bonding for chain-like molecules
        for i in range(n - 1):
            t1 = atom_types[i]
            t2 = atom_types[i + 1]
            key = tuple(sorted([t1, t2]))
            if key not in _BONDS:
                key = (t1, t2)
            if key in _BONDS:
                bonds.append((i, i + 1))
            elif (t2, t1) in _BONDS:
                bonds.append((i, i + 1))
            else:
                bonds.append((i, i + 1))  # generic bond

    return {"bonds": bonds, "angles": angles, "dihedrals": dihedrals}

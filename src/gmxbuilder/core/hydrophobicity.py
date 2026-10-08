"""Wimley–White water-to-POPC-interface whole-residue energies, kcal/mol.

Source: Stephen White laboratory, Table 1 (interface column):
https://blanco.biomol.uci.edu/hydrophobicity_scales.html
Wimley & White (1996), Nature Structural Biology 3:842–848.
These measurements are not octanol energies or a calibrated bilayer-core
potential. Their use in the orientation score remains a heuristic.
"""

WW_INTERFACE = {
    "ALA": 0.17,
    "ARG": 0.81,
    "ASN": 0.42,
    "ASP": 1.23,
    "ASH": -0.07,
    "CYS": -0.24,
    "GLN": 0.58,
    "GLU": 2.02,
    "GLH": -0.01,
    "GLY": 0.01,
    "HIS": 0.17,
    "HIP": 0.96,
    "ILE": -0.31,
    "LEU": -0.56,
    "LYS": 0.99,
    "MET": -0.23,
    "PHE": -1.13,
    "PRO": 0.45,
    "SER": 0.13,
    "THR": 0.14,
    "TRP": -1.85,
    "TYR": -0.94,
    "VAL": 0.07,
}
# Force-field spelling aliases; neutral HIS is shared by its two tautomers.
for alias, source in {
    "ASPP": "ASH",
    "GLUP": "GLH",
    "HID": "HIS",
    "HIE": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIP",
}.items():
    WW_INTERFACE[alias] = WW_INTERFACE[source]

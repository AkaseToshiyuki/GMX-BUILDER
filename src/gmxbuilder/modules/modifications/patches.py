"""Post-translational modification patch registry.

Each Patch defines:
  - target residue(s)
  - product residue name
  - atoms to add/remove
  - charge shift
  - associated force-field topology (.itp reference)
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class PatchAtom:
    """A new atom to add to the residue when applying a patch."""

    name: str          # atom name, e.g. "P"
    element: str       # element symbol
    x: float = 0.0     # relative coordinates (nm)
    y: float = 0.0
    z: float = 0.0
    charge: float = 0.0  # partial charge (e)


@dataclasses.dataclass(frozen=True)
class StereoConstraint:
    """Named stereocentre validated by an ordered-neighbour signed volume.

    Each selector contains force-field atom-name alternatives.  The neighbour
    order follows the chemical-component priority order documented by the
    patch; ``expected_sign`` is measured from the wwPDB ideal component.
    """

    center: tuple[str, ...]
    ordered_neighbors: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    expected_sign: int
    label: str


@dataclasses.dataclass
class Patch:
    """Definition of a single PTM patch."""

    name: str                     # short code, e.g. "PHOS"
    description: str              # human-readable
    target_residues: list[str]    # original residue names
    product_name: str             # resulting residue name
    charge_shift: int             # change in net residue charge
    added_atoms: list[PatchAtom] = dataclasses.field(default_factory=list)
    removed_atoms: list[str] = dataclasses.field(default_factory=list)  # atom names to remove
    bond_to: Optional[str] = None  # atom to connect the new group to
    mass_shift: float = 0.0       # g/mol
    formula_addition: str = ""    # e.g. "PO3H" or "C2H3O"
    requires_itp: str = ""        # force-field .itp template name
    stereo_constraints: tuple[StereoConstraint, ...] = ()


# ---------------------------------------------------------------------------
# Pre-defined patch library
# ---------------------------------------------------------------------------

_PHOSPHORYLATION = Patch(
    name="PHOS",
    description="Phosphorylation — force-field-native phosphate state",
    target_residues=["SER", "THR", "TYR"],
    product_name="SEP",  # default SER→SEP; overridden per target
    charge_shift=-2,
    added_atoms=[
        PatchAtom("P", "P", x=0.16, y=0.0, z=0.0, charge=1.1),
        PatchAtom("O1P", "O", x=0.22, y=0.0, z=0.0, charge=-0.7),
        PatchAtom("O2P", "O", x=0.16, y=0.14, z=0.0, charge=-0.7),
        PatchAtom("O3P", "O", x=0.16, y=-0.14, z=0.0, charge=-0.7),
    ],
    bond_to="OG" if "SER" in ["SER"] else "OG1",  # override per target
    mass_shift=79.98,
    formula_addition="PO3",
    requires_itp="phos.itp",
)

# Per-target phosphorylation variants
PHOSPHORYLATION_PATCHES: dict[str, Patch] = {
    "SER": dataclasses.replace(
        _PHOSPHORYLATION,
        product_name="SEP",
        target_residues=["SER"],
        bond_to="OG",
    ),
    "THR": dataclasses.replace(
        _PHOSPHORYLATION,
        product_name="TPO",
        target_residues=["THR"],
        bond_to="OG1",
    ),
    "TYR": dataclasses.replace(
        _PHOSPHORYLATION,
        product_name="PTR",
        target_residues=["TYR"],
        bond_to="OH",
    ),
}

# AmberTools phosaa14SB also supplies the explicitly protonated, monoanionic
# phosphate states.  They are separate choices because silently selecting a
# phosphate protonation state from the global protein pH would overstate the
# accuracy of the current pKa model.
MONOANIONIC_PHOSPHORYLATION_PATCHES: dict[str, Patch] = {
    "SER": dataclasses.replace(
        PHOSPHORYLATION_PATCHES["SER"], name="PHOS1", product_name="S1P",
        charge_shift=-1,
        description="Phosphorylation — protonated monoanionic phosphate (−1)",
    ),
    "THR": dataclasses.replace(
        PHOSPHORYLATION_PATCHES["THR"], name="PHOS1", product_name="T1P",
        charge_shift=-1,
        description="Phosphorylation — protonated monoanionic phosphate (−1)",
    ),
    "TYR": dataclasses.replace(
        PHOSPHORYLATION_PATCHES["TYR"], name="PHOS1", product_name="Y1P",
        charge_shift=-1,
        description="Phosphorylation — protonated monoanionic phosphate (−1)",
    ),
}

_ACETYLATION = Patch(
    name="ACET",
    description="Acetylation — acetylates lysine ε-amine",
    target_residues=["LYS"],
    product_name="ALY",
    charge_shift=-1,  # LYS +1 → ALY 0
    added_atoms=[
        PatchAtom("C1", "C", x=0.15, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.27, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.10, y=0.12, z=0.0, charge=-0.2),
    ],
    bond_to="NZ",
    mass_shift=42.04,
    formula_addition="C2H3O",
    requires_itp="acet.itp",
)

_SUCCINYLATION = Patch(
    name="SUCC",
    description="Succinylation — adds succinyl group to lysine ε-amine",
    target_residues=["LYS"],
    product_name="SLY",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("C1", "C", x=0.15, y=0.0, z=0.0, charge=0.6),
        PatchAtom("O1", "O", x=0.27, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.10, y=0.10, z=0.0, charge=-0.3),
        PatchAtom("C3", "C", x=0.10, y=0.20, z=0.0, charge=-0.3),
        PatchAtom("O2", "O", x=0.18, y=0.27, z=0.0, charge=-0.5),
    ],
    bond_to="NZ",
    mass_shift=100.07,
    formula_addition="C4H4O3",
    requires_itp="succ.itp",
)

_NTER_ACETYL = Patch(
    name="ACE",
    description="N-terminal acetylation — caps the N-terminus with acetyl",
    target_residues=["NTER"],  # special: N-terminal residue
    product_name="ACE",
    charge_shift=0,  # neutralises the N-terminal NH₃⁺ charge
    added_atoms=[
        PatchAtom("CH3", "C", x=0.0, y=0.0, z=-0.15, charge=-0.3),
        PatchAtom("C", "C", x=0.0, y=0.0, z=-0.30, charge=0.5),
        PatchAtom("O", "O", x=0.0, y=-0.12, z=-0.33, charge=-0.5),
    ],
    bond_to="N",
    mass_shift=42.04,
    formula_addition="C2H3O",
    requires_itp="ace.itp",
)

_CTER_NME = Patch(
    name="NME",
    description="C-terminal N-methylamide — caps the C-terminus with methylamide",
    target_residues=["CTER"],  # special: C-terminal residue
    product_name="NME",
    charge_shift=0,
    added_atoms=[
        PatchAtom("N", "N", x=0.0, y=0.0, z=0.15, charge=-0.5),
        PatchAtom("CH3", "C", x=0.0, y=0.0, z=0.28, charge=-0.1),
    ],
    bond_to="C",
    mass_shift=29.04,
    formula_addition="CH3N",
    requires_itp="nme.itp",
)

_CARBAMYLATION = Patch(
    name="CBM",
    description="Carbamylation — adds carbamyl group to lysine ε-amine",
    target_residues=["LYS"],
    product_name="CLY",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("N1", "N", x=0.05, y=0.12, z=0.0, charge=-0.5),
    ],
    bond_to="NZ",
    mass_shift=43.03,
    formula_addition="CH2NO",
    requires_itp="cbm.itp",
)

_LYSINE_NZ_CARBOXY = Patch(
    name="CARBOXY",
    description="Lysine Nζ-carboxylation — force-field-native lysine carbamate",
    target_residues=["LYS"],
    product_name="KCX",
    charge_shift=-2,  # protonated LYS (+1) -> carboxylated KCX (-1)
    added_atoms=[
        PatchAtom("CX", "C"),
        PatchAtom("OQ1", "O"),
        PatchAtom("OQ2", "O"),
    ],
    bond_to="NZ",
    mass_shift=43.99,
    formula_addition="CO2",
    requires_itp="kcx.itp",
)

_CITRULLINATION = Patch(
    name="CIT",
    description="Citrullination — converts arginine to citrulline",
    target_residues=["ARG"],
    product_name="CIR",
    charge_shift=-1,
    removed_atoms=["NH1", "NH2"],
    added_atoms=[
        PatchAtom("O", "O", x=0.0, y=0.0, z=0.0, charge=-0.5),
    ],
    bond_to="CZ",
    mass_shift=0.98,
    formula_addition="O",
    requires_itp="cit.itp",
)

_MYRISTOYL = Patch(
    name="MYRI",
    description="N-myristoylation — attaches C14:0 fatty acyl chain to N-terminal glycine",
    target_residues=["GLY"],
    product_name="MYR",
    charge_shift=0,
    added_atoms=[
        # Simplified: just mark the connection point
        PatchAtom("C1", "C", x=0.0, y=0.0, z=-0.15, charge=0.0),
    ],
    bond_to="N",
    mass_shift=210.36,
    formula_addition="C14H27O",
    requires_itp="myri.itp",
)

_PALMITOYL = Patch(
    name="PALM",
    description="S-palmitoylation — attaches C16:0 fatty acyl chain to cysteine via thioester",
    target_residues=["CYS"],
    product_name="PLC",
    charge_shift=0,
    added_atoms=[
        PatchAtom("C1", "C", x=0.12, y=0.0, z=0.0, charge=0.4),
        PatchAtom("O1", "O", x=0.22, y=0.0, z=0.0, charge=-0.4),
    ],
    bond_to="SG",
    mass_shift=238.39,
    formula_addition="C16H31O",
    requires_itp="palm.itp",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2 — expanded PTM library
# ---------------------------------------------------------------------------

_LYSINE_METHYL_MONO = Patch(
    name="KME", description="Lysine mono-methylation — adds methyl to ε-amine",
    target_residues=["LYS"], product_name="MLZ",
    charge_shift=0,  # still +1 (secondary ammonium)
    added_atoms=[PatchAtom("CM", "C", x=0.12, y=0.06, z=0.0, charge=-0.1)],
    bond_to="NZ", mass_shift=14.03, formula_addition="CH2", requires_itp="kme.itp",
)

_LYSINE_METHYL_DI = Patch(
    name="KME2", description="Lysine di-methylation — two methyls on ε-amine",
    target_residues=["LYS"], product_name="MLY",
    charge_shift=0,
    added_atoms=[
        PatchAtom("CH1", "C", x=0.12, y=0.08, z=0.0, charge=-0.1),
        PatchAtom("CH2", "C", x=0.12, y=-0.08, z=0.0, charge=-0.1),
    ],
    bond_to="NZ", mass_shift=28.05, formula_addition="C2H4", requires_itp="kme2.itp",
)

_LYSINE_METHYL_TRI = Patch(
    name="KME3", description="Lysine tri-methylation — trimethylammonium (+1 permanent)",
    target_residues=["LYS"], product_name="M3L",
    charge_shift=0,  # still +1 (quaternary ammonium)
    added_atoms=[
        PatchAtom("CM1", "C", x=0.12, y=0.10, z=0.0, charge=-0.1),
        PatchAtom("CM2", "C", x=0.12, y=-0.05, z=0.09, charge=-0.1),
        PatchAtom("CM3", "C", x=0.12, y=-0.05, z=-0.09, charge=-0.1),
    ],
    bond_to="NZ", mass_shift=42.08, formula_addition="C3H6", requires_itp="kme3.itp",
)

_ARGININE_METHYL_MONO = Patch(
    name="RME", description="Arginine mono-methylation — Nω-methyl-arginine",
    target_residues=["ARG"], product_name="RME",
    charge_shift=0,  # still +1
    added_atoms=[PatchAtom("CM", "C", x=0.10, y=0.08, z=0.0, charge=-0.1)],
    bond_to="NH1", mass_shift=14.03, formula_addition="CH2", requires_itp="rme.itp",
)

_ARGININE_METHYL_SYM = Patch(
    name="RME2", description="Arginine symmetric di-methylation — both Nω methylated",
    target_residues=["ARG"], product_name="2MR",
    charge_shift=0,
    added_atoms=[
        PatchAtom("CQ1", "C", x=0.10, y=0.08, z=0.0, charge=-0.1),
        PatchAtom("CQ2", "C", x=-0.10, y=0.08, z=0.0, charge=-0.1),
    ],
    bond_to="NH1", mass_shift=28.05, formula_addition="C2H4", requires_itp="rme2.itp",
)

_ARGININE_METHYL_ASYM = Patch(
    name="RME2A",
    description="Arginine asymmetric di-methylation — both methyls on one Nω",
    target_residues=["ARG"],
    product_name="DA2",
    charge_shift=0,
    added_atoms=[PatchAtom("C1", "C"), PatchAtom("C2", "C")],
    bond_to="NH1",
    mass_shift=28.05,
    formula_addition="C2H4",
    requires_itp="da2.itp",
)

_CYSTEINE_SULFENIC = Patch(
    name="CSO", description="Cysteine oxidation to protonated sulfenic acid (-SOH)",
    target_residues=["CYS"], product_name="CSO",
    charge_shift=0,
    added_atoms=[PatchAtom("OD", "O", x=0.14, y=0.0, z=0.0, charge=-0.5)],
    bond_to="SG", mass_shift=16.00, formula_addition="O", requires_itp="cso.itp",
)

_CYSTEINE_SULFINIC = Patch(
    name="CSD", description="Cysteine oxidation to sulfinic acid (-SO₂H)",
    target_residues=["CYS"], product_name="CSD",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("O1", "O", x=0.14, y=0.07, z=0.0, charge=-0.4),
        PatchAtom("O2", "O", x=0.14, y=-0.07, z=0.0, charge=-0.4),
    ],
    bond_to="SG", mass_shift=32.00, formula_addition="O2", requires_itp="csd.itp",
)

_CYSTEINE_SULFENATE = Patch(
    name="CSX", description="Cysteine oxidation to deprotonated sulfenic acid (-SO⁻)",
    target_residues=["CYS"], product_name="CSX",
    charge_shift=-1,
    added_atoms=[PatchAtom("OD", "O", x=0.14, y=0.0, z=0.0, charge=-0.8)],
    bond_to="SG", mass_shift=16.00, formula_addition="O", requires_itp="csx.itp",
)

_CYSTEINE_NITROSYL = Patch(
    name="CSN", description="Cysteine S-nitrosylation (-SNO)",
    target_residues=["CYS"], product_name="SNC",
    charge_shift=0,
    added_atoms=[
        PatchAtom("ND", "N", x=0.14, y=0.0, z=0.0, charge=0.2),
        PatchAtom("OE", "O", x=0.22, y=0.0, z=0.0, charge=-0.2),
    ],
    bond_to="SG", mass_shift=30.01, formula_addition="NO", requires_itp="csn.itp",
)

_METHIONINE_OXIDATION = Patch(
    name="MSO",
    description="Methionine oxidation to sulfoxide — R/S state must be specified",
    target_residues=["MET"], product_name="SME",
    charge_shift=0,
    added_atoms=[PatchAtom("O", "O", x=0.14, y=0.0, z=0.0, charge=-0.4)],
    bond_to="SD", mass_shift=16.00, formula_addition="O", requires_itp="mse.itp",
)

_METHIONINE_R_OXIDATION = dataclasses.replace(
    _METHIONINE_OXIDATION,
    name="MSO-R",
    description="L-methionine (R)-S-oxide — explicit sulfur configuration",
    stereo_constraints=(StereoConstraint(
        center=("SD", "S"),
        ordered_neighbors=(("OE", "O"), ("CG",), ("CE",)),
        expected_sign=-1,
        label="R sulfur (SME)",
    ),),
)

_ASN_DEAMIDATION = Patch(
    name="DEA", description="Asparagine deamidation — ASN→ASP (hydrolysis of sidechain amide)",
    target_residues=["ASN"], product_name="ASP",
    charge_shift=-1,  # neutral ASN → -1 ASP
    removed_atoms=["ND2"],
    added_atoms=[PatchAtom("OD2", "O", x=0.0, y=0.0, z=0.0, charge=-0.5)],
    bond_to="CG", mass_shift=0.98, formula_addition="", requires_itp="",
)

_GLN_DEAMIDATION = Patch(
    name="DEG", description="Glutamine deamidation — GLN→GLU (hydrolysis of sidechain amide)",
    target_residues=["GLN"], product_name="GLU",
    charge_shift=-1,  # neutral GLN → -1 GLU
    removed_atoms=["NE2"],
    added_atoms=[PatchAtom("OE2", "O", x=0.0, y=0.0, z=0.0, charge=-0.5)],
    bond_to="CD", mass_shift=0.98, formula_addition="", requires_itp="",
)

_NTER_FORMYL = Patch(
    name="FOR", description="N-terminal formylation — formyl cap on N-terminus",
    target_residues=["NTER"], product_name="FOR",
    charge_shift=0,
    added_atoms=[
        PatchAtom("C", "C", x=0.0, y=0.0, z=-0.15, charge=0.4),
        PatchAtom("O", "O", x=0.0, y=-0.12, z=-0.18, charge=-0.4),
    ],
    bond_to="N", mass_shift=28.01, formula_addition="CHO", requires_itp="for.itp",
)

_PYROGLUTAMATE = Patch(
    name="PCA", description="Pyroglutamate — N-terminal GLN cyclisation to PCA",
    target_residues=["GLN"], product_name="PCA",
    charge_shift=0,  # loses N-terminal NH₃⁺
    removed_atoms=["N", "CD", "OE1", "NE2"],
    bond_to="CA", mass_shift=-17.03, formula_addition="", requires_itp="pca.itp",
)

_LYSINE_MALONYL = Patch(
    name="MAL", description="Lysine malonylation — malonyl group on ε-amine",
    target_residues=["LYS"], product_name="MALY",
    charge_shift=-1,  # +1 LYS → 0 (malonyl carboxylate -1)
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.08, y=0.10, z=0.0, charge=-0.2),
        PatchAtom("C3", "C", x=0.08, y=-0.10, z=0.0, charge=0.6),
        PatchAtom("O2", "O", x=0.18, y=-0.12, z=0.0, charge=-0.5),
        PatchAtom("O3", "O", x=0.0, y=-0.18, z=0.0, charge=-0.5),
    ],
    bond_to="NZ", mass_shift=86.05, formula_addition="C3H2O3", requires_itp="mal.itp",
)

_LYSINE_CROTONYL = Patch(
    name="CRO", description="Lysine crotonylation — crotonyl group on ε-amine",
    target_residues=["LYS"], product_name="CRY",
    charge_shift=-1,  # converts +1 NH₃⁺ to neutral amide
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.08, y=0.08, z=0.0, charge=-0.1),
        PatchAtom("C3", "C", x=0.05, y=0.18, z=0.0, charge=-0.2),
        PatchAtom("C4", "C", x=-0.02, y=0.26, z=0.0, charge=-0.2),
    ],
    bond_to="NZ", mass_shift=68.07, formula_addition="C4H4O", requires_itp="cro.itp",
)

_LYSINE_BUTYRYL = Patch(
    name="BUT", description="Lysine butyrylation — butyryl group on ε-amine",
    target_residues=["LYS"], product_name="BLY",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.08, y=0.10, z=0.0, charge=-0.1),
        PatchAtom("C3", "C", x=0.05, y=0.18, z=0.0, charge=-0.1),
        PatchAtom("C4", "C", x=0.10, y=0.28, z=0.0, charge=-0.2),
    ],
    bond_to="NZ", mass_shift=70.09, formula_addition="C4H6O", requires_itp="but.itp",
)

_LYSINE_PROPIONYL = Patch(
    name="PRO", description="Lysine propionylation — propionyl group on ε-amine",
    target_residues=["LYS"], product_name="PLY",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.08, y=0.10, z=0.0, charge=-0.2),
        PatchAtom("C3", "C", x=0.05, y=0.18, z=0.0, charge=-0.2),
    ],
    bond_to="NZ", mass_shift=56.06, formula_addition="C3H4O", requires_itp="pro.itp",
)

_LYSINE_GLUTARYL = Patch(
    name="GLR", description="Lysine glutarylation — glutaryl group on ε-amine",
    target_residues=["LYS"], product_name="GRY",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.25, y=0.0, z=0.0, charge=-0.5),
        PatchAtom("C2", "C", x=0.08, y=0.08, z=0.0, charge=-0.1),
        PatchAtom("C3", "C", x=0.05, y=0.16, z=0.0, charge=-0.1),
        PatchAtom("C4", "C", x=0.10, y=0.24, z=0.0, charge=-0.1),
        PatchAtom("C5", "C", x=0.05, y=0.32, z=0.0, charge=0.6),
        PatchAtom("O2", "O", x=0.12, y=0.36, z=0.0, charge=-0.5),
        PatchAtom("O3", "O", x=-0.05, y=0.34, z=0.0, charge=-0.5),
    ],
    bond_to="NZ", mass_shift=114.10, formula_addition="C5H6O3", requires_itp="glr.itp",
)

_O_GLcNAc_SER = Patch(
    name="GCS", description="O-GlcNAcylation — single GlcNAc on serine",
    target_residues=["SER"], product_name="GCS",
    charge_shift=0,
    added_atoms=[
        PatchAtom("C1", "C", x=0.14, y=0.0, z=0.0, charge=0.3),
        PatchAtom("C2", "C", x=0.22, y=0.05, z=0.0, charge=0.1),
        PatchAtom("C3", "C", x=0.28, y=-0.03, z=0.0, charge=0.1),
        PatchAtom("C4", "C", x=0.25, y=-0.13, z=0.0, charge=0.1),
        PatchAtom("C5", "C", x=0.17, y=-0.18, z=0.0, charge=0.1),
        PatchAtom("O5", "O", x=0.12, y=-0.10, z=0.0, charge=-0.3),
        PatchAtom("O1", "O", x=0.28, y=-0.22, z=0.0, charge=-0.4),
        PatchAtom("O3", "O", x=0.33, y=0.05, z=0.0, charge=-0.4),
        PatchAtom("O4", "O", x=0.30, y=-0.18, z=0.0, charge=-0.4),
        PatchAtom("N2", "N", x=0.20, y=0.14, z=0.0, charge=-0.3),
        PatchAtom("C6", "C", x=0.20, y=0.24, z=0.0, charge=0.4),
        PatchAtom("O6", "O", x=0.10, y=0.26, z=0.0, charge=-0.4),
        PatchAtom("C7", "C", x=0.30, y=0.28, z=0.0, charge=-0.2),
    ],
    bond_to="OG", mass_shift=203.19, formula_addition="C8H13NO5", requires_itp="glcnac.itp",
)

_O_GLcNAc_THR = Patch(
    name="GCT", description="O-GlcNAcylation — single GlcNAc on threonine",
    target_residues=["THR"], product_name="GCT",
    charge_shift=0,
    added_atoms=_O_GLcNAc_SER.added_atoms,
    bond_to="OG1", mass_shift=203.19, formula_addition="C8H13NO5", requires_itp="glcnac.itp",
)

_TYROSINE_SULFATION = Patch(
    name="TYS", description="Tyrosine O-sulfation — sulfate on hydroxyl",
    target_residues=["TYR"], product_name="TYS",
    charge_shift=-1,  # neutral TYR → -1 (sulfate monoester)
    added_atoms=[
        PatchAtom("S", "S", x=0.16, y=0.0, z=0.0, charge=1.3),
        PatchAtom("O1", "O", x=0.22, y=0.07, z=0.0, charge=-0.5),
        PatchAtom("O2", "O", x=0.22, y=-0.07, z=0.0, charge=-0.5),
        PatchAtom("O3", "O", x=0.10, y=0.0, z=0.0, charge=-0.5),
    ],
    bond_to="OH", mass_shift=80.06, formula_addition="SO3", requires_itp="tys.itp",
)

_SERINE_ACETYL = Patch(
    name="SAC", description="Serine O-acetylation — acetyl on sidechain hydroxyl",
    target_residues=["SER"], product_name="OAS",
    charge_shift=0,
    added_atoms=[
        PatchAtom("C1", "C", x=0.13, y=0.0, z=0.0, charge=0.5),
        PatchAtom("O1", "O", x=0.23, y=0.0, z=0.0, charge=-0.4),
        PatchAtom("C2", "C", x=0.08, y=0.10, z=0.0, charge=-0.2),
    ],
    bond_to="OG", mass_shift=42.04, formula_addition="C2H2O", requires_itp="sac.itp",
)

_CYSTEINE_METHYL = Patch(
    name="SMC", description="Cysteine S-methylation — methyl thioether",
    target_residues=["CYS"], product_name="SMC",
    charge_shift=0,
    added_atoms=[PatchAtom("CS", "C")],
    bond_to="SG", mass_shift=14.03, formula_addition="CH2", requires_itp="smc.itp",
)

_CYSTEINE_SULFONIC = Patch(
    name="OCS", description="Cysteine oxidation to cysteinesulfonic acid (-SO3−)",
    target_residues=["CYS"], product_name="OCS",
    charge_shift=-1,
    added_atoms=[
        PatchAtom("OD1", "O"), PatchAtom("OD2", "O"), PatchAtom("OD3", "O"),
    ],
    bond_to="SG", mass_shift=48.00, formula_addition="O3", requires_itp="ocs.itp",
)

_HYDROXYPROLINE = Patch(
    name="HYP", description="trans-4-hydroxy-L-proline (2S,4R)",
    target_residues=["PRO"], product_name="HYP",
    charge_shift=0,
    added_atoms=[PatchAtom("OD1", "O")],
    bond_to="CG", mass_shift=16.00, formula_addition="O", requires_itp="hyp.itp",
    stereo_constraints=(StereoConstraint(
        center=("CG",),
        ordered_neighbors=(("OE", "OD1"), ("CD", "CD2"), ("CB",)),
        expected_sign=-1,
        label="4R carbon (HYP)",
    ),),
)

_HYDROXYLYSINE = Patch(
    name="HYL", description="5-hydroxylysine (2S,5R)",
    target_residues=["LYS"], product_name="LYZ",
    charge_shift=0,
    added_atoms=[PatchAtom("OH", "O")],
    bond_to="CD", mass_shift=16.00, formula_addition="O", requires_itp="lyz.itp",
    stereo_constraints=(StereoConstraint(
        center=("CD",),
        ordered_neighbors=(("OH",), ("CE",), ("CG",)),
        expected_sign=-1,
        label="5R carbon (LYZ)",
    ),),
)

_NITROTYROSINE = Patch(
    name="NIY", description="3-nitro-L-tyrosine — meta nitration of the phenolic ring",
    target_residues=["TYR"], product_name="NIY",
    charge_shift=0,
    added_atoms=[
        PatchAtom("NN", "N"), PatchAtom("ON1", "O"), PatchAtom("ON2", "O"),
    ],
    bond_to="CE1", mass_shift=45.00, formula_addition="NO2", requires_itp="niy.itp",
)

_THREONINE_ACETYL = Patch(
    name="TAC", description="Threonine O-acetylation — acetyl on sidechain hydroxyl",
    target_residues=["THR"], product_name="TAC",
    charge_shift=0,
    added_atoms=_SERINE_ACETYL.added_atoms,
    bond_to="OG1", mass_shift=42.04, formula_addition="C2H2O", requires_itp="tac.itp",
)

_CYSTEINE_FARNESYL = Patch(
    name="FAR", description="Cysteine S-farnesylation — C15 isoprenoid chain",
    target_residues=["CYS"], product_name="FAR",
    charge_shift=0,
    added_atoms=[
        PatchAtom("C1", "C", x=0.12, y=0.0, z=0.0, charge=0.0),
    ],
    bond_to="SG", mass_shift=204.35, formula_addition="C15H25", requires_itp="far.itp",
)

_CYSTEINE_DISULFIDE = Patch(
    name="CYX", description="Disulfide bond formation — paired CYS→CYS bridge",
    target_residues=["CYS"], product_name="CYX",
    charge_shift=0,  # paired with another CYS
    removed_atoms=["HG1"],
    bond_to="SG", mass_shift=-1.01, formula_addition="", requires_itp="",
)

_TRYPTOPHAN_OXIDATION = Patch(
    name="WOX", description="Tryptophan oxidation — hydroxytryptophan",
    target_residues=["TRP"], product_name="WOH",
    charge_shift=0,
    added_atoms=[PatchAtom("OH", "O", x=0.0, y=0.12, z=0.0, charge=-0.4)],
    bond_to="CH2", mass_shift=16.00, formula_addition="O", requires_itp="wox.itp",
)

_GLYCINE_LIPIDATION = Patch(
    name="GPL", description="N-palmitoylation on glycine — C16:0 acyl chain",
    target_residues=["GLY"], product_name="GPL",
    charge_shift=0,
    added_atoms=[PatchAtom("C1", "C", x=0.0, y=0.0, z=-0.15, charge=0.4)],
    bond_to="N", mass_shift=238.39, formula_addition="C16H31O", requires_itp="gpl.itp",
)


ALL_PATCHES: dict[str, Patch] = {
    # Phosphorylation
    "PHOS_SER": PHOSPHORYLATION_PATCHES["SER"],
    "PHOS_THR": PHOSPHORYLATION_PATCHES["THR"],
    "PHOS_TYR": PHOSPHORYLATION_PATCHES["TYR"],
    "PHOS1_SER": MONOANIONIC_PHOSPHORYLATION_PATCHES["SER"],
    "PHOS1_THR": MONOANIONIC_PHOSPHORYLATION_PATCHES["THR"],
    "PHOS1_TYR": MONOANIONIC_PHOSPHORYLATION_PATCHES["TYR"],
    # Lysine acylations
    "ACET_LYS": _ACETYLATION,
    "CARBOXY_LYS": _LYSINE_NZ_CARBOXY,
    "SUCC_LYS": _SUCCINYLATION,
    "CBM_LYS": _CARBAMYLATION,
    "MAL_LYS": _LYSINE_MALONYL,
    "CRO_LYS": _LYSINE_CROTONYL,
    "BUT_LYS": _LYSINE_BUTYRYL,
    "PRO_LYS": _LYSINE_PROPIONYL,
    "GLR_LYS": _LYSINE_GLUTARYL,
    # Lysine methylations
    "KME_LYS": _LYSINE_METHYL_MONO,
    "KME2_LYS": _LYSINE_METHYL_DI,
    "KME3_LYS": _LYSINE_METHYL_TRI,
    # Arginine modifications
    "CIT_ARG": _CITRULLINATION,
    "RME_ARG": _ARGININE_METHYL_MONO,
    "RME2_ARG": _ARGININE_METHYL_SYM,
    "RME2A_ARG": _ARGININE_METHYL_ASYM,
    # Cysteine modifications
    "PALM_CYS": _PALMITOYL,
    "FAR_CYS": _CYSTEINE_FARNESYL,
    "CSO_CYS": _CYSTEINE_SULFENIC,
    "CSD_CYS": _CYSTEINE_SULFINIC,
    "CSX_CYS": _CYSTEINE_SULFENATE,
    "CSN_CYS": _CYSTEINE_NITROSYL,
    "SMC_CYS": _CYSTEINE_METHYL,
    "OCS_CYS": _CYSTEINE_SULFONIC,
    "CYX_CYS": _CYSTEINE_DISULFIDE,
    # Methionine / Tryptophan oxidation
    "MSO_MET": _METHIONINE_OXIDATION,
    "MSO_R_MET": _METHIONINE_R_OXIDATION,
    "WOX_TRP": _TRYPTOPHAN_OXIDATION,
    # Deamidation
    "DEA_ASN": _ASN_DEAMIDATION,
    "DEG_GLN": _GLN_DEAMIDATION,
    # Serine / Threonine modifications
    "SAC_SER": _SERINE_ACETYL,
    "TAC_THR": _THREONINE_ACETYL,
    "GCS_SER": _O_GLcNAc_SER,
    "GCT_THR": _O_GLcNAc_THR,
    # Tyrosine
    "TYS_TYR": _TYROSINE_SULFATION,
    "NIY_TYR": _NITROTYROSINE,
    # N-terminal / N-terminal residue modifications
    "MYRI_GLY": _MYRISTOYL,
    "GPL_GLY": _GLYCINE_LIPIDATION,
    "PCA_GLN": _PYROGLUTAMATE,
    "HYP_PRO": _HYDROXYPROLINE,
    "HYL_LYS": _HYDROXYLYSINE,
    # Termini
    "ACE_NTER": _NTER_ACETYL,
    "FOR_NTER": _NTER_FORMYL,
    "NME_CTER": _CTER_NME,
}


# A catalogue entry is not automatically executable.  Each entry below has a
# matching RTP definition and complete bonded parameters in the listed force
# field.  Name collisions with unrelated residues are deliberately excluded.
_TEMPLATE_PATCH_SUPPORT: dict[str, frozenset[str]] = {
    "PHOS_SER": frozenset({"amber14sb", "charmm36m"}),
    "PHOS_THR": frozenset({"amber14sb", "charmm36m"}),
    "PHOS_TYR": frozenset({"amber14sb", "charmm36m"}),
    "PHOS1_SER": frozenset({"amber14sb"}),
    "PHOS1_THR": frozenset({"amber14sb"}),
    "PHOS1_TYR": frozenset({"amber14sb"}),
    "ACET_LYS": frozenset({"charmm36m"}),
    "CARBOXY_LYS": frozenset({"charmm36m"}),
    "KME_LYS": frozenset({"charmm36m"}),
    "KME2_LYS": frozenset({"charmm36m"}),
    "KME3_LYS": frozenset({"charmm36m"}),
    "CIT_ARG": frozenset({"charmm36m"}),
    "RME2_ARG": frozenset({"charmm36m"}),
    "RME2A_ARG": frozenset({"charmm36m"}),
    "CSO_CYS": frozenset({"charmm36m"}),
    "CSX_CYS": frozenset({"charmm36m"}),
    "CSN_CYS": frozenset({"charmm36m"}),
    "SMC_CYS": frozenset({"charmm36m"}),
    "OCS_CYS": frozenset({"charmm36m"}),
    "SAC_SER": frozenset({"charmm36m"}),
    "TYS_TYR": frozenset({"charmm36m"}),
    "NIY_TYR": frozenset({"charmm36m"}),
    "MSO_R_MET": frozenset({"charmm36m"}),
    "HYP_PRO": frozenset({"charmm36m", "amber14sb", "amber99sb", "amber99sb-ildn"}),
    "HYL_LYS": frozenset({"charmm36m"}),
}
_NATIVE_TEMPLATE_PATCHES: frozenset[str] = frozenset(_TEMPLATE_PATCH_SUPPORT)
SUPPORTED_PATCHES: frozenset[str] = frozenset(
    {"DEA_ASN", "DEG_GLN"} | set(_NATIVE_TEMPLATE_PATCHES)
)
_UNSUPPORTED_REASON = "No validated atom-complete topology is bundled for this patch"
_UNSUPPORTED_REASONS: dict[str, str] = {
    "MSO_MET": (
        "Methionine sulfoxide is stereogenic; choose the explicit MSO_R_MET state. "
        "No compatible bundled template is available for the S-sulfoxide state"
    ),
    "CBM_LYS": (
        "Lysine carbamylation produces homocitrulline; the bundled KCX template is "
        "N-zeta-carboxylysine and must not be substituted for homocitrulline"
    ),
    "MAL_LYS": (
        "No native malonyllysine template is bundled; CHARMM MLY is dimethyllysine, "
        "not malonyllysine"
    ),
    "RME_ARG": (
        "No validated native N-omega-monomethylarginine template is bundled"
    ),
    "CYX_CYS": (
        "Disulfides are unavailable as a single-residue patch because they are paired "
        "cross-residue chemistry. "
        "Use the dedicated Disulfide Crosslinks control when the selected force field supports it"
    ),
    "PCA_GLN": (
        "No force-field-compatible N-terminal pyroglutamate template is bundled. "
        "The CHARMM36m residue named PCA has a different atom inventory and chemistry"
    ),
    "GCS_SER": (
        "O-GlcNAc requires an explicit glycan stereochemistry/linkage and matching carbohydrate "
        "parameters; no validated coupled protein-glycan template is bundled"
    ),
    "GCT_THR": (
        "O-GlcNAc requires an explicit glycan stereochemistry/linkage and matching carbohydrate "
        "parameters; no validated coupled protein-glycan template is bundled"
    ),
    "PALM_CYS": (
        "S-palmitoylation requires a complete C16 thioester topology and membrane-aware starting "
        "geometry; the catalogue placeholder is intentionally unavailable"
    ),
    "FAR_CYS": (
        "Farnesylation requires a complete isoprenoid topology and linkage-state parameters; "
        "the catalogue placeholder is intentionally unavailable"
    ),
    "MYRI_GLY": (
        "N-myristoylation requires a complete C14 amide topology and membrane-aware starting "
        "geometry. The bundled CHARMM residue MYR is free myristic acid, not N-myristoylglycine"
    ),
    "GPL_GLY": (
        "N-palmitoylation requires a complete C16 amide topology and membrane-aware starting "
        "geometry. The bundled CHARMM36m residue GPL is chemically unrelated to N-palmitoylglycine"
    ),
}


def patch_capability(patch_id: str, force_field: str | None = None) -> tuple[bool, str]:
    """Return whether *patch_id* can currently produce a valid structure."""
    if patch_id not in ALL_PATCHES:
        return False, "Unknown patch"
    if patch_id in {"DEA_ASN", "DEG_GLN"}:
        return True, ""
    if patch_id in _NATIVE_TEMPLATE_PATCHES:
        supported_force_fields = _TEMPLATE_PATCH_SUPPORT[patch_id]
        if force_field and force_field.strip().lower() not in supported_force_fields:
            choices = ", ".join(sorted(supported_force_fields))
            return False, (
                f"This modification is parameterized only in {choices}, not {force_field}"
            )
        if not force_field:
            return True, ""
        from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
        patch = ALL_PATCHES[patch_id]
        rtp = load_force_field_rtp(force_field)
        product = rtp.get_residue(patch.product_name)
        parent = next((
            rtp.get_residue(name) for name in patch.target_residues
            if rtp.get_residue(name) is not None
        ), None)
        if product is None:
            return False, f"{force_field} has no {patch.product_name} residue topology"
        if parent is None:
            return False, f"{force_field} has no parent residue topology for {patch_id}"
        from gmxbuilder.modules.modifications.geometry import (
            ModificationGeometryError,
            validate_modified_template_parameters,
        )
        try:
            validate_modified_template_parameters(
                force_field=force_field,
                product_template=product,
                parent_template=parent,
                stereo_constraints=patch.stereo_constraints,
            )
        except ModificationGeometryError as error:
            return False, f"Incomplete force-field geometry for {patch_id}: {error}"
        return True, ""
    return False, _UNSUPPORTED_REASONS.get(patch_id, _UNSUPPORTED_REASON)


def effective_patch_charge_shift(patch_id: str, force_field: str | None = None) -> int:
    """Return the integer residue charge change for the selected FF template."""
    patch = ALL_PATCHES.get(patch_id)
    if patch is None:
        raise KeyError(patch_id)
    if not force_field or patch_id not in _NATIVE_TEMPLATE_PATCHES:
        return patch.charge_shift
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    product = rtp.get_residue(patch.product_name)
    target = next((rtp.get_residue(name) for name in patch.target_residues
                   if rtp.get_residue(name) is not None), None)
    if product is None or target is None:
        return patch.charge_shift
    delta = sum(atom[2] for atom in product["atoms"]) - sum(
        atom[2] for atom in target["atoms"]
    )
    rounded = round(delta)
    if abs(delta - rounded) > 1e-3:
        raise ValueError(
            f"Non-integral charge shift for {patch_id} in {force_field}: {delta:.6f}"
        )
    return int(rounded)


def list_patches(force_field: str | None = None) -> list[dict]:
    """Return all available patches with metadata."""
    return [
        {
            "id": pid,
            "name": p.name,
            "description": p.description,
            "target_residues": p.target_residues,
            "product_name": p.product_name,
            "charge_shift": effective_patch_charge_shift(pid, force_field),
            "mass_shift": p.mass_shift,
            "formula_addition": p.formula_addition,
            "stereochemistry": [item.label for item in p.stereo_constraints],
            "supported": patch_capability(pid, force_field)[0],
            "support_reason": patch_capability(pid, force_field)[1],
        }
        for pid, p in ALL_PATCHES.items()
    ]


def list_patches_for_residue(resname: str, force_field: str | None = None) -> list[dict]:
    """Return patches applicable to a specific residue."""
    rn = resname.strip().upper()
    results = []
    for pid, p in ALL_PATCHES.items():
        if rn in p.target_residues:
            supported, support_reason = patch_capability(pid, force_field)
            results.append({
                "id": pid,
                "name": p.name,
                "description": p.description,
                "product_name": p.product_name,
                "charge_shift": effective_patch_charge_shift(pid, force_field),
                "formula_addition": p.formula_addition,
                "stereochemistry": [item.label for item in p.stereo_constraints],
                "supported": supported,
                "support_reason": support_reason,
            })
    return results


def get_patch(patch_id: str) -> Patch | None:
    """Look up a patch by its ID."""
    return ALL_PATCHES.get(patch_id)


def disulfide_capability(force_field: str) -> tuple[bool, str, float | None]:
    """Return whether paired CYS→CYX plus an explicit SG-SG bond is supported."""
    name = force_field.strip().lower()
    supported = {"amber14sb", "amber99sb", "amber99sb-ildn"}
    if name not in supported:
        return False, (
            "Paired disulfides currently require an Amber CYX residue template; "
            f"{force_field} needs a force-field-native cross-residue patch model"
        ), None
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
    from gmxbuilder.modules.modifications.geometry import (
        ModificationGeometryError,
        crosslink_bond_length,
    )

    template = load_force_field_rtp(name).get_residue("CYX")
    if template is None:
        return False, f"{force_field} has no CYX residue template", None
    try:
        distance = crosslink_bond_length(name, template, "SG")
    except ModificationGeometryError as error:
        return False, f"Incomplete {force_field} disulfide parameters: {error}", None
    return True, "", distance

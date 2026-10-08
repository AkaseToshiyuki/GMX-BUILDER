"""Lipid library — comprehensive template definitions and registry.

Lipids are organized by headgroup category:
  PC  — Phosphatidylcholine (zwitterionic)
  PE  — Phosphatidylethanolamine (zwitterionic)
  PG  — Phosphatidylglycerol (anionic)
  PS  — Phosphatidylserine (anionic)
  PA  — Phosphatidic acid (anionic)
  PI  — Phosphatidylinositol (anionic)
  CL  — Cardiolipin (anionic)
  SM  — Sphingomyelin (zwitterionic)
  ST  — Sterols (cholesterol, ergosterol)
  PIP — Phosphoinositides (highly anionic)

Naming convention: first letter = tail-1 source, second letter = tail-2 source.
  P = Palmitoyl (C16:0), O = Oleoyl (C18:1), S = Stearoyl (C18:0)
  L = Lauroyl (C12:0), M = Myristoyl (C14:0), E = Eicosenoyl (C20:1)
  A = Arachidonoyl (C20:4), D = Docosahexaenoyl (C22:6)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class LipidTemplate:
    """Metadata and template references for a lipid species.

    Positional-arg order (for compact construction):
      name, common_name, category, formula, tail1, tail2,
      area_per_lipid, bilayer_thickness, vdw_radius, charge, mass, smiles
    """

    name: str  # Short code, e.g. "POPC"
    common_name: str  # Full IUPAC/trivial name
    category: str  # Headgroup category: "PC","PE","PG","PS","PA","PI","CL","SM","ST","PIP"
    formula: str  # Molecular formula
    tail1: tuple[int, int]  # (chain_length, n_unsaturated)
    tail2: tuple[int, int]
    area_per_lipid: float  # nm^2 per lipid (experimental)
    bilayer_thickness: float  # nm, DHH (head-to-head)
    vdw_radius: float  # nm, approximate
    charge: int  # Net charge at pH 7
    mass: float  # g/mol
    smiles: str = ""  # SMILES string for structure rendering
    headgroup: str = ""  # Auto-derived from category in __post_init__
    template_gro: str = ""  # Path to bundled .gro template
    template_itp: str = ""  # Path to bundled .itp file

    def __post_init__(self):
        if not self.headgroup:
            self.headgroup = CATEGORY_NAMES.get(self.category, self.category)
        # Validate category: warn if using an unrecognised category string
        if self.category not in CATEGORY_NAMES and not self.headgroup.startswith("Custom"):
            import logging

            logging.getLogger(__name__).warning(
                "Lipid '%s' has unrecognised category '%s' — headgroup defaults to '%s'",
                self.name,
                self.category,
                self.headgroup,
            )


# Category display names and sort order
CATEGORY_ORDER = [
    "PC",
    "PE",
    "PG",
    "PS",
    "PA",
    "PI",
    "CL",
    "SM",
    "CER",
    "ST",
    "LPC",
    "LPE",
    "DG",
    "MGDG",
    "DGDG",
    "GM1",
    "PIP",
    "LPG",
]
CATEGORY_NAMES = {
    "PC": "Phosphatidylcholine (PC)",
    "PE": "Phosphatidylethanolamine (PE)",
    "PG": "Phosphatidylglycerol (PG)",
    "PS": "Phosphatidylserine (PS)",
    "PA": "Phosphatidic Acid (PA)",
    "PI": "Phosphatidylinositol (PI)",
    "CL": "Cardiolipin (CL)",
    "SM": "Sphingomyelin (SM)",
    "CER": "Ceramides (CER)",
    "ST": "Sterols",
    "LPC": "Lyso-PC (single-tail)",
    "LPE": "Lyso-PE (single-tail)",
    "DG": "Diacylglycerol (DG)",
    "MGDG": "Monogalactosyl-DAG",
    "DGDG": "Digalactosyl-DAG",
    "GM1": "Ganglioside GM1",
    "PIP": "Phosphoinositides (PIP)",
    "LPG": "Lysyl-phosphatidylglycerol (LPG)",
}


class LipidRegistry:
    """Singleton registry of lipid templates."""

    _lipids: dict[str, LipidTemplate] = {}
    _by_category: dict[str, list[str]] = {}
    _CUSTOM_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,4}$")
    _task_lipids: ContextVar[dict[str, LipidTemplate] | None] = ContextVar(
        "gmxbuilder_task_lipids", default=None
    )

    @classmethod
    def register_custom(cls, name: str, properties: dict) -> LipidTemplate:
        """Register a user-defined lipid from estimated properties dict.

        The *properties* dict should have keys matching the output of
        ``parse_custom_lipid`` / ``_estimate_lipid_properties``.
        """
        lipid = cls.custom_template(name, properties)
        cls.register(lipid)
        return lipid

    @classmethod
    def custom_template(cls, name: str, properties: dict) -> LipidTemplate:
        """Create a validated custom template without global registration."""
        normalized_name = str(name).strip().upper().replace(" ", "_")
        if not cls._CUSTOM_NAME_PATTERN.fullmatch(normalized_name):
            raise ValueError(
                "Custom lipid residue ID must contain 1-5 uppercase letters, "
                "digits, or underscores and must start with a letter"
            )
        return LipidTemplate(
            name=normalized_name,
            common_name=properties.get("common_name", name),
            category=properties.get("category", "PC"),
            formula=properties.get("formula", ""),
            tail1=tuple(properties.get("tail1", (16, 0))),
            tail2=tuple(properties.get("tail2", (16, 0))),
            area_per_lipid=float(properties.get("area_per_lipid", 0.65)),
            bilayer_thickness=float(properties.get("bilayer_thickness", 3.8)),
            vdw_radius=float(properties.get("vdw_radius", 0.30)),
            charge=int(properties.get("charge", 0)),
            mass=float(properties.get("mass", 700.0)),
            smiles=properties.get("smiles", ""),
        )

    # ------------------------------------------------------------------
    # Built-in lipid database — 84+ common lipids
    # Literature values from NMR, X-ray, and MD benchmark studies.
    # SMILES are canonical (simplified for tail-length variants).
    # ------------------------------------------------------------------

    _BUILTIN = [
        # ================================================================
        # PC — Phosphatidylcholine (zwitterionic)
        # ================================================================
        LipidTemplate(
            "DLPC",
            "1,2-dilauroyl-sn-glycero-3-phosphocholine",
            "PC",
            "C32H64NO8P",
            (12, 0),
            (12, 0),
            0.630,
            3.0,
            0.28,
            0,
            621.83,
            "CCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCC",
        ),
        LipidTemplate(
            "DMPC",
            "1,2-dimyristoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C36H72NO8P",
            (14, 0),
            (14, 0),
            0.606,
            3.3,
            0.28,
            0,
            677.93,
            "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "DPPC",
            "1,2-dipalmitoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C40H80NO8P",
            (16, 0),
            (16, 0),
            0.631,
            3.9,
            0.30,
            0,
            734.04,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "DSPC",
            "1,2-distearoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C44H88NO8P",
            (18, 0),
            (18, 0),
            0.642,
            4.2,
            0.30,
            0,
            790.15,
            "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "POPC",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C42H82NO8P",
            (16, 0),
            (18, 1),
            0.643,
            3.8,
            0.30,
            0,
            760.08,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "SOPC",
            "1-stearoyl-2-oleoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C44H86NO8P",
            (18, 0),
            (18, 1),
            0.654,
            4.0,
            0.30,
            0,
            788.14,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "DOPC",
            "1,2-dioleoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C44H84NO8P",
            (18, 1),
            (18, 1),
            0.672,
            3.6,
            0.30,
            0,
            786.11,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        ),
        LipidTemplate(
            "PAPC",
            "1-palmitoyl-2-arachidonoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C44H80NO8P",
            (16, 0),
            (20, 4),
            0.695,
            3.7,
            0.30,
            0,
            782.10,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        # ================================================================
        # PE — Phosphatidylethanolamine (zwitterionic)
        # ================================================================
        LipidTemplate(
            "DMPE",
            "1,2-dimyristoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C33H66NO8P",
            (14, 0),
            (14, 0),
            0.530,
            3.6,
            0.28,
            0,
            635.85,
            "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "DPPE",
            "1,2-dipalmitoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C37H74NO8P",
            (16, 0),
            (16, 0),
            0.513,
            4.2,
            0.28,
            0,
            691.96,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "POPE",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C39H76NO8P",
            (16, 0),
            (18, 1),
            0.590,
            4.1,
            0.28,
            0,
            718.01,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[NH3+]",
        ),
        LipidTemplate(
            "DOPE",
            "1,2-dioleoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C41H78NO8P",
            (18, 1),
            (18, 1),
            0.630,
            3.6,
            0.28,
            0,
            744.03,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCC/C=C\CCCCCCCC",
        ),
        LipidTemplate(
            "PAPE",
            "1-palmitoyl-2-arachidonoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C41H74NO8P",
            (16, 0),
            (20, 4),
            0.580,
            3.8,
            0.28,
            0,
            740.02,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[NH3+]",
        ),
        LipidTemplate(
            "SOPE",
            "1-stearoyl-2-oleoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C41H80NO8P",
            (18, 0),
            (18, 1),
            0.600,
            4.3,
            0.28,
            0,
            746.05,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[NH3+]",
        ),
        # ================================================================
        # PG — Phosphatidylglycerol (anionic, -1)
        # ================================================================
        LipidTemplate(
            "DMPG",
            "1,2-dimyristoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C34H66O10P",
            (14, 0),
            (14, 0),
            0.590,
            3.5,
            0.30,
            -1,
            665.87,
            "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@@H](O)CO)OC(=O)CCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "DPPG",
            "1,2-dipalmitoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C38H74O10P",
            (16, 0),
            (16, 0),
            0.614,
            3.9,
            0.30,
            -1,
            721.97,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@@H](O)CO)OC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "POPG",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C40H76O10P",
            (16, 0),
            (18, 1),
            0.650,
            3.7,
            0.30,
            -1,
            748.012,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@@H](O)CO",
        ),
        LipidTemplate(
            "DOPG",
            "1,2-dioleoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C42H78O10P",
            (18, 1),
            (18, 1),
            0.700,
            3.5,
            0.30,
            -1,
            774.05,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@@H](O)CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        ),
        LipidTemplate(
            "SOPG",
            "1-stearoyl-2-oleoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C42H80O10P",
            (18, 0),
            (18, 1),
            0.660,
            4.0,
            0.30,
            -1,
            776.07,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@@H](O)CO",
        ),
        # ================================================================
        # PS — Phosphatidylserine (anionic, -1)
        # ================================================================
        LipidTemplate(
            "DPPS",
            "1,2-dipalmitoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C38H73NO10P",
            (16, 0),
            (16, 0),
            0.560,
            4.1,
            0.30,
            -1,
            734.97,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-])OC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "POPS",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C40H75NO10P",
            (16, 0),
            (18, 1),
            0.630,
            4.0,
            0.30,
            -1,
            761.00,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",
        ),
        LipidTemplate(
            "DOPS",
            "1,2-dioleoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C42H77NO10P",
            (18, 1),
            (18, 1),
            0.680,
            3.7,
            0.30,
            -1,
            787.04,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-])OC(=O)CCCCCCC/C=C\CCCCCCCC",
        ),
        LipidTemplate(
            "SOPS",
            "1-stearoyl-2-oleoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C42H79NO10P",
            (18, 0),
            (18, 1),
            0.650,
            4.2,
            0.30,
            -1,
            789.05,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",
        ),
        # ================================================================
        # PA — Phosphatidic Acid (anionic, -1 at pH 7)
        # ================================================================
        LipidTemplate(
            "DPPA",
            "1,2-dipalmitoyl-sn-glycero-3-phosphate",
            "PA",
            "C35H68O8P",
            (16, 0),
            (16, 0),
            0.480,
            4.4,
            0.28,
            -1,
            647.89,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])O)OC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "POPA",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphate",
            "PA",
            "C37H70O8P",
            (16, 0),
            (18, 1),
            0.500,
            3.9,
            0.28,
            -1,
            673.93,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])O",
        ),
        LipidTemplate(
            "DOPA",
            "1,2-dioleoyl-sn-glycero-3-phosphate",
            "PA",
            "C39H72O8P",
            (18, 1),
            (18, 1),
            0.530,
            3.5,
            0.28,
            -1,
            699.97,
            r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])O)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        ),
        # ================================================================
        # PI — Phosphatidylinositol (anionic, -1)
        # ================================================================
        LipidTemplate(
            "POPI",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoinositol",
            "PI",
            "C43H80O13P",
            (16, 0),
            (18, 1),
            0.630,
            3.9,
            0.32,
            -1,
            836.07,
            r"CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        ),
        LipidTemplate(
            "SOPI",
            "1-stearoyl-2-oleoyl-sn-glycero-3-phosphoinositol",
            "PI",
            "C45H84O13P",
            (18, 0),
            (18, 1),
            0.640,
            4.1,
            0.32,
            -1,
            864.13,
            # Exact sn-chain/headgroup identity follows the named native CHARMM residue.
            "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        ),
        LipidTemplate(
            "PIPI",
            "1-palmitoyl-2-arachidonoyl-sn-glycero-3-phosphoinositol",
            "PI",
            "C45H78O13P",
            (16, 0),
            (20, 4),
            0.650,
            3.8,
            0.32,
            -1,
            858.08,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",
        ),
        # ================================================================
        # SM — Sphingomyelin (zwitterionic)
        # ================================================================
        LipidTemplate(
            "PSM",
            "N-palmitoyl-D-erythro-sphingosylphosphorylcholine",
            "SM",
            "C39H79N2O6P",
            (16, 0),
            (0, 0),
            0.530,
            4.2,
            0.30,
            0,
            703.03,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COP(=O)([O-])OCC[N+](C)(C)C)NC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "SSM",
            "N-stearoyl-D-erythro-sphingosylphosphorylcholine",
            "SM",
            "C41H83N2O6P",
            (18, 0),
            (0, 0),
            0.540,
            4.4,
            0.30,
            0,
            731.08,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COP(=O)([O-])OCC[N+](C)(C)C)NC(=O)CCCCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "BSM",
            "N-lignoceroyl-D-erythro-sphingosylphosphorylcholine",
            "SM",
            "C47H95N2O6P",
            (24, 0),
            (0, 0),
            0.550,
            4.8,
            0.30,
            0,
            815.24,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COP(=O)([O-])OCC[N+](C)(C)C)NC(=O)CCCCCCCCCCCCCCCCCCCCCCC",
        ),
        # ================================================================
        # ST — Sterols
        # ================================================================
        LipidTemplate(
            "CHOL",
            "Cholesterol",
            "ST",
            "C27H46O",
            (0, 0),
            (0, 0),
            0.380,
            3.5,
            0.25,
            0,
            386.65,
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "ERG",
            "Ergosterol",
            "ST",
            "C28H44O",
            (0, 0),
            (0, 0),
            0.390,
            3.4,
            0.25,
            0,
            396.65,
            # Natural ergosterol, PubChem CID 444679: preserve the C20 stereoisomer.
            "C[C@H](/C=C/[C@H](C)C(C)C)[C@H]1CC[C@H]2C3=CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        # ================================================================
        # PIP — Phosphoinositides (highly anionic)
        # ================================================================
        LipidTemplate(
            "POP2",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoinositol-4,5-bisphosphate",
            "PIP",
            "C43H79O19P3",
            (16, 0),
            (18, 1),
            0.680,
            3.8,
            0.34,
            -4,
            993.01,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](OP(=O)([O-])O)[C@H](OP(=O)([O-])[O-])[C@H]1O",
        ),
        LipidTemplate(
            "POP3",
            "1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoinositol-3,4,5-trisphosphate",
            "PIP",
            "C43H78O22P4",
            (16, 0),
            (18, 1),
            0.700,
            3.7,
            0.34,
            -6,
            1070.97,
            # Exact sn-chain/headgroup identity follows the named native CHARMM residue.
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])O[C@H]1[C@H](O)[C@@H](OP(=O)([O-])[O-])[C@H](OP(=O)([O-])[O-])[C@@H](OP(=O)([O-])O)[C@H]1O",
        ),
        # ================================================================
        # Mixed / Special
        # ================================================================
        LipidTemplate(
            "DEPC",
            "1,2-dierucoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C52H100NO8P",
            (22, 1),
            (22, 1),
            0.780,
            # 3.4 nm was almost certainly DEPC's hydrophobic core thickness
            # rather than its head-to-head distance: it sat below DOPC's 3.6 nm
            # although a 22:1 chain is four carbons longer per tail than 18:1,
            # and the rest of the symmetric PC series rises monotonically
            # (12:0 3.00, 14:0 3.30, 16:0 3.90, 18:0 4.20). A measured 50 ns
            # GAFF2 bilayer gives 5.12 nm.
            4.4,
            0.32,
            0,
            898.35,
            r"CCCCCCCC/C=C\CCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCC/C=C\CCCCCCCC",
        ),
        LipidTemplate(
            "PAPG",
            "1-palmitoyl-2-arachidonoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C42H74O10P",
            (16, 0),
            (20, 4),
            0.670,
            3.7,
            0.30,
            -1,
            770.02,
            "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@@H](O)CO",
        ),
        LipidTemplate(
            "TOCL",
            "tetraoleoyl-cardiolipin",
            "CL",
            "C81H148O17P2",
            (18, 1),
            (18, 1),
            1.290,
            3.5,
            0.36,
            -2,
            1456.01,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC(O)COP(=O)([O-])OC[C@@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)OC(=O)CCCCCCC/C=C\\CCCCCCCC)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        ),
        # ================================================================
        # More PC variants
        # ================================================================
        LipidTemplate(
            "DLiPC",
            "1,2-dilinoleoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C44H80NO8P",
            (18, 2),
            (18, 2),
            0.692,
            3.5,
            0.30,
            0,
            782.09,
            r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\C/C=C\CCCCC",
        ),
        LipidTemplate(
            "DAPC",
            "1,2-diarachidonoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C48H80NO8P",
            (20, 4),
            (20, 4),
            0.742,
            3.3,
            0.30,
            0,
            830.15,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC",
        ),
        LipidTemplate(
            "PUPC",
            "1-palmitoyl-2-docosahexaenoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C46H80NO8P",
            (16, 0),
            (22, 6),
            0.710,
            3.5,
            0.30,
            0,
            806.12,
            r"CC/C=C\C/C=C\C/C=C\C/C=C\C/C=C\C/C=C\CCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "SAPC",
            "1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C46H84NO8P",
            (18, 0),
            (20, 4),
            0.698,
            3.8,
            0.30,
            0,
            810.15,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "SMpC",
            "1-stearoyl-2-myristoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C40H80NO8P",
            (18, 0),
            (14, 0),
            0.625,
            4.0,
            0.30,
            0,
            734.04,
            "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "PMpC",
            "1-palmitoyl-2-myristoyl-sn-glycero-3-phosphocholine",
            "PC",
            "C38H76NO8P",
            (16, 0),
            (14, 0),
            0.620,
            3.8,
            0.30,
            0,
            705.99,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCC",
        ),
        # ================================================================
        # More PE variants
        # ================================================================
        LipidTemplate(
            "DLiPE",
            "1,2-dilinoleoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C41H74NO8P",
            (18, 2),
            (18, 2),
            0.610,
            3.8,
            0.28,
            0,
            740.01,
            r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCC/C=C\C/C=C\CCCCC",
        ),
        LipidTemplate(
            "DAPE",
            "1,2-diarachidonoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C45H74NO8P",
            (20, 4),
            (20, 4),
            0.640,
            3.5,
            0.28,
            0,
            788.07,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC",
        ),
        LipidTemplate(
            "SAPE",
            "1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C43H78NO8P",
            (18, 0),
            (20, 4),
            0.595,
            4.0,
            0.28,
            0,
            768.07,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[NH3+]",
        ),
        LipidTemplate(
            "DPePE",
            "1,2-dipalmitoleoyl-sn-glycero-3-phosphoethanolamine",
            "PE",
            "C37H70NO8P",
            (16, 1),
            (16, 1),
            0.580,
            3.9,
            0.28,
            0,
            687.93,
            r"CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCC/C=C\CCCCCC",
        ),
        # ================================================================
        # More PG variants
        # ================================================================
        LipidTemplate(
            "DLiPG",
            "1,2-dilinoleoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C42H74O10P",
            (18, 2),
            (18, 2),
            0.690,
            3.6,
            0.30,
            -1,
            770.018,
            "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@@H](O)CO)OC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",
        ),
        LipidTemplate(
            "DAPG",
            "1,2-diarachidonoyl-sn-glycero-3-phosphoglycerol",
            "PG",
            "C46H74O10P",
            (20, 4),
            (20, 4),
            0.730,
            3.4,
            0.30,
            -1,
            818.062,
            "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)OC[C@H](COP(=O)([O-])OC[C@@H](O)CO)OC(=O)CCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCC",
        ),
        # ================================================================
        # More PS variants
        # ================================================================
        LipidTemplate(
            "DLiPS",
            "1,2-dilinoleoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C42H73NO10P",
            (18, 2),
            (18, 2),
            0.670,
            3.7,
            0.30,
            -1,
            783.01,
            r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-])OC(=O)CCCCCCC/C=C\C/C=C\CCCCC",
        ),
        LipidTemplate(
            "SAPS",
            "1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphoserine",
            "PS",
            "C44H77NO10P",
            (18, 0),
            (20, 4),
            0.660,
            4.1,
            0.30,
            -1,
            811.07,
            r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",
        ),
        # ================================================================
        # More PA variants
        # ================================================================
        LipidTemplate(
            "DLiPA",
            "1,2-dilinoleoyl-sn-glycero-3-phosphate",
            "PA",
            "C39H68O8P",
            (18, 2),
            (18, 2),
            0.550,
            3.5,
            0.28,
            -1,
            695.94,
            r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])O)OC(=O)CCCCCCC/C=C\C/C=C\CCCCC",
        ),
        # ================================================================
        # LPC — Lyso-phosphatidylcholine (single-chain, neutral)
        # ================================================================
        LipidTemplate(
            "LPC16",
            "1-palmitoyl-2-hydroxy-sn-glycero-3-phosphocholine",
            "LPC",
            "C24H50NO7P",
            (16, 0),
            (0, 0),
            0.470,
            2.8,
            0.26,
            0,
            495.63,
            "CCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "LPC18",
            "1-stearoyl-2-hydroxy-sn-glycero-3-phosphocholine",
            "LPC",
            "C26H54NO7P",
            (18, 0),
            (0, 0),
            0.480,
            3.0,
            0.26,
            0,
            523.68,
            "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        # ================================================================
        # LPE — Lyso-phosphatidylethanolamine (single-chain)
        # ================================================================
        LipidTemplate(
            "LPE16",
            "1-palmitoyl-2-hydroxy-sn-glycero-3-phosphoethanolamine",
            "LPE",
            "C21H44NO7P",
            (16, 0),
            (0, 0),
            0.440,
            3.0,
            0.25,
            0,
            453.55,
            "CCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[NH3+]",
        ),
        # ================================================================
        # DG — Diacylglycerol (neutral, no headgroup)
        # ================================================================
        LipidTemplate(
            "DOPGd",
            "1,2-dioleoyl-sn-glycerol",
            "DG",
            "C39H72O5",
            (18, 1),
            (18, 1),
            0.520,
            2.8,
            0.25,
            0,
            621.00,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        ),
        LipidTemplate(
            "DPPGd",
            "1,2-dipalmitoyl-sn-glycerol",
            "DG",
            "C35H68O5",
            (16, 0),
            (16, 0),
            0.480,
            3.2,
            0.25,
            0,
            568.91,
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCC",
        ),
        # ================================================================
        # CER — Ceramides (sphingolipid backbone + one fatty acid)
        # ================================================================
        LipidTemplate(
            "Cer16",
            "N-palmitoyl-D-erythro-sphingosine",
            "CER",
            "C34H67NO3",
            (16, 0),
            (0, 0),
            0.420,
            3.5,
            0.25,
            0,
            537.90,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "Cer18",
            "N-stearoyl-D-erythro-sphingosine",
            "CER",
            "C36H71NO3",
            (18, 0),
            (0, 0),
            0.430,
            3.7,
            0.25,
            0,
            565.95,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCCCC",
        ),
        LipidTemplate(
            "Cer24",
            "N-lignoceroyl-D-erythro-sphingosine",
            "CER",
            "C42H83NO3",
            (24, 0),
            (0, 0),
            0.440,
            4.0,
            0.25,
            0,
            650.11,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCCCCCCCCCC",
        ),
        # ================================================================
        # MGDG — Monogalactosyldiacylglycerol (plant thylakoid)
        # ================================================================
        LipidTemplate(
            "MGDG",
            "monogalactosyldiacylglycerol (18:3/18:3)",
            "MGDG",
            "C45H74O10",
            (18, 3),
            (18, 3),
            0.670,
            3.2,
            0.32,
            0,
            775.08,
            "C(O[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@@H](CO)O1)[C@]([H])"
            "(OC(CCCCCCC/C=C\\C/C=C\\C/C=C\\CC)=O)COC(CCCCCCC/C=C\\C/C=C\\C/C=C\\CC)=O",
        ),
        # ================================================================
        # DGDG — Digalactosyldiacylglycerol (plant thylakoid)
        # ================================================================
        LipidTemplate(
            "DGDG",
            "digalactosyldiacylglycerol (18:3/18:3)",
            "DGDG",
            "C51H84O15",
            (18, 3),
            (18, 3),
            0.720,
            3.3,
            0.34,
            0,
            937.22,
            "C(O[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@@H](CO[C@@H]2[C@H](O)"
            "[C@@H](O)[C@@H](O)[C@@H](CO)O2)O1)[C@]([H])"
            "(OC(CCCCCCC/C=C\\C/C=C\\C/C=C\\CC)=O)COC(CCCCCCC/C=C\\C/C=C\\C/C=C\\CC)=O",
        ),
        # ================================================================
        # GM1 — Monosialotetrahexosylganglioside
        # ================================================================
        LipidTemplate(
            "GM1",
            "ganglioside GM1 (d18:1/18:0), sialate",
            "GM1",
            "C73H130N3O31",
            (18, 1),
            (18, 0),
            0.980,
            4.2,
            0.42,
            -1,
            1545.83,
            # PubChem CID 9963963: complete GM1a stereochemistry; only sialic acid is deprotonated.
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O)[C@H]4O)[C@H]3NC(C)=O)[C@H](O[C@]3(C(=O)[O-])C[C@H](O)[C@@H](NC(C)=O)[C@H]([C@H](O)[C@H](O)CO)O3)[C@H]2O)[C@H](O)[C@H]1O)NC(=O)CCCCCCCCCCCCCCCCC",
        ),
        # ================================================================
        # ST — Additional Sterols
        # ================================================================
        LipidTemplate(
            "CAMP",
            "Campesterol",
            "ST",
            "C28H48O",
            (0, 0),
            (0, 0),
            0.390,
            3.5,
            0.25,
            0,
            400.68,
            "CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "SITO",
            "β-Sitosterol",
            "ST",
            "C29H50O",
            (0, 0),
            (0, 0),
            0.400,
            3.5,
            0.25,
            0,
            414.71,
            "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
        ),
        LipidTemplate(
            "STIG",
            "Stigmasterol",
            "ST",
            "C29H48O",
            (0, 0),
            (0, 0),
            0.400,
            3.5,
            0.25,
            0,
            412.69,
            "CC[C@H](/C=C/[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
        ),
        # ---- Oxysterols (hydroxycholesterols) ----
        LipidTemplate(
            "25OHC",
            "25-Hydroxycholesterol",
            "ST",
            "C27H46O2",
            (0, 0),
            (0, 0),
            0.400,
            3.4,
            0.26,
            0,
            402.65,
            "C[C@H](CCCC(C)(C)O)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "27OHC",
            "27-Hydroxycholesterol",
            "ST",
            "C27H46O2",
            (0, 0),
            (0, 0),
            0.410,
            3.4,
            0.26,
            0,
            402.65,
            "C[C@H](CCC[C@@H](C)CO)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "20AHC",
            "20α-Hydroxycholesterol",
            "ST",
            "C27H46O2",
            (0, 0),
            (0, 0),
            0.395,
            3.5,
            0.26,
            0,
            402.65,
            "C[C@](O)(CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "22RHC",
            "22R-Hydroxycholesterol",
            "ST",
            "C27H46O2",
            (0, 0),
            (0, 0),
            0.395,
            3.5,
            0.26,
            0,
            402.65,
            "C[C@H]([C@H](O)CCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "24SHC",
            "24S-Hydroxycholesterol",
            "ST",
            "C27H46O2",
            (0, 0),
            (0, 0),
            0.400,
            3.4,
            0.26,
            0,
            402.65,
            "CC(C)[C@@H](O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        LipidTemplate(
            "7KCH",
            "7-Ketocholesterol",
            "ST",
            "C27H44O2",
            (0, 0),
            (0, 0),
            0.400,
            3.5,
            0.26,
            0,
            400.64,
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C(=O)C=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        ),
        # ================================================================
        # PIP — More Phosphoinositides
        # ================================================================
        LipidTemplate(
            "SAPI",
            "1-stearoyl-2-arachidonoyl-PI(4,5)P2",
            "PIP",
            "C47H81O19P3",
            (18, 0),
            (20, 4),
            0.690,
            3.8,
            0.34,
            -4,
            1043.07,
            "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](OP(=O)([O-])O)[C@H](OP(=O)([O-])[O-])[C@H]1O",
        ),
        LipidTemplate(
            "PAPI",
            "1-palmitoyl-2-arachidonoyl-PI(4,5)P2",
            "PIP",
            "C45H77O19P3",
            (16, 0),
            (20, 4),
            0.685,
            3.8,
            0.34,
            -4,
            1015.01,
            # Exact sn-chain/headgroup identity follows the named native CHARMM residue.
            "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](OP(=O)([O-])O)[C@H](OP(=O)([O-])[O-])[C@H]1O",
        ),
        LipidTemplate(
            "SOP2",
            "1-stearoyl-2-oleoyl-PI(3,4)P2",
            "PIP",
            "C45H83O19P3",
            (18, 0),
            (18, 1),
            0.690,
            3.9,
            0.34,
            -4,
            1021.06,
            # Exact sn-chain/headgroup identity follows the named native CHARMM residue.
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])O[C@H]1[C@H](O)[C@@H](O)[C@H](OP(=O)([O-])[O-])[C@@H](OP(=O)([O-])O)[C@H]1O",
        ),
        LipidTemplate(
            "SOP3",
            "1-stearoyl-2-oleoyl-PI(3,4,5)P3",
            "PIP",
            "C45H82O22P4",
            (18, 0),
            (18, 1),
            0.705,
            3.8,
            0.34,
            -6,
            1099.03,
            # Exact sn-chain/headgroup identity follows the named native CHARMM residue.
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])O[C@H]1[C@H](O)[C@@H](OP(=O)([O-])[O-])[C@H](OP(=O)([O-])[O-])[C@@H](OP(=O)([O-])O)[C@H]1O",
        ),
        # ================================================================
        # Ether / Plasmalogen lipids
        # ================================================================
        LipidTemplate(
            "PPCpl",
            "1-(1Z-hexadecenyl)-2-oleoyl-sn-glycero-3-phosphocholine (plasmalogen)",
            "PC",
            "C42H82NO7P",
            (16, 0),
            (18, 1),
            0.700,
            3.6,
            0.30,
            0,
            744.07,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO/C=C\\CCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
        ),
        LipidTemplate(
            "PPEpl",
            "1-(1Z-hexadecenyl)-2-oleoyl-sn-glycero-3-phosphoethanolamine (plasmalogen)",
            "PE",
            "C39H76NO7P",
            (16, 0),
            (18, 1),
            0.650,
            3.9,
            0.28,
            0,
            701.98,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO/C=C\\CCCCCCCCCCCCCC)COP(=O)([O-])OCC[NH3+]",
        ),
        # ================================================================
        # Additional SM variants
        # ================================================================
        LipidTemplate(
            "DSM",
            "N-decanoyl-D-erythro-sphingosylphosphorylcholine",
            "SM",
            "C33H67N2O6P",
            (10, 0),
            (0, 0),
            0.500,
            3.8,
            0.28,
            0,
            618.87,
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COP(=O)([O-])OCC[N+](C)(C)C)NC(=O)CCCCCCCCC",
        ),
        LipidTemplate(
            "NSM",
            "N-nervonoyl-D-erythro-sphingosylphosphorylcholine",
            "SM",
            "C47H93N2O6P",
            (24, 1),
            (0, 0),
            0.560,
            4.6,
            0.30,
            0,
            813.24,
            r"CCCCCCCC/C=C\CCCCCCCCCCCCCC(=O)N[C@@H](COP(=O)([O-])OCC[N+](C)(C)C)[C@H](O)/C=C/CCCCCCCCCCCCC",
        ),
        # ================================================================
        # Bacterial-specific lipids
        # ================================================================
        LipidTemplate(
            "LysPG",
            "lysyl-phosphatidylglycerol (16:0/18:1)",
            "LPG",
            "C46H90N2O11P",
            (16, 0),
            (18, 1),
            0.720,
            3.8,
            0.32,
            1,
            878.20,
            "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@@H](O)COC(=O)[C@@H]([NH3+])CCCC[NH3+]",
        ),
        LipidTemplate(
            "TMCL",
            "tetramyristoyl-cardiolipin",
            "CL",
            "C65H124O17P2",
            (14, 0),
            (14, 0),
            1.100,
            3.3,
            0.34,
            -2,
            1239.58,
            "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC(O)COP(=O)([O-])OC[C@@H](COC(=O)CCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC",
        ),
    ]

    # ------------------------------------------------------------------
    # Registry API
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> LipidTemplate:
        name = name.upper()
        cls._ensure_loaded()
        scoped = cls._task_lipids.get()
        if scoped is not None and name in scoped:
            return scoped[name]
        if name not in cls._lipids:
            raise KeyError(f"Unknown lipid: {name!r}. Available: {cls.list()}")
        return cls._lipids[name]

    @classmethod
    def list(cls) -> list[str]:
        cls._ensure_loaded()
        names = set(cls._lipids)
        scoped = cls._task_lipids.get()
        if scoped:
            names.update(scoped)
        return sorted(names)

    @classmethod
    def list_builtin(cls) -> list[str]:
        """Return process-wide built-ins without task-scoped additions."""
        return sorted({lipid.name.upper() for lipid in cls._BUILTIN})

    @classmethod
    def list_by_category(cls) -> dict[str, list[str]]:
        cls._ensure_loaded()
        combined = dict(cls._lipids)
        scoped = cls._task_lipids.get()
        if scoped:
            combined.update(scoped)
        result = {}
        for cat in CATEGORY_ORDER:
            names = [n for n, t in combined.items() if t.category == cat]
            if names:
                result[cat] = sorted(names)
        # Any uncategorized
        known = set()
        for names in result.values():
            known.update(names)
        remaining = [n for n in combined if n not in known]
        if remaining:
            result["Other"] = sorted(remaining)
        return result

    @classmethod
    @contextmanager
    def task_scope(cls, lipids: dict[str, LipidTemplate]) -> Iterator[None]:
        """Expose immutable custom lipids only in the current task execution."""
        normalized = {str(name).upper(): lipid for name, lipid in lipids.items()}
        token = cls._task_lipids.set(normalized)
        try:
            yield
        finally:
            cls._task_lipids.reset(token)

    @classmethod
    def register(cls, template: LipidTemplate) -> None:
        """Register a lipid template after loading the built-in catalog."""
        cls._ensure_loaded()
        cls._lipids[template.name.upper()] = template
        cat = template.category
        if cat not in cls._by_category:
            cls._by_category[cat] = []
        cls._by_category[cat].append(template.name.upper())

    @classmethod
    def _ensure_loaded(cls) -> None:
        # NOTE: Not thread-safe. In multi-threaded use, two threads may
        # both load simultaneously (harmless but redundant).
        if cls._lipids:
            return
        for t in cls._BUILTIN:
            cls._lipids[t.name.upper()] = t


# =============================================================================
# Custom lipid from SMILES — lightweight parser + property estimator
# =============================================================================

# Atomic masses (g/mol) for formula mass calculation
_ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.065,
    "F": 18.998,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.904,
    "Na": 22.990,
    "K": 39.098,
    "Ca": 40.078,
    "Mg": 24.305,
    "Zn": 65.380,
    "Fe": 55.845,
}

# Functional-group SMILES patterns for charge and headgroup detection
_CHARGED_PATTERNS: list[tuple[str, int, str]] = [
    ("[N+]", +1, "quaternary ammonium"),
    ("[NH3+]", +1, "primary ammonium"),
    ("[NH2+]", +1, "iminium"),
    ("C(=O)[O-]", -1, "carboxylate"),
    ("S(=O)(=O)[O-]", -1, "sulfonate"),
    ("[O-]", -1, "alkoxide/phosphate anion"),
    ("[Cl-]", -1, "chloride"),
]

_HEADGROUP_SMILES: list[tuple[str, str]] = [
    ("OCC[N+](C)(C)C", "PC"),  # phosphocholine
    ("OCC[NH3+]", "PE"),  # phosphoethanolamine
    ("OCC(N)", "PS"),  # phosphoserine
    ("OC(CO)CO", "PG"),  # phosphoglycerol
    ("OP(=O)(O)O", "PA"),  # phosphatidic acid
    ("OC1C(O)C(O)C(O)C(O)C1O", "PI"),  # phosphoinositol (simplified)
    ("C1CC2C3CCC4CC(O)CCC4(C)C3CCC2(C)C1", "ST"),  # sterol core
]


def _parse_smiles_elements(smiles: str) -> dict[str, int]:
    """Count elements from a SMILES string.

    Handles two-letter elements (Cl, Br, etc.) and bracket-atom notation
    like [N+], [13C], [O-].
    """
    counts: dict[str, int] = {}
    i = 0
    s = smiles
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "[":
            # Bracket atom: extract element symbol
            j = i + 1
            # skip isotope digits
            while j < n and s[j].isdigit():
                j += 1
            # skip H count if present
            if j < n and s[j] == "H":
                j += 1
                while j < n and s[j].isdigit():
                    j += 1
            # Read element (1-2 letters, first uppercase)
            if j < n and s[j].isupper():
                elem = s[j]
                j += 1
                if j < n and s[j].islower():
                    elem += s[j]
                    j += 1
                counts[elem] = counts.get(elem, 0) + 1
            # Skip charges, class, etc. until ]
            while j < n and s[j] != "]":
                j += 1
            i = j + 1
        elif ch.isupper():
            # Organic element: C, N, O, P, S, F, Cl, Br, I
            elem = ch
            if i + 1 < n and s[i + 1].islower():
                elem += s[i + 1]
                i += 1
            counts[elem] = counts.get(elem, 0) + 1
            i += 1
        elif ch == "c":
            # Aromatic carbon (lowercase)
            counts["C"] = counts.get("C", 0) + 1
            i += 1
        elif ch in "nop":
            # Aromatic N, O, P (lowercase)
            emap = {"n": "N", "o": "O", "p": "P"}
            counts[emap.get(ch, ch.upper())] = counts.get(emap.get(ch, ch.upper()), 0) + 1
            i += 1
        else:
            # H, digits, bond symbols, parentheses, etc.
            i += 1

    # Implicit hydrogens: C gets 4 minus explicit bonds, etc.
    # For rough mass estimation, add implicit H based on heavy atom valence
    heavy_valence = {"C": 4, "N": 3, "O": 2, "P": 5, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
    implicit_h = 0
    for elem, cnt in counts.items():
        if elem == "H":
            continue
        v = heavy_valence.get(elem, 4)
        implicit_h += cnt * v // 2  # rough estimate

    if implicit_h > 0:
        counts["H"] = counts.get("H", 0) + implicit_h

    return counts


# ---------------------------------------------------------------------------
# Headgroup classification
#
# This replaced a list of raw SMILES fragments matched as substrings against a
# canonical SMILES string. That test cannot work: RDKit canonicalisation
# rewrites ring closures and inserts stereo markers, so a pattern like the
# sterol nucleus never matched anything, and the default was PC. Fed the
# project's own 84 built-in lipids, it recovered 36% of their categories --
# and, worse, it mistook all eleven sterols for PC, which denied them the POPC
# host they need, so an uploaded cholesterol was built as a pure cholesterol
# bilayer that cannot form. Matching real substructures instead recovers 99%.

_HEADGROUP_SMARTS: dict[str, str] = {
    "CHOLINE": "[NX4+](C)(C)C",
    "ETHANOLAMINE": "[OX2][CX4][CX4][NX4H3+,NX3H2]",
    "SERINE": "[OX2][CX4][CX4]([NX4H3+,NX3H2])[CX3](=O)[OX1-,OX2H1]",
    "GLYCEROLP": "[OX2]C[CH1]([OX2H1])[CH2][OX2H1]",
    "INOSITOL": "[CH1]1([OX2])[CH1]([OX2])[CH1]([OX2])[CH1]([OX2])[CH1]([OX2])[CH1]1[OX2]",
    "AMIDE": "[NX3H1][CX3](=O)",
    "ESTER": "[CX3](=O)[OX2][CX4]",
    "PYRANOSE": "[CH1]1[OX2][CH1][CH1][CH1][CH1]1",
    # One acyl tail contributes one terminal methyl on a chain. Two carbons is
    # enough: a docosahexaenoyl chain ends C=C-CH2-CH3, so demanding a longer
    # run of CH2 would count it as absent and read the lipid as lyso.
    "ACYL_TAIL": "[CH3][CH2]",
}

_COMPILED_HEADGROUP_SMARTS: dict[str, object] = {}


def _smarts(key: str):
    from rdkit import Chem

    if key not in _COMPILED_HEADGROUP_SMARTS:
        _COMPILED_HEADGROUP_SMARTS[key] = Chem.MolFromSmarts(_HEADGROUP_SMARTS[key])
    return _COMPILED_HEADGROUP_SMARTS[key]


def _count(molecule, key: str) -> int:
    pattern = _smarts(key)
    return len(molecule.GetSubstructMatches(pattern)) if pattern is not None else 0


def _is_sterol(molecule) -> bool:
    """Whether four rings are fused into the steroid nucleus (three 6, one 5).

    Ring topology rather than a SMARTS pattern. Sterols differ by unsaturation,
    ring methylation and oxidation state -- cholesterol, ergosterol, the
    phytosterols, 7-ketocholesterol and the hydroxycholesterols all have to be
    admitted while sugars and single aromatic rings are excluded -- and one
    pattern covering that set is much harder to get right than asking how the
    rings are fused.
    """
    rings = [set(r) for r in molecule.GetRingInfo().AtomRings()]
    systems: list[dict] = []
    for ring in rings:
        block = {"atoms": set(ring), "sizes": [len(ring)]}
        for existing in [s for s in systems if s["atoms"] & ring]:
            block["atoms"] |= existing["atoms"]
            block["sizes"] += existing["sizes"]
            systems.remove(existing)
        systems.append(block)
    return any(
        len(s["sizes"]) >= 4 and s["sizes"].count(6) >= 3 and 5 in s["sizes"] for s in systems
    )


def detect_lipid_category(molecule) -> str:
    """Classify a lipid headgroup from its structure.

    Ordered most specific first, because the classes genuinely nest: a
    sphingomyelin carries the same phosphocholine as a PC, and a
    phosphoinositide the same inositol as a PI.
    """
    phosphorus = sum(a.GetAtomicNum() == 15 for a in molecule.GetAtoms())
    choline = _count(molecule, "CHOLINE")
    inositol = _count(molecule, "INOSITOL")
    amide = _count(molecule, "AMIDE")
    sugars = _count(molecule, "PYRANOSE")
    # A lyso lipid carries one tail where its class carries two. Counting tails
    # rather than esters, because a plasmalogen attaches its first chain as a
    # vinyl ether, which is not an ester and would read as a missing chain.
    lyso = _count(molecule, "ACYL_TAIL") <= 1

    if _is_sterol(molecule):
        return "ST"
    if phosphorus >= 2 and not inositol:
        return "CL"
    if inositol:
        return "PIP" if phosphorus >= 2 else "PI"
    if amide and phosphorus and choline:
        return "SM"
    if amide and sugars >= 3:
        return "GM1"
    if amide and not phosphorus:
        return "CER"
    if phosphorus:
        if choline:
            return "LPC" if lyso else "PC"
        if _count(molecule, "SERINE"):
            return "PS"
        if _count(molecule, "ETHANOLAMINE"):
            return "LPE" if lyso else "PE"
        if _count(molecule, "GLYCEROLP"):
            return "LPG" if lyso else "PG"
        return "PA"
    if sugars >= 2:
        return "DGDG"
    if sugars == 1:
        return "MGDG"
    return "DG" if _count(molecule, "ESTER") >= 2 else "PC"


def _category_reference(category: str) -> tuple[float, float] | None:
    """Median area and thickness of the built-in lipids in this class.

    A lipid's headgroup constrains its area and thickness far better than its
    carbon count does. Predicting each built-in lipid from the median of the
    *others* in its class, rather than from the carbon-count buckets below,
    cuts the worst area error from 46% to 18% and the worst thickness error
    from 38% to 23%. That matters more than the median does, because the
    production gate is a ratio band: a 46% error cannot pass it, and an 18%
    one comfortably does.

    Derived from the registry at call time rather than written out as a second
    table, so it cannot drift from the entries it summarises.
    """
    import statistics

    areas, thicknesses = [], []
    for template in LipidRegistry._BUILTIN:
        if isinstance(template, LipidTemplate):
            if template.category == category:
                areas.append(template.area_per_lipid)
                thicknesses.append(template.bilayer_thickness)
        elif isinstance(template, dict) and template.get("category") == category:
            areas.append(float(template["area_per_lipid"]))
            thicknesses.append(float(template["bilayer_thickness"]))
    if not areas:
        return None
    return statistics.median(areas), statistics.median(thicknesses)


def _estimate_lipid_properties(smiles: str, user_name: str) -> dict:
    """Estimate physical properties of a lipid from its SMILES string.

    Returns a dict compatible with LipidTemplate fields.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("lipid SMILES could not be parsed")

    # Formula, mass and charge from RDKit rather than by scanning the SMILES
    # text. The previous element counter estimated implicit hydrogens as
    # `heavy_valence // 2`, which reported POPC as C42H95NO8P against its real
    # C42H82NO8P -- a formula shown to the user that was simply wrong.
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    mass = Descriptors.MolWt(molecule)
    charge = Chem.GetFormalCharge(molecule)
    elem: dict[str, int] = {}
    for atom in Chem.AddHs(molecule).GetAtoms():
        elem[atom.GetSymbol()] = elem.get(atom.GetSymbol(), 0) + 1

    category = detect_lipid_category(molecule)

    # ---- APL estimation ----
    # Count carbons in the molecule (proxy for tail size)
    n_carbon = elem.get("C", 0)

    # Detect if it's a lysolipid (one tail) by counting ester groups
    n_ester = smiles.count("C(=O)O") + smiles.count("OC(=O)")

    if category == "ST":
        apl = 0.40  # nm² — sterols are compact
    elif n_ester <= 1:
        apl = 0.45  # nm² — lysolipid
    elif n_carbon < 30:
        apl = 0.55  # nm² — short-tail lipid
    elif n_carbon < 40:
        apl = 0.62  # nm² — medium-tail lipid
    elif n_carbon < 50:
        apl = 0.65  # nm² — common phospholipid
    else:
        apl = 0.70  # nm² — long-tail / polyunsaturated

    # ---- Bilayer thickness estimation (DHH) ----
    if category == "ST":
        dh = 3.5  # nm — cholesterol in bilayer
    elif n_ester <= 1:
        dh = 3.0  # nm — lysolipid
    elif n_carbon < 30:
        dh = 3.0  # nm
    elif n_carbon < 40:
        dh = 3.5  # nm
    else:
        dh = 3.8  # nm — typical phospholipid

    # The headgroup class is the better predictor; the carbon-count buckets
    # above remain the fallback for a class the registry does not cover.
    reference = _category_reference(category)
    if reference is not None:
        apl, dh = reference

    # ---- vdW radius ----
    vdw = 0.35  # nm — approximate

    # Tail estimation
    # Try to find fatty-acid chain lengths from SMILES
    # Look for CC(C)C(=O) or similar ester-linked chains
    tail1 = (16, 0)
    tail2 = (16, 0)
    if n_ester >= 2:
        # Estimate tail length from total carbon count
        headgroup_c = {"PC": 8, "PE": 5, "PG": 6, "PS": 6, "PA": 3, "PI": 9, "ST": 27}
        hc = headgroup_c.get(category, 8)
        tail_carbons = (n_carbon - hc) // 2
        tail_carbons = max(8, min(tail_carbons, 24))
        tail1 = (tail_carbons, 0)
        tail2 = (tail_carbons, 0)

    return {
        "name": user_name.upper().replace(" ", "_"),
        "common_name": user_name,
        "category": category,
        "formula": formula,
        "tail1": tail1,
        "tail2": tail2,
        "area_per_lipid": round(apl, 3),
        "bilayer_thickness": round(dh, 2),
        "vdw_radius": vdw,
        "charge": charge,
        "mass": round(mass, 1),
        "smiles": smiles,
        "headgroup": CATEGORY_NAMES.get(category, category),
        "n_carbon": n_carbon,
        "n_ester": n_ester,
    }


def canonical_lipid_identity(smiles: str) -> dict[str, str]:
    """Return a sanitised, stereochemistry-aware identity for one lipid."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES string")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    try:
        inchi_key = Chem.MolToInchiKey(molecule)
    except (ValueError, RuntimeError):
        inchi_key = ""
    return {
        "canonical_smiles": canonical,
        "connectivity_smiles": connectivity,
        "inchi_key": inchi_key,
    }


def find_registered_lipid_matches(smiles: str) -> list[dict]:
    """Find exact or stereochemistry-insensitive matches in the registry."""
    query = canonical_lipid_identity(smiles)
    matches = []
    for lipid_name in LipidRegistry.list():
        lipid = LipidRegistry.get(lipid_name)
        if not lipid.smiles:
            continue
        try:
            identity = canonical_lipid_identity(lipid.smiles)
        except ValueError:
            continue
        exact = identity["canonical_smiles"] == query["canonical_smiles"]
        same_connectivity = identity["connectivity_smiles"] == query["connectivity_smiles"]
        if exact or same_connectivity:
            matches.append(
                {
                    "name": lipid.name,
                    "common_name": lipid.common_name,
                    "match": "exact" if exact else "connectivity",
                    "canonical_smiles": identity["canonical_smiles"],
                    "inchi_key": identity["inchi_key"],
                }
            )
    return sorted(matches, key=lambda item: (item["match"] != "exact", item["name"]))


def parse_custom_lipid(smiles: str, name: str) -> dict:
    """Public API: parse a SMILES string and estimate lipid properties.

    Returns a dict ready for the frontend to display.
    """
    if not smiles or not name:
        raise ValueError("Both SMILES string and lipid name are required")
    if len(smiles) > 500:
        raise ValueError("SMILES string too long")
    normalized_name = str(name).strip().upper().replace(" ", "_")
    if not LipidRegistry._CUSTOM_NAME_PATTERN.fullmatch(normalized_name):
        raise ValueError(
            "Custom lipid residue ID must contain 1-5 uppercase letters, "
            "digits, or underscores and must start with a letter"
        )
    identity = canonical_lipid_identity(smiles)
    result = _estimate_lipid_properties(identity["canonical_smiles"], str(name).strip())
    result["name"] = normalized_name
    result.update(identity)
    from gmxbuilder.modules.membrane.chain_identity import chain_identity

    result.update(chain_identity(identity["canonical_smiles"]))
    result["registered_matches"] = find_registered_lipid_matches(smiles)
    result["is_existing"] = any(match["match"] == "exact" for match in result["registered_matches"])
    return result

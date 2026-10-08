"""Evidence that library production started from a completely water-free slab."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.modules.solvation.membrane_exclusion import (
    SURFACE_MARGIN_NM,
    assert_membrane_water_free,
    membrane_slab,
)

POLICY = "complete-lipid-z-envelope-v1"
WATER = {"SOL", "HOH", "WAT", "TIP3", "TIP4", "TIP3P", "TIP4P"}
IONS = {"NA", "CL", "K", "CA", "MG", "SOD", "CLA", "POT", "CAL"}


def initial_water_evidence(work):
    """Measure saved pre-EM GRO files from this pure/host lipid-only builder.

    It emits whole, centered lipids before solvent, so use the actual full lipid
    envelope without a guessed core thickness. Include every water site and the
    same rounding margin used by the interactive atomistic and CG workflows.
    """
    files = {}
    for filename in ("solvated.gro", "ionized.gro"):
        path = work / filename
        structure = GROReader().read(path)
        names = np.asarray([str(n).upper() for n in structure.resnames])
        lipid = np.flatnonzero(~np.isin(names, list(WATER | IONS)))
        water = np.flatnonzero(np.isin(names, list(WATER)))
        if not len(lipid) or not len(water):
            raise ValueError("Initial library bilayer must contain both lipids and external water")
        system = System(
            structure=structure,
            components=[
                Component("Library membrane", ComponentKind.MEMBRANE, lipid),
                Component("Library water", ComponentKind.SOLVENT, water),
            ],
        )
        assert_membrane_water_free(system)
        files[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "lipid_atoms": len(lipid),
            "water_sites": len(water),
            "water_sites_in_membrane": 0,
            "lipid_z_bounds_nm": list(membrane_slab(system)[:2]),
        }
    result = {"policy": POLICY, "margin_nm": SURFACE_MARGIN_NM, "files": files}
    if not initial_water_evidence_valid(result):
        raise ValueError("Initial solvation/ionization lipid counts differ")
    return result


def initial_water_evidence_valid(evidence):
    """Reject legacy, incomplete, nonzero or malformed construction evidence."""
    try:
        if evidence["policy"] != POLICY or evidence["margin_nm"] != SURFACE_MARGIN_NM:
            return False
        files = evidence["files"]
        if set(files) != {"solvated.gro", "ionized.gro"}:
            return False
        counts = []
        for record in files.values():
            if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                return False
            for key in ("lipid_atoms", "water_sites", "water_sites_in_membrane"):
                if type(record[key]) is not int:
                    return False
            if min(record["lipid_atoms"], record["water_sites"]) <= 0:
                return False
            if record["water_sites_in_membrane"] != 0:
                return False
            lower, upper = record["lipid_z_bounds_nm"]
            if not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper:
                return False
            counts.append(record["lipid_atoms"])
        return len(set(counts)) == 1
    except (KeyError, TypeError, ValueError, AttributeError):
        return False

"""Exact Amber Lipid21 templates shipped with GMXBUILDER.

The bundled files are generated from AmberTools' official ``lipid21.lib`` and
``lipid21.dat``.  They retain the explicit GROMACS 1-4 pair parameters written
by ParmEd, including Lipid21's special polyunsaturated-chain scaling.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


_TAIL_MODULES = {
    (12, 0): "LAL",
    (14, 0): "MY",
    (16, 0): "PA",
    (18, 0): "ST",
    (18, 1): "OL",
    (20, 4): "AR",
    (22, 6): "DHA",
}
_HEAD_MODULES = {"PC": "PC", "PE": "PE", "PG": "PGR", "PS": "PS", "PA": "PH-"}
_NON_ESTER_IDENTITIES = {"PPCPL", "PPEPL"}


def lipid21_sequence(lipid_name: str) -> tuple[str, ...] | None:
    """Return the exact Amber Lipid21 module sequence for a built-in lipid."""
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    name = str(lipid_name).strip().upper()
    # Plasmalogens share the same summary tail tuple as diacyl lipids but have
    # an ether/vinyl-ether linkage, so a diacyl Lipid21 sequence is not exact.
    if name in _NON_ESTER_IDENTITIES:
        return None
    if name == "CHOL":
        return ("CHL",)
    if name == "PSM":
        return ("PA", "SPM", "SA")
    if name == "SSM":
        return ("ST", "SPM", "SA")
    try:
        lipid = LipidRegistry.get(name)
    except KeyError:
        return None
    head = _HEAD_MODULES.get(lipid.category)
    tail1 = _TAIL_MODULES.get(tuple(lipid.tail1))
    tail2 = _TAIL_MODULES.get(tuple(lipid.tail2))
    if not (head and tail1 and tail2):
        return None
    return tail1, head, tail2


def lipid21_capability(lipid_name: str) -> tuple[bool, str]:
    """Return whether an exact bundled Lipid21 topology exists."""
    name = str(lipid_name).strip().upper()
    sequence = lipid21_sequence(name)
    if sequence is None:
        return False, "not represented by an exact Amber Lipid21 module combination"
    if not (_data_root() / "itp" / f"{name}.itp").is_file():
        return False, "exact Lipid21 source is known but the bundled template is missing"
    return True, "exact Amber Lipid21 v1.0 parameters"


def _data_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "lipid21"


@lru_cache(maxsize=1)
def _templates() -> dict:
    path = _data_root() / "templates.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def load_lipid21_geometry(lipid_name: str) -> tuple[np.ndarray, list[str]]:
    """Load one exact-topology coordinate template in nanometres."""
    name = str(lipid_name).strip().upper()
    entry = _templates().get(name)
    if entry is None:
        raise KeyError(f"No bundled Amber Lipid21 template for {name}")
    coordinates = np.asarray(entry["coordinates_nm"], dtype=float)
    atom_names = [str(value) for value in entry["atom_names"]]
    if coordinates.shape != (len(atom_names), 3):
        raise ValueError(f"Corrupt Amber Lipid21 coordinate template for {name}")
    return coordinates.copy(), atom_names


def lipid21_itp_path(lipid_name: str) -> Path:
    """Return the exact per-molecule GROMACS include path."""
    name = str(lipid_name).strip().upper()
    path = _data_root() / "itp" / f"{name}.itp"
    if not path.is_file():
        raise KeyError(f"No bundled Amber Lipid21 topology for {name}")
    return path


def lipid21_atomtypes_path() -> Path:
    """Return the namespaced Lipid21 non-bonded atom-type include."""
    path = _data_root() / "lipid21_atomtypes.itp"
    if not path.is_file():
        raise FileNotFoundError("Bundled Amber Lipid21 atom types are missing")
    return path


def lipid21_lipids() -> tuple[str, ...]:
    """Return all exact Lipid21 lipids currently bundled by GMXBUILDER."""
    return tuple(sorted(_templates()))

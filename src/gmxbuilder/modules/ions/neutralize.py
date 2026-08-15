"""Net charge calculation for neutralization."""

from __future__ import annotations

from gmxbuilder.core.system import System

# Use System._RESIDUE_CHARGES as the single source of truth for residue charges
_RESIDUE_CHARGES = System._RESIDUE_CHARGES

_ION_CHARGES: dict[str, float] = {
    "NA": 1.0,
    "K": 1.0,
    "CS": 1.0,
    "LI": 1.0,
    "CA": 2.0,
    "ZN": 2.0,
    "MG": 2.0,
    "CL": -1.0,
    "BR": -1.0,
    "I": -1.0,
}


def compute_net_charge(system: System) -> float:
    """Compute non-solvent formal charge from the central System contract."""
    return system.total_charge()

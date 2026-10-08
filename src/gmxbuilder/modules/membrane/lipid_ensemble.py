"""Shared population rules for final-frame and trajectory orientation checks.

Population fractions are geometry diagnostics, not equilibrium tests. Species
fractions are reported separately so a dilute guest cannot hide in its host.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.modules.membrane.lipid_orientation import (
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
)

# The reviewed population screen tolerates up to 2% orientation outliers.
# Callers choose the gated population; a reported species subgroup is not always a hard gate.
MIN_ORIENTED_FRACTION = 0.98
# These guests have polar groups at both ends; physical interfacial/tilted
# populations must not be rejected as misoriented single-headed amphiphiles.
SIDE_CHAIN_OXYSTEROLS = frozenset({"20AHC", "22RHC", "24SHC", "25OHC", "27OHC"})


def orientation_summary(projections, cosines):
    projections = np.asarray(projections, dtype=float)
    cosines = np.asarray(cosines, dtype=float)
    if projections.ndim != 1 or projections.shape != cosines.shape:
        raise ValueError("Orientation projections and cosines must be matching vectors")
    valid = (
        np.isfinite(projections)
        & np.isfinite(cosines)
        & (projections >= MIN_INWARD_PROJECTION_NM)
        & (cosines >= MIN_INWARD_COSINE)
    )
    fraction = float(valid.mean()) if len(valid) else 0.0
    return {
        "n_lipids": len(valid),
        "outlier_count": int((~valid).sum()),
        "correct_fraction": fraction,
        "passed": bool(len(valid) and fraction >= MIN_ORIENTED_FRACTION),
        "minimum_projection_nm": (
            float(projections.min()) if len(valid) and np.isfinite(projections).all() else None
        ),
        "minimum_cosine": (
            float(cosines.min()) if len(valid) and np.isfinite(cosines).all() else None
        ),
    }


def orientation_population(lipid_name, projections, cosines, host_projections, host_cosines):
    """Use exactly the same chemical exception in both analysis paths."""
    if str(lipid_name).upper() in SIDE_CHAIN_OXYSTEROLS:
        return (
            np.asarray(host_projections),
            np.asarray(host_cosines),
            "host-bilayer-plus-canonicalized-side-chain-oxysterol",
        )
    return np.asarray(projections), np.asarray(cosines), "all-lipids-single-headgroup"

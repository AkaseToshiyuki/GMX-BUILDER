"""Track initial leaflet membership by the immutable trajectory atom order."""

from __future__ import annotations

import numpy as np


def initial_leaflets(initial, current, groups):
    """Label using initial coordinates, never relabel a molecule after migration.

    GROMACS trajectories preserve topology atom order. Reject an imported or
    reordered snapshot that violates that correspondence instead of silently
    assigning the first half of its residues to the upper leaflet.
    """
    if initial.num_atoms != current.num_atoms:
        raise ValueError("Initial/trajectory atom counts differ")
    for field in ("atom_names", "resnames", "resids"):
        if list(getattr(initial, field)) != list(getattr(current, field)):
            raise ValueError(f"Initial/trajectory {field} order differs")
    if not groups or len(groups) % 2:
        raise ValueError("Initial membrane requires an even nonzero molecule count")
    heights = np.array([initial.coordinates[indices, 2].mean() for indices in groups])
    middle = float(np.median(heights))
    if not np.isfinite(heights).all() or np.any(np.isclose(heights, middle, atol=1e-6)):
        raise ValueError("Initial leaflet identities are ambiguous")
    flags = heights > middle
    if int(flags.sum()) != len(groups) // 2:
        raise ValueError("Initial leaflet molecule counts are unequal")
    return [(indices, bool(upper)) for indices, upper in zip(groups, flags, strict=True)]

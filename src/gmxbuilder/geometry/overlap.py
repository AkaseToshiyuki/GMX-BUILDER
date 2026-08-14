"""Van der Waals overlap detection and removal using KD-trees."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gmxbuilder.geometry.periodic import wrap_periodic_coordinates

# Approximate vdW radii (nm) for common elements
_DEFAULT_VDW_RADII: dict[str, float] = {
    "H": 0.110,  "HE": 0.140,
    "C": 0.170,  "N": 0.155,  "O": 0.152,  "F": 0.147,
    "P": 0.180,  "S": 0.180,  "CL": 0.175,
    "NA": 0.227, "K": 0.275,  "CA": 0.231, "MG": 0.173,
    "ZN": 0.139, "FE": 0.200, "BR": 0.185, "I": 0.198,
}


def find_overlapping_atoms(
    mobile: np.ndarray,
    fixed: np.ndarray,
    vdw_radii_mobile: np.ndarray | float = 0.15,
    vdw_radii_fixed: np.ndarray | float = 0.15,
    scale: float = 0.8,
    box_dimensions: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of *mobile* atoms that overlap with *fixed*.

    An overlap occurs when the distance between a mobile atom and a fixed
    atom is less than scale * (r_mobile + r_fixed).

    Parameters
    ----------
    mobile : (M, 3) ndarray
    fixed : (N, 3) ndarray
    vdw_radii_mobile : float or (M,) ndarray
    vdw_radii_fixed : float or (N,) ndarray
    scale : float
        Fraction of the sum of vdW radii to use as cutoff.
    box_dimensions : (3,) ndarray or None
        Orthorhombic periodic box lengths. When supplied, overlaps across
        opposite box faces are detected with minimum-image distances.

    Returns
    -------
    overlapping : (M,) bool ndarray
    """
    if len(fixed) == 0 or len(mobile) == 0:
        return np.zeros(len(mobile), dtype=bool)

    if np.isscalar(vdw_radii_mobile):
        vdw_radii_mobile = np.full(len(mobile), vdw_radii_mobile)
    if np.isscalar(vdw_radii_fixed):
        vdw_radii_fixed = np.full(len(fixed), vdw_radii_fixed)
    vdw_radii_mobile = np.asarray(vdw_radii_mobile, dtype=float)
    vdw_radii_fixed = np.asarray(vdw_radii_fixed, dtype=float)
    if vdw_radii_mobile.shape != (len(mobile),):
        raise ValueError("vdw_radii_mobile must be scalar or match mobile atoms")
    if vdw_radii_fixed.shape != (len(fixed),):
        raise ValueError("vdw_radii_fixed must be scalar or match fixed atoms")
    if (
        not np.isfinite(vdw_radii_mobile).all()
        or not np.isfinite(vdw_radii_fixed).all()
        or np.any(vdw_radii_mobile < 0)
        or np.any(vdw_radii_fixed < 0)
    ):
        raise ValueError("van der Waals radii must be finite and non-negative")

    tree_options = {}
    fixed_for_tree = fixed
    mobile_for_tree = mobile
    if box_dimensions is not None:
        box = np.asarray(box_dimensions, dtype=float)
        if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
            raise ValueError("box_dimensions must contain three positive finite lengths")
        # Periodic cKDTree inputs must be in [0, L); source structures can be
        # centred around zero, so wrap copies rather than mutating callers.
        fixed_for_tree = wrap_periodic_coordinates(fixed, box)
        mobile_for_tree = wrap_periodic_coordinates(mobile, box)
        tree_options["boxsize"] = box

    # Build KD-trees for both atom sets.
    tree_fixed = cKDTree(fixed_for_tree, **tree_options)
    tree_mobile = cKDTree(mobile_for_tree, **tree_options)

    # Single C-level call: find all (fixed, mobile) pairs within worst-case cutoff.
    # This avoids the per-atom Python loop — all neighbor finding happens in C.
    max_mobile_vdw = np.max(vdw_radii_mobile)
    max_fixed_vdw = np.max(vdw_radii_fixed)
    global_cutoff = scale * (max_mobile_vdw + max_fixed_vdw)
    mat = tree_fixed.sparse_distance_matrix(tree_mobile, global_cutoff)

    if mat.count_nonzero() == 0:
        return np.zeros(len(mobile), dtype=bool)

    # Convert to COO for fast row/col/data access
    mat = mat.tocoo()
    pairs_fixed = mat.row
    pairs_mobile = mat.col
    dists = mat.data

    # Vectorized per-pair threshold check using exact per-atom vdW radii
    thresholds = scale * (vdw_radii_mobile[pairs_mobile] + vdw_radii_fixed[pairs_fixed])
    overlap_pairs = dists < thresholds

    # Collect unique mobile atoms that have at least one overlapping neighbor
    overlapping = np.zeros(len(mobile), dtype=bool)
    overlapping[np.unique(pairs_mobile[overlap_pairs])] = True
    return overlapping

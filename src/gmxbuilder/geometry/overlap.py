"""Van der Waals overlap detection and removal using KD-trees."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gmxbuilder.core.system import System


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

    if isinstance(vdw_radii_mobile, float):
        vdw_radii_mobile = np.full(len(mobile), vdw_radii_mobile)
    if isinstance(vdw_radii_fixed, float):
        vdw_radii_fixed = np.full(len(fixed), vdw_radii_fixed)

    tree_options = {}
    fixed_for_tree = fixed
    mobile_for_tree = mobile
    if box_dimensions is not None:
        box = np.asarray(box_dimensions, dtype=float)
        if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
            raise ValueError("box_dimensions must contain three positive finite lengths")
        # Periodic cKDTree inputs must be in [0, L); source structures can be
        # centred around zero, so wrap copies rather than mutating callers.
        fixed_for_tree = np.mod(fixed, box)
        mobile_for_tree = np.mod(mobile, box)
        tree_options["boxsize"] = box

    # Build KD-trees for both atom sets.
    tree_fixed = cKDTree(fixed_for_tree, **tree_options)
    tree_mobile = cKDTree(mobile_for_tree, **tree_options)

    # Single C-level call: find all (fixed, mobile) pairs within worst-case cutoff.
    # This avoids the per-atom Python loop — all neighbor finding happens in C.
    max_mobile_vdw = np.max(vdw_radii_mobile)
    max_fixed_vdw = np.max(vdw_radii_fixed)
    global_cutoff = scale * (max_mobile_vdw + max_fixed_vdw) + 0.02
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


def remove_overlapping_molecules(
    system: System,
    template_system: System,
    vdw_radii: dict[str, float] | None = None,
    overlap_scale: float = 0.8,
    remove_whole_residues: bool = True,
) -> tuple[System, list[int]]:
    """Remove molecules from *template_system* that overlap with *system*.

    Parameters
    ----------
    system : System
        The already-placed system (e.g., protein + membrane).
    template_system : System
        Molecules to check for overlap (e.g., water box).
    vdw_radii : dict or None
        Element -> radius (nm). Uses defaults if None.
    overlap_scale : float
    remove_whole_residues : bool
        If True, remove entire residues that have any overlapping atom.

    Returns
    -------
    filtered_system : System
        *template_system* with overlapping atoms/residues removed.
    removed_resids : list[int]
        Residue IDs of removed molecules.
    """
    if vdw_radii is None:
        vdw_radii = _DEFAULT_VDW_RADII

    coords_a = system.structure.coordinates
    coords_b = template_system.structure.coordinates

    # Assign vdW radii based on element
    def _radii(structure):
        return np.array([
            vdw_radii.get(e.upper(), 0.15) if e else 0.15
            for e in structure.elements
        ])

    radii_a = _radii(system.structure)
    radii_b = _radii(template_system.structure)

    overlap_mask = find_overlapping_atoms(
        coords_b, coords_a,
        vdw_radii_mobile=radii_b,
        vdw_radii_fixed=radii_a,
        scale=overlap_scale,
    )

    if not overlap_mask.any():
        return template_system.copy(), []

    # Determine which residues to keep
    resids = np.array(template_system.structure.resids)
    if remove_whole_residues and len(resids) == len(overlap_mask):
        overlapping_resids = np.unique(resids[overlap_mask])
        keep_mask = ~np.isin(resids, overlapping_resids)
    else:
        keep_mask = ~overlap_mask

    # Build filtered system
    keep_indices = np.where(keep_mask)[0]
    if len(keep_indices) == 0:
        # Return empty system with same box
        from gmxbuilder.core.structure import Structure
        empty = Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=template_system.structure.box_vectors.copy(),
        )
        return System(structure=empty), list(set(resids[overlap_mask]))

    new_coords = coords_b[keep_indices]
    new_structure = template_system.structure.copy()
    new_structure.coordinates = new_coords
    new_structure.atom_names = list(np.array(new_structure.atom_names)[keep_indices])
    new_structure.resnames = list(np.array(new_structure.resnames)[keep_indices])
    new_structure.resids = list(np.array(new_structure.resids)[keep_indices])
    new_structure.chain_ids = list(np.array(new_structure.chain_ids)[keep_indices])
    new_structure.segids = list(np.array(new_structure.segids)[keep_indices])
    new_structure.elements = list(np.array(new_structure.elements)[keep_indices])
    new_structure.occupancies = list(np.array(new_structure.occupancies)[keep_indices])
    new_structure.tempfactors = list(np.array(new_structure.tempfactors)[keep_indices])

    filtered = System(
        structure=new_structure,
        topology=None,  # topology indices are not remapped to filtered structure
        metadata=template_system.metadata.copy(),
    )

    removed = list(set(resids[overlap_mask]))
    return filtered, removed

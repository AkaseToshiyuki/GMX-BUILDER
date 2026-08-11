"""Simple repulsion-based relaxation for lipid placement.

After placing lipids on a grid, some molecules may be too close to
each other.  This module applies short-range repulsive displacements
at the whole-lipid level — each lipid is moved as a rigid body so
covalent bond geometry is strictly preserved.

This is NOT a full energy minimization, just a "nudge" to relieve
the worst inter-lipid VDW clashes.  True equilibration happens in MD.
"""

from __future__ import annotations

import numpy as np


def relax_interleaflet_clashes_xy(
    upper: np.ndarray,
    lower: np.ndarray,
    upper_sizes: list[int],
    lower_sizes: list[int],
    *,
    cutoff: float = 0.20,
    displacement: float = 0.025,
    n_iterations: int = 120,
    rng: np.random.Generator | None = None,
    box_xy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Relieve cross-leaflet tail clashes without changing membrane DHH.

    The lower leaflet is translated as one rigid periodic sheet. Moving
    individual lipids in response to thousands of tail contacts can collapse
    an otherwise uniform lattice because the many-body pair vectors do not
    define a stable per-molecule descent direction. A bounded rigid offset
    preserves both leaflets' APL and lateral ordering while breaking exact
    upper/lower tail coincidences before energy minimization.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not len(upper) or not len(lower):
        return upper, lower
    if sum(upper_sizes) != len(upper) or sum(lower_sizes) != len(lower):
        raise ValueError("lipid_sizes must partition both leaflets")
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")

    from scipy.spatial import cKDTree

    del rng
    z_origin = min(float(upper[:, 2].min()), float(lower[:, 2].min())) - 1.0
    z_box = (
        max(float(upper[:, 2].max()), float(lower[:, 2].max()))
        - z_origin
        + 1.0
    )
    upper_search = upper.copy()
    if box_xy is not None:
        upper_search[:, :2] = np.mod(upper_search[:, :2], box_xy)
    upper_search[:, 2] -= z_origin
    tree_options = (
        {"boxsize": np.asarray([box_xy, box_xy, z_box])}
        if box_xy is not None else {}
    )
    tree = cKDTree(upper_search, **tree_options)

    # The established arguments retain their meaning as search controls, so
    # this robustness fix does not introduce a new caller-facing parameter.
    max_shift = max(float(cutoff), float(displacement) * 4.0)
    samples = min(17, max(5, int(np.sqrt(max(1, n_iterations))) | 1))
    candidate_offsets = np.linspace(-max_shift, max_shift, samples)
    best_score: tuple[float, float, float] | None = None
    best_shift = np.zeros(2, dtype=float)
    for shift_x in candidate_offsets:
        for shift_y in candidate_offsets:
            candidate = lower.copy()
            candidate[:, :2] += np.asarray([shift_x, shift_y])
            if box_xy is not None:
                candidate[:, :2] = np.mod(candidate[:, :2], box_xy)
            candidate[:, 2] -= z_origin
            nearest = tree.query(candidate, k=1)[0]
            score = tuple(
                float(value) for value in np.percentile(nearest, [0.0, 0.1, 1.0])
            )
            if best_score is None or score > best_score:
                best_score = score
                best_shift[:] = (shift_x, shift_y)
    lower[:, :2] += best_shift
    return upper, lower


def scale_lipid_centres_xy(
    coords: np.ndarray,
    lipid_sizes: list[int],
    target_extent: float,
    *,
    tolerance: float = 0.005,
    max_iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a leaflet in XY by translating whole lipids as rigid bodies.

    Only each lipid's centre of geometry is scaled relative to the leaflet
    centre.  Atom coordinates within a lipid receive one common translation,
    so all internal distances, angles, and the Z profile remain unchanged.

    Returns the mutated coordinates and the accumulated ``[scale_x, scale_y]``.
    """
    if len(coords) == 0:
        return coords, np.ones(2, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
    if not lipid_sizes or any(int(size) <= 0 for size in lipid_sizes):
        raise ValueError("lipid_sizes must contain positive atom counts")
    if sum(int(size) for size in lipid_sizes) != len(coords):
        raise ValueError("sum(lipid_sizes) must equal the number of coordinates")
    if not np.isfinite(target_extent) or target_extent <= 0.0:
        raise ValueError("target_extent must be a positive finite number")

    offsets = np.cumsum([0] + [int(size) for size in lipid_sizes])
    total_scale = np.ones(2, dtype=float)

    for _ in range(max_iterations):
        xy_min = coords[:, :2].min(axis=0)
        xy_max = coords[:, :2].max(axis=0)
        extents = xy_max - xy_min
        errors = target_extent - extents
        if np.all(np.abs(errors) <= tolerance):
            break

        lipid_centres = np.array([
            coords[offsets[i]:offsets[i + 1], :2].mean(axis=0)
            for i in range(len(lipid_sizes))
        ])
        centre_spans = np.ptp(lipid_centres, axis=0)
        factors = np.ones(2, dtype=float)
        adjustable = centre_spans > tolerance
        factors[adjustable] = 1.0 + errors[adjustable] / centre_spans[adjustable]
        # Do not reverse the order of lipids when the requested box is
        # smaller than their combined rigid molecular envelopes.
        factors = np.maximum(factors, 0.05)
        if np.allclose(factors, 1.0):
            break

        leaflet_centre = (xy_min + xy_max) / 2.0
        for i, centre in enumerate(lipid_centres):
            shift_xy = (centre - leaflet_centre) * (factors - 1.0)
            coords[offsets[i]:offsets[i + 1], :2] += shift_xy
        total_scale *= factors

    return coords, total_scale


def relax_lipid_clashes(
    coords: np.ndarray,
    atom_names: list[str],
    n_lipids: int = 0,
    *,
    lipid_sizes: list[int] | None = None,
    vdw_cutoff: float = 0.25,      # nm — lipids closer than this are "clashing"
    displacement: float = 0.02,    # nm — push distance per iteration
    n_iterations: int = 20,
    rng: np.random.Generator | None = None,
    freeze_headgroups: bool = True,
    box_xy: float | None = None,
) -> np.ndarray:
    """Apply rigid-body repulsive displacements to relieve lipid-lipid clashes.

    Each lipid is treated as a rigid body — inter-lipid repulsion moves
    entire molecules without distorting internal covalent geometry.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Coordinates of all lipids in one leaflet.  Modified in-place.
    atom_names : list[str]
        Atom names corresponding to each row.
    n_lipids : int
        Number of lipid molecules.  Ignored if *lipid_sizes* is provided.
    lipid_sizes : list[int] or None
        Atom count per lipid (supports mixed-size compositions).
    vdw_cutoff : float
        Lipids whose centre-of-mass distance is less than this (nm) are
        pushed apart.
    displacement : float
        Maximum COM displacement per iteration (nm).
    n_iterations : int
        Number of relaxation iterations.
    rng : np.random.Generator or None
    freeze_headgroups : bool
        Ignored (kept for backward compat; rigid-body moves the whole lipid).

    Returns
    -------
    coords : (N, 3) ndarray (mutated in-place)
    """
    if rng is None:
        rng = np.random.default_rng()

    if lipid_sizes is not None:
        n_lipids = len(lipid_sizes)
    if n_lipids == 0 or len(coords) == 0:
        return coords
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")
    atoms_per_lipid = len(coords) // n_lipids
    if atoms_per_lipid == 0:
        return coords

    # Build per-lipid offsets
    if lipid_sizes is not None:
        sizes = list(lipid_sizes)
    else:
        sizes = [atoms_per_lipid] * n_lipids
    offsets = np.cumsum([0] + sizes)

    from scipy.spatial import cKDTree

    for _ in range(n_iterations):
        centres = np.asarray([
            coords[offsets[index]:offsets[index + 1], :2].mean(axis=0)
            for index in range(n_lipids)
        ])
        search_centres = centres.copy()
        tree_options = {}
        if box_xy is not None:
            search_centres = np.mod(search_centres, box_xy)
            tree_options = {"boxsize": np.asarray([box_xy, box_xy])}
        lipid_pairs = cKDTree(search_centres, **tree_options).query_pairs(
            r=vdw_cutoff, output_type="ndarray"
        )
        if len(lipid_pairs) == 0:
            break
        lipid_i, lipid_j = lipid_pairs[:, 0], lipid_pairs[:, 1]
        centre_delta = centres[lipid_i] - centres[lipid_j]
        if box_xy is not None:
            centre_delta -= box_xy * np.round(centre_delta / box_xy)
        distance = np.linalg.norm(centre_delta, axis=1)
        overlap = np.maximum(vdw_cutoff - distance, 0.0)
        degenerate = distance < 1e-8
        if np.any(degenerate):
            centre_delta[degenerate] = rng.normal(
                0.0, 1.0, (degenerate.sum(), 2)
            )
            distance[degenerate] = np.linalg.norm(
                centre_delta[degenerate], axis=1
            )
        push = centre_delta / distance[:, None] * overlap[:, None] * 0.5
        disp = np.zeros((n_lipids, 2))
        np.add.at(disp, lipid_i, push)
        np.add.at(disp, lipid_j, -push)

        for li in range(n_lipids):
            magnitude = np.linalg.norm(disp[li])
            if magnitude < 1e-12:
                continue
            shift = disp[li]
            if magnitude > displacement:
                shift *= displacement / magnitude
            start, end = offsets[li], offsets[li + 1]
            coords[start:end, :2] += shift

    return coords


def rotate_lipids_away_from_clashes(
    coords: np.ndarray,
    lipid_sizes: list[int],
    *,
    min_distance: float = 0.035,
    angle_samples: int = 24,
    max_rounds: int = 4,
    box_xy: float | None = None,
) -> tuple[np.ndarray, float]:
    """Resolve singular contacts through rigid azimuthal lipid rotations.

    Dense mixed leaflets can contain a rare nearly coincident atom pair even
    after centre translation relaxation. Rotating a whole lipid about its
    centre is a physical in-plane degree of freedom and removes that contact
    without changing APL, DHH, molecular centres, or internal geometry.
    """
    if not lipid_sizes or len(coords) == 0:
        return coords, float("inf")
    sizes = [int(value) for value in lipid_sizes]
    if any(value <= 0 for value in sizes) or sum(sizes) != len(coords):
        raise ValueError("lipid_sizes must contain a complete positive partition")
    if min_distance <= 0.0 or angle_samples < 2 or max_rounds < 1:
        raise ValueError("clash-rotation controls must be positive")
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")

    from scipy.spatial import cKDTree

    offsets = np.cumsum([0] + sizes)
    owners = np.repeat(np.arange(len(sizes)), sizes)
    z_origin = float(coords[:, 2].min()) - 1.0
    z_box = float(np.ptp(coords[:, 2])) + 2.0

    def _tree_coords(values: np.ndarray) -> np.ndarray:
        transformed = values.copy()
        if box_xy is not None:
            transformed[:, :2] = np.mod(transformed[:, :2], box_xy)
        transformed[:, 2] -= z_origin
        return transformed

    tree_options = (
        {"boxsize": np.asarray([box_xy, box_xy, z_box])}
        if box_xy is not None else {}
    )
    angles = np.linspace(0.0, 2.0 * np.pi, angle_samples, endpoint=False)[1:]

    for _ in range(max_rounds):
        full_tree = cKDTree(_tree_coords(coords), **tree_options)
        pairs = full_tree.query_pairs(min_distance, output_type="ndarray")
        if len(pairs) == 0:
            return coords, min_distance
        inter = owners[pairs[:, 0]] != owners[pairs[:, 1]]
        pairs = pairs[inter]
        if len(pairs) == 0:
            return coords, min_distance
        offenders = np.unique(np.concatenate((owners[pairs[:, 0]], owners[pairs[:, 1]])))

        improved = False
        for lipid_index in offenders:
            start, end = offsets[lipid_index], offsets[lipid_index + 1]
            fixed = np.vstack((coords[:start], coords[end:]))
            fixed_tree = cKDTree(_tree_coords(fixed), **tree_options)
            molecule = coords[start:end].copy()
            centre_xy = molecule[:, :2].mean(axis=0)
            local_xy = molecule[:, :2] - centre_xy
            current_clearance = float(
                fixed_tree.query(_tree_coords(molecule), k=1)[0].min()
            )
            best_clearance = current_clearance
            best_xy = molecule[:, :2]
            for angle in angles:
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.asarray([[cosine, -sine], [sine, cosine]])
                candidate = molecule.copy()
                candidate[:, :2] = local_xy @ rotation.T + centre_xy
                clearance = float(
                    fixed_tree.query(_tree_coords(candidate), k=1)[0].min()
                )
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_xy = candidate[:, :2].copy()
            if best_clearance > current_clearance + 1e-6:
                coords[start:end, :2] = best_xy
                improved = True
        if not improved:
            break

    full_tree = cKDTree(_tree_coords(coords), **tree_options)
    pairs = full_tree.query_pairs(min_distance, output_type="ndarray")
    pairs = pairs[owners[pairs[:, 0]] != owners[pairs[:, 1]]] if len(pairs) else pairs
    if len(pairs) == 0:
        return coords, min_distance
    distances = full_tree.sparse_distance_matrix(full_tree, min_distance).tocoo()
    valid = (
        distances.row < distances.col
    ) & (owners[distances.row] != owners[distances.col])
    minimum = float(distances.data[valid].min()) if np.any(valid) else min_distance
    return coords, minimum


def rotate_lipids_away_from_external_clashes(
    coords: np.ndarray,
    lipid_sizes: list[int],
    external: np.ndarray,
    *,
    min_distance: float = 0.05,
    angle_samples: int = 36,
    max_rounds: int = 3,
    box_xy: float | None = None,
) -> tuple[np.ndarray, float]:
    """Rotate whole lipids in XY to remove contacts with another leaflet.

    Molecular centres, Z coordinates, APL, and internal geometry are fixed.
    Candidate rotations are also scored against the other lipids in the same
    leaflet, preventing a cross-leaflet improvement from creating a lateral
    singularity.
    """
    sizes = [int(value) for value in lipid_sizes]
    if not sizes or len(coords) == 0 or len(external) == 0:
        return coords, float("inf")
    if any(value <= 0 for value in sizes) or sum(sizes) != len(coords):
        raise ValueError("lipid_sizes must contain a complete positive partition")
    if min_distance <= 0.0 or angle_samples < 2 or max_rounds < 1:
        raise ValueError("external clash-rotation controls must be positive")
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")

    from scipy.spatial import cKDTree

    offsets = np.cumsum([0] + sizes)
    z_origin = min(float(coords[:, 2].min()), float(external[:, 2].min())) - 1.0
    z_box = (
        max(float(coords[:, 2].max()), float(external[:, 2].max()))
        - z_origin
        + 1.0
    )

    def tree_coords(values: np.ndarray) -> np.ndarray:
        transformed = values.copy()
        if box_xy is not None:
            transformed[:, :2] = np.mod(transformed[:, :2], box_xy)
        transformed[:, 2] -= z_origin
        return transformed

    tree_options = (
        {"boxsize": np.asarray([box_xy, box_xy, z_box])}
        if box_xy is not None else {}
    )
    angles = np.linspace(0.0, 2.0 * np.pi, angle_samples, endpoint=False)[1:]
    external_tree = cKDTree(tree_coords(external), **tree_options)

    for _ in range(max_rounds):
        external_distances = external_tree.query(tree_coords(coords), k=1)[0]
        offenders = np.unique(np.repeat(np.arange(len(sizes)), sizes)[
            external_distances < min_distance
        ])
        if len(offenders) == 0:
            return coords, min_distance
        improved = False
        for lipid_index in offenders:
            start, end = offsets[lipid_index], offsets[lipid_index + 1]
            same_leaflet = np.vstack((coords[:start], coords[end:]))
            fixed = np.vstack((external, same_leaflet))
            fixed_tree = cKDTree(tree_coords(fixed), **tree_options)
            molecule = coords[start:end].copy()
            centre_xy = molecule[:, :2].mean(axis=0)
            local_xy = molecule[:, :2] - centre_xy

            def score(
                candidate: np.ndarray, fixed_tree: cKDTree = fixed_tree,
            ) -> tuple[float, float, float]:
                nearest = fixed_tree.query(tree_coords(candidate), k=1)[0]
                return tuple(
                    float(value)
                    for value in np.percentile(nearest, [0.0, 1.0, 10.0])
                )

            best_score = score(molecule)
            best_xy = molecule[:, :2]
            for angle in angles:
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.asarray([[cosine, -sine], [sine, cosine]])
                candidate = molecule.copy()
                candidate[:, :2] = local_xy @ rotation.T + centre_xy
                candidate_score = score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_xy = candidate[:, :2].copy()
            if not np.array_equal(best_xy, molecule[:, :2]):
                coords[start:end, :2] = best_xy
                improved = True
        if not improved:
            break

    minimum = float(external_tree.query(tree_coords(coords), k=1)[0].min())
    return coords, minimum

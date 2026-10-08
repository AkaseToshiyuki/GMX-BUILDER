"""2D grid generators for lipid placement."""

from __future__ import annotations

import numpy as np

MAX_GRID_CANDIDATES = 1_000_000


def _validated_grid_inputs(
    xy_extent: tuple[float, float],
    center: np.ndarray | None,
    jitter: float,
    max_points: int,
) -> tuple[float, float, float, float, float, int]:
    try:
        extent = np.asarray(xy_extent, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("xy_extent must contain two positive finite dimensions") from exc
    if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0.0):
        raise ValueError("xy_extent must contain two positive finite dimensions")

    if center is None:
        center_array = extent / 2.0
    else:
        try:
            center_array = np.asarray(center, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("center must contain two finite coordinates") from exc
        if center_array.shape != (2,) or not np.isfinite(center_array).all():
            raise ValueError("center must contain two finite coordinates")

    try:
        jitter_value = float(jitter)
    except (TypeError, ValueError) as exc:
        raise ValueError("jitter must be a non-negative finite number") from exc
    if not np.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("jitter must be a non-negative finite number")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be a positive integer")
    point_budget = int(max_points)
    if point_budget <= 0:
        raise ValueError("max_points must be a positive integer")
    return (
        float(extent[0]),
        float(extent[1]),
        float(center_array[0]),
        float(center_array[1]),
        jitter_value,
        point_budget,
    )


def _bounded_ceil_ratio(numerator: float, denominator: float, point_budget: int) -> int:
    ratio = numerator / denominator
    if not np.isfinite(ratio) or ratio > point_budget:
        raise ValueError(f"grid would exceed the {point_budget} candidate-point budget")
    return int(np.ceil(ratio))


def hexagonal_grid(
    xy_extent: tuple[float, float],
    spacing: float,
    center: np.ndarray | None = None,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    max_points: int = MAX_GRID_CANDIDATES,
) -> np.ndarray:
    """Generate a hexagonal (triangular) grid of points in the XY plane.

    Parameters
    ----------
    xy_extent : (x_size, y_size)
        Size of the rectangular region to fill (nm).
    spacing : float
        Distance between neighbouring points (nm).
        For hexagonal packing: area_per_point = spacing^2 * sqrt(3)/2.
    center : (2,) ndarray or None
        (cx, cy) of the grid. If None, uses (x_size/2, y_size/2).
    jitter : float
        Standard deviation of random XY displacement (nm). 0 = no jitter.

    Returns
    -------
    points : (N, 2) ndarray
        (x, y) positions.
    """
    x_size, y_size, cx, cy, jitter, point_budget = _validated_grid_inputs(
        xy_extent, center, jitter, max_points
    )
    try:
        spacing = float(spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must be a positive finite number") from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be a positive finite number")

    # Number of rows / cols needed to cover the rectangle
    # Hexagonal lattice vectors:
    #   a1 = (spacing, 0)
    #   a2 = (spacing/2, spacing * sqrt(3)/2)
    row_height = spacing * np.sqrt(3) / 2.0
    if not np.isfinite(row_height) or row_height <= 0.0:
        raise ValueError("spacing is too small to form a finite hexagonal grid")

    n_cols = _bounded_ceil_ratio(x_size, spacing, point_budget)
    n_rows = _bounded_ceil_ratio(y_size, row_height, point_budget)
    candidate_count = (2 * n_rows + 3) * (2 * n_cols + 3)
    if candidate_count > point_budget:
        raise ValueError(
            f"hexagonal grid would inspect {candidate_count} candidates, "
            f"exceeding the {point_budget} point budget"
        )

    points = []
    for row in range(-n_rows - 1, n_rows + 2):
        y_pos = row * row_height
        x_offset = (spacing / 2.0) if row % 2 != 0 else 0.0
        for col in range(-n_cols - 1, n_cols + 2):
            x_pos = col * spacing + x_offset
            if (
                abs(x_pos) <= x_size / 2.0 + spacing * 0.5
                and abs(y_pos) <= y_size / 2.0 + spacing * 0.5
            ):
                points.append([x_pos, y_pos])

    points = np.array(points, dtype=np.float64)
    if len(points) == 0:
        # Minimum: return a single point at center
        return np.array([[0.0, 0.0]])

    # Apply jitter (before sorting so order is meaningful)
    if jitter > 0:
        if rng is None:
            rng = np.random.default_rng()
        points += rng.normal(0, jitter, size=points.shape)

    # Translate to center
    points[:, 0] += cx
    points[:, 1] += cy

    # Sort by distance from center
    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    points = points[np.argsort(dists)]

    return points


def rectangular_grid(
    xy_extent: tuple[float, float],
    spacing: tuple[float, float],
    center: np.ndarray | None = None,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    max_points: int = MAX_GRID_CANDIDATES,
) -> np.ndarray:
    """Generate a rectangular grid of points in the XY plane.

    Parameters
    ----------
    xy_extent : (x_size, y_size)
    spacing : (dx, dy)
    center : (2,) ndarray or None
    jitter : float

    Returns
    -------
    points : (N, 2) ndarray
    """
    x_size, y_size, cx, cy, jitter, point_budget = _validated_grid_inputs(
        xy_extent, center, jitter, max_points
    )
    try:
        spacing_array = np.asarray(spacing, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must contain two positive finite values") from exc
    if (
        spacing_array.shape != (2,)
        or not np.isfinite(spacing_array).all()
        or np.any(spacing_array <= 0.0)
    ):
        raise ValueError("spacing must contain two positive finite values")
    dx, dy = map(float, spacing_array)

    n_x = _bounded_ceil_ratio(x_size, dx, point_budget) + 1
    n_y = _bounded_ceil_ratio(y_size, dy, point_budget) + 1
    candidate_count = n_x * n_y
    if candidate_count > point_budget:
        raise ValueError(
            f"rectangular grid would allocate {candidate_count} points, "
            f"exceeding the {point_budget} point budget"
        )
    x_vals = np.linspace(-x_size / 2, x_size / 2, n_x)
    y_vals = np.linspace(-y_size / 2, y_size / 2, n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    if jitter > 0:
        if rng is None:
            rng = np.random.default_rng()
        points += rng.normal(0, jitter, size=points.shape)

    points[:, 0] += cx
    points[:, 1] += cy
    return points

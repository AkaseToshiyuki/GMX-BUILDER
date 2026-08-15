"""2D grid generators for lipid placement."""

from __future__ import annotations

import numpy as np


def hexagonal_grid(
    xy_extent: tuple[float, float],
    spacing: float,
    center: np.ndarray | None = None,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
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
    x_size, y_size = xy_extent
    if center is None:
        cx, cy = x_size / 2.0, y_size / 2.0
    else:
        cx, cy = center[0], center[1]

    # Number of rows / cols needed to cover the rectangle
    # Hexagonal lattice vectors:
    #   a1 = (spacing, 0)
    #   a2 = (spacing/2, spacing * sqrt(3)/2)
    row_height = spacing * np.sqrt(3) / 2.0

    n_cols = int(np.ceil(x_size / spacing))
    n_rows = int(np.ceil(y_size / row_height))

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
    dists = np.sqrt(((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2))
    points = points[np.argsort(dists)]

    return points


def rectangular_grid(
    xy_extent: tuple[float, float],
    spacing: tuple[float, float],
    center: np.ndarray | None = None,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
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
    x_size, y_size = xy_extent
    dx, dy = spacing
    if center is None:
        cx, cy = x_size / 2.0, y_size / 2.0
    else:
        cx, cy = center[0], center[1]

    n_x = int(np.ceil(x_size / dx)) + 1
    n_y = int(np.ceil(y_size / dy)) + 1
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

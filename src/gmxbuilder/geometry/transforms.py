"""Geometric transformations — rotation, translation, alignment."""

from __future__ import annotations

import numpy as np


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return rotation matrix that rotates *v1* onto *v2*.

    Uses Rodrigues' rotation formula. Both vectors must be nonzero.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    cos_theta = np.dot(v1_u, v2_u)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    if cos_theta > 0.9999:
        return np.eye(3)
    if cos_theta < -0.9999:
        # 180-degree rotation: pick any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp, v1_u)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1_u, perp)
        axis /= np.linalg.norm(axis)
        return rotation_matrix_from_axis_angle(axis, np.pi)

    axis = np.cross(v1_u, v2_u)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(cos_theta)
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return (3,3) rotation matrix for a given axis and angle (radians)."""
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis

    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)    ],
    ])


def rotation_matrix_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return (3,3) rotation matrix from ZYZ Euler angles (radians)."""
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(psi), np.sin(psi)

    return np.array([
        [c1 * c2 * c3 - s1 * s3,  -c1 * c2 * s3 - s1 * c3,  c1 * s2],
        [s1 * c2 * c3 + c1 * s3,  -s1 * c2 * s3 + c1 * c3,  s1 * s2],
        [-s2 * c3,                 s2 * s3,                  c2     ],
    ])


def align_principal_axis(
    coords: np.ndarray,
    target_axis: np.ndarray = np.array([0, 0, 1]),
) -> np.ndarray:
    """Compute rotation matrix that aligns the principal axis of *coords*
    to *target_axis*.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Atom coordinates.
    target_axis : (3,) ndarray
        Desired direction for the principal axis.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray
    """
    from gmxbuilder.geometry.align import compute_principal_axes
    principal = compute_principal_axes(coords)
    return rotation_matrix_from_vectors(principal, np.asarray(target_axis))

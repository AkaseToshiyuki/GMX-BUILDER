"""Principal component analysis and protein-membrane orientation."""

from __future__ import annotations

import numpy as np

from gmxbuilder.geometry.transforms import rotation_matrix_from_vectors


def compute_principal_axes(
    coords: np.ndarray, masses: np.ndarray | None = None
) -> np.ndarray:
    """Compute principal axes of a point set via weighted PCA.

    Returns the three principal axes as rows of a (3,3) matrix, sorted by
    decreasing eigenvalue.  The **longest** axis (largest variance) is the
    first returned row.

    Parameters
    ----------
    coords : (N, 3) ndarray
    masses : (N,) ndarray or None

    Returns
    -------
    axes : (3, 3) ndarray
        axes[0] = longest axis, axes[1] = medium, axes[2] = shortest.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if masses is None:
        center = coords.mean(axis=0)
    else:
        center = np.average(coords, axis=0, weights=masses)

    centered = coords - center
    if masses is not None:
        centered = centered * np.sqrt(masses)[:, np.newaxis]

    cov = centered.T @ centered / (len(coords) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; reverse for descending
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T

    return axes


def orient_protein_to_membrane(
    coords: np.ndarray,
    method: str = "pca",
    target_axis: np.ndarray | None = None,
    ca_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return a rotation matrix that aligns a protein to the membrane normal.

    Parameters
    ----------
    coords : (N, 3) ndarray
        All-atom or CA coordinates of the protein.
    method : str
        "pca" — use the principal axis.
        "com" — use the vector from N- to C-terminus (simple fallback).
    target_axis : (3,) ndarray or None
        The membrane normal direction (default Z).
    ca_indices : (M,) ndarray or None
        If provided, use only these indices for the calculation.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray
    """
    if target_axis is None:
        target_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        target_axis = np.asarray(target_axis, dtype=np.float64)

    if ca_indices is not None and len(ca_indices) >= 3:
        subset = coords[ca_indices]
    else:
        subset = coords

    if len(subset) < 2:
        # Single atom or empty — no meaningful orientation
        return np.eye(3)
    if method == "pca" and len(subset) >= 3:
        axes = compute_principal_axes(subset)
        principal = axes[0]  # Longest axis
    elif method == "com" or len(subset) < 3:
        # Fallback: use vector from N-term to C-term
        principal = subset[-1] - subset[0]
        if np.linalg.norm(principal) < 1e-8:
            principal = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown orientation method: {method}")

    # Ensure the principal axis points in the +Z hemisphere
    if principal[2] < 0:
        principal = -principal

    return rotation_matrix_from_vectors(principal, target_axis)

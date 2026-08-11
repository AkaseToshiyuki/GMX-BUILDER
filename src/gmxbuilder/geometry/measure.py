"""Distance and geometric measurements using numpy."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def center_of_mass(coords: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Weighted center of mass. If *masses* is None, use uniform weights."""
    coords = np.asarray(coords)
    if masses is None:
        return coords.mean(axis=0)
    return np.average(coords, axis=0, weights=masses)


def center_of_geometry(coords: np.ndarray) -> np.ndarray:
    """Unweighted center of geometry."""
    return np.asarray(coords).mean(axis=0)


def minimal_distance(set1: np.ndarray, set2: np.ndarray) -> float:
    """Minimum distance between any point in *set1* and any point in *set2*."""
    tree = cKDTree(set2)
    dists, _ = tree.query(set1, k=1)
    return float(dists.min())


def all_pairwise_distances(set1: np.ndarray, set2: np.ndarray) -> np.ndarray:
    """All pairwise Euclidean distances between *set1* (M x 3) and *set2* (N x 3).

    Returns (M, N) array.
    """
    diff = set1[:, np.newaxis, :] - set2[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

"""Numerically safe coordinate wrapping for periodic neighbour searches."""

from __future__ import annotations

import numpy as np


def wrap_periodic_coordinates(
    coordinates: np.ndarray,
    box_lengths: float | np.ndarray,
) -> np.ndarray:
    """Return coordinates strictly inside the half-open interval ``[0, L)``.

    ``numpy.remainder`` can round a tiny negative coordinate to exactly ``L``.
    SciPy's periodic ``cKDTree`` correctly rejects that boundary value, so
    clamp the wrapped copy to the greatest representable value below each box
    length. Callers are never mutated.
    """
    values = np.asarray(coordinates, dtype=float)
    lengths = np.asarray(box_lengths, dtype=float)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("periodic coordinates must have a non-empty final axis")
    if lengths.ndim == 0:
        lengths = np.full(values.shape[-1], float(lengths))
    if lengths.shape != (values.shape[-1],):
        raise ValueError("box lengths must be scalar or match the coordinate axis")
    if not np.isfinite(values).all():
        raise ValueError("periodic coordinates must be finite")
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("periodic box lengths must be positive and finite")
    wrapped = np.remainder(values, lengths)
    upper = np.nextafter(lengths, np.zeros_like(lengths))
    return np.minimum(np.maximum(wrapped, 0.0), upper)

"""Separate deposited crystallographic evidence from usable display bounds."""

import numpy as np


def classify_cell(box):
    if box is None:
        return "missing"
    box = np.asarray(box)
    if not np.isfinite(box).all() or np.linalg.det(box) <= 0:
        return "invalid"
    if np.allclose(box, np.eye(3) * 0.1, atol=1e-7, rtol=0):
        return "placeholder"
    return "deposited"


def display_envelope(coordinates):
    extent = np.ptp(coordinates, axis=0)
    return np.eye(3) * max(float(extent.max()) * 1.3, 3.0)

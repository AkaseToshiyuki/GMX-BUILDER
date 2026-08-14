from __future__ import annotations

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.builder import _validate_membrane_quality


def _system(coordinates: np.ndarray) -> System:
    count = len(coordinates)
    return System(
        structure=Structure(
            coordinates=np.asarray(coordinates, dtype=float),
            box_vectors=np.diag([6.0, 6.0, 20.0]),
            atom_names=["CA"] + ["C"] * (count - 1),
            resnames=["ALA"] + ["POPC"] * (count - 1),
            resids=list(range(1, count + 1)),
            chain_ids=["A"] + ["M"] * (count - 1),
            elements=["C"] * count,
        )
    )


def test_membrane_quality_rejects_protein_outside_bilayer_envelope():
    system = _system(np.array([
        [0.0, 0.0, 8.0],
        [-1.0, -1.0, 2.0],
        [1.0, -1.0, 2.0],
        [-1.0, 1.0, -2.0],
        [1.0, 1.0, -2.0],
    ]))

    with pytest.raises(ModuleConfigError, match="Z envelopes do not intersect"):
        _validate_membrane_quality(
            system,
            np.arange(1, 5),
            1,
            6.0,
            20.0,
            True,
            [],
        )


def test_membrane_quality_accepts_intersecting_protein_envelope():
    system = _system(np.array([
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 2.0],
        [1.0, -1.0, 2.0],
        [-1.0, 1.0, -2.0],
        [1.0, 1.0, -2.0],
    ]))

    _validate_membrane_quality(
        system,
        np.arange(1, 5),
        1,
        6.0,
        20.0,
        True,
        [],
    )

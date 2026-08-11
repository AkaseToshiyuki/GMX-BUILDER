"""Scientific invariants for lipid head/tail orientation."""

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.membrane.builder import _validate_bilayer_structure
from gmxbuilder.modules.membrane.lipid_orientation import (
    LipidOrientationError,
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
)


def _amphiphile():
    names = ["P", "O1", "N", "C1", "C2", "C3", "C4", "C5", "H1"]
    coordinates = np.asarray([
        [0.0, 0.0, -0.9],
        [0.1, 0.0, -0.8],
        [-0.1, 0.0, -0.7],
        [0.0, 0.0, -0.2],
        [0.1, 0.0, 0.2],
        [0.2, 0.0, 0.6],
        [0.1, 0.0, 1.0],
        [0.0, 0.0, 1.4],
        [0.0, 0.1, -0.9],
    ])
    return coordinates, names


@pytest.mark.parametrize("upper", [True, False])
def test_orientation_is_a_rigid_body_correction_with_heads_outward(upper):
    coordinates, names = _amphiphile()
    distances_before = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )

    oriented = orient_lipid_to_outward_normal(
        coordinates, names, upper=upper,
    )
    profile = infer_lipid_orientation(oriented, names)
    projection, cosine = outward_orientation(profile, upper=upper)

    distances_after = np.linalg.norm(
        oriented[:, None, :] - oriented[None, :, :], axis=2
    )
    assert projection > 0.1
    assert cosine == pytest.approx(1.0)
    assert np.allclose(distances_after, distances_before)


def test_non_amphiphilic_molecule_is_rejected_explicitly():
    with pytest.raises(LipidOrientationError, match="no polar"):
        infer_lipid_orientation(
            np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            ["C1", "C2", "C3"],
        )


def test_final_bilayer_gate_rejects_one_solvent_facing_tail():
    coordinates, names = _amphiphile()
    upper = orient_lipid_to_outward_normal(coordinates, names, upper=False)
    lower = orient_lipid_to_outward_normal(coordinates, names, upper=False)
    upper[:, 2] += 1.0
    lower[:, 2] -= 1.0

    def leaflet(values):
        return System(
            Structure(
                coordinates=values,
                box_vectors=np.eye(3) * 5.0,
                atom_names=names,
            ),
            metadata={
                "n_lipids": 1,
                "lipid_sizes": [len(names)],
                "headgroup_anchor_local_indices": [0],
            },
        )

    with pytest.raises(ModuleConfigError, match="orientation validation failed"):
        _validate_bilayer_structure(leaflet(upper), leaflet(lower), [])


def test_bilayer_quality_gate_is_identical_with_parallel_kdtree_workers(
    monkeypatch,
):
    coordinates, names = _amphiphile()
    upper = orient_lipid_to_outward_normal(coordinates, names, upper=True)
    lower = orient_lipid_to_outward_normal(coordinates, names, upper=False)
    upper[:, 2] += 1.5
    lower[:, 2] -= 1.5

    def leaflet(values):
        return System(
            Structure(
                coordinates=values,
                box_vectors=np.eye(3) * 5.0,
                atom_names=names,
            ),
            metadata={
                "n_lipids": 1,
                "lipid_sizes": [len(names)],
                "headgroup_anchor_local_indices": [0],
            },
        )

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.builder.configured_task_threads",
        lambda: 1,
    )
    serial = _validate_bilayer_structure(
        leaflet(upper.copy()), leaflet(lower.copy()), []
    )
    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.builder.configured_task_threads",
        lambda: 4,
    )
    parallel = _validate_bilayer_structure(
        leaflet(upper.copy()), leaflet(lower.copy()), []
    )

    assert parallel == serial

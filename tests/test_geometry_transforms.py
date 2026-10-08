"""Degenerate-input guards for shared rigid-body geometry."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gmxbuilder.geometry.align import compute_principal_axes
from gmxbuilder.geometry.grid import hexagonal_grid, rectangular_grid
from gmxbuilder.geometry.periodic import wrap_periodic_coordinates
from gmxbuilder.geometry.transforms import (
    rotation_matrix_from_axis_angle,
    rotation_matrix_from_vectors,
)


@pytest.mark.parametrize(
    "source,target",
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([1.0, 0.0, 0.0], [np.nan, 0.0, 1.0]),
    ],
)
def test_vector_alignment_rejects_zero_or_nonfinite_vectors(source, target):
    with pytest.raises(ValueError):
        rotation_matrix_from_vectors(source, target)


def test_axis_angle_rejects_degenerate_axis_and_angle():
    with pytest.raises(ValueError):
        rotation_matrix_from_axis_angle([0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        rotation_matrix_from_axis_angle([1.0, 0.0, 0.0], np.inf)


def test_principal_axes_reject_degenerate_coordinates_and_masses():
    with pytest.raises(ValueError, match="non-degenerate"):
        compute_principal_axes(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="masses"):
        compute_principal_axes(np.eye(3), masses=np.asarray([1.0, 0.0, 1.0]))


def test_valid_vector_alignment_remains_a_proper_rotation():
    rotation = rotation_matrix_from_vectors(
        np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation @ np.asarray([1.0, 0.0, 0.0]) == pytest.approx(np.asarray([0.0, 0.0, 1.0]))


def test_nearly_degenerate_principal_axes_use_deterministic_proper_basis():
    square = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    first = compute_principal_axes(square)
    second = compute_principal_axes(square.copy())

    assert np.array_equal(first, second)
    assert first @ first.T == pytest.approx(np.eye(3))
    assert np.linalg.det(first) == pytest.approx(1.0)


def test_periodic_wrap_never_rounds_tiny_negative_values_to_box_length():
    box = np.asarray([4.0, 5.0, 6.0])
    coordinates = np.asarray(
        [
            [-np.finfo(float).eps, 5.0, 12.0 + np.finfo(float).eps],
            [4.0, -np.finfo(float).eps, 6.0],
        ]
    )

    wrapped = wrap_periodic_coordinates(coordinates, box)

    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < box)
    # SciPy is the downstream authority that rejected the former DAPE input.
    tree = cKDTree(wrapped, boxsize=box)
    assert tree.n == 2


@pytest.mark.parametrize(
    "factory,args",
    [
        (hexagonal_grid, ((float("inf"), 1.0), 0.5)),
        (hexagonal_grid, ((1.0, 1.0), 0.0)),
        (rectangular_grid, ((1.0, -1.0), (0.5, 0.5))),
        (rectangular_grid, ((1.0, 1.0), (float("nan"), 0.5))),
    ],
)
def test_grid_generators_reject_nonfinite_or_nonpositive_geometry(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


def test_grid_generators_fail_before_exceeding_candidate_budget():
    with pytest.raises(ValueError, match="point budget"):
        hexagonal_grid((100.0, 100.0), 0.01, max_points=10_000)
    with pytest.raises(ValueError, match="point budget"):
        rectangular_grid((100.0, 100.0), (0.01, 0.01), max_points=10_000)

    assert hexagonal_grid((2.0, 2.0), 1.0, max_points=100).shape[1] == 2

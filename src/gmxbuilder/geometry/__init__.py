"""Geometry module — pure numpy/scipy geometric operations."""

from gmxbuilder.geometry.transforms import (
    rotation_matrix_from_vectors,
    rotation_matrix_from_axis_angle,
    rotation_matrix_from_euler,
    align_principal_axis,
)
from gmxbuilder.geometry.measure import (
    center_of_mass,
    center_of_geometry,
    minimal_distance,
    all_pairwise_distances,
)
from gmxbuilder.geometry.align import (
    compute_principal_axes,
    orient_protein_to_membrane,
)
from gmxbuilder.geometry.grid import (
    hexagonal_grid,
    rectangular_grid,
)
from gmxbuilder.geometry.overlap import (
    find_overlapping_atoms,
)

__all__ = [
    "rotation_matrix_from_vectors",
    "rotation_matrix_from_axis_angle",
    "rotation_matrix_from_euler",
    "align_principal_axis",
    "center_of_mass",
    "center_of_geometry",
    "minimal_distance",
    "all_pairwise_distances",
    "compute_principal_axes",
    "orient_protein_to_membrane",
    "hexagonal_grid",
    "rectangular_grid",
    "find_overlapping_atoms",
]

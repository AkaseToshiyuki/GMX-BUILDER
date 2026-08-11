"""Scientific geometry regressions for force-field-native modifications."""

import numpy as np
import pytest

from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
from gmxbuilder.modules.modifications.patches import (
    ALL_PATCHES,
    _TEMPLATE_PATCH_SUPPORT,
)
from gmxbuilder.modules.modifications.geometry import (
    ModificationGeometryError,
    _angle,
    build_modified_heavy_atom_geometry,
)
from gmxbuilder.modules.modifications.processor import StructureProcessor
from tests.test_gromacs_smoke import _three_residue_modification_system


_TYR_HEAVY_COORDINATES = {
    "N": [1.1047, 1.0181, -1.3842],
    "CA": [1.1030, 1.1191, -1.2772],
    "C": [1.2337, 1.1987, -1.2724],
    "O": [1.2291, 1.3194, -1.2490],
    "CB": [1.0715, 1.0538, -1.1424],
    "CG": [0.9311, 1.0816, -1.0967],
    "CD1": [0.9021, 1.1987, -1.0233],
    "CD2": [0.8258, 0.9922, -1.1291],
    "CE1": [0.7712, 1.2259, -0.9815],
    "CE2": [0.6962, 1.0195, -1.0881],
    "CZ": [0.6669, 1.1361, -1.0137],
    "OH": [0.5385, 1.1623, -0.9726],
}


_SUPPORTED_NATIVE_CASES = [
    (force_field, patch_id)
    for patch_id, force_fields in _TEMPLATE_PATCH_SUPPORT.items()
    for force_field in sorted(force_fields)
]


def _build_ptr(coordinates):
    template = load_force_field_rtp("charmm36m").get_residue("PTR")
    assert template is not None
    return build_modified_heavy_atom_geometry(
        force_field="charmm36m",
        template=template,
        retained_coordinates={
            name: np.asarray(position, dtype=float)
            for name, position in coordinates.items()
        },
    )


def test_ptr_uses_force_field_tetrahedral_geometry():
    coordinates, quality = _build_ptr(_TYR_HEAVY_COORDINATES)

    phosphate_angles = [
        _angle(coordinates[first], coordinates["P"], coordinates[second])
        for first, second in (
            ("OH", "O1P"),
            ("OH", "O2P"),
            ("OH", "O3P"),
            ("O1P", "O2P"),
            ("O1P", "O3P"),
            ("O2P", "O3P"),
        )
    ]

    assert quality.max_bond_error_nm < 0.001
    assert quality.max_angle_error_deg < 2.0
    assert min(phosphate_angles) > 95.0
    assert max(phosphate_angles) < 125.0


def test_modified_geometry_is_rotation_and_translation_covariant():
    original, original_quality = _build_ptr(_TYR_HEAVY_COORDINATES)
    angle = np.deg2rad(63.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([2.1, -0.7, 1.3])
    transformed_input = {
        name: rotation @ np.asarray(position) + translation
        for name, position in _TYR_HEAVY_COORDINATES.items()
    }
    transformed, transformed_quality = _build_ptr(transformed_input)

    local_group = ("OH", "P", "O1P", "O2P", "O3P")
    for first_index, first in enumerate(local_group):
        for second in local_group[first_index + 1:]:
            original_distance = np.linalg.norm(original[first] - original[second])
            transformed_distance = np.linalg.norm(
                transformed[first] - transformed[second]
            )
            assert transformed_distance == pytest.approx(original_distance, abs=2e-3)
    assert transformed_quality.max_bond_error_nm == pytest.approx(
        original_quality.max_bond_error_nm, abs=1e-7
    )
    assert transformed_quality.max_angle_error_deg == pytest.approx(
        original_quality.max_angle_error_deg, abs=0.05
    )


def test_modified_geometry_rejects_an_unavoidable_external_overlap():
    count = 300
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(1.0 - z * z)
    directions = np.column_stack((
        radius * np.cos(golden_angle * indices),
        radius * np.sin(golden_angle * indices),
        z,
    ))
    attachment = np.asarray(_TYR_HEAVY_COORDINATES["OH"])
    blocked_environment = attachment + 0.161 * directions
    template = load_force_field_rtp("charmm36m").get_residue("PTR")

    with pytest.raises(ModificationGeometryError, match="outside tolerance|overlap"):
        build_modified_heavy_atom_geometry(
            force_field="charmm36m",
            template=template,
            retained_coordinates={
                name: np.asarray(position, dtype=float)
                for name, position in _TYR_HEAVY_COORDINATES.items()
            },
            environment_coordinates=blocked_environment,
        )


@pytest.mark.parametrize("force_field,patch_id", _SUPPORTED_NATIVE_CASES)
def test_every_enabled_native_template_passes_geometry_validation(
    force_field, patch_id
):
    patch = ALL_PATCHES[patch_id]
    system = _three_residue_modification_system(
        force_field, patch.target_residues[0]
    )
    result = StructureProcessor().run(
        system,
        {
            "skip_protonation": True,
            "modifications": [{"index": 1, "patch_id": patch_id}],
        },
    )

    report = result.system.metadata["modification_geometry"]
    assert len(report) == 1
    assert report[0]["patch_id"] == patch_id
    assert report[0]["status"] == "passed"
    assert report[0]["max_bond_error_nm"] < 0.005
    assert report[0]["max_angle_error_deg"] < 10.0
    assert report[0]["min_nonbonded_distance_nm"] >= 0.08
    assert report[0]["stereo_centres"] == [
        constraint.label for constraint in patch.stereo_constraints
    ]

    target_indices = [
        index for index, residue_name in enumerate(result.system.structure.resnames)
        if residue_name == patch.product_name
    ]
    target_coordinates = {
        result.system.structure.atom_names[index].strip():
            result.system.structure.coordinates[index]
        for index in target_indices
    }
    for constraint in patch.stereo_constraints:
        selectors = (constraint.center, *constraint.ordered_neighbors)
        center, first, second, third = [
            next(name for name in selector if name in target_coordinates)
            for selector in selectors
        ]
        origin = target_coordinates[center]
        signed_volume = float(np.dot(
            np.cross(
                target_coordinates[first] - origin,
                target_coordinates[second] - origin,
            ),
            target_coordinates[third] - origin,
        ))
        assert constraint.expected_sign * signed_volume > 2.0e-4


def test_multiple_native_modifications_across_chains_are_independent():
    tyrosine = _three_residue_modification_system("charmm36m", "TYR")
    lysine = _three_residue_modification_system("charmm36m", "LYS")
    lysine.structure.chain_ids = ["B"] * lysine.num_atoms
    lysine.structure.coordinates += np.array([0.0, 2.0, 0.0])
    combined = tyrosine.merge(lysine)

    result = StructureProcessor().run(
        combined,
        {
            "skip_protonation": True,
            "modifications": [
                {"index": 1, "patch_id": "PHOS_TYR"},
                {"index": 4, "patch_id": "KME3_LYS"},
            ],
        },
    )

    assert result.success
    assert {"PTR", "M3L"}.issubset(set(result.system.structure.resnames))
    assert [
        record["patch_id"]
        for record in result.system.metadata["modification_geometry"]
    ] == ["PHOS_TYR", "KME3_LYS"]

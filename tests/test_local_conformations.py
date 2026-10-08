"""Internal coordinates checked against independent analytical/MDTraj references."""

import numpy as np
import pytest

from gmxbuilder.modules.membrane.local_conformations import (
    canonical_coordinates,
    dihedrals,
    internal_descriptors,
    validate_definition,
    validate_intrinsic_geometry,
)
from tests.local_relaxation_fixture import molecule_definition


def test_signed_dihedral_matches_mdtraj_under_rigid_transforms():
    import mdtraj as md
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(77)
    xyz = rng.normal(size=(30, 7, 3)).astype("float32")
    quads = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
    topology = md.Topology()
    residue = topology.add_residue("MOL", topology.add_chain())
    for _ in range(7):
        topology.add_atom("C", md.element.carbon, residue)
    expected = md.compute_dihedrals(md.Trajectory(xyz, topology), quads, periodic=False)
    actual = dihedrals(xyz, quads)
    np.testing.assert_allclose(np.exp(1j * actual), np.exp(1j * expected), atol=2e-6)
    rotated = xyz @ Rotation.from_rotvec([0.4, -1.2, 0.6]).as_matrix().T + [5, 3, -4]
    np.testing.assert_allclose(
        np.exp(1j * dihedrals(rotated, quads)), np.exp(1j * expected), atol=2e-6
    )


def test_shape_distances_and_radius_have_hand_calculated_reference():
    xyz = np.array([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    definition = {
        "heavy": [0, 1, 2, 3],
        "polar": [0],
        "tails": [2, 3],
        "anchor": 0,
        "endpoints": [3],
        "atom_names": ["O", "C1", "C2", "C3"],
        "torsions": [],
    }
    measured = internal_descriptors(xyz, definition)
    assert measured["shape:rg_nm"][0] == pytest.approx(np.sqrt(1.25))
    assert measured["shape:head_tail_nm"][0] == 2.5
    assert measured["shape:end_C3_nm"][0] == 3


def test_canonical_frame_preserves_all_pair_distances_and_signed_volume():
    from scipy.spatial.distance import pdist

    _, definition = molecule_definition()
    xyz = np.random.default_rng(88).normal(size=(len(definition["atom_names"]), 3))
    transformed = canonical_coordinates(xyz, definition)
    np.testing.assert_allclose(pdist(xyz), pdist(transformed), atol=1e-12)
    np.testing.assert_allclose(transformed[definition["anchor"]], 0, atol=1e-12)
    assert np.linalg.det(xyz[1:4] - xyz[0]) == pytest.approx(
        np.linalg.det(transformed[1:4] - transformed[0])
    )


def test_descriptor_mapping_rejects_tampered_torsions():
    import copy

    _, original = molecule_definition()
    validate_definition(original)
    edited = copy.deepcopy(original)
    edited["torsions"] = []
    with pytest.raises(ValueError, match="Changed"):
        validate_definition(edited)


def test_intrinsic_guards_reject_broken_bond_and_nonlocal_self_overlap():
    elements = ("C",) * 6
    bonds = tuple((i, i + 1) for i in range(5))
    xyz = np.array([[i * 0.154, 0.0, 0] for i in range(6)])
    validate_intrinsic_geometry(xyz, elements, bonds)
    broken = xyz.copy()
    broken[-1, 0] += 0.4
    with pytest.raises(ValueError, match="covalent"):
        validate_intrinsic_geometry(broken, elements, bonds)
    # Regular five-edge almost-closed hexagon: covalent lengths remain valid,
    # but atoms 0 and 5 overlap despite being five graph bonds apart.
    angles = np.arange(6) * 2 * np.pi / 5
    overlap = (
        np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
        * 0.154
        / (2 * np.sin(np.pi / 5))
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_intrinsic_geometry(overlap, elements, bonds)


def test_hydration_uses_periodic_water_contacts_with_a_hand_count():
    from gmxbuilder.modules.membrane.v4_trajectory import TrajectorySampler

    sampler = TrajectorySampler.__new__(TrajectorySampler)
    sampler.water_oxygens = np.array([2, 3, 4])
    sampler.target_group = {"indices": np.array([[0], [1]]), "polar": np.array([0])}
    xyz = np.array(
        [[0.05, 0.5, 0.5], [1, 0.5, 0.5], [1.95, 0.5, 0.5], [0.3, 0.5, 0.5], [1.2, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(sampler.hydration(xyz, np.eye(3) * 2), [2, 1])
    np.testing.assert_array_equal(sampler.hydration(xyz + [8, -6, 2], np.eye(3) * 2), [2, 1])


def test_bond_lengths_and_angles_match_independent_mdtraj_reference():
    import mdtraj as md

    _, definition = molecule_definition()
    xyz = (
        np.random.default_rng(188)
        .normal(size=(20, len(definition["atom_names"]), 3))
        .astype("float32")
    )
    topology = md.Topology()
    residue = topology.add_residue("MOL", topology.add_chain())
    for name, element in zip(definition["atom_names"], definition["elements"], strict=True):
        topology.add_atom(name, md.element.get_by_symbol(element), residue)
    trajectory = md.Trajectory(xyz, topology)
    distances = md.compute_distances(trajectory, definition["heavy_bonds"], periodic=False)
    angles = np.degrees(md.compute_angles(trajectory, definition["heavy_angles"], periodic=False))
    actual = internal_descriptors(xyz, definition)
    for index, pair in enumerate(definition["heavy_bonds"]):
        key = "bond:" + "-".join(definition["atom_names"][i] for i in pair)
        np.testing.assert_allclose(actual[key], distances[:, index], atol=1e-6)
    for index, triple in enumerate(definition["heavy_angles"]):
        key = "angle:" + "-".join(definition["atom_names"][i] for i in triple)
        np.testing.assert_allclose(actual[key], angles[:, index], atol=1e-4)


def test_torsion_sector_boundaries_form_an_exact_partition(monkeypatch):
    from gmxbuilder.modules.membrane import local_conformations as module

    _, definition = molecule_definition()
    boundary = np.array([-np.pi, -2 * np.pi / 3, 0, 2 * np.pi / 3, np.pi])
    monkeypatch.setattr(
        module, "dihedrals", lambda xyz, quads: np.repeat(boundary[:, None], len(quads), axis=1)
    )
    measured = internal_descriptors(np.zeros((5, len(definition["atom_names"]), 3)), definition)
    for key in measured:
        if ":trans" in key:
            np.testing.assert_array_equal(
                measured[key]
                + measured[key.replace(":trans", ":minus")]
                + measured[key.replace(":trans", ":plus")],
                np.ones(5),
            )


def test_folded_source_direction_is_counted_as_outlier_not_a_molecule_veto():
    from gmxbuilder.modules.membrane.lipid_orientation import (
        LipidOrientationError,
        infer_lipid_orientation,
        outward_orientation,
    )
    from gmxbuilder.modules.membrane.local_conformations import source_orientation

    xyz = np.array([[0.0, 0.0, 0.05], [-0.2, 0, 0], [0, 0.2, 0], [0.2, -0.2, 0]])
    with pytest.raises(LipidOrientationError):
        infer_lipid_orientation(xyz, ["O", "C1", "C2", "C3"])
    profile = source_orientation(xyz, {"polar": [0], "tails": [1, 2, 3]})
    projection, cosine = outward_orientation(profile, upper=True)
    assert projection == pytest.approx(0.05)
    assert cosine == pytest.approx(1.0)
    assert projection < 0.10  # The existing population gate sees the outlier.
    np.testing.assert_array_equal(profile.tail_centroid, [0.0, 0.0, 0.0])
    xyz[0, 2] = 0
    profile = source_orientation(xyz, {"polar": [0], "tails": [1, 2, 3]})
    assert outward_orientation(profile, upper=True) == (0.0, 0.0)

"""Physical-invariant tests for rigid-body leaflet scaling."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.relax import (
    relax_interleaflet_clashes_xy,
    relax_lipid_clashes,
    rotate_lipids_away_from_clashes,
    rotate_lipids_away_from_external_clashes,
    scale_lipid_centres_xy,
)
from gmxbuilder.modules.membrane.builder import (
    MembraneBuilder,
    _headgroup_anchor_index,
    _leaflet_headgroup_plane,
    _weighted_leaflet_apl,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry


def _distance_matrix(coords):
    delta = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(delta, axis=2)


def test_xy_scaling_moves_whole_lipids_without_distorting_geometry():
    lipid_a = np.array(
        [
            [-2.1, -1.0, 0.3],
            [-1.9, -1.0, 0.3],
            [-2.0, -0.8, -0.2],
        ]
    )
    lipid_b = np.array(
        [
            [1.9, 1.0, 0.4],
            [2.1, 1.0, 0.4],
            [2.0, 1.2, -0.1],
        ]
    )
    coords = np.vstack([lipid_a, lipid_b])
    before_a = _distance_matrix(coords[:3].copy())
    before_b = _distance_matrix(coords[3:].copy())
    before_z = coords[:, 2].copy()

    scaled, factors = scale_lipid_centres_xy(coords, [3, 3], target_extent=6.0)

    assert np.allclose(_distance_matrix(scaled[:3]), before_a)
    assert np.allclose(_distance_matrix(scaled[3:]), before_b)
    assert np.array_equal(scaled[:, 2], before_z)
    assert np.allclose(np.ptp(scaled[:, :2], axis=0), [6.0, 6.0], atol=0.005)
    assert np.all(factors > 1.0)


def test_xy_scaling_validates_lipid_partition():
    coords = np.zeros((3, 3))

    with pytest.raises(ValueError, match="sum\\(lipid_sizes\\)"):
        scale_lipid_centres_xy(coords, [2, 2], target_extent=4.0)


def test_cross_leaflet_relaxation_preserves_z_coordinates():
    upper = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 1.9]])
    lower = np.array([[0.01, 0.0, -0.1], [0.0, 0.0, -1.9]])
    upper_z = upper[:, 2].copy()
    lower_z = lower[:, 2].copy()

    relax_interleaflet_clashes_xy(upper, lower, [2], [2])

    assert np.array_equal(upper[:, 2], upper_z)
    assert np.array_equal(lower[:, 2], lower_z)
    assert np.linalg.norm(upper[0] - lower[0]) >= 0.19


def test_cross_leaflet_relaxation_preserves_each_leaflet_lattice():
    upper = np.array(
        [
            [0.0, 0.0, 0.05],
            [0.8, 0.0, 0.05],
        ]
    )
    lower = np.array(
        [
            [0.0, 0.0, -0.05],
            [0.8, 0.0, -0.05],
        ]
    )
    upper_before = upper.copy()
    lower_vector = lower[1] - lower[0]

    relax_interleaflet_clashes_xy(
        upper,
        lower,
        [1, 1],
        [1, 1],
        cutoff=0.20,
        displacement=0.05,
        n_iterations=20,
        box_xy=4.0,
    )

    assert np.array_equal(upper, upper_before)
    assert np.allclose(lower[1] - lower[0], lower_vector)


def test_same_leaflet_relaxation_detects_periodic_face_clashes():
    coordinates = np.array(
        [
            [0.01, 1.0, 0.0],
            [5.99, 1.0, 0.0],
        ]
    )

    relaxed = relax_lipid_clashes(
        coordinates,
        ["C1", "C1"],
        lipid_sizes=[1, 1],
        vdw_cutoff=0.12,
        displacement=0.03,
        n_iterations=20,
        rng=np.random.default_rng(5),
        box_xy=6.0,
    )

    delta = relaxed[0] - relaxed[1]
    delta[:2] -= 6.0 * np.round(delta[:2] / 6.0)
    assert np.linalg.norm(delta) >= 0.119


def test_same_leaflet_relaxation_rejects_invalid_partitions():
    coordinates = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError, match="partition"):
        relax_lipid_clashes(
            coordinates,
            ["C1", "C1", "C1"],
            lipid_sizes=[1, 1],
        )


def test_cross_leaflet_relaxation_detects_periodic_face_clashes():
    upper = np.array([[0.01, 1.0, 0.05]])
    lower = np.array([[5.99, 1.0, -0.05]])

    relax_interleaflet_clashes_xy(
        upper,
        lower,
        [1],
        [1],
        cutoff=0.20,
        displacement=0.05,
        n_iterations=10,
        box_xy=6.0,
    )

    delta = upper[0] - lower[0]
    delta[:2] -= 6.0 * np.round(delta[:2] / 6.0)
    assert np.linalg.norm(delta) >= 0.19


def test_azimuthal_declashing_preserves_centres_and_internal_geometry():
    first = np.asarray([[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    second = np.asarray([[0.0, 0.005, 0.0], [0.0, 0.105, 0.0], [0.0, 0.205, 0.0]])
    coordinates = np.vstack((first, second))
    centres_before = np.asarray([first.mean(axis=0), second.mean(axis=0)])
    internal_before = _distance_matrix(second)

    rotated, clearance = rotate_lipids_away_from_clashes(
        coordinates,
        [3, 3],
        min_distance=0.035,
    )

    centres_after = np.asarray([rotated[:3].mean(axis=0), rotated[3:].mean(axis=0)])
    assert clearance >= 0.035
    assert np.allclose(centres_after, centres_before)
    assert np.allclose(_distance_matrix(rotated[3:]), internal_before)


def test_external_azimuthal_declashing_preserves_centre_and_geometry():
    lipid = np.asarray(
        [
            [-0.20, 0.0, 0.0],
            [0.00, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ]
    )
    external = np.asarray([[0.20, 0.01, 0.0]])
    centre_before = lipid.mean(axis=0)
    internal_before = _distance_matrix(lipid)

    rotated, clearance = rotate_lipids_away_from_external_clashes(
        lipid.copy(),
        [3],
        external,
        min_distance=0.05,
        angle_samples=24,
        max_rounds=2,
    )

    assert np.allclose(rotated.mean(axis=0), centre_before)
    assert np.allclose(_distance_matrix(rotated), internal_before)
    assert clearance >= 0.049


def test_lipid_trimming_retains_whole_molecules_and_aligned_atom_fields():
    structure = Structure(
        coordinates=np.arange(27, dtype=float).reshape(9, 3),
        box_vectors=np.eye(3) * 5.0,
        atom_names=[f"A{i}" for i in range(9)],
        resnames=["L1"] * 2 + ["L2"] * 3 + ["L3"] * 4,
        resids=[1] * 2 + [2] * 3 + [3] * 4,
        chain_ids=["A"] * 9,
        segids=["MEMB"] * 9,
        elements=["C"] * 9,
        occupancies=[1.0] * 9,
        tempfactors=[float(i) for i in range(9)],
    )
    leaflet = System(
        structure=structure,
        metadata={"n_lipids": 3, "lipid_sizes": [2, 3, 4]},
    )

    removed = MembraneBuilder._trim_leaflet_to_count(
        leaflet, 2, np.random.default_rng(42), "upper", []
    )

    assert removed == 1
    assert leaflet.metadata["n_lipids"] == 2
    assert sum(leaflet.metadata["lipid_sizes"]) == leaflet.num_atoms
    for field_name in (
        "atom_names",
        "resnames",
        "resids",
        "chain_ids",
        "segids",
        "elements",
        "occupancies",
        "tempfactors",
    ):
        assert len(getattr(leaflet.structure, field_name)) == leaflet.num_atoms


def test_explicit_lipid_count_is_the_final_output_contract(empty_system):
    empty_system.metadata["seed"] = 42

    result = MembraneBuilder().run(
        empty_system,
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 64},
    )

    membrane = result.system.component_by_name("MEMBRANE_POPC(100%)")
    assert result.success
    assert membrane is not None
    assert membrane.metadata["n_lipids_upper"] == 64
    assert membrane.metadata["n_lipids_lower"] == 64


def test_popc_headgroup_spacing_matches_registered_bilayer_thickness(empty_system):
    empty_system.metadata.update({"seed": 20260713, "force_field": "charmm36m"})

    result = MembraneBuilder().run(
        empty_system,
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 64},
    )

    membrane = result.system.component_by_name("MEMBRANE_POPC(100%)")
    names = np.asarray(result.system.structure.atom_names)
    phosphate_z = result.system.coordinates[names == "P", 2]
    upper = phosphate_z[phosphate_z > 0]
    lower = phosphate_z[phosphate_z < 0]
    measured_dhh = float(upper.mean() - lower.mean())
    assert result.success
    assert len(upper) == len(lower) == 64
    assert measured_dhh == pytest.approx(3.8, abs=0.25)
    assert membrane.metadata["bilayer_thickness"] == pytest.approx(measured_dhh, abs=1e-8)


def test_explicit_lipid_count_preserves_apl_derived_periodic_box(empty_system):
    lipid = LipidRegistry.get("DAPC")
    count = 64

    result = MembraneBuilder(use_equilibrated_library=False).run(
        empty_system,
        {"lipid_type": "DAPC", "n_lipids_per_leaflet": count},
    )

    expected_xy = np.sqrt(count * lipid.area_per_lipid)
    membrane = result.system.component_by_name("MEMBRANE_DAPC(100%)")
    assert result.success
    assert result.system.structure.dimensions()[0] == pytest.approx(expected_xy)
    assert membrane.metadata["box_xy"] == pytest.approx(expected_xy)
    assert any("Box XY retained at APL target" in line for line in result.log)
    assert all("protein extent dominates" not in line for line in result.log)


def test_bootstrap_geometry_retries_a_compact_rdkit_conformer(monkeypatch):
    calls = []

    def fake_geometry(*args, seed, **kwargs):
        calls.append(seed)
        names = ["N", "C1", "C2", "C3"]
        if len(calls) == 1:
            # The polar and carbon centroids nearly coincide, so the first
            # conformer is not a usable amphiphile axis.
            coordinates = np.asarray(
                [
                    [0.0, 0.0, 0.05],
                    [-0.05, 0.0, 0.0],
                    [0.00, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
                ]
            )
        else:
            coordinates = np.asarray(
                [
                    [0.0, 0.0, 0.60],
                    [-0.10, 0.0, 0.0],
                    [0.00, 0.0, 0.0],
                    [0.10, 0.0, 0.0],
                ]
            )
        return coordinates, names

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.builder.build_rdkit_lipid_geometry",
        fake_geometry,
    )
    leaflet = MembraneBuilder(use_equilibrated_library=False)._build_mixed_leaflet(
        np.asarray([[0.0, 0.0]]),
        1.9,
        ["DAPE"],
        np.random.default_rng(20260713),
        force_field="charmm36m",
        lipid_ff="charmm36m",
        box_xy=4.0,
    )

    assert len(calls) == 2
    assert leaflet.metadata["bootstrap_conformer_retries"] == 1
    assert leaflet.num_atoms == 4


def test_mixed_gaff2_bilayer_has_inward_tails_and_a_sealed_core(empty_system):
    empty_system.metadata.update({"seed": 20260716, "force_field": "amber14sb"})
    result = MembraneBuilder(use_equilibrated_library=False).run(
        empty_system,
        {
            "lipid_composition": {
                "upper": [
                    {"name": "POPC", "ratio": 50},
                    {"name": "POPE", "ratio": 50},
                ],
                "lower": [
                    {"name": "POPC", "ratio": 25},
                    {"name": "POPE", "ratio": 75},
                ],
            },
            "n_lipids_per_leaflet": 64,
        },
    )

    membrane = result.system.component_by_name("MEMBRANE_POPC(50%)+POPE(50%)_asym")
    assert result.success
    assert membrane is not None
    quality = membrane.metadata["orientation_quality"]
    assert quality["passed"] is True
    assert quality["n_lipids_checked"] == 128
    assert quality["minimum_inward_projection_nm"] >= 0.10
    assert quality["minimum_inward_cosine"] >= 0.10
    assert quality["tail_core_gap_nm"] <= quality["maximum_tail_core_gap_nm"]
    sizes = membrane.metadata["lipid_sizes"]
    split = sum(sizes[: membrane.metadata["n_lipids_upper"]])
    coordinates = result.system.coordinates[membrane.atom_indices]
    box = result.system.structure.dimensions()
    z_origin = float(coordinates[:, 2].min()) - 1.0
    z_box = float(np.ptp(coordinates[:, 2])) + 2.0
    upper = coordinates[:split].copy()
    lower = coordinates[split:].copy()
    upper[:, :2] = np.mod(upper[:, :2], box[:2])
    lower[:, :2] = np.mod(lower[:, :2], box[:2])
    upper[:, 2] -= z_origin
    lower[:, 2] -= z_origin
    clearance = (
        cKDTree(upper, boxsize=np.asarray([box[0], box[1], z_box])).query(lower, k=1)[0].min()
    )
    # The geometric pre-relaxation need only avoid singular contacts; the
    # exported minimization performs force-field-specific VDW relaxation.
    assert clearance >= 0.05


def test_asymmetric_box_uses_the_larger_leaflet_natural_area():
    upper = _weighted_leaflet_apl([("POPC", 100)])
    lower = _weighted_leaflet_apl([("POPC", 80), ("TOCL", 20)])

    assert upper == pytest.approx(LipidRegistry.get("POPC").area_per_lipid)
    assert lower > upper


def test_nonphospholipid_anchor_uses_outer_polar_geometry_not_gaff_o3():
    coordinates = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.2],
        ]
    )

    index = _headgroup_anchor_index(coordinates, ["C17", "O14", "O3"])

    assert index == 1


def test_headgroup_plane_reuses_recorded_nonphospholipid_anchor():
    leaflet = System(
        structure=Structure(
            coordinates=np.asarray(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.2],
                ]
            ),
            box_vectors=np.eye(3) * 4.0,
            atom_names=["C17", "O14", "O3"],
        ),
        metadata={
            "lipid_sizes": [3],
            "headgroup_anchor_local_indices": [1],
        },
    )

    assert _leaflet_headgroup_plane(leaflet, upper=True) == pytest.approx(0.8)

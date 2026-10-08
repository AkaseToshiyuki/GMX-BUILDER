import json

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.equilibrated_library import (
    ACCEPTED_METHOD,
    MIN_CONFORMERS,
    SCHEMA_VERSION,
    EquilibratedLipidLibrary,
    lipid_parameter_family,
    topology_signature,
)
from gmxbuilder.modules.membrane.lipid_equilibration import (
    MIN_ORIENTED_FRACTION,
    LipidEquilibrationBuilder,
    _orientation_gate,
    _outer_headgroup_anchor,
    _simulation_lipid_resname_map,
)
from gmxbuilder.modules.membrane.lipids import (
    LipidRegistry,
    find_registered_lipid_matches,
    parse_custom_lipid,
)
from tests.dry_initial_fixture import dry_initial_fixture
from tests.prerequisites import requires_gaff_runtime


def test_side_chain_oxysterol_gate_uses_oriented_host_bilayer():
    all_projection = np.asarray([0.7, 0.8, -0.02])
    all_cosine = np.asarray([0.9, 0.8, -0.03])
    host_projection = np.asarray([0.7, 0.8])
    host_cosine = np.asarray([0.9, 0.8])

    passed, profile, gate_projection, _ = _orientation_gate(
        "24SHC",
        all_projection,
        all_cosine,
        host_projection,
        host_cosine,
        upper_count=26,
        lower_count=26,
    )
    assert passed
    assert profile.startswith("host-bilayer")
    assert np.array_equal(gate_projection, host_projection)

    standard_passed, _, _, _ = _orientation_gate(
        "CHOL",
        all_projection,
        all_cosine,
        host_projection,
        host_cosine,
        upper_count=26,
        lower_count=26,
    )
    assert not standard_passed


def test_side_chain_oxysterol_gate_rejects_disordered_host():
    passed, _, _, _ = _orientation_gate(
        "25OHC",
        np.asarray([0.7, -0.02]),
        np.asarray([0.8, -0.03]),
        np.asarray([0.7, -0.02]),
        np.asarray([0.8, -0.03]),
        upper_count=26,
        lower_count=26,
    )
    assert not passed


def test_production_orientation_gate_uses_ensemble_fraction_not_single_outlier():
    projections = np.full(100, 0.7)
    cosines = np.full(100, 0.8)
    projections[0] = -0.1
    cosines[0] = -0.2

    passed, profile, _, _ = _orientation_gate(
        "CHOL",
        projections,
        cosines,
        np.empty(0),
        np.empty(0),
        upper_count=50,
        lower_count=50,
    )

    assert passed
    assert profile == "all-lipids-single-headgroup"
    assert MIN_ORIENTED_FRACTION == 0.98


def test_production_orientation_gate_rejects_more_than_two_percent_outliers():
    projections = np.full(100, 0.7)
    cosines = np.full(100, 0.8)
    projections[:3] = -0.1
    cosines[:3] = -0.2

    passed, _, _, _ = _orientation_gate(
        "CHOL",
        projections,
        cosines,
        np.empty(0),
        np.empty(0),
        upper_count=50,
        lower_count=50,
    )

    assert not passed


def _write_entry(root, *, method=ACCEPTED_METHOD, quality=True, family="charmm36m-lipid"):
    directory = root / family / "POPC"
    directory.mkdir(parents=True)
    names = ["P", "O1", "C1", "C2", "C3"]
    coords = np.asarray(
        [
            [0.0, 0.0, 0.8],
            [0.1, 0.0, 0.7],
            [0.0, 0.0, 0.1],
            [0.1, 0.0, -0.3],
            [-0.1, 0.0, -0.7],
        ]
    )
    for index in range(MIN_CONFORMERS):
        np.savez_compressed(
            directory / f"conf_{index:04d}.npz",
            coords=coords,
            atom_names=np.asarray(names),
        )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "lipid_name": "POPC",
                "parameter_fingerprint": __import__(
                    "gmxbuilder.modules.membrane.parameter_provenance",
                    fromlist=["parameter_fingerprint"],
                ).parameter_fingerprint("POPC", "charmm36m", "charmm36m"),
                "schema_version": SCHEMA_VERSION,
                "coordinate_handedness": "preserved",
                "leaflet_transform": "proper_rotation",
                "status": "ready",
                "method": method,
                "parameter_family": family,
                "force_field": "charmm36m",
                "lipid_ff": "charmm36m",
                "canonical_smiles": LipidRegistry.get("POPC").smiles,
                "topology_sha256": topology_signature(names, "charmm36m", "charmm36m"),
                "atom_names": names,
                "n_conformations": MIN_CONFORMERS,
                "quality": {
                    "initial_water_exclusion": dry_initial_fixture(),
                    "passed": quality,
                    "orientation": {
                        "passed": quality,
                        "n_lipids_checked": MIN_CONFORMERS,
                    },
                },
            }
        )
    )
    return directory


@requires_gaff_runtime
def test_numeric_gaff_sterol_output_name_maps_back_to_registry(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "bcc")

    mapping = _simulation_lipid_resname_map(
        {"POPC", "20AHC"},
        "amber14sb",
        "gaff2",
    )

    assert mapping["L_20A"] == "20AHC"
    assert mapping["POPC"] == "POPC"


def test_strict_library_accepts_only_validated_explicit_solvent_npt(tmp_path):
    _write_entry(tmp_path)
    library = EquilibratedLipidLibrary([tmp_path])
    assert library.has("POPC", "charmm36m", "charmm36m")
    coords, names = library.load_one("POPC", "charmm36m", rng=np.random.default_rng(3))
    assert coords.shape == (5, 3)
    assert names == ["P", "O1", "C1", "C2", "C3"]


@pytest.mark.parametrize("method,quality", [("geometric_fallback", True), (ACCEPTED_METHOD, False)])
def test_strict_library_rejects_bootstrap_or_failed_quality(tmp_path, method, quality):
    _write_entry(tmp_path, method=method, quality=quality)
    assert not EquilibratedLipidLibrary([tmp_path]).has("POPC", "charmm36m")


def test_strict_library_invalidates_entries_without_orientation_schema(tmp_path):
    directory = _write_entry(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = SCHEMA_VERSION - 1
    metadata_path.write_text(json.dumps(metadata))

    assert not EquilibratedLipidLibrary([tmp_path]).has("POPC", "charmm36m")


def test_strict_library_invalidates_stale_registry_identity(tmp_path):
    directory = _write_entry(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["canonical_smiles"] = "C"
    metadata_path.write_text(json.dumps(metadata))

    assert not EquilibratedLipidLibrary([tmp_path]).has("POPC", "charmm36m")


def test_strict_library_recomputes_topology_signature(tmp_path):
    directory = _write_entry(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["topology_sha256"] = "not-the-stored-topology"
    metadata_path.write_text(json.dumps(metadata))

    assert not EquilibratedLipidLibrary([tmp_path]).has("POPC", "charmm36m")


def test_current_production_gate_failure_is_classified_unavailable(tmp_path):
    directory = _write_entry(tmp_path)
    failed = directory.with_name(directory.name + ".failed")
    directory.rename(failed)
    metadata_path = failed / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "status": "failed",
            "test_mode": False,
            "npt_ps": 1000.0,
        }
    )
    metadata["quality"]["passed"] = False
    metadata_path.write_text(json.dumps(metadata))
    library = EquilibratedLipidLibrary([tmp_path])

    failure = library.inspect_failure(
        "POPC",
        "charmm36m",
        "charmm36m",
        min_npt_ps=1000.0,
    )

    assert failure is not None
    assert failure.path == failed
    assert (
        library.inspect_failure(
            "POPC",
            "charmm36m",
            "charmm36m",
            min_npt_ps=1001.0,
        )
        is None
    )


def test_stale_or_identity_mismatched_failure_is_not_terminal(tmp_path):
    directory = _write_entry(tmp_path)
    failed = directory.with_name(directory.name + ".failed")
    directory.rename(failed)
    metadata_path = failed / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "status": "failed",
            "test_mode": False,
            "npt_ps": 1000.0,
            "canonical_smiles": "C",
            "schema_version": SCHEMA_VERSION - 1,
        }
    )
    metadata["quality"]["passed"] = False
    metadata_path.write_text(json.dumps(metadata))

    assert (
        EquilibratedLipidLibrary([tmp_path]).inspect_failure(
            "POPC",
            "charmm36m",
            "charmm36m",
        )
        is None
    )


def test_strict_library_rechecks_selected_conformer_direction(tmp_path):
    directory = _write_entry(tmp_path)
    for path in directory.glob("conf_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            coordinates = np.asarray(data["coords"], dtype=float)
            names = np.asarray(data["atom_names"])
        coordinates[:, 2] *= -1.0
        np.savez_compressed(path, coords=coordinates, atom_names=names)

    library = EquilibratedLipidLibrary([tmp_path])
    with pytest.raises(ValueError, match="outward polar head"):
        library.load_one("POPC", "charmm36m", rng=np.random.default_rng(3))


def test_force_fields_share_only_a_compatible_lipid_parameter_family():
    assert lipid_parameter_family("charmm36m") != lipid_parameter_family("charmm36")
    assert lipid_parameter_family("charmm36m") == "charmm36m-lipid"
    assert lipid_parameter_family("charmm36") == "charmm36-lipid"
    assert lipid_parameter_family("amber99sb", "gaff2") == lipid_parameter_family(
        "amber99sb-ildn", "gaff2"
    )
    with pytest.raises(ValueError):
        lipid_parameter_family("unknown")


def test_exact_charmm_release_is_part_of_library_compatibility(tmp_path):
    _write_entry(tmp_path)
    library = EquilibratedLipidLibrary([tmp_path])
    assert library.has("POPC", "charmm36m", "charmm36m")
    assert not library.has("POPC", "charmm36", "charmm36")


def test_custom_lipid_parser_detects_registered_structure_before_building():
    popc = LipidRegistry.get("POPC")
    result = parse_custom_lipid(popc.smiles, "DUPL")
    assert result["is_existing"] is True
    assert any(
        match["name"] == "POPC" and match["match"] == "exact"
        for match in result["registered_matches"]
    )
    assert find_registered_lipid_matches(popc.smiles)
    assert result["canonical_smiles"]
    assert result["inchi_key"]


def test_offline_repack_moves_whole_lipids_to_staggered_lattices():
    molecule = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, -0.2]])
    coordinates = np.vstack([molecule for _ in range(8)])
    system = System(
        Structure(
            coordinates=coordinates,
            box_vectors=np.eye(3) * 4.0,
            atom_names=["P", "C1"] * 8,
            resnames=["TEST"] * 16,
            resids=np.repeat(np.arange(1, 9), 2).tolist(),
        ),
        components=[
            Component(
                "MEMBRANE_TEST",
                ComponentKind.MEMBRANE,
                np.arange(16),
                metadata={
                    "lipid_sizes": [2] * 8,
                    "n_lipids_upper": 4,
                    "n_lipids_lower": 4,
                },
            )
        ],
    )
    before = np.linalg.norm(system.coordinates[0] - system.coordinates[1])
    LipidEquilibrationBuilder._repack_bootstrap_bilayer(system, spacing=1.2)
    after = np.linalg.norm(system.coordinates[0] - system.coordinates[1])
    centers = np.asarray(
        [system.coordinates[index : index + 2, :2].mean(axis=0) for index in range(0, 8, 2)]
    )
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    distances[distances == 0] = np.inf
    upper_inner = float(system.coordinates[:8, 2].min())
    lower_inner = float(system.coordinates[8:, 2].max())
    assert after == pytest.approx(before)
    assert distances.min() >= 1.2
    assert upper_inner - lower_inner >= 0.18 - 1e-8


def test_offline_repack_spacing_uses_rotation_invariant_molecular_radius():
    molecule = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, -0.2],
        ]
    )
    coordinates = np.vstack([molecule for _ in range(8)])
    system = System(
        Structure(
            coordinates=coordinates,
            box_vectors=np.eye(3) * 4.0,
            atom_names=["P", "C1"] * 8,
            resnames=["TEST"] * 16,
            resids=np.repeat(np.arange(1, 9), 2).tolist(),
        ),
        components=[
            Component(
                "MEMBRANE_TEST",
                ComponentKind.MEMBRANE,
                np.arange(16),
                metadata={
                    "lipid_sizes": [2] * 8,
                    "n_lipids_upper": 4,
                    "n_lipids_lower": 4,
                },
            )
        ],
    )
    expected_spacing = 2.0 * np.sqrt(2.0) + 0.15

    LipidEquilibrationBuilder._repack_bootstrap_bilayer(system, spacing=1.2)

    assert system.structure.dimensions()[0] == pytest.approx(2.0 * expected_spacing)


def test_offline_reimage_moves_whole_lipids_to_intended_z_images():
    upper_molecule = np.asarray([[0.0, 0.0, 0.4], [0.1, 0.0, 0.0]])
    lower_molecule = np.asarray([[0.0, 0.0, -0.4], [0.1, 0.0, 0.0]])
    coordinates = np.vstack(
        (
            upper_molecule + [0.0, 0.0, 6.0],
            lower_molecule - [0.0, 0.0, 6.0],
        )
    )
    system = System(
        Structure(
            coordinates=coordinates,
            box_vectors=np.diag([4.0, 4.0, 6.0]),
            atom_names=["O1", "C1"] * 2,
        ),
        components=[
            Component(
                "MEMBRANE_TEST",
                ComponentKind.MEMBRANE,
                np.arange(4),
                metadata={
                    "lipid_sizes": [2, 2],
                    "n_lipids_upper": 1,
                    "n_lipids_lower": 1,
                    "bilayer_thickness": 3.8,
                },
            )
        ],
    )
    internal_before = np.linalg.norm(system.coordinates[0] - system.coordinates[1])

    LipidEquilibrationBuilder._reimage_bilayer_z(system)

    assert system.coordinates[0, 2] == pytest.approx(0.4)
    assert system.coordinates[2, 2] == pytest.approx(-0.4)
    assert np.linalg.norm(system.coordinates[0] - system.coordinates[1]) == pytest.approx(
        internal_before
    )


@pytest.mark.parametrize(
    "coordinates,expected_index,upper",
    [
        (np.asarray([[0, 0, 5.2], [0, 0, 6.1], [0, 0, 5.8]]), 1, True),
        (np.asarray([[0, 0, 4.8], [0, 0, 3.9], [0, 0, 4.2]]), 1, False),
    ],
)
def test_headgroup_anchor_uses_outward_polar_geometry_not_gaff_atom_numbering(
    coordinates,
    expected_index,
    upper,
):
    index, is_upper = _outer_headgroup_anchor(
        coordinates,
        ["C17", "O14", "O3"],
        box_midplane_z=5.0,
    )
    assert index == expected_index
    assert is_upper is upper


def test_extreme_anionic_library_systems_receive_a_larger_solvent_reservoir():
    assert LipidEquilibrationBuilder._solvent_padding(-4) == 2.0
    assert LipidEquilibrationBuilder._solvent_padding(-3) == 1.2


def test_genion_retries_only_solvent_exhaustion_and_restores_topology(tmp_path):
    (tmp_path / "topol.top").write_text("original topology")
    builder = object.__new__(LipidEquilibrationBuilder)
    builder.gmx = "gmx"
    attempts = []

    def fake_run(args, cwd, **kwargs):
        attempts.append(args[-1])
        (cwd / "topol.top").write_text("partially modified")
        if args[-1] == "0.40":
            raise RuntimeError("Fatal error: No more replaceable solvent!")
        (cwd / "ionized.gro").write_text("ions")

    builder._run = fake_run

    assert builder._genion_with_retry(tmp_path) == pytest.approx(0.35)
    assert attempts == ["0.40", "0.35"]
    assert (tmp_path / "topol.top").read_text() == "partially modified"


def test_postion_minimization_uses_a_conservative_step_size():
    mdp = LipidEquilibrationBuilder._ion_minimization_mdp(test_mode=False)
    assert "emstep = 0.001" in mdp
    assert "nsteps = 20000" in mdp


def test_nonfinite_minimization_is_rejected_before_nvt(tmp_path):
    log = tmp_path / "em.log"
    log.write_text("Potential Energy  =  2.3e+17\nMaximum force     =  inf on atom 42\n")
    with pytest.raises(RuntimeError, match="unresolved atomic overlaps"):
        LipidEquilibrationBuilder._assert_finite_minimization(log)


# --------------------------------------------------------------------------
# Replica seeding (R3): one seed per entry is why a V3 area cannot be checked


def test_the_mdp_seed_defaults_to_the_one_every_v3_entry_was_built_with():
    """Rebuilding an existing entry must reproduce it, not perturb it."""
    from gmxbuilder.modules.membrane.lipid_equilibration import (
        DEFAULT_REPLICA_SEED,
        LipidEquilibrationBuilder,
    )

    for stage in ("nvt", "npt"):
        mdp = LipidEquilibrationBuilder._mdp(stage, 1000, 310.0)
        assert f"gen-seed = {DEFAULT_REPLICA_SEED}" in mdp


def test_a_replica_asks_for_its_own_seed_and_gets_it():
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    mdp = LipidEquilibrationBuilder._mdp("nvt", 1000, 310.0, 987654321)
    assert "gen-seed = 987654321" in mdp
    assert "gen-vel = yes" in mdp, "the seed only means anything where velocities are drawn"


def test_two_replica_seeds_produce_two_different_inputs():
    """Otherwise 'two replicas' would be one trajectory run twice."""
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    first = LipidEquilibrationBuilder._mdp("nvt", 1000, 310.0, 11)
    second = LipidEquilibrationBuilder._mdp("nvt", 1000, 310.0, 22)
    assert first != second


def test_the_build_accepts_a_replica_seed_and_a_retained_work_directory():
    """R2 and R3 need these; a signature change would break both silently."""
    import inspect

    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    parameters = inspect.signature(LipidEquilibrationBuilder._build_once).parameters
    assert "replica_seed" in parameters
    assert "retain_work" in parameters
    assert parameters["retain_work"].default is None, "retention must be opt-in"


def test_lipid21_backbone_phosphorus_is_found_by_its_amber_name():
    """Lipid21 calls it P31. Missing it silently moves the reference plane.

    Without this the anchor falls through to 'whichever polar atom sits
    furthest out', which is a different atom for every headgroup, so all 39
    Lipid21 templates get a bilayer thickness measured from an inconsistent
    plane -- low enough that POPC, POPG, DPPC and DMPC all came to rest within
    0.03 of the rejection threshold.
    """
    coordinates = np.asarray([[0, 0, 5.2], [0, 0, 5.9], [0, 0, 6.4]])
    index, _ = _outer_headgroup_anchor(
        coordinates,
        ["C21", "P31", "O33"],
        box_midplane_z=5.0,
        upper_leaflet=True,
    )
    assert index == 1, "P31 is the phosphorus, not the outermost oxygen"


def test_a_headgroup_phosphate_never_displaces_the_backbone_one():
    """CHARMM names cardiolipin's and PIP2's other phosphates PA, PB, P4, P5.

    Those residues carry a bare P as well, so the exact match wins; but a
    letter-suffixed name must never be accepted as a fallback, or a
    phosphoinositide would be measured from its inositol ring.
    """
    coordinates = np.asarray([[0, 0, 4.0], [0, 0, 5.0], [0, 0, 6.0], [0, 0, 7.0]])
    index, _ = _outer_headgroup_anchor(
        coordinates,
        ["C1", "P", "PA", "O4"],
        box_midplane_z=5.0,
        upper_leaflet=True,
    )
    assert index == 1, "the bare P is the backbone phosphate"

    index, _ = _outer_headgroup_anchor(
        coordinates,
        ["C1", "PA", "PB", "O4"],
        box_midplane_z=5.0,
        upper_leaflet=True,
    )
    assert index == 3, "with no backbone P, geometry decides -- not a headgroup phosphate"


def test_bilayer_thickness_is_not_reduced_to_its_minimum_image():
    """A bilayer thicker than half its box is the ordinary case, not an extreme.

    Reducing the head-plane separation to its minimum image returns the
    distance the other way round the box -- through the water -- so a 4.6 nm
    membrane in a 7.6 nm box is reported as 2.9 nm. That is what rejected four
    perfectly good DPPA, DOPA, DLiPA and DGDG bilayers after two GPU-hours
    each, and what left 57 library entries sitting on the rejection threshold.
    """
    from gmxbuilder.modules.membrane.lipid_equilibration import head_to_head_distance

    box_z = 7.556
    # Upper heads at 6.06, lower at 1.46: 4.60 nm apart, well over box_z / 2.
    head_z = np.concatenate([np.full(64, 6.06), np.full(64, 1.46)])
    # Each lipid's own tail centre, 1.39 nm inward from its own head.
    tail_offsets = np.concatenate([np.full(64, -1.39), np.full(64, 1.39)])
    upper = np.concatenate([np.ones(64, dtype=bool), np.zeros(64, dtype=bool)])

    measured = head_to_head_distance(head_z, tail_offsets, upper, box_z)
    assert measured == pytest.approx(4.60, abs=0.01)
    assert measured != pytest.approx(box_z - 4.60, abs=0.1), "that is the water gap"


def test_bilayer_thickness_still_handles_a_membrane_across_the_box_edge():
    """The wrap is genuinely needed when a leaflet straddles the boundary.

    Removing the minimum image altogether would break this case, which is why
    it is applied to the two core planes -- always close together -- and never
    to the head planes.
    """
    from gmxbuilder.modules.membrane.lipid_equilibration import head_to_head_distance

    box_z = 7.556
    # The same 4.60 nm bilayer shifted up by 1.80 nm: the upper leaflet's heads
    # have wrapped past the top of the box to z = 0.30, and its tail centres,
    # 1.39 nm inward, are still at the top at z = 6.47.
    head_z = np.concatenate([np.full(64, 0.30), np.full(64, 3.26)])
    tail_offsets = np.concatenate([np.full(64, -1.39), np.full(64, 1.39)])
    upper = np.concatenate([np.ones(64, dtype=bool), np.zeros(64, dtype=bool)])

    measured = head_to_head_distance(head_z, tail_offsets, upper, box_z)
    assert measured == pytest.approx(4.60, abs=0.01), (
        "the same membrane, whichever periodic image it is written in"
    )


def test_an_ambiguous_box_is_refused_rather_than_measured_wrongly():
    """The old code returned a plausible number for an unmeasurable box."""
    from gmxbuilder.modules.membrane.lipid_equilibration import head_to_head_distance

    # A box with almost no water: the leaflet cores end up 1.8 nm apart in a
    # 4.0 nm box, so which way round the box the membrane lies is no longer
    # decidable and the answer must be refused rather than guessed.
    box_z = 4.0
    head_z = np.concatenate([np.full(8, 0.5), np.full(8, 3.7)])
    tail_offsets = np.concatenate([np.full(8, -1.5), np.full(8, 1.5)])
    upper = np.concatenate([np.ones(8, dtype=bool), np.zeros(8, dtype=bool)])

    with pytest.raises(RuntimeError, match="ambiguous"):
        head_to_head_distance(head_z, tail_offsets, upper, box_z)


def test_a_gaff2_entry_the_topology_writer_would_reject_is_not_served(tmp_path, monkeypatch):
    """A GAFF2 entry is addressed by atom name, and ACPYPE's names can move.

    Where the registry SMILES was rewritten, ACPYPE can return the same molecule
    in a different order -- CER16 does -- and the entry's conformers then
    disagree with the topology the writer builds about which atom is which. That
    surfaces as an error about coordinate order half way through a build. It is
    a fact about the entry, not a policy: refuse it and fall back to a generated
    conformer, which is at least self-consistent.
    """
    import dataclasses

    from gmxbuilder.modules.forcefield.gaff_backend import GAFFTemplate
    from gmxbuilder.modules.membrane.equilibrated_library import _explicit_atom_count

    family = lipid_parameter_family("amber14sb", "gaff2")
    directory = tmp_path / family / "DPPC"
    directory.mkdir(parents=True)
    registered = LipidRegistry.get("DPPC")
    names = [f"C{index}" for index in range(_explicit_atom_count(registered.smiles))]
    for number in range(MIN_CONFORMERS):
        np.savez(
            directory / f"conf_{number:04d}.npz",
            coords=np.zeros((len(names), 3)),
            atom_names=np.asarray(names),
        )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ready",
                "method": ACCEPTED_METHOD,
                "coordinate_handedness": "preserved",
                "leaflet_transform": "proper_rotation",
                "parameter_family": family,
                "force_field": "amber14sb",
                "lipid_ff": "gaff2",
                "lipid_name": "DPPC",
                "canonical_smiles": registered.smiles,
                "atom_names": names,
                "n_conformations": MIN_CONFORMERS,
                "topology_sha256": topology_signature(names, "amber14sb", "gaff2"),
                "quality": {
                    "initial_water_exclusion": dry_initial_fixture(),
                    "passed": True,
                    "orientation": {"passed": True, "n_lipids_checked": MIN_CONFORMERS},
                },
            }
        )
    )

    def matching(name, smiles, charge, *, charge_method=None):
        return GAFFTemplate(
            name=name,
            atom_names=tuple(names),
            coordinates=np.zeros((len(names), 3)),
            itp_path=directory / "lipid.itp",
            atomtypes_path=directory / "atomtypes.itp",
            charge_method="bcc",
        )

    def reordered(name, smiles, charge, *, charge_method=None):
        return dataclasses.replace(
            matching(name, smiles, charge), atom_names=tuple(reversed(names))
        )

    import gmxbuilder.modules.forcefield.gaff_backend as backend

    monkeypatch.setattr(backend, "cached_gaff_template", matching)
    from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

    (directory / "lipid.itp").write_text("; synthetic molecule")
    (directory / "atomtypes.itp").write_text("; synthetic types")
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
    (directory / "metadata.json").write_text(json.dumps(metadata))

    assert EquilibratedLipidLibrary([tmp_path]).has("DPPC", "amber14sb", "gaff2"), (
        "an entry whose names match the fitted template stays usable"
    )

    monkeypatch.setattr(backend, "cached_gaff_template", reordered)
    assert not EquilibratedLipidLibrary([tmp_path]).has("DPPC", "amber14sb", "gaff2"), (
        "an entry the topology writer would reject must not be served"
    )


def _bilayer_structure(gap_nm: float, core_waters: int = 0):
    """A minimal two-leaflet structure with a chosen gap between the tails."""
    from gmxbuilder.core.structure import Structure

    coordinates = []
    atom_names = []
    resnames = []
    resids = []
    resid = 1
    for upper in (True, False):
        sign = 1.0 if upper else -1.0
        head = sign * (gap_nm / 2.0 + 1.5)
        for column in range(4):
            for row in range(4):
                x, y = 0.6 * column, 0.6 * row
                # P at the outside, then three tail carbons reaching the core.
                for offset, name in (
                    (0.0, "P"),
                    (0.5, "C1"),
                    (1.0, "C2"),
                    (1.5, "C3"),
                ):
                    coordinates.append([x, y, head - sign * offset])
                    atom_names.append(name)
                    resnames.append("POPC")
                    resids.append(resid)
                resid += 1
    for index in range(core_waters):
        coordinates.append([0.1 * index, 0.1 * index, 0.0])
        atom_names.append("OW")
        resnames.append("SOL")
        resids.append(resid)
        resid += 1
    return Structure(
        coordinates=np.asarray(coordinates, dtype=float),
        box_vectors=np.eye(3) * 8.0,
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=[""] * len(atom_names),
        segids=[""] * len(atom_names),
        elements=[name[0] for name in atom_names],
        occupancies=[1.0] * len(atom_names),
        tempfactors=[0.0] * len(atom_names),
    )


def test_an_open_core_is_measured_before_the_water_goes_in():
    """The gap the production gate reports, asked while it can still be acted on.

    22RHC was solvated with a 2.2 nm hole in its hydrophobic core, came out of
    NVT with a water slab in the middle, and 50 ns of NPT changed neither: the
    run was decided before it started. Two hours of GPU time per entry to learn
    something the coordinates say in eighty seconds.
    """
    from gmxbuilder.modules.membrane.lipid_equilibration import construction_core_gap

    sealed = construction_core_gap(_bilayer_structure(gap_nm=0.0))
    open_core = construction_core_gap(_bilayer_structure(gap_nm=1.5))
    assert sealed == pytest.approx(0.0, abs=0.15), sealed
    assert open_core == pytest.approx(1.5, abs=0.15), open_core
    assert open_core > sealed


def test_water_in_the_hydrophobic_core_is_counted():
    """A bilayer that has just been solvated has none there, and it never will.

    Dewetting a slab of water out of a hydrophobic core is a rare event on any
    timescale this project simulates, so water placed there at construction
    stays for the whole run.
    """
    from gmxbuilder.modules.membrane.lipid_equilibration import core_water_count

    assert core_water_count(_bilayer_structure(gap_nm=0.0)) == 0
    assert core_water_count(_bilayer_structure(gap_nm=1.5, core_waters=9)) == 9


def test_molecule_boundaries_beat_residue_boundaries():
    """A residue is not a molecule, and Lipid21 is where that bites.

    It builds DOPC out of three residues -- OL, PC, OL -- so grouping atoms by
    residue hands half a tail to anything that asks what a lipid looks like:
    which leaflet it is in, where its head is, how far its chains reach.
    """
    from gmxbuilder.modules.membrane.lipid_equilibration import _molecule_slices

    resids = [1, 1, 2, 2, 3, 3]
    resnames = ["OL", "OL", "PC", "PC", "OL", "OL"]
    by_residue = _molecule_slices(resids, resnames)
    by_molecule = _molecule_slices(resids, resnames, offsets=[0, 6])
    assert len(by_residue) == 3, "three residues"
    assert len(by_molecule) == 1, "one molecule"
    assert by_molecule[0] == slice(0, 6)


def test_the_library_quarantine_does_not_block_its_own_repair():
    """DOPC has no validated Lipid21 library, which is what the queue builds.

    Outside a rebuild the lipid stays unavailable, which is the point of the
    quarantine. Inside one it must resolve to Lipid21, or the backend falls
    back to GAFF2, the builder produces GAFF-named atoms, and the topology
    writer refuses them against the Lipid21 order it was asked for -- the
    circle that kept DOPA and DOPC out of the library for three releases.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import (
        amber_lipid_backend,
        rebuilding_library_entry,
    )

    for lipid in ("DOPC", "DOPA"):
        outside, _reason = amber_lipid_backend([lipid])
        assert outside == "lipid21", "old-library failure must not switch the parameter model"
        with rebuilding_library_entry():
            inside, _reason = amber_lipid_backend([lipid])
        assert inside == "lipid21", f"{lipid} must be buildable by the queue"


def test_core_water_is_removed_from_the_slab_and_nowhere_else():
    """The bilayer keeps its solvent; the chains do not."""
    import numpy as np

    from gmxbuilder.core.component import Component
    from gmxbuilder.core.enums import ComponentKind
    from gmxbuilder.core.system import System
    from gmxbuilder.modules.membrane.lipid_equilibration import (
        LipidEquilibrationBuilder,
        core_water_count,
    )

    structure = _bilayer_structure(gap_nm=0.0)
    lipid_atoms = structure.num_atoms
    coordinates = list(structure.coordinates)
    atom_names = list(structure.atom_names)
    resnames = list(structure.resnames)
    resids = list(structure.resids)
    resid = max(resids) + 1
    depths = [0.0, 0.2, 3.0, 3.2]  # two among the chains, two in the bulk
    for depth in depths:
        for offset, name in ((0.0, "OW"), (0.01, "HW1"), (0.02, "HW2")):
            coordinates.append([0.2 + offset, 0.2, depth])
            atom_names.append(name)
            resnames.append("SOL")
            resids.append(resid)
        resid += 1
    from gmxbuilder.core.structure import Structure

    solvated = Structure(
        coordinates=np.asarray(coordinates, dtype=float),
        box_vectors=structure.box_vectors.copy(),
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=[""] * len(atom_names),
        segids=[""] * len(atom_names),
        elements=[name[0] for name in atom_names],
        occupancies=[1.0] * len(atom_names),
        tempfactors=[0.0] * len(atom_names),
    )
    system = System(
        structure=solvated,
        topology=None,
        components=[
            Component(
                name="MEMBRANE",
                kind=ComponentKind.MEMBRANE,
                atom_indices=np.arange(lipid_atoms),
                metadata={"lipid_sizes": [4] * (lipid_atoms // 4)},
            ),
            Component(
                name="SOLVENT_TIP3P",
                kind=ComponentKind.SOLVENT,
                atom_indices=np.arange(lipid_atoms, len(atom_names)),
                metadata={"n_molecules": len(depths)},
            ),
        ],
        metadata={},
    )

    assert core_water_count(solvated) > 0, "the fixture must start with water in the chains"
    removed = LipidEquilibrationBuilder._drain_core_water(system)
    assert removed == 2, f"only the two among the chains, removed {removed}"
    assert core_water_count(system.structure) == 0
    solvent = next(c for c in system.components if c.kind is ComponentKind.SOLVENT)
    assert solvent.metadata["n_molecules"] == len(depths) - 2
    assert len(solvent.atom_indices) == (len(depths) - 2) * 3


def test_a_published_entry_is_what_the_measurement_returns():
    """The build's return value is the entry path, not None.

    Trivial to state and easy to lose: the measurement and publication used to
    be the tail of ``_build_once``, whose last statement was the return, and
    splitting it into its own method so a continuation could reach it left the
    return behind. Everything still ran, every gate still passed, and every
    caller got ``None`` for the path it had just published.
    """
    import inspect

    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    source = inspect.getsource(LipidEquilibrationBuilder._measure_and_publish)
    assert source.rstrip().endswith("return output_dir")


def test_a_continuation_will_not_shorten_a_run(tmp_path):
    """Asking for less than the entry already has is a mistake, not a no-op."""
    import json

    import pytest

    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    from gmxbuilder.modules.membrane.parameter_provenance import (
        fingerprint_from_metadata,
        record_work_parameters,
    )

    metadata = {
        "npt_ps": 100000.0,
        "genion_rmin_nm": 0.2,
        "lipid_name": "POPC",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
    }
    metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
    (entry / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "work").mkdir()
    (tmp_path / "work/topol.top").write_text("")
    record_work_parameters(tmp_path / "work", metadata["parameter_fingerprint"])

    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.library = EquilibratedLipidLibrary(roots=[tmp_path, tmp_path])
    with pytest.raises(ValueError, match="already run"):
        builder._extend_once(
            "POPC", "charmm36m", "charmm36m", to_ps=50000.0, work_dir=tmp_path / "work"
        )


def test_a_continuation_needs_the_checkpoint_it_continues_from(tmp_path):
    """Without npt.cpt there is nothing to continue, and no partial run is made."""
    import json

    import pytest

    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    from gmxbuilder.modules.membrane.parameter_provenance import (
        fingerprint_from_metadata,
        record_work_parameters,
    )

    metadata = {
        "npt_ps": 50000.0,
        "genion_rmin_nm": 0.2,
        "lipid_name": "POPC",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
    }
    metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
    (entry / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "work").mkdir()
    (tmp_path / "work/topol.top").write_text("")
    record_work_parameters(tmp_path / "work", metadata["parameter_fingerprint"])
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "npt.tpr").write_bytes(b"x")
    (work / "topol.top").write_text("")

    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.library = EquilibratedLipidLibrary(roots=[tmp_path, tmp_path])
    with pytest.raises(RuntimeError, match="npt.cpt"):
        builder._extend_once("POPC", "charmm36m", "charmm36m", to_ps=60000.0, work_dir=work)


def test_a_continuation_that_integrated_nothing_is_a_failure(tmp_path, monkeypatch):
    """A resumed run is judged by its energy file, not by its exit status.

    ``_mdrun`` accepts a GPU run that failed during device teardown when the
    ``.gro`` is present, because a final frame is only written after every
    requested step. A continuation always inherits the previous run's ``.gro``,
    so the same reasoning would accept a resume that never started.
    """
    import pytest

    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.grompp_policy.ensure_dynamics_policy", lambda *a: None
    )
    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.gmx = "gmx"
    builder.threads = 1

    builder._run = lambda *a, **k: None
    builder._energy_end_time = lambda work, stage: 50000.0
    with pytest.raises(RuntimeError, match="reached 50000 ps"):
        builder._mdrun_continue("npt", tmp_path, until_ps=60000.0, timeout=10)

    # and one energy frame short of the target is not short at all
    builder._energy_end_time = lambda work, stage: 59999.5
    assert builder._mdrun_continue("npt", tmp_path, until_ps=60000.0, timeout=10) == 59999.5


def test_an_unreadable_energy_file_fails_a_continuation(tmp_path, monkeypatch):
    """NaN compares false against every bound, so the check states a lower one."""
    import pytest

    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.grompp_policy.ensure_dynamics_policy", lambda *a: None
    )
    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.gmx = "gmx"
    builder.threads = 1
    builder._run = lambda *a, **k: None
    builder._energy_end_time = lambda work, stage: float("nan")
    with pytest.raises(RuntimeError, match="reached nan ps"):
        builder._mdrun_continue("npt", tmp_path, until_ps=60000.0, timeout=10)


def test_the_continuation_ladder_preserves_the_ten_sample_queue_policy():
    """The queue carries an entry further in steps, and stops at the maximum."""
    import importlib.util
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_v4_library", root / "scripts" / "build_v4_library.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.INITIAL_NS < module.MAXIMUM_NS
    # every step adds at least one independent sample at the longest
    # autocorrelation time the library has measured (10.3 ns)
    assert module.EXTENSION_NS >= 10.0
    assert module.MINIMUM_EFFECTIVE_SAMPLES == 10.0
    # The cap is a resource limit and must not redefine statistical acceptance.
    assert not hasattr(module, "LENGTH_LADDER_NS"), "the rerun ladder is gone"


def test_the_queue_calls_the_builder_with_arguments_it_actually_takes(tmp_path, monkeypatch):
    """The queue hands each replica a source string, which no import checks.

    ``launch`` and ``launch_extension`` build the child process's program as
    text, so a renamed parameter on the builder produces a queue that starts
    every replica and fails every one of them at the first call -- hours in,
    with the failure reported only as a non-zero exit status. Parsing what the
    queue writes and checking it against the builder's real signature is the
    only place that mismatch can be caught before the run.
    """
    import ast
    import importlib.util
    import inspect
    import subprocess
    from pathlib import Path as _Path

    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    root = _Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_v4_library", root / "scripts" / "build_v4_library.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    written: list[str] = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            written.append(args[2])

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    class Arguments:
        v4_root = tmp_path
        threads = 12
        replicas = 2

    entry = {
        "family": "charmm36m-lipid",
        "lipid": "POPC",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
    }
    module.launch(entry, 1, 0, Arguments, 50.0)
    module.launch_extension(entry, 1, 0, Arguments, 60.0)
    assert len(written) == 2

    for source, method in zip(written, ("build", "extend"), strict=True):
        call = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
        )
        target = getattr(LipidEquilibrationBuilder, f"_{method}_once")
        accepted = set(inspect.signature(target).parameters)
        passed = {keyword.arg for keyword in call.keywords}
        assert passed <= accepted, f"{method} is passed {passed - accepted}, which it does not take"

    assert "to_ps=60000.0" in written[1]
    assert "npt_ps=50000.0" in written[0]


@pytest.fixture(autouse=True)
def _storage_parameter_sources(parameter_sources):
    return parameter_sources


def test_source_balanced_draws_do_not_overweight_larger_replica(tmp_path, monkeypatch):
    from gmxbuilder.modules.membrane.equilibrated_library import LibraryEntry

    directory = _write_entry(tmp_path)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["conformer_sampling"] = "equal-replica-then-uniform-conformer"
    metadata["conformer_provenance"] = [
        {"file": f"conf_{i:04d}.npz", "replica": 1 if i == 0 else 2} for i in range(MIN_CONFORMERS)
    ]
    library = EquilibratedLipidLibrary([tmp_path])
    monkeypatch.setattr(library, "inspect", lambda *args: LibraryEntry(directory, metadata))
    original = np.load
    selected = []

    def track(path, **kwargs):
        selected.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(np, "load", track)
    rng = np.random.default_rng(801)
    for _ in range(400):
        library.load_one("POPC", "charmm36m", rng=rng)
    # Uniform file sampling would select the one-file replica only about 5%.
    assert 0.4 < selected.count("conf_0000.npz") / len(selected) < 0.6
    assert len(set(selected)) == MIN_CONFORMERS


def test_build_reader_preserves_draws_and_rejects_source_changes(tmp_path, monkeypatch):
    from gmxbuilder.modules.membrane.conformer_reader import conformer_read_scope

    directory = _write_entry(tmp_path)
    library = EquilibratedLipidLibrary([tmp_path])
    rng = np.random.default_rng(340)
    expected = [library.load_one("POPC", "charmm36m", rng=rng) for _ in range(8)]
    original = library.inspect
    calls = []

    def inspect(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(library, "inspect", inspect)
    rng = np.random.default_rng(340)
    with conformer_read_scope():
        actual = [library.load_one("POPC", "charmm36m", rng=rng) for _ in range(8)]
    assert len(calls) == 1
    for (a, names), (b, expected_names) in zip(actual, expected, strict=True):
        assert np.array_equal(a, b)
        assert names == expected_names
    with pytest.raises(ValueError, match="changed"):
        with conformer_read_scope():
            library.load_one("POPC", "charmm36m", rng=rng)
            # Even an unselected conformer must invalidate the completed build.
            path = directory / "conf_0000.npz"
            path.write_bytes(path.read_bytes() + b"changed")


def test_admission_cache_invalidates_changed_metadata_policy_and_estimator(monkeypatch):
    from gmxbuilder.modules.membrane import v4_evidence
    from gmxbuilder.modules.membrane.equilibrated_library import _assembled_admission

    calls = []

    def validate(metadata):
        calls.append(metadata)
        return metadata["accepted"]

    monkeypatch.setattr(v4_evidence, "assembled_evidence_valid", validate)
    _assembled_admission.cache_clear()
    try:
        valid, invalid = '{"accepted": true}', '{"accepted": false}'
        assert _assembled_admission(valid, "policy-a", "estimator-a")
        assert _assembled_admission(valid, "policy-a", "estimator-a")
        assert len(calls) == 1
        assert not _assembled_admission(invalid, "policy-a", "estimator-a")
        assert _assembled_admission(valid, "policy-b", "estimator-a")
        assert len(calls) == 3
        assert _assembled_admission(valid, "policy-b", "estimator-b")
        assert len(calls) == 4
    finally:
        _assembled_admission.cache_clear()


def test_admission_cache_keeps_more_than_one_full_option_catalog(monkeypatch):
    from gmxbuilder.modules.membrane import v4_evidence
    from gmxbuilder.modules.membrane.equilibrated_library import _assembled_admission

    calls = []

    def validate(metadata):
        calls.append(metadata["lipid"])
        return True

    monkeypatch.setattr(v4_evidence, "assembled_evidence_valid", validate)
    _assembled_admission.cache_clear()
    try:
        metadata = [json.dumps({"lipid": index}) for index in range(17)]
        for _ in range(2):
            for payload in metadata:
                assert _assembled_admission(payload, "current-policy", "current-estimator")
        assert calls == list(range(17))
        assert _assembled_admission(
            json.dumps({"lipid": 0, "changed": True}), "current-policy", "current-estimator"
        )
        assert calls[-1] == 0
    finally:
        _assembled_admission.cache_clear()

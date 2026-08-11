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
from gmxbuilder.modules.membrane.lipids import (
    LipidRegistry,
    find_registered_lipid_matches,
    parse_custom_lipid,
)
from gmxbuilder.modules.membrane.lipid_equilibration import (
    LipidEquilibrationBuilder,
    _outer_headgroup_anchor,
    _simulation_lipid_resname_map,
)


def _write_entry(root, *, method=ACCEPTED_METHOD, quality=True, family="charmm36m-lipid"):
    directory = root / family / "POPC"
    directory.mkdir(parents=True)
    names = ["P", "O1", "C1", "C2", "C3"]
    coords = np.asarray([
        [0.0, 0.0, 0.8],
        [0.1, 0.0, 0.7],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, -0.3],
        [-0.1, 0.0, -0.7],
    ])
    for index in range(MIN_CONFORMERS):
        np.savez_compressed(
            directory / f"conf_{index:04d}.npz",
            coords=coords,
            atom_names=np.asarray(names),
        )
    (directory / "metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
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
            "passed": quality,
            "orientation": {
                "passed": quality,
                "n_lipids_checked": MIN_CONFORMERS,
            },
        },
    }))
    return directory


def test_numeric_gaff_sterol_output_name_maps_back_to_registry(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "bcc")

    mapping = _simulation_lipid_resname_map(
        {"POPC", "20AHC"}, "amber14sb", "gaff2",
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
    assert any(match["name"] == "POPC" and match["match"] == "exact" for match in result["registered_matches"])
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
        components=[Component(
            "MEMBRANE_TEST",
            ComponentKind.MEMBRANE,
            np.arange(16),
            metadata={
                "lipid_sizes": [2] * 8,
                "n_lipids_upper": 4,
                "n_lipids_lower": 4,
            },
        )],
    )
    before = np.linalg.norm(system.coordinates[0] - system.coordinates[1])
    LipidEquilibrationBuilder._repack_bootstrap_bilayer(system, spacing=1.2)
    after = np.linalg.norm(system.coordinates[0] - system.coordinates[1])
    centers = np.asarray([
        system.coordinates[index:index + 2, :2].mean(axis=0)
        for index in range(0, 8, 2)
    ])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    distances[distances == 0] = np.inf
    upper_inner = float(system.coordinates[:8, 2].min())
    lower_inner = float(system.coordinates[8:, 2].max())
    assert after == pytest.approx(before)
    assert distances.min() >= 1.2
    assert upper_inner - lower_inner >= 0.18 - 1e-8


def test_offline_reimage_moves_whole_lipids_to_intended_z_images():
    upper_molecule = np.asarray([[0.0, 0.0, 0.4], [0.1, 0.0, 0.0]])
    lower_molecule = np.asarray([[0.0, 0.0, -0.4], [0.1, 0.0, 0.0]])
    coordinates = np.vstack((
        upper_molecule + [0.0, 0.0, 6.0],
        lower_molecule - [0.0, 0.0, 6.0],
    ))
    system = System(
        Structure(
            coordinates=coordinates,
            box_vectors=np.diag([4.0, 4.0, 6.0]),
            atom_names=["O1", "C1"] * 2,
        ),
        components=[Component(
            "MEMBRANE_TEST",
            ComponentKind.MEMBRANE,
            np.arange(4),
            metadata={
                "lipid_sizes": [2, 2],
                "n_lipids_upper": 1,
                "n_lipids_lower": 1,
                "bilayer_thickness": 3.8,
            },
        )],
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
    coordinates, expected_index, upper,
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

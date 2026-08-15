"""Scientific and integration contracts for the independent Martini 3 workflow."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from click.testing import CliRunner

from gmxbuilder.app import main
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.coarse_grained import (
    CGExportModule,
    CGSystemCheckModule,
)
from gmxbuilder.modules.coarse_grained.assets import (
    load_manifest,
    public_capabilities,
    verify_assets,
)
from gmxbuilder.modules.coarse_grained.backend import (
    _membrane_command,
    normalize_environment,
    normalize_solvation,
    validate_protein_box,
)
from gmxbuilder.modules.coarse_grained.common import (
    normalize_composition,
    write_cg_viewer_pdb,
)
from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol
from gmxbuilder.modules.martini3_bilayer.orientation import CGOrientationModule
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.pipeline.step_executor import StepRunner, _get_module, get_pipeline_steps
from gmxbuilder.web import server
from gmxbuilder.web.server import app, task_manager
from gmxbuilder.web.task_types import get_all_task_types, get_task_type


def test_martini_task_types_and_workflow_modules_are_independent():
    bilayer = get_task_type("martini3-bilayer")
    solvent = get_task_type("martini3-solvent")
    assert bilayer is not None and bilayer.enabled and not bilayer.requires_input
    assert bilayer.pipeline == "martini_bilayer"
    assert solvent is not None and solvent.enabled and solvent.requires_input
    assert solvent.pipeline == "martini_solvent"
    advertised = {item["id"] for item in get_all_task_types()}
    assert {"martini3-bilayer", "martini3-solvent"} <= advertised
    assert "coarse-grained" not in advertised
    bilayer_steps = [
        "input",
        "cg_model",
        "cg_mapping",
        "cg_orientation",
        "cg_environment",
        "cg_solvation",
        "cg_system",
        "topology",
        "export",
    ]
    solvent_steps = [
        "input",
        "cg_model",
        "cg_mapping",
        "cg_environment",
        "cg_solvation",
        "cg_system",
        "topology",
        "export",
    ]
    assert get_pipeline_steps("martini3-bilayer") == bilayer_steps
    assert get_pipeline_steps("martini3-solvent") == solvent_steps
    for pipeline, expected in (
        ("martini3-bilayer", bilayer_steps),
        ("martini3-solvent", solvent_steps),
    ):
        classes = []
        for step in expected:
            module = _get_module(step, pipeline)
            assert module.__class__.__module__.startswith(
                f"gmxbuilder.modules.{pipeline.replace('-', '_')}"
            )
            classes.append(module.__class__)
        assert len(set(classes)) == len(expected)


def test_martini_assets_and_public_boundaries_are_explicit():
    verified = verify_assets()
    capabilities = public_capabilities()

    assert "martini_v3.0.0.itp" in verified
    assert capabilities["ready"] is True
    assert capabilities["force_field"] == "Martini 3.0.0"
    assert {item["name"] for item in capabilities["lipids"]} >= {
        "POPC",
        "POPE",
        "POPG",
        "POPS",
        "CHOL",
        "DLPC",
        "DAPE",
        "SAPS",
        "BSM",
    }
    assert len(capabilities["lipids"]) >= 170
    # An exact PE topology is not enough by itself: the pinned COBY release
    # lacks an OAPE LTF placement scaffold, so it must not be advertised.
    assert "OAPE" not in {item["name"] for item in capabilities["lipids"]}
    assert capabilities["boundaries"] == {
        "standard_protein_residues": True,
        "elastic_network": True,
        "custom_molecules": False,
        "post_translational_modifications": False,
        "curved_membranes": False,
        "backmapping": False,
    }
    for lipid, definition in load_manifest()["lipids"].items():
        assert definition["head_beads"], lipid
        assert definition["midplane_beads"], lipid
        assert definition["tail_beads"], lipid


def test_cg_input_ignores_crystallographic_water_explicitly(tmp_path):
    pdb = tmp_path / "protein_with_water.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  O   HOH A 101       5.000   5.000   5.000  1.00  0.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    runner = StepRunner(tmp_path / "task", pipeline_type="martini3-solvent")

    result = runner.run_step(
        "input",
        {
            "pdb": str(pdb),
            "include_protein": True,
            "environment": "solution",
        },
    )

    assert result["status"] == "ok", result
    system = runner.load_system("input")
    assert system is not None
    assert system.structure.resnames == ["ALA"]
    assert any("Ignored 1 crystallographic water" in line for line in result["log"])


def test_composition_and_protocol_reject_silent_scientific_drift():
    composition = normalize_composition(
        [{"name": "POPC", "ratio": 3}, {"name": "CHOL", "ratio": 1}],
        label="Upper",
    )
    assert sum(item["ratio"] for item in composition) == pytest.approx(1.0)
    with pytest.raises(ModuleConfigError, match="unavailable"):
        normalize_composition([{"name": "NOT_A_LIPID", "ratio": 1}], label="Upper")
    with pytest.raises(ModuleConfigError, match="integers"):
        normalize_protocol({"threads": 8.5}, has_membrane=True)
    with pytest.raises(ModuleConfigError, match="duplicates"):
        normalize_protocol(
            {"threads": 8, "mpi_ranks": 2, "gpu_ids": "0,0"},
            has_membrane=True,
        )
    protocol = normalize_protocol(
        {"threads": 8, "mpi_ranks": 2, "gpu_ids": "0,1"},
        has_membrane=True,
    )
    assert protocol["has_membrane"] is True
    assert protocol["threads"] // protocol["mpi_ranks"] == 4
    assert normalize_protocol(protocol, has_membrane=True) == protocol
    assert protocol["eq1_timestep_fs"] == 10.0
    assert protocol["production_timestep_fs"] == 20.0


def test_exact_lipid_corrections_keep_each_coby_parameter_library():
    environment = normalize_environment(
        {
            "n_lipids_per_leaflet": 200,
            "upper_leaflet": [
                {"name": "POPC", "ratio": 0.9},
                {"name": "CHOL", "ratio": 0.1},
            ],
        },
        {"cg_environment": "bilayer"},
    )
    command = _membrane_command(
        environment,
        {
            "upper": {"POPC": -106, "CHOL": -12},
            "lower": {"POPC": -159, "CHOL": -18},
        },
    )

    assert "lipid_extra:name:CHOL:params:default:extra_type:absolute:extra_val:-12" in command
    assert "lipid_extra:name:POPC:params:LTF:extra_type:absolute:extra_val:-106" in command
    assert "lipid_extra:name:CHOL:params:default:extra_type:absolute:extra_val:-18" in command
    assert "lipid_extra:name:POPC:params:LTF:extra_type:absolute:extra_val:-159" in command


def test_cg_viewer_preserves_authoritative_martinize_connectivity(tmp_path):
    source = tmp_path / "steps" / "cg_mapping" / "martinize" / "cg_protein.pdb"
    source.parent.mkdir(parents=True)
    source.write_text(
        "ATOM      1  BB  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2 SC1  ALA A   1       3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  BB  GLY A   2       6.000   0.000   0.000  1.00  0.00           C\n"
        "CONECT    1    2    3\nEND\n",
        encoding="utf-8",
    )
    structure = Structure(
        coordinates=np.asarray([[0.0, 0.1, 0.2], [0.3, 0.1, 0.2], [0.6, 0.1, 0.2]]),
        box_vectors=np.eye(3) * 5.0,
        atom_names=["BB", "SC1", "BB"],
        resnames=["ALA", "ALA", "GLY"],
        resids=[1, 1, 2],
        chain_ids=["A"] * 3,
        elements=["C"] * 3,
    )
    system = System(
        structure=structure,
        components=[
            Component(
                name="Martini 3 Protein",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(3, dtype=np.int64),
            )
        ],
        metadata={
            "cg_protein_pdb": "steps/cg_orientation/oriented_protein.pdb",
            "cg_connectivity_pdb": "steps/cg_mapping/martinize/cg_protein.pdb",
        },
    )
    viewer = tmp_path / "viewer.pdb"

    write_cg_viewer_pdb(system, viewer, task_dir=tmp_path)

    viewer_text = viewer.read_text(encoding="utf-8")
    assert "CONECT    1    2    3" in viewer_text
    assert viewer_text.index("CONECT") < viewer_text.index("END")
    assert PDBParser().parse(viewer).coordinates[0].tolist() == pytest.approx([0.0, 0.1, 0.2])


def test_cg_viewer_uses_bundled_lipid_topology_connections(tmp_path):
    structure = Structure(
        coordinates=np.asarray(
            [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 0.6],
                [0.2, 0.0, 0.6],
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -0.3],
                [0.0, 0.0, -0.6],
                [0.2, 0.0, 0.3],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, -0.3],
                [0.2, 0.0, -0.6],
            ]
        ),
        box_vectors=np.eye(3) * 5.0,
        atom_names=[
            "NC3",
            "PO4",
            "GL1",
            "GL2",
            "C1A",
            "D2A",
            "C3A",
            "C4A",
            "C1B",
            "C2B",
            "C3B",
            "C4B",
        ],
        resnames=["POPC"] * 12,
        resids=[1] * 12,
        chain_ids=[""] * 12,
        elements=["C"] * 12,
    )
    system = System(
        structure=structure,
        components=[
            Component(
                name="Martini 3 Membrane",
                kind=ComponentKind.MEMBRANE,
                atom_indices=np.arange(12, dtype=np.int64),
            )
        ],
    )
    viewer = tmp_path / "viewer.pdb"

    write_cg_viewer_pdb(system, viewer, task_dir=tmp_path)

    connections = [
        line
        for line in viewer.read_text(encoding="utf-8").splitlines()
        if line.startswith("CONECT")
    ]
    assert len(connections) == 10
    assert sum(len(line[6:].split()) - 1 for line in connections) == 12
    assert "CONECT    1    2" in connections
    assert "CONECT   11   12" in connections


def test_asymmetric_bilayer_requires_an_explicit_lower_leaflet():
    with pytest.raises(ModuleConfigError, match="explicit lower-leaflet"):
        normalize_environment({"asymmetric": True}, {"cg_environment": "bilayer"})
    with pytest.raises(ModuleConfigError, match="enable asymmetric"):
        normalize_environment(
            {
                "asymmetric": False,
                "upper_leaflet": [{"name": "POPC", "ratio": 1}],
                "lower_leaflet": [{"name": "POPE", "ratio": 1}],
            },
            {"cg_environment": "bilayer"},
        )
    with pytest.raises(ModuleConfigError, match="include_solvent must be true or false"):
        normalize_solvation({"include_solvent": "false"}, {"cg_environment": "bilayer"})
    with pytest.raises(ModuleConfigError, match="asymmetric must be true or false"):
        normalize_environment({"asymmetric": "false"}, {"cg_environment": "bilayer"})
    with pytest.raises(ModuleConfigError, match="Random seed must be an integer"):
        normalize_environment({"seed": 1.5}, {"cg_environment": "bilayer"})
    with pytest.raises(ModuleConfigError, match="confirmation endpoint"):
        CGSystemCheckModule().validate_config({"confirm_system": True})


def test_rotated_protein_must_fit_periodic_box_before_coby_wraps_it():
    coordinates = np.array([[0.0, 0.0, 0.0], [15.0, 1.0, 2.0]], dtype=float)
    system = SimpleNamespace(
        num_atoms=2,
        structure=SimpleNamespace(coordinates=coordinates),
    )
    environment = normalize_environment(
        {},
        {"cg_environment": "solution", "cg_include_protein": True},
        coordinates,
    )
    environment["box_xy"] = 12.0
    validate_protein_box(system, environment)
    assert environment["box_xy"] == pytest.approx(18.0)
    assert environment["box_z"] == pytest.approx(6.0)
    assert environment["automatic_box_adjustments"]
    validate_protein_box(system, environment)


def test_cg_orientation_keeps_detected_tm_segment_in_membrane_core(tmp_path):
    helix = np.column_stack((np.linspace(-3.6, 3.6, 19), np.zeros(19), np.zeros(19)))
    soluble = np.asarray(
        [
            [0.2 * (index % 7), 2.0 + 0.25 * (index // 7), 2.5 + 0.08 * (index % 5)]
            for index in range(42)
        ]
    )
    coordinates = np.vstack((helix, soluble))
    names = ["LEU"] * len(helix) + ["GLU"] * len(soluble)
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 20.0,
        atom_names=["BB"] * len(coordinates),
        resnames=names,
        resids=list(range(1, len(coordinates) + 1)),
        chain_ids=["A"] * len(coordinates),
        elements=["C"] * len(coordinates),
    )
    system = System(
        structure=structure,
        components=[
            Component(
                name="Martini 3 Protein",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(len(coordinates), dtype=np.int64),
            )
        ],
        metadata={"cg_environment": "bilayer", "cg_include_protein": True},
    )
    result = CGOrientationModule().execute(
        system,
        {
            "method": "ppm",
            "half_thickness": 1.4,
            "_task_dir": str(tmp_path),
            "_step_dir": str(tmp_path / "step"),
        },
    )
    metrics = result.system.metadata["cg_orientation"]
    assert metrics["tm_window_residues"] >= 15
    assert metrics["tm_core_fraction"] >= 0.65
    assert (tmp_path / "step" / "oriented_protein.pdb").is_file()


def test_cg_manual_orientation_is_relative_to_ppm_and_preserves_geometry(tmp_path):
    coordinates = np.column_stack(
        (
            np.linspace(-3.6, 3.6, 19),
            np.zeros(19),
            np.zeros(19),
        )
    )
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 20.0,
        atom_names=["BB"] * len(coordinates),
        resnames=["LEU"] * len(coordinates),
        resids=list(range(1, len(coordinates) + 1)),
        chain_ids=["A"] * len(coordinates),
        elements=["C"] * len(coordinates),
    )
    system = System(
        structure=structure,
        components=[
            Component(
                name="Martini 3 Protein",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(len(coordinates), dtype=np.int64),
            )
        ],
        metadata={"cg_environment": "bilayer", "cg_include_protein": True},
    )
    base = {
        "method": "ppm",
        "half_thickness": 1.4,
        "_task_dir": str(tmp_path / "ppm"),
        "_step_dir": str(tmp_path / "ppm" / "step"),
    }
    automatic = CGOrientationModule().execute(system, base).system
    manual = (
        CGOrientationModule()
        .execute(
            system,
            {
                "method": "manual",
                "half_thickness": 1.4,
                "z_offset": 0.35,
                "tilt": 17.0,
                "phi": 65.0,
                "_task_dir": str(tmp_path / "manual"),
                "_step_dir": str(tmp_path / "manual" / "step"),
            },
        )
        .system
    )

    assert manual.metadata["cg_orientation"]["method"] == "manual"
    assert np.isclose(
        manual.structure.center_of_geometry()[2] - automatic.structure.center_of_geometry()[2],
        0.35,
        atol=1e-10,
    )
    assert np.allclose(
        np.linalg.norm(np.diff(manual.structure.coordinates, axis=0), axis=1),
        np.linalg.norm(np.diff(automatic.structure.coordinates, axis=0), axis=1),
        atol=1e-10,
    )


def test_cg_orientation_preview_matches_manual_check_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    server._step_runners.clear()
    server._cg_orientation_preview_cache.clear()
    coordinates = np.column_stack(
        (
            np.linspace(-3.6, 3.6, 19),
            np.zeros(19),
            np.zeros(19),
        )
    )
    mapped = System(
        structure=Structure(
            coordinates=coordinates,
            box_vectors=np.eye(3) * 20.0,
            atom_names=["BB"] * len(coordinates),
            resnames=["LEU"] * len(coordinates),
            resids=list(range(1, len(coordinates) + 1)),
            chain_ids=["A"] * len(coordinates),
            elements=["C"] * len(coordinates),
        ),
        components=[
            Component(
                name="Martini 3 Protein",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(len(coordinates), dtype=np.int64),
            )
        ],
        metadata={"cg_environment": "bilayer", "cg_include_protein": True},
    )
    config = {
        "method": "manual",
        "half_thickness": 1.4,
        "z_offset": -0.4,
        "tilt": 21.0,
        "phi": 125.0,
    }

    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"task_type": "martini3-bilayer"})
        task_id = created.json()["task_id"]
        connectivity_path = (
            tmp_path / task_id / "steps" / "cg_mapping" / "martinize" / "cg_protein.pdb"
        )
        connectivity_path.parent.mkdir(parents=True)
        mapped.write_viewer_pdb(connectivity_path)
        connectivity_text = connectivity_path.read_text(encoding="utf-8")
        connectivity_path.write_text(
            connectivity_text.replace("END\n", "CONECT    1    2\nEND\n"),
            encoding="utf-8",
        )
        mapped.metadata.update(
            {
                "cg_protein_pdb": "steps/cg_mapping/martinize/cg_protein.pdb",
                "cg_connectivity_pdb": "steps/cg_mapping/martinize/cg_protein.pdb",
            }
        )
        mapped.save_checkpoint(tmp_path / task_id / "steps" / "cg_mapping")
        response = client.post(
            f"/api/cg-orient-preview/{task_id}",
            json={"config": config},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    expected = (
        CGOrientationModule()
        .execute(
            mapped,
            {
                **config,
                "_task_dir": str(tmp_path / "expected"),
                "_step_dir": str(tmp_path / "expected" / "step"),
            },
        )
        .system
    )
    preview_path = tmp_path / "preview.pdb"
    preview_path.write_text(payload["oriented_pdb"], encoding="utf-8")
    preview = PDBParser().parse(preview_path)
    assert "CONECT    1    2" in payload["oriented_pdb"]
    assert payload["orientation"] == expected.metadata["cg_orientation"]
    assert np.allclose(
        preview.coordinates,
        expected.structure.coordinates,
        atol=1.1e-4,
    )


def test_coarse_grained_capabilities_and_protein_free_input_api(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    server._step_runners.clear()
    with TestClient(app) as client:
        capabilities = client.get("/api/coarse-grained/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["ready"] is True
        created = client.post("/api/tasks", json={"task_type": "martini3-bilayer"})
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        checked = client.post(
            f"/api/step/{task_id}/input",
            json={"config": {"include_protein": False, "environment": "bilayer"}},
        )
        assert checked.status_code == 200, checked.text
        assert checked.json()["status"] == "ok"
        assert (tmp_path / task_id / "steps" / "input" / "system.npz").is_file()

        configs = {
            "cg_model": {"model": "martini3", "water_model": "W"},
            "cg_mapping": {
                "protein_model": "folded",
                "secondary_structure": "auto",
                "elastic": True,
            },
            "cg_orientation": {"method": "ppm", "half_thickness": 1.4},
            "cg_environment": {
                "seed": 99,
                "asymmetric": False,
                "n_lipids_per_leaflet": 64,
                "upper_leaflet": [{"name": "POPC", "ratio": 1}],
            },
            "cg_solvation": {"include_solvent": False, "salt_molarity": 0.15},
            "cg_system": {"salt_molarity": 0.15, "confirm_system": False},
        }
        for step, config in configs.items():
            response = client.post(f"/api/step/{task_id}/{step}", json={"config": config})
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "ok", response.text

        runner = server._get_step_runner(task_id, "martini3-bilayer")
        before = runner.load_system("cg_system")
        assert before is not None and before.metadata["system_confirmed"] is False
        exact_coordinates = np.array(before.structure.coordinates, copy=True)
        confirmed = client.post(f"/api/step/{task_id}/cg_system/confirm")
        assert confirmed.status_code == 200, confirmed.text
        after = runner.load_system("cg_system")
        assert after is not None and after.metadata["system_confirmed"] is True
        assert np.array_equal(after.structure.coordinates, exact_coordinates)
        steps = client.get(f"/api/steps/{task_id}").json()["steps"]
        final_record = next(record for record in steps if record["name"] == "cg_system")
        assert final_record["preview_available"] is True
        assert final_record["has_checkpoint"] is True
        resumed = client.get(f"/api/task/{task_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["step_input_config"]["include_protein"] is False


def test_coarse_grained_frontend_keeps_step_validation_and_viewer_confirmation_separate():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "src/gmxbuilder/web/static/app.js").read_text()
    template = (root / "src/gmxbuilder/web/templates/index.html").read_text()

    assert "function buildModuleConfig(focusStep)" in app_source
    assert "buildModuleConfig(stepName)[stepName]" in app_source
    assert "stepName === 'input' && !isCoarseGrainedWorkflow()" in app_source
    assert "this browser does not provide a working WebGL context" in app_source
    assert "cgConfirmation.disabled = cgViewerRendered !== true" in app_source
    assert "await _doCheckStep(spec[1], spec[2], spec[0]);" in app_source
    assert "if (spec[1] !== 'cg_model') await renderCoarseGrainedViewer" not in app_source
    assert 'id="cg-mapping-controls"' in template
    assert "Execution Hardware" in template
    assert 'class="btn primary cg-check-button" id="cg-mapping-check"' in template
    assert 'id="panel-cg_orientation"' in template
    assert 'id="cg-orientation-check"' in template
    assert 'id="cg-orient-tab-ppm"' in template
    assert 'id="cg-orient-tab-manual"' in template
    assert 'class="orient-tab cg-orient-tab' not in template
    assert 'id="cg-orient-manual-z"' in template
    assert 'id="cg-orient-manual-tilt"' in template
    assert 'id="cg-orient-manual-phi"' in template
    assert "/api/cg-orient-preview/" in app_source
    assert "viewer.removeAllShapes()" in app_source
    assert "function addCgOrientationPlaneMarkers" in app_source
    assert "Number(halfThicknessNm) * 10.0" in app_source
    assert "stepName !== 'cg_mapping' && stepName !== 'cg_orientation'" in app_source
    assert "stick: {radius: 0.065" in app_source
    assert "not an energy-minimized or equilibrated membrane" in template


def _run(runner: StepRunner, step: str, config: dict) -> None:
    result = runner.run_step(step, config)
    assert result["status"] == "ok", result


def _write_glycine_hairpin(path: Path) -> None:
    lines: list[str] = []
    serial = 1
    ca_points = np.asarray(
        [
            [0.00, 0.00],
            [3.80, 0.00],
            [7.60, 0.00],
            [11.40, 0.00],
            [11.40, 3.80],
            [7.60, 3.80],
            [3.80, 3.80],
            [0.00, 3.80],
        ]
    )
    for index, ca in enumerate(ca_points):
        previous = ca_points[max(0, index - 1)]
        following = ca_points[min(len(ca_points) - 1, index + 1)]
        tangent = following - previous
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.asarray([-tangent[1], tangent[0]])
        for name, element, xy in (
            ("N", "N", ca - 1.45 * tangent),
            ("CA", "C", ca),
            ("C", "C", ca + 1.45 * tangent),
            ("O", "O", ca + 1.45 * tangent + normal),
        ):
            lines.append(
                f"ATOM  {serial:5d} {name:^4s} GLY A{index + 1:4d}    "
                f"{xy[0]:8.3f}{xy[1]:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          "
                f"{element:>2s}"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def test_real_martinize_and_coby_solution_protein_path(tmp_path):
    """Exercise the previously uncovered atomistic-to-CG protein path."""
    pdb = tmp_path / "gly_hairpin.pdb"
    _write_glycine_hairpin(pdb)
    runner = StepRunner(tmp_path / "task", pipeline_type="martini3-solvent")
    _run(
        runner,
        "input",
        {
            "pdb": str(pdb),
            "include_protein": True,
            "environment": "solution",
        },
    )
    _run(runner, "cg_model", {"model": "martini3", "water_model": "W"})
    _run(
        runner,
        "cg_mapping",
        {
            "protein_model": "folded",
            "secondary_structure": "manual",
            "secondary_structure_string": "HHHHHHHH",
            "elastic": True,
            "elastic_lower": 0.3,
            "elastic_upper": 0.9,
        },
    )
    mapped = runner.load_system("cg_mapping")
    assert mapped is not None
    assert mapped.component_by_kind(ComponentKind.PROTEIN)
    assert mapped.metadata["cg_mapping"]["beads"] == 8
    assert mapped.metadata["cg_mapping"]["elastic_network"] is True
    topology_text = "\n".join(mapped.metadata["cg_topology_texts"].values())
    assert "Rubber band" in topology_text or "RUBBER" in topology_text.upper()
    _run(
        runner,
        "cg_environment",
        {
            "seed": 2718,
        },
    )
    _run(
        runner,
        "cg_solvation",
        {
            "include_solvent": True,
            "salt_molarity": 0.15,
        },
    )
    _run(
        runner,
        "cg_system",
        {
            "salt_molarity": 0.15,
            "confirm_system": False,
        },
    )
    final = runner.load_system("cg_system")
    assert final is not None
    quality = final.metadata["cg_scientific_check"]
    assert quality["passed"] is True
    assert quality["net_charge_e"] == pytest.approx(0.0, abs=1e-5)
    assert quality["actual_salt_molarity"] == pytest.approx(0.15, abs=0.02)
    assert final.component_by_kind(ComponentKind.PROTEIN)


def test_real_coby_mixed_bilayer_exports_exact_neutral_package(tmp_path):
    """Run the pinned builder, not a mock, across the complete pure-bilayer path."""
    runner = StepRunner(tmp_path / "task", pipeline_type="martini3-bilayer")
    _run(runner, "input", {"include_protein": False, "environment": "bilayer"})
    _run(runner, "cg_model", {"model": "martini3", "water_model": "W"})
    _run(
        runner,
        "cg_mapping",
        {
            "protein_model": "folded",
            "secondary_structure": "auto",
            "elastic": True,
        },
    )
    _run(runner, "cg_orientation", {"method": "ppm", "half_thickness": 1.4})
    _run(
        runner,
        "cg_environment",
        {
            "seed": 1729,
            "n_lipids_per_leaflet": 64,
            "asymmetric": True,
            "upper_leaflet": [
                {"name": "POPC", "ratio": 3},
                {"name": "CHOL", "ratio": 1},
            ],
            "lower_leaflet": [
                {"name": "POPE", "ratio": 1},
                {"name": "POPG", "ratio": 1},
            ],
        },
    )
    _run(runner, "cg_solvation", {"include_solvent": True, "salt_molarity": 0.15})
    _run(runner, "cg_system", {"salt_molarity": 0.15, "confirm_system": False})

    final = runner.load_system("cg_system")
    assert final is not None
    quality = final.metadata["cg_scientific_check"]
    assert quality["passed"] is True
    assert quality["net_charge_e"] == pytest.approx(0.0, abs=1e-5)
    assert quality["actual_salt_molarity"] == pytest.approx(0.15, abs=0.02)
    assert quality["bilayer_orientation"]["correct_fraction"] >= 0.98
    assert quality["bilayer_orientation"]["upper_leaflet_lipids"] == 64
    assert quality["bilayer_orientation"]["lower_leaflet_lipids"] == 64
    assert 2.5 <= quality["bilayer_orientation"]["headgroup_separation_nm"] <= 6.0
    assert quality["solvent_layers"]["water_beads_below"] >= 10
    assert quality["solvent_layers"]["water_beads_above"] >= 10
    assert quality["protein_placement"] is None
    source_coordinates = np.array(final.structure.coordinates, copy=True)

    final.metadata["system_confirmed"] = True
    final.save_checkpoint(runner.step_dir("cg_system"))
    result = runner.finalize_from_checkpoint(
        "cg_system",
        simparams={
            "temperature": 310,
            "pressure": 1,
            "production_ns": 10,
            "output_interval_ps": 100,
            "equilibration_1": True,
            "equilibration_2": True,
            "use_gpu": False,
            "gpu_ids": "0",
            "threads": 2,
            "mpi_ranks": 1,
            "system_name": "cg_acceptance",
        },
        export_config={"write_mdp": True, "system_name": "cg_acceptance"},
    )
    assert result["status"] == "ok", result
    archive_path = Path(result["zip_path"])
    assert archive_path.is_file()
    exported = runner.load_system("cg_system")
    assert exported is not None
    assert np.array_equal(exported.structure.coordinates, source_coordinates)
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        assert {"input.gro", "input.pdb", "topol.top", "index.ndx", "run_md.sh"} <= members
        assert {
            "mdp/mini.mdp",
            "mdp/equilibration_1.mdp",
            "mdp/equilibration_2.mdp",
            "mdp/production.mdp",
        } <= members
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["coordinate_source"] == "exact cg_system Check checkpoint"
        assert manifest["simulation_ready"] is True
        for member in members:
            assert not member.startswith("/") and ".." not in Path(member).parts


def test_coarse_grained_cli_builds_dry_bilayer_package(tmp_path, monkeypatch):
    output = tmp_path / "cli-output"

    def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("dry geometry export must not require GROMACS")

    monkeypatch.setattr(CGExportModule, "_validate_with_gromacs", unexpected_validation)
    result = CliRunner().invoke(
        main,
        [
            "martini3-bilayer",
            "--dry",
            "--yes",
            "--lipids-per-leaflet",
            "64",
            "--production-ns",
            "10",
            "--threads",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "input.gro").is_file()
    assert (output / "topol.top").is_file()
    assert not (output / "run_md.sh").exists()
    archives = list(output.glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert "run_md.sh" not in archive.namelist()
        assert not any(name.startswith("mdp/") for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["gromacs_validation"] == "not-requested"

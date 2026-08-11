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
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained import (
    CGEnvironmentModule,
    CGExportModule,
    CGInputModule,
    CGMappingModule,
    CGModelModule,
    CGSolvationModule,
    CGSystemCheckModule,
    CGTopologyModule,
)
from gmxbuilder.modules.coarse_grained.assets import public_capabilities, verify_assets
from gmxbuilder.modules.coarse_grained.backend import (
    normalize_environment,
    normalize_solvation,
    validate_protein_box,
)
from gmxbuilder.modules.coarse_grained.common import normalize_composition
from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol
from gmxbuilder.pipeline.step_executor import StepRunner, _get_module, get_pipeline_steps
from gmxbuilder.web import server
from gmxbuilder.web.server import app, task_manager
from gmxbuilder.web.task_types import get_task_type


def test_coarse_grained_task_and_modules_are_independent():
    task = get_task_type("coarse-grained")
    assert task is not None and task.enabled and not task.requires_input
    assert task.pipeline == "coarse_grained"
    assert get_pipeline_steps("coarse-grained") == [
        "input", "cg_model", "cg_mapping", "cg_environment",
        "cg_solvation", "cg_system", "topology", "export",
    ]
    expected = {
        "input": CGInputModule,
        "cg_model": CGModelModule,
        "cg_mapping": CGMappingModule,
        "cg_environment": CGEnvironmentModule,
        "cg_solvation": CGSolvationModule,
        "cg_system": CGSystemCheckModule,
        "topology": CGTopologyModule,
        "export": CGExportModule,
    }
    for step, module_type in expected.items():
        module = _get_module(step, "coarse-grained")
        assert isinstance(module, module_type)
        assert module.__class__.__module__.startswith("gmxbuilder.modules.coarse_grained")


def test_martini_assets_and_public_boundaries_are_explicit():
    verified = verify_assets()
    capabilities = public_capabilities()

    assert "martini_v3.0.0.itp" in verified
    assert capabilities["ready"] is True
    assert capabilities["force_field"] == "Martini 3.0.0"
    assert {item["name"] for item in capabilities["lipids"]} >= {
        "POPC", "POPE", "POPG", "POPS", "CHOL",
    }
    assert capabilities["boundaries"] == {
        "standard_protein_residues": True,
        "elastic_network": True,
        "custom_molecules": False,
        "post_translational_modifications": False,
        "curved_membranes": False,
        "backmapping": False,
    }


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
        {"box_xy": 12.0, "box_z": 14.0},
        {"cg_environment": "solution", "cg_include_protein": True},
    )
    with pytest.raises(ModuleConfigError, match="periodic wrapping"):
        validate_protein_box(system, environment)
    environment["box_xy"] = 18.0
    validate_protein_box(system, environment)


def test_coarse_grained_capabilities_and_protein_free_input_api(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    server._step_runners.clear()
    with TestClient(app) as client:
        capabilities = client.get("/api/coarse-grained/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["ready"] is True
        created = client.post("/api/tasks", json={"task_type": "coarse-grained"})
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
                "protein_model": "folded", "secondary_structure": "auto",
                "elastic": True,
            },
            "cg_environment": {
                "environment": "bilayer", "box_xy": 5.0, "box_z": 8.0,
                "seed": 99, "asymmetric": False,
                "upper_leaflet": [{"name": "POPC", "ratio": 1}],
            },
            "cg_solvation": {"include_solvent": False, "salt_molarity": 0.15},
            "cg_system": {"salt_molarity": 0.15, "confirm_system": False},
        }
        for step, config in configs.items():
            response = client.post(f"/api/step/{task_id}/{step}", json={"config": config})
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "ok", response.text

        runner = server._get_step_runner(task_id, "coarse-grained")
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
    assert "stepName !== 'cg_mapping'" in app_source


def _run(runner: StepRunner, step: str, config: dict) -> None:
    result = runner.run_step(step, config)
    assert result["status"] == "ok", result


def test_real_coby_mixed_bilayer_exports_exact_neutral_package(tmp_path):
    """Run the pinned builder, not a mock, across the complete pure-bilayer path."""
    runner = StepRunner(tmp_path / "task", pipeline_type="coarse-grained")
    _run(runner, "input", {"include_protein": False, "environment": "bilayer"})
    _run(runner, "cg_model", {"model": "martini3", "water_model": "W"})
    _run(runner, "cg_mapping", {
        "protein_model": "folded", "secondary_structure": "auto", "elastic": True,
    })
    _run(runner, "cg_environment", {
        "environment": "bilayer", "box_xy": 6.0, "box_z": 10.0, "seed": 1729,
        "asymmetric": True,
        "upper_leaflet": [
            {"name": "POPC", "ratio": 3}, {"name": "CHOL", "ratio": 1},
        ],
        "lower_leaflet": [
            {"name": "POPE", "ratio": 1}, {"name": "POPG", "ratio": 1},
        ],
    })
    _run(runner, "cg_solvation", {"include_solvent": True, "salt_molarity": 0.15})
    _run(runner, "cg_system", {"salt_molarity": 0.15, "confirm_system": False})

    final = runner.load_system("cg_system")
    assert final is not None
    quality = final.metadata["cg_scientific_check"]
    assert quality["passed"] is True
    assert quality["net_charge_e"] == pytest.approx(0.0, abs=1e-5)
    assert quality["actual_salt_molarity"] == pytest.approx(0.15, abs=0.02)
    assert quality["bilayer_orientation"]["correct_fraction"] >= 0.98
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
            "temperature": 310, "pressure": 1, "production_ns": 10,
            "output_interval_ps": 100, "equilibration_1": True,
            "equilibration_2": True, "use_gpu": False, "gpu_ids": "0",
            "threads": 2, "mpi_ranks": 1, "system_name": "cg_acceptance",
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
        assert {"mdp/mini.mdp", "mdp/equilibration_1.mdp", "mdp/equilibration_2.mdp", "mdp/production.mdp"} <= members
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["coordinate_source"] == "exact cg_system Check checkpoint"
        assert manifest["simulation_ready"] is True
        for member in members:
            assert not member.startswith("/") and ".." not in Path(member).parts


def test_coarse_grained_cli_builds_dry_bilayer_package(tmp_path):
    output = tmp_path / "cli-output"
    result = CliRunner().invoke(main, [
        "coarse-grained", "--dry", "--yes", "--box-xy", "5",
        "--box-z", "8", "--production-ns", "10", "--threads", "2",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert (output / "input.gro").is_file()
    assert (output / "topol.top").is_file()
    assert not (output / "run_md.sh").exists()
    archives = list(output.glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert "run_md.sh" not in archive.namelist()
        assert not any(name.startswith("mdp/") for name in archive.namelist())

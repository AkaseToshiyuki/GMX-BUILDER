"""Finalization, force-field protocol, and security-boundary regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.io.mdp import _nonbond_params
from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.modules.pure_membrane.export import PureMembraneExportModule
from gmxbuilder.pipeline.step_executor import StepRunner, get_pipeline_steps
from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager


def _checkpoint_system() -> System:
    return System(
        structure=Structure(
            coordinates=np.array([[1.23456, 2.34567, 3.45678]], dtype=float),
            box_vectors=np.diag([5.0, 6.0, 7.0]),
            atom_names=["CA"],
            resnames=["ALA"],
            resids=[1],
            chain_ids=["A"],
            elements=["C"],
        ),
        metadata={
            "force_field": "amber99sb-ildn",
            "lipid_ff": "gaff2",
            "water_model": "tip3p",
            "seed": 8128,
        },
    )


def test_finalization_exports_exact_checkpoint_without_coordinate_rebuild(tmp_path):
    runner = StepRunner(tmp_path, pipeline_type="pure-membrane")
    source = _checkpoint_system()
    source.save_checkpoint(runner.step_dir("membrane"))

    result = runner.finalize_from_checkpoint(
        "membrane",
        export_config={"system_name": "exact", "write_mdp": False},
    )

    assert result["status"] == "ok"
    exported = GROReader().read(runner.step_dir("export") / "input.gro")
    assert exported.num_atoms == source.num_atoms
    assert np.allclose(exported.coordinates, source.coordinates, atol=5.1e-4, rtol=0)
    assert np.allclose(exported.box_vectors, source.structure.box_vectors, atol=5.1e-6)
    assert any("confirmed membrane checkpoint" in line for line in result["log"])
    assert result["package_contents"] == {
        "simulation_ready": False,
        "run_script": None,
        "mdp_files": [],
        "dry_export": True,
    }
    with zipfile.ZipFile(result["zip_path"]) as archive:
        assert "run_md.sh" not in archive.namelist()
        assert not any(name.startswith("mdp/") for name in archive.namelist())
    assert "verify" not in get_pipeline_steps("pure-membrane")


def test_dry_export_archive_excludes_stale_and_unrelated_files(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    (output / "unrelated-secret.txt").write_text("must not ship")
    stale_mdp = output / "mdp"
    stale_mdp.mkdir()
    (stale_mdp / "obsolete.mdp").write_text("stale")
    (output / "run_md.sh").write_text("stale")

    result = PureMembraneExportModule().run(
        _checkpoint_system(),
        {
            "output_dir": output,
            "system_name": "dry",
            "write_mdp": False,
        },
    )

    assert result.success is True
    with zipfile.ZipFile(output / "dry.zip") as archive:
        names = set(archive.namelist())
    assert "unrelated-secret.txt" not in names
    assert "run_md.sh" not in names
    assert not any(name.startswith("mdp/") for name in names)


def test_finalization_ignores_client_controlled_output_directory(tmp_path):
    runner = StepRunner(tmp_path / "task", pipeline_type="pure-membrane")
    source = _checkpoint_system()
    source.save_checkpoint(runner.step_dir("membrane"))
    outside = tmp_path / "attacker-selected"

    result = runner.finalize_from_checkpoint(
        "membrane",
        export_config={
            "system_name": "confined",
            "write_mdp": False,
            "output_dir": str(outside),
        },
    )

    assert result["status"] == "ok"
    assert not outside.exists()
    assert (runner.step_dir("export") / "confined.zip").is_file()


def test_wet_finalization_contract_includes_launcher_and_mdp_without_path_leaks(
    tmp_path,
):
    runner = StepRunner(tmp_path, pipeline_type="membrane-bilayer")
    source = _checkpoint_system()
    source.save_checkpoint(runner.step_dir("ions"))

    result = runner.finalize_from_checkpoint(
        "ions",
        export_config={"system_name": "wet"},
    )

    assert result["status"] == "ok"
    contents = result["package_contents"]
    assert contents["simulation_ready"] is True
    assert contents["run_script"] == "run_md.sh"
    assert "mdp/mini.mdp" in contents["mdp_files"]
    assert any(name.startswith("mdp/production") for name in contents["mdp_files"])
    with zipfile.ZipFile(result["zip_path"]) as archive:
        names = archive.namelist()
        assert "run_md.sh" in names
        assert "mdp/mini.mdp" in names
        assert (archive.getinfo("run_md.sh").external_attr >> 16) & 0o111
    public_log = "\n".join(result["log"])
    assert str(tmp_path) not in public_log
    assert "/home/" not in public_log
    assert "/tmp/" not in public_log


def test_task_download_prefers_current_export_over_larger_legacy_zip(
    tmp_path, monkeypatch,
):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("example.pdb")
    task_id = task["task_id"]
    legacy = manager.get_output_dir(task_id) / "legacy.zip"
    legacy.write_bytes(b"legacy archive is deliberately larger")
    export_dir = manager.get_task_dir(task_id) / "steps" / "export"
    export_dir.mkdir(parents=True)
    current = export_dir / "current.zip"
    current.write_bytes(b"current")
    monkeypatch.setattr(server, "task_manager", manager)

    assert server._authoritative_task_zip(task_id) == current


def test_legacy_download_route_survives_process_restart(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("complete.pdb")
    task_id = task["task_id"]
    export_dir = manager.get_task_dir(task_id) / "steps" / "export"
    export_dir.mkdir(parents=True)
    archive = export_dir / "current.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    manager.update_state(task_id, {"build_status": {"status": "completed"}})
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_tasks", {})

    response = asyncio.run(server.api_download(task_id))

    assert Path(response.path) == archive


def test_public_build_log_redacts_server_paths():
    assert (
        server._redact_server_paths(
            "Wrote /home/example/.local/share/gmxbuilder/tasks/abc/export/system.zip"
        )
        == "Wrote <server-path>"
    )


def test_path_redaction_uses_configured_task_root(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        server, "task_manager", SimpleNamespace(root=Path("/workspace/gmx/tasks"))
    )

    assert server._redact_server_paths(
        "Failed at /workspace/gmx/tasks/abc/steps/ions/system.npz"
    ) == "Failed at <server-path>"


def test_finalization_requires_the_confirmed_checkpoint(tmp_path):
    result = StepRunner(tmp_path, "membrane-bilayer").finalize_from_checkpoint("ions")
    assert result["status"] == "error"
    assert "Ion" in result["error"] or "ions" in result["error"]


def test_nonbond_defaults_are_force_field_specific():
    amber = _nonbond_params({"force_field": "amber14sb"})
    charmm = _nonbond_params({"force_field": "charmm36m"})

    assert "vdw-modifier            = Potential-shift" in amber
    assert "rvdw                    = 1.0" in amber
    assert "DispCorr                = EnerPres" in amber
    assert "rvdw_switch" not in amber
    assert "vdw-modifier            = Force-switch" in charmm
    assert "rvdw_switch             = 1.0" in charmm
    assert "rvdw                    = 1.2" in charmm
    assert "DispCorr                = no" in charmm


def test_charmm_dispersion_correction_conflict_is_rejected():
    with pytest.raises(ValueError, match="DispCorr=no"):
        _nonbond_params({"force_field": "charmm36", "dispcorr": "EnerPres"})


def test_lipid_library_rejects_path_traversal(tmp_path):
    library = EquilibratedLipidLibrary([tmp_path])
    with pytest.raises(ValueError, match="Unsafe lipid name"):
        library.entry_dir("../../../../TMP/AUDIT-PROBE", "amber14sb", "gaff2")
    assert not (tmp_path.parent / "TMP" / "AUDIT-PROBE").exists()


def test_new_task_ids_have_full_entropy_and_private_files(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("private.pdb")
    task_dir = manager.get_task_dir(task["task_id"])

    assert len(task["task_id"]) == 32
    assert task_dir.stat().st_mode & 0o077 == 0
    assert (task_dir / "state.json").stat().st_mode & 0o077 == 0


def test_online_lipid_build_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_ALLOW_ONLINE_LIPID_BUILD", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/build-lipid-library",
            json={
                "lipid_name": "AUDIT",
                "is_custom": True,
                "force_field": "amber14sb",
                "lipid_ff": "gaff2",
            },
        )
    assert response.status_code == 403
    assert "administrator" in response.json()["error"].lower()

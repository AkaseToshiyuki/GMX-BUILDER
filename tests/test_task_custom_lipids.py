"""Task ownership, lifecycle and routing regressions for custom lipids."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.modules.forcefield import gaff_backend
from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid
from gmxbuilder.web import server
from gmxbuilder.web.custom_lipids import (
    CustomLipidStore,
    task_custom_lipid_scope,
)
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager, task_manager
from tests.prerequisites import requires_lfs_assets

NEW_SMILES = "CCCCCCCCCCCCCCCCCC(=O)OCC(O)CO"


def _web_task() -> str:
    task = task_manager.create_task("custom-test.pdb")
    task_manager.update_state(
        task["task_id"],
        {
            "task_type_id": "membrane-bilayer",
            "task_type": {
                "id": "membrane-bilayer",
                "route_slug": "BilayerBuilder",
                "requires_input": True,
            },
        },
    )
    return task["task_id"]


def test_web_lipid_submission_is_retired_before_any_work(monkeypatch):
    task_id = _web_task()
    paths = {
        "parse": "/api/custom-lipid",
        "submit": f"/api/task/{task_id}/custom-lipids",
        "retry": f"/api/task/{task_id}/custom-lipids/PVA/retry",
        "build": "/api/build-lipid-library",
    }
    monkeypatch.setenv("GMXBUILDER_ADMIN_TOKEN", "test-administrator-token-only")

    def forbidden(*args, **kwargs):
        pytest.fail("A retired Web request must not parse, save or schedule a lipid")

    monkeypatch.setattr("gmxbuilder.web.custom_lipids.run_custom_lipid_build", forbidden)
    monkeypatch.setattr(CustomLipidStore, "save_submission", forbidden)
    monkeypatch.setattr("gmxbuilder.modules.membrane.lipids.parse_custom_lipid", forbidden)
    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.lipid_equilibration.LipidEquilibrationBuilder.build", forbidden
    )
    try:
        with TestClient(app) as client:
            for route, path in paths.items():
                for body in (b'{"name":"PVA","smiles":"CC","is_custom":true}', b"not-json"):
                    response = client.post(
                        path,
                        content=body,
                        headers={
                            "Content-Type": "application/json",
                            "X-Admin-Token": "test-administrator-token-only",
                        },
                    )
                    assert response.status_code == 403, route
                    assert response.json()["code"] == "lipid_submission_offline_only", route
                    assert "SMILES" in response.json()["error"], route
                    assert response.json()["contact_url"] == "/", route
        assert not (task_manager.get_task_dir(task_id) / "custom_lipids").exists()
    finally:
        task_manager.delete_task(task_id)


@pytest.mark.parametrize("legacy_state", ["queued", "running", "failed", "ready"])
def test_legacy_lipids_are_readable_but_never_restarted(monkeypatch, legacy_state):
    task_a = _web_task()
    task_b = _web_task()
    store = CustomLipidStore(task_manager.get_task_dir(task_a))
    store.save_submission(parse_custom_lipid(NEW_SMILES, "PVA"), "amber14sb")
    store.update_status("PVA", state=legacy_state, phase=legacy_state, progress=0, message="legacy")
    before = store.public_record("PVA")

    def forbidden(*args, **kwargs):
        pytest.fail("Web startup must not recover an offline-only lipid calculation")

    monkeypatch.setattr("gmxbuilder.web.custom_lipids.run_custom_lipid_build", forbidden)
    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.lipid_equilibration.LipidEquilibrationBuilder.build", forbidden
    )
    monkeypatch.setattr("gmxbuilder.modules.forcefield.gaff_backend.prepare_gaff_lipid", forbidden)
    try:
        with TestClient(app) as client:
            own = client.get(f"/api/task/{task_a}/custom-lipids").json()
            other = client.get(f"/api/task/{task_b}/custom-lipids").json()
            detail = client.get(f"/api/task/{task_a}/custom-lipids/PVA").json()
            retry = client.post(f"/api/task/{task_a}/custom-lipids/PVA/retry")
        assert own["lipids"] == [before]
        assert detail == before
        assert other["lipids"] == []
        assert retry.status_code == 403
        assert store.public_record("PVA") == before
        if legacy_state == "ready":
            assert server._require_task_custom_lipids_ready(task_a) == [before]
        else:
            with pytest.raises(ValueError, match="Contact the administrator"):
                server._require_task_custom_lipids_ready(task_a)
    finally:
        task_manager.delete_task(task_a)
        task_manager.delete_task(task_b)


@requires_lfs_assets
def test_ready_task_lipid_uses_task_cache_and_cleanup_removes_all(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("private.pdb")
    task_dir = manager.get_task_dir(task["task_id"])
    store = CustomLipidStore(task_dir)
    properties = parse_custom_lipid(NEW_SMILES, "PVB")
    store.save_submission(properties, "amber14sb")
    store.update_status(
        "PVB",
        state="ready",
        phase="complete",
        progress=100,
        message="ready",
    )
    secret = store.gaff_cache / "private.dat"
    secret.parent.mkdir(parents=True)
    secret.write_text("task-owned")

    assert "PVB" not in LipidRegistry.list()
    with task_custom_lipid_scope(task_dir):
        assert LipidRegistry.get("PVB").name == "PVB"
        assert gaff_backend._cache_root("PVB") == store.gaff_cache.resolve()
        assert gaff_backend._cache_root("POPC") != store.gaff_cache.resolve()
    assert "PVB" not in LipidRegistry.list()

    state = manager.get_state(task["task_id"])
    state["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    manager._write_state(task_dir, state)
    assert manager.cleanup_expired() == [task["task_id"]]
    assert not task_dir.exists()


def test_membrane_config_uses_server_definition_and_rejects_cross_task_reference():
    task_a = _web_task()
    task_b = _web_task()
    try:
        store = CustomLipidStore(task_manager.get_task_dir(task_a))
        properties = parse_custom_lipid(NEW_SMILES, "PVC")
        store.save_submission(properties, "amber14sb")
        store.update_status(
            "PVC",
            state="ready",
            phase="complete",
            progress=100,
            message="ready",
        )
        untrusted = {
            "lipid_composition": {
                "upper": [
                    {
                        "name": "PVC",
                        "ratio": 100,
                        "category": "ST",
                        "charge": 99,
                        "smiles": "C",
                    }
                ],
                "lower": None,
            }
        }
        trusted = server._trusted_membrane_config(task_a, untrusted)
        entry = trusted["lipid_composition"]["upper"][0]
        assert entry["ratio"] == 100
        assert entry["smiles"] == properties["smiles"]
        assert entry["charge"] == properties["charge"]
        assert entry["category"] == properties["category"]
        with pytest.raises(ValueError, match="does not belong to this task"):
            server._trusted_membrane_config(task_b, untrusted)
    finally:
        task_manager.delete_task(task_a)
        task_manager.delete_task(task_b)


def test_custom_lipid_residue_id_is_limited_to_five_safe_characters():
    parsed = parse_custom_lipid(NEW_SMILES, "MyLip")
    assert parsed["name"] == "MYLIP"
    assert parsed["common_name"] == "MyLip"
    with pytest.raises(ValueError, match="1-5 uppercase"):
        parse_custom_lipid(NEW_SMILES, "TOOLONG")


def test_cleanup_skips_active_task_then_deletes_it(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("active.pdb")
    task_dir = manager.get_task_dir(task["task_id"])
    state = manager.get_state(task["task_id"])
    state["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    manager._write_state(task_dir, state)

    with manager.active_task(task["task_id"]):
        assert manager.cleanup_expired() == []
        assert task_dir.exists()
    assert manager.cleanup_expired() == [task["task_id"]]


def test_workflow_routes_hide_task_ids_and_legacy_links_return_home():
    task_id = _web_task()
    try:
        with TestClient(app) as client:
            assert client.get("/BilayerBuilder/Step1").status_code == 200
            page = client.get(f"/BilayerBuilder/{task_id}/Step2", follow_redirects=False)
            assert page.status_code == 307
            assert page.headers["location"] == "/"
            mismatch = client.get(f"/Solvator/{task_id}/Step1", follow_redirects=False)
            assert mismatch.status_code == 307
            assert mismatch.headers["location"] == "/"
            assert client.get("/UnknownWorkflow/Step1").status_code == 404
    finally:
        task_manager.delete_task(task_id)

"""Task ownership, lifecycle and routing regressions for custom lipids."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from gmxbuilder.modules.forcefield import gaff_backend
from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid
from gmxbuilder.web import server
from gmxbuilder.web.custom_lipids import (
    CustomLipidStore,
    task_custom_lipid_scope,
)
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager, task_manager


NEW_SMILES = "CCCCCCCCCCCCCCCCCC(=O)OCC(O)CO"


def _web_task() -> str:
    task = task_manager.create_task("custom-test.pdb")
    task_manager.update_state(task["task_id"], {
        "task_type_id": "membrane-bilayer",
        "task_type": {
            "id": "membrane-bilayer",
            "route_slug": "BilayerBuilder",
            "requires_input": True,
        },
    })
    return task["task_id"]


def test_submission_rejects_standard_library_identity(monkeypatch):
    task_id = _web_task()
    monkeypatch.setattr(server, "_schedule_custom_lipid_build", lambda *_: True)
    try:
        popc = LipidRegistry.get("POPC")
        with TestClient(app) as client:
            response = client.post(
                f"/api/task/{task_id}/custom-lipids",
                json={
                    "name": "DUPL",
                    "smiles": popc.smiles,
                    "force_field": "amber14sb",
                    "lipid_ff": "gaff2",
                },
            )
        assert response.status_code == 409
        assert response.json()["existing_lipid"] == "POPC"
        assert not (task_manager.get_task_dir(task_id) / "custom_lipids").exists()
    finally:
        task_manager.delete_task(task_id)


def test_task_submission_is_private_and_not_selectable_until_ready(monkeypatch):
    task_a = _web_task()
    task_b = _web_task()
    monkeypatch.setattr(server, "_schedule_custom_lipid_build", lambda *_: True)
    try:
        with TestClient(app) as client:
            submitted = client.post(
                f"/api/task/{task_a}/custom-lipids",
                json={
                    "name": "PVA",
                    "smiles": NEW_SMILES,
                    "force_field": "amber14sb",
                    "lipid_ff": "gaff2",
                },
            )
            own = client.get(f"/api/task/{task_a}/custom-lipids").json()
            other = client.get(f"/api/task/{task_b}/custom-lipids").json()
        assert submitted.status_code == 202
        assert own["lipids"][0]["state"] == "queued"
        assert other["lipids"] == []
        assert "PVA" not in LipidRegistry.list()
        with pytest.raises(ValueError, match="must finish successfully"):
            server._require_task_custom_lipids_ready(task_a)
        with task_custom_lipid_scope(task_manager.get_task_dir(task_a)):
            with pytest.raises(KeyError):
                LipidRegistry.get("PVA")
        with task_custom_lipid_scope(
            task_manager.get_task_dir(task_a), include_unready={"PVA"}
        ):
            assert LipidRegistry.get("PVA").smiles
        with task_custom_lipid_scope(task_manager.get_task_dir(task_b)):
            with pytest.raises(KeyError):
                LipidRegistry.get("PVA")
    finally:
        task_manager.delete_task(task_a)
        task_manager.delete_task(task_b)


def test_ready_task_lipid_uses_task_cache_and_cleanup_removes_all(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("private.pdb")
    task_dir = manager.get_task_dir(task["task_id"])
    store = CustomLipidStore(task_dir)
    properties = parse_custom_lipid(NEW_SMILES, "PVB")
    store.save_submission(properties, "amber14sb")
    store.update_status(
        "PVB", state="ready", phase="complete", progress=100,
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
    state["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
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
            "PVC", state="ready", phase="complete", progress=100,
            message="ready",
        )
        untrusted = {
            "lipid_composition": {
                "upper": [{
                    "name": "PVC", "ratio": 100,
                    "category": "ST", "charge": 99, "smiles": "C",
                }],
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
    state["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
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
            page = client.get(
                f"/BilayerBuilder/{task_id}/Step2", follow_redirects=False
            )
            assert page.status_code == 307
            assert page.headers["location"] == "/"
            mismatch = client.get(
                f"/Solvator/{task_id}/Step1", follow_redirects=False
            )
            assert mismatch.status_code == 307
            assert mismatch.headers["location"] == "/"
            assert client.get("/UnknownWorkflow/Step1").status_code == 404
    finally:
        task_manager.delete_task(task_id)


def test_custom_lipid_gpu_allocator_limits_and_rotates_devices(monkeypatch):
    monkeypatch.setattr(server, "_CUSTOM_GPU_IDS", (0, 1, 2))
    monkeypatch.setattr(server, "_CUSTOM_GPU_CONCURRENCY", 2)
    with server._custom_gpu_condition:
        server._custom_gpu_in_use.clear()
        server._custom_gpu_cursor = 0
    first = server._acquire_custom_gpu()
    second = server._acquire_custom_gpu()
    try:
        assert first == 0
        assert second == 1
        server._release_custom_gpu(first)
        first = None
        third = server._acquire_custom_gpu()
        try:
            assert third == 2
            assert len(server._custom_gpu_in_use) == 2
        finally:
            server._release_custom_gpu(third)
    finally:
        server._release_custom_gpu(first)
        server._release_custom_gpu(second)

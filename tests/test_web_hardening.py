"""Regression tests for bounded Web input and capability handling."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.server_parts.admission import (
    BoundedAdmission,
    TaskAdmission,
    WorkQueueFull,
)
from gmxbuilder.web.server_parts.task_security import (
    redact_task_capabilities,
    task_log_reference,
)
from gmxbuilder.web.task_manager import TaskManager
from gmxbuilder.web.task_types import get_task_type_detail

PDB_TWO_ATOMS = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
    "END\n"
)


def test_malformed_json_and_non_string_task_capabilities_are_client_errors():
    with TestClient(app) as client:
        malformed = client.post(
            "/api/build", content=b"{", headers={"content-type": "application/json"}
        )
        non_object = client.post("/api/build", json=[])
        malformed_id = client.get("/api/task/not-a-capability")

    assert malformed.status_code == 400
    assert non_object.status_code == 400
    assert malformed_id.status_code == 400
    assert malformed_id.json() == {"error": "Invalid task ID format"}


def test_task_capability_log_references_are_irreversible_and_redactable():
    task_id = "a" * 32
    reference = task_log_reference(task_id)
    assert task_id not in reference
    assert redact_task_capabilities(f"failed task {task_id}") == ("failed task <task-capability>")
    assert server._rate_policy(f"/api/task/{task_id}/resume", "GET")[0] == "api-read"


def test_interactive_admission_rejects_before_executor_submission(monkeypatch):
    admission = BoundedAdmission(1)
    assert admission.try_acquire() is True
    monkeypatch.setattr(server, "_interactive_admission", admission)

    with pytest.raises(WorkQueueFull):
        asyncio.run(server._run_interactive(lambda: "must not run"))


def test_cancelled_queued_interactive_work_returns_its_capacity(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    admission = BoundedAdmission(2)
    release, started = Event(), Event()

    def running():
        started.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(server, "_interactive_admission", admission)
        monkeypatch.setattr(server, "_get_interactive_executor", lambda: executor)
        first = server._submit_interactive(running)
        try:
            assert started.wait(2)
            queued = server._submit_interactive(lambda: pytest.fail("cancelled work executed"))
            assert not admission.try_acquire()
            assert queued.cancel()
            assert admission.try_acquire(), "cancelled work leaked its admission slot"
            admission.release()
        finally:
            release.set()
            first.result(timeout=2)


def test_running_interactive_work_keeps_capacity_until_completion(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    admission = BoundedAdmission(1)
    release, started = Event(), Event()

    def running():
        started.set()
        assert release.wait(5)
        raise ValueError("original failure")

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(server, "_interactive_admission", admission)
        monkeypatch.setattr(server, "_get_interactive_executor", lambda: executor)
        future = server._submit_interactive(running)
        try:
            assert started.wait(2)
            assert not future.cancel()
            assert not admission.try_acquire()
        finally:
            release.set()
        with pytest.raises(ValueError, match="original failure"):
            future.result(timeout=2)
    assert admission.try_acquire()
    assert not admission.try_acquire()
    admission.release()


@pytest.mark.parametrize("submission_failure", [False, True])
def test_step_not_executed_releases_task_and_progress(tmp_path, monkeypatch, submission_failure):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from starlette.requests import Request

    manager = TaskManager(tmp_path)
    task_id = manager.create_task("cancelled")["task_id"]
    detail = get_task_type_detail("martini3-bilayer")
    manager.update_state(task_id, {"task_type": detail, "task_type_id": "martini3-bilayer"})
    admission = TaskAdmission(1)
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_step_admission", admission)
    monkeypatch.setattr(server, "_resources", None)
    monkeypatch.setattr(server, "_step_runners", {})
    future = Future()
    future.cancel()

    def submit(*args):
        if submission_failure:
            raise RuntimeError("executor unavailable")
        return future

    monkeypatch.setattr(server, "_get_step_executor", lambda: SimpleNamespace(submit=submit))

    async def receive():
        return {"type": "http.request", "body": b'{"config":{"include_protein":false}}'}

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    with pytest.raises(RuntimeError if submission_failure else asyncio.CancelledError):
        asyncio.run(server.api_run_step(task_id, "input", request))
    assert not admission.active(task_id)
    assert task_id not in server._step_progress
    assert admission.try_acquire(task_id) == "accepted"
    admission.release(task_id)


def test_same_task_check_is_rejected_before_duplicate_executor_submission(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("duplicate.pdb")
    task_id = task["task_id"]
    manager.save_uploaded_pdb(task_id, "duplicate.pdb", PDB_TWO_ATOMS.encode())
    detail = get_task_type_detail("solvator")
    manager.update_state(
        task_id,
        {"task_type": detail, "task_type_id": "solvator", "seed": 42},
    )
    admission = TaskAdmission(2)
    assert admission.try_acquire(task_id) == "accepted"
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_step_admission", admission)
    monkeypatch.setattr(server, "_building_tasks", set())
    monkeypatch.setattr(server, "_build_queue", [])
    server._step_runners.pop(task_id, None)

    with TestClient(app) as client:
        response = client.post(f"/api/step/{task_id}/input", json={"config": {}})

    admission.release(task_id)
    assert response.status_code == 409
    assert "already running" in response.json()["error"]


def test_upload_enforces_atom_complexity_limit_before_full_processing(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setenv("GMXBUILDER_MAX_STRUCTURE_ATOMS", "1")

    with TestClient(app) as client:
        response = client.post(
            "/api/upload-pdb",
            files={"file": ("two.pdb", PDB_TWO_ATOMS, "chemical/x-pdb")},
            data={"task_type": "solvator"},
        )

    assert response.status_code == 400
    assert "more than 1 atom" in response.json()["error"]


def test_resume_reuses_cached_structure_summary(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    original = PDB_TWO_ATOMS.replace("ALA A", "ALA R")

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/upload-pdb",
            files={"file": ("cached.pdb", original, "chemical/x-pdb")},
            data={"task_type": "solvator"},
        )
        assert uploaded.status_code == 200, uploaded.text
        task_id = uploaded.json()["task_id"]
        monkeypatch.setattr(
            server,
            "summarize_resume_structure",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cached upload must not be re-parsed")
            ),
        )
        resumed = client.get(f"/api/task/{task_id}/resume")

    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["pdb_info_full"]["num_atoms"] == 2
    assert resumed.json()["chain_mapping"] == {"R": "A"}
    atom_lines = [
        line for line in resumed.json()["pdb_content"].splitlines() if line.startswith("ATOM")
    ]
    assert atom_lines and all(line[21] == "A" for line in atom_lines)
    assert manager.get_filter_source(task_id).read_text() == original

"""Queue persistence, estimates, and first-incomplete-step resume."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from starlette.requests import Request

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.web import server
from gmxbuilder.web.task_manager import TaskManager
from gmxbuilder.web.task_types import get_task_type_detail


def _empty_system() -> System:
    return System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3),
        )
    )


def _json_request(payload: dict) -> Request:
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/build",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


class _NoSlotSemaphore:
    def acquire(self, blocking=False):
        assert blocking is False
        return False


def test_build_request_is_private_and_restart_loadable(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("restart")
    request = {"task_id": task["task_id"], "modules": {"simparams": {}}}

    path = manager.save_build_request(task["task_id"], request)

    assert manager.load_build_request(task["task_id"]) == request
    assert path.stat().st_mode & 0o077 == 0


def test_queue_status_has_task_id_and_start_estimate(monkeypatch):
    task_id = "a" * 32
    monkeypatch.setattr(server, "_MAX_CONCURRENT_BUILDS", 1)
    monkeypatch.setattr(server, "_build_queue", [(task_id, {})])
    monkeypatch.setattr(server, "_queue_enqueued_at", {task_id: time.time()})
    monkeypatch.setattr(server, "_building_tasks", {"b" * 32})
    monkeypatch.setattr(server, "_build_started_at", {"b" * 32: time.time() - 5.0})
    server._build_duration_history.clear()
    server._build_duration_history.append(20.0)

    status = asyncio.run(server.api_build_queue_status(task_id))

    assert status["status"] == "queued"
    assert status["task_id"] == task_id
    assert status["queue_position"] == 1
    assert status["estimated_wait_seconds"] >= 1
    assert status["estimated_start_at"].endswith("+00:00")


def test_queue_estimate_accounts_for_positions_beyond_currently_free_slots(monkeypatch):
    now = time.time()
    monkeypatch.setattr(server, "_MAX_CONCURRENT_BUILDS", 4)
    monkeypatch.setattr(server, "_building_tasks", {"active"})
    monkeypatch.setattr(server, "_build_started_at", {"active": now})
    server._build_duration_history.clear()
    server._build_duration_history.append(20.0)

    assert server._queue_estimate(3)["estimated_wait_seconds"] == 0
    assert server._queue_estimate(4)["estimated_wait_seconds"] >= 19


def test_resume_selects_first_incomplete_visible_step(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("pure")
    task_id = task["task_id"]
    detail = get_task_type_detail("pure-membrane")
    manager.update_state(
        task_id,
        {
            "task_type": detail,
            "task_type_id": "pure-membrane",
            "current_step": "forcefield",
        },
    )
    _empty_system().save_checkpoint(manager.get_task_dir(task_id) / "steps" / "forcefield")
    monkeypatch.setattr(server, "task_manager", manager)
    server._step_runners.pop(task_id, None)

    resumed = asyncio.run(server.api_task_resume(task_id))

    assert resumed["resume_step"] == "membrane"
    assert resumed["resume_step_number"] == 2
    assert resumed["resume_url"] == "/PureBilayerSystem/Step2"
    assert task_id not in resumed["resume_url"]


def test_queue_admission_persists_request_and_returns_restorable_task_id(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("pure")
    task_id = task["task_id"]
    detail = get_task_type_detail("pure-membrane")
    manager.update_state(
        task_id,
        {
            "task_type": detail,
            "task_type_id": "pure-membrane",
        },
    )
    _empty_system().save_checkpoint(manager.get_task_dir(task_id) / "steps" / "membrane")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_build_semaphore", _NoSlotSemaphore())
    monkeypatch.setattr(server, "_build_queue", [])
    monkeypatch.setattr(server, "_queue_enqueued_at", {})
    monkeypatch.setattr(server, "_tasks", {})
    monkeypatch.setattr(server, "_building_tasks", set())
    monkeypatch.setattr(server, "_MAX_QUEUED_BUILDS", 2)
    server._step_runners.pop(task_id, None)
    payload = {
        "task_id": task_id,
        "task_type": "pure-membrane",
        "system_name": "queued",
        "modules": {
            "solvation": {"enabled": False},
            "simparams": {},
            "export": {"write_mdp": False},
        },
    }

    response = asyncio.run(server.api_build(_json_request(payload)))
    result = json.loads(response.body)

    assert response.status_code == 200
    assert result["status"] == "queued"
    assert result["task_id"] == task_id
    assert result["queue_position"] == 1
    assert result["estimated_start_at"].endswith("+00:00")
    saved_request = manager.load_build_request(task_id)
    assert saved_request["task_id"] == task_id
    assert saved_request["task_type"] == "pure-membrane"
    assert saved_request["system_name"] == "queued"
    assert saved_request["source_step"] == "membrane"
    assert saved_request["modules"]["simparams"]["schema_version"] == 2
    assert set(saved_request["modules"]["simparams"]) == {
        "schema_version",
        "minimization",
        "eq_stages",
        "prod_iters",
    }
    assert saved_request["modules"]["execution"]["cpu_threads"] >= 1
    assert manager.get_state(task_id)["build_status"]["status"] == "queued"


def test_full_queue_does_not_persist_an_unaccepted_build(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("pure")
    task_id = task["task_id"]
    detail = get_task_type_detail("pure-membrane")
    manager.update_state(
        task_id,
        {
            "task_type": detail,
            "task_type_id": "pure-membrane",
        },
    )
    _empty_system().save_checkpoint(manager.get_task_dir(task_id) / "steps" / "membrane")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_build_semaphore", _NoSlotSemaphore())
    monkeypatch.setattr(server, "_build_queue", [])
    monkeypatch.setattr(server, "_tasks", {})
    monkeypatch.setattr(server, "_building_tasks", set())
    monkeypatch.setattr(server, "_MAX_QUEUED_BUILDS", 0)
    server._step_runners.pop(task_id, None)
    payload = {
        "task_id": task_id,
        "task_type": "pure-membrane",
        "modules": {
            "solvation": {"enabled": False},
            "simparams": {},
            "export": {"write_mdp": False},
        },
    }

    response = asyncio.run(server.api_build(_json_request(payload)))

    assert response.status_code == 503
    assert manager.load_build_request(task_id) is None

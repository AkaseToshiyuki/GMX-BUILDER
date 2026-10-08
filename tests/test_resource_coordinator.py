"""Resource admission against independent fake process and storage state."""

import asyncio
import time
from datetime import datetime, timedelta, timezone

from gmxbuilder.web.resource_coordinator import ResourceCoordinator, expensive_request
from gmxbuilder.web.resource_policy import GIB
from gmxbuilder.web.task_manager import TaskManager


class Workers:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.grants = []
        self.states = {}

    async def launch(self, operation, threads):
        self.started.append((operation["id"], threads))
        self.states[operation["unit"]] = "active"
        return operation["unit"]

    async def state(self, unit):
        return self.states.get(unit, "inactive")

    async def stop(self, unit):
        self.stopped.append(unit)
        self.states[unit] = "inactive"

    async def memory_current(self, unit):
        return 0

    async def rebalance(self, operations):
        self.grants.append(len(operations))


def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("GMXBUILDER_CPU_CORES", "8")
    monkeypatch.setenv("GMXBUILDER_MAX_BUILDS", "4")
    monkeypatch.setenv("GMXBUILDER_TASK_MEMORY_GIB", "16")
    workers = Workers()
    manager = TaskManager(tmp_path / "tasks")
    return ResourceCoordinator(tmp_path, manager, workers=workers, require_mount=False)


def ticket(co, *, age=0, expires=None):
    state = co.manager.create_task()
    state["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=age)).isoformat()
    co.manager._write_state(co.manager.get_task_dir(state["task_id"]), state)
    operation = co.queue.enqueue(state["task_id"], "input", "step", expires or time.time() + 86400)
    co.job_directory(operation).mkdir(parents=True)
    return operation


def test_oldest_nonrunning_is_reclaimed_first(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    running = ticket(co, age=3)
    oldest_waiting = ticket(co, age=2)
    younger = ticket(co, age=1)
    co.queue.update(running["id"], status="running")
    monkeypatch.setattr(
        co,
        "free_bytes",
        lambda: 4 * GIB if not co.manager.get_task_dir(oldest_waiting["task_id"]).exists() else 0,
    )
    assert co.make_room(0)
    assert co.manager.get_task_dir(running["task_id"]).exists()
    assert co.manager.get_task_dir(younger["task_id"]).exists()
    assert co.queue.get(oldest_waiting["id"])["status"] == "cancelled"


def test_all_tasks_leased_pauses_without_deletion(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    operation = ticket(co, age=3)
    monkeypatch.setattr(co, "free_bytes", lambda: 0)
    with co.manager.active_task(operation["task_id"]):
        assert not co.make_room(1)
        assert co.manager.get_task_dir(operation["task_id"]).exists()


def test_memory_reservations_limit_starts_not_queue_length(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    for _ in range(8):
        ticket(co)
    monkeypatch.setattr(co, "free_bytes", lambda: 100 * GIB)
    monkeypatch.setattr("gmxbuilder.web.resource_coordinator.available_memory", lambda: 20 * GIB)
    asyncio.run(co.tick())
    assert len(co.workers.started) == 1
    assert co.workers.started[0][1] == 8
    assert co.queue.counts() == {"queued": 7, "running": 1}
    assert co.pause_reason == "Waiting for sufficient memory."
    for operation_id in list(co._leases):
        co.release(operation_id)


def test_no_waiting_work_does_not_poll_worker_memory(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    operation = ticket(co)
    unit = "gmxbuilder-op-" + operation["id"] + ".service"
    co.queue.update(operation["id"], status="running", unit=unit)
    co.workers.states[unit] = "active"

    async def unexpected_read(unit):
        raise AssertionError("No memory admission is needed without queued work")

    monkeypatch.setattr(co.workers, "memory_current", unexpected_read)
    asyncio.run(co.tick())
    assert co.queue.get(operation["id"])["status"] == "running"


def test_expiry_stops_process_before_removing_directory(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    operation = ticket(co, expires=time.time() - 1)
    unit = "gmxbuilder-op-" + operation["id"] + ".service"
    co.queue.update(operation["id"], status="running", unit=unit)
    co.workers.states[unit] = "active"
    deleted = []
    original = co.manager.delete_task

    def delete(task_id):
        assert unit in co.workers.stopped
        deleted.append(task_id)
        return original(task_id)

    monkeypatch.setattr(co.manager, "delete_task", delete)
    asyncio.run(co.tick())
    assert deleted == [operation["task_id"]]
    assert not co.manager.get_task_dir(operation["task_id"]).exists()


def test_expired_queue_is_never_started(tmp_path, monkeypatch):
    co = coordinator(tmp_path, monkeypatch)
    operation = ticket(co, expires=time.time() - 1)
    asyncio.run(co.tick())
    assert co.workers.started == []
    assert co.queue.get(operation["id"])["status"] == "cancelled"


def test_expensive_entry_points_share_admission():
    for path in (
        "/api/upload-pdb",
        "/api/protonate",
        "/api/step/abc/input",
        "/api/forcefield-compatibility/abc",
        "/api/ligand-chemistry/abc",
        "/api/build",
        "/api/cgenff-upload/abc",
        "/api/preview-pdb",
    ):
        assert expensive_request("POST", path)
    assert expensive_request("GET", "/api/step/abc/input/viewer.pdb")
    assert not expensive_request("POST", "/api/custom-lipid")
    assert not expensive_request("GET", "/api/operations/abc")


def test_unchanged_grants_do_not_repeat_systemd_writes(tmp_path, monkeypatch):
    from gmxbuilder.web.resource_policy import ResourcePolicy
    from gmxbuilder.web.resource_workers import SystemdWorkers

    monkeypatch.setenv("GMXBUILDER_CPU_CORES", "8")
    workers = SystemdWorkers(tmp_path, ResourcePolicy.from_environment())
    calls = []

    async def command(*args):
        calls.append(args)
        return 0, "active", ""

    monkeypatch.setattr(workers, "command", command)
    first = {"unit": "gmxbuilder-op-" + "a" * 32 + ".service", "threads": 8}
    second = {"unit": "gmxbuilder-op-" + "b" * 32 + ".service", "threads": 4}
    asyncio.run(workers.rebalance([first]))
    asyncio.run(workers.rebalance([first]))
    assert len(calls) == 1
    asyncio.run(workers.rebalance([first, second]))
    assert len(calls) == 3
    assert calls[1][-2:] == (first["unit"], "CPUQuota=400%")


def test_malformed_managed_forms_release_admission_without_tasks(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    co = coordinator(tmp_path, monkeypatch)
    app = FastAPI()

    @app.post("/api/upload-pdb")
    async def upload(request: Request):
        return await co.enqueue_request(request)

    ingress_before = co._ingress._value
    spools = []
    from starlette import formparsers

    original = formparsers.SpooledTemporaryFile

    def tracked_spool(*args, **kwargs):
        spool = original(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", tracked_spool)
    bodies = [
        b'--b\\r\\nContent-Disposition: form-data; name="file"\\r\\n',
        b'--b\nContent-Disposition: form-data; name="file"\n',
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.pdb"\r\n'
        b"\r\nATOM incomplete",
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.pdb"\r\n'
        b"\r\nATOM\r\n--b\r\nInvalid Header@\r\n",
    ]
    with TestClient(app, raise_server_exceptions=False) as client:
        for body in bodies:
            result = client.post(
                "/api/upload-pdb",
                content=body,
                headers={"content-type": "multipart/form-data; boundary=b"},
            )
            assert result.status_code == 400, result.text
            assert "Malformed multipart" in result.json()["error"]
            assert co._ingress._value == ingress_before
            assert co._upload_clients == {}
            assert co.queue.counts() == {}
            assert not list(co.manager.root.glob("*/state.json"))
    assert spools and all(spool.closed for spool in spools)

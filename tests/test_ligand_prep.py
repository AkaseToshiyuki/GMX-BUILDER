"""Parameterizing a task's ligands before the user asks for them.

The Force Field step on a real task took 214 s, of which 99.98% was one
`acpype` call. Nothing about that calculation depends on the Force Field panel,
so it starts when the input Check saves its checkpoint and overlaps the time
the user spends choosing force fields.

Because it is speculative, what matters most is what it must never do: take a
build slot, surface its own failures, parameterize at a charge the user has not
seen, or outlive the task.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.web import ligand_prep, server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager


@pytest.fixture(autouse=True)
def clean_registry():
    """Module state is process-wide; no test may inherit another's."""
    yield
    with ligand_prep._guard:
        ligand_prep._status.clear()
        ligand_prep._futures.clear()
        ligand_prep._cancelled.clear()


@pytest.fixture
def pool():
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# The status snapshot


def test_an_unknown_task_is_idle_rather_than_missing():
    assert ligand_prep.status("never-started") == {"state": "idle", "molecules": {}}


def test_the_status_is_a_snapshot_and_not_a_live_reference():
    """The worker mutates from another thread while a caller reads.

    Returning the live mapping let a caller observe a state that never
    existed, and let the JSON encoder walk a dict while it changed size.
    """
    ligand_prep._set_molecule("task", "UK4", "computing")
    first = ligand_prep.status("task")
    ligand_prep._set_molecule("task", "UK4", "ready")
    second = ligand_prep.status("task")

    assert first["molecules"] == {"UK4": "computing"}
    assert second["molecules"] == {"UK4": "ready"}
    assert first["molecules"] is not second["molecules"]


def test_a_snapshot_survives_concurrent_mutation():
    """Reading while the worker writes must not raise."""
    stop = threading.Event()

    def churn():
        index = 0
        while not stop.is_set():
            ligand_prep._set_molecule("task", f"L{index % 12}", "computing")
            index += 1

    writer = threading.Thread(target=churn, daemon=True)
    writer.start()
    try:
        for _ in range(500):
            dict(ligand_prep.status("task")["molecules"])  # must not raise
    finally:
        stop.set()
        writer.join(timeout=5)


# --------------------------------------------------------------------------
# Running, and not running


# Terminal states that mean "nothing to do", as opposed to "something broke".
NO_WORK_STATES = {"idle", "unavailable", "done"}


def test_a_task_with_no_input_checkpoint_finishes_without_work(pool, tmp_path):
    """Which no-work state depends on whether GAFF2 is installed here.

    Asserting `idle` passed locally and failed in CI, where the GAFF2
    environment is absent and the answer is `unavailable`. Both branches are
    covered explicitly below; what this asserts is the property they share.
    """
    assert ligand_prep.start("task", tmp_path, pool) is True
    _wait_for(lambda: ligand_prep.status("task")["state"] != "running")
    assert ligand_prep.status("task")["state"] in NO_WORK_STATES


def test_an_absent_gaff_environment_is_reported_rather_than_failing(pool, tmp_path, monkeypatch):
    monkeypatch.setattr("gmxbuilder.modules.forcefield.gaff_backend.gaff_available", lambda: False)
    ligand_prep.start("task", tmp_path, pool)
    _wait_for(lambda: ligand_prep.status("task")["state"] != "running")
    assert ligand_prep.status("task")["state"] == "unavailable"


def test_a_present_gaff_environment_with_no_checkpoint_is_idle(pool, tmp_path, monkeypatch):
    monkeypatch.setattr("gmxbuilder.modules.forcefield.gaff_backend.gaff_available", lambda: True)
    ligand_prep.start("task", tmp_path, pool)
    _wait_for(lambda: ligand_prep.status("task")["state"] != "running")
    assert ligand_prep.status("task")["state"] == "idle"


def test_a_second_start_while_one_runs_is_refused(pool, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow(task_id, task_dir):
        started.set()
        release.wait(timeout=10)
        ligand_prep._set(task_id, state="done")

    monkeypatch.setattr(ligand_prep, "_prewarm", slow)
    assert ligand_prep.start("task", tmp_path, pool) is True
    assert started.wait(timeout=5)
    try:
        assert ligand_prep.start("task", tmp_path, pool) is False
    finally:
        release.set()


def test_a_failure_is_contained_and_reported_as_state(pool, tmp_path, monkeypatch):
    """Speculative work must never surface an error the user did not cause."""

    def explode(task_id, task_dir):
        raise RuntimeError("obabel is not installed")

    # Exercise the real wrapper's own guard rather than replacing it.
    monkeypatch.setattr(
        "gmxbuilder.core.system.System.load_checkpoint",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    (tmp_path / "steps" / "input").mkdir(parents=True)
    (tmp_path / "steps" / "input" / "system.npz").write_bytes(b"not a checkpoint")

    assert ligand_prep.start("task", tmp_path, pool) is True
    _wait_for(lambda: ligand_prep.status("task")["state"] in ("error", "done", "unavailable"))
    assert ligand_prep.status("task")["state"] in ("error", "unavailable")


def test_cancelling_forgets_the_task(pool, tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_prep, "_prewarm", lambda task_id, task_dir: None)
    ligand_prep.start("task", tmp_path, pool)
    ligand_prep.cancel("task")
    assert ligand_prep.status("task") == {"state": "idle", "molecules": {}}


def test_a_cancelled_task_stops_before_the_next_molecule():
    """Cancellation is checked between molecules, not inside a subprocess."""
    import inspect

    source = inspect.getsource(ligand_prep._prewarm)
    loop = source[source.index("for name, instances in groups.items():") :]
    assert "_cancelled" in loop.split("estimate_gaff_net_charge")[0]


# --------------------------------------------------------------------------
# What it must never do


def test_the_default_ph_matches_what_the_panel_prefills():
    """A different pH computes a different cache entry, so this must agree."""
    import inspect

    from gmxbuilder.web import server as web_server

    assert ligand_prep.PREWARM_PH == 7.0
    suggestions = inspect.getsource(web_server.api_ligand_charge_suggestions)
    assert 'data.get("pH", 7.0)' in suggestions


def test_an_ambiguous_charge_is_left_for_the_user():
    """Parameterizing at an unseen charge is a guess, not a head start."""
    import inspect

    source = inspect.getsource(ligand_prep._prewarm)
    assert '"ambiguous"' in source
    branch = source[source.index("if len(charges) != 1:") :]
    assert "continue" in branch[: branch.index("net_charge = charges.pop()")]


def test_it_never_runs_on_the_build_queue():
    """Speculative work must not delay a build the user actually asked for."""
    import inspect

    from gmxbuilder.web import server as web_server

    source = inspect.getsource(web_server.api_run_step)
    call = source[source.index("ligand_prep.start(") :][:200]
    assert "_get_custom_lipid_executor()" in call
    assert "_get_step_executor" not in call and "_get_executor" not in call


def test_it_starts_when_the_input_checkpoint_exists():
    import inspect

    from gmxbuilder.web import server as web_server

    source = inspect.getsource(web_server.api_run_step)
    input_branch = source[source.index('if step_name == "input":') :]
    input_branch = input_branch[: input_branch.index('if step_name == "structure":')]
    assert "ligand_prep.start(" in input_branch


def test_an_expired_task_has_its_preparation_cancelled():
    import inspect

    from gmxbuilder.web import server as web_server

    source = inspect.getsource(web_server.startup_background_tasks)
    assert "ligand_prep.cancel(tid)" in source


# --------------------------------------------------------------------------
# The endpoint


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_RATE_LIMIT_DB", str(tmp_path / "rate.sqlite3"))
    monkeypatch.setattr(server, "task_manager", TaskManager(tmp_path / "tasks"))
    with TestClient(app) as started:
        yield started


def test_status_for_an_unknown_task_is_a_404(client):
    assert client.get("/api/ligand-prep/" + "a" * 32).status_code == 404


def test_status_for_a_task_with_no_preparation_is_idle(client):
    task = client.post("/api/tasks", json={"task_type": "pure-membrane"}).json()
    body = client.get(f"/api/ligand-prep/{task['task_id']}").json()
    assert body == {"state": "idle", "molecules": {}}


def test_status_reports_what_the_worker_recorded(client):
    task = client.post("/api/tasks", json={"task_type": "pure-membrane"}).json()
    ligand_prep._set_molecule(task["task_id"], "UK4", "computing")
    body = client.get(f"/api/ligand-prep/{task['task_id']}").json()
    assert body["molecules"] == {"UK4": "computing"}
    assert body["state"] == "running"


def _wait_for(predicate, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was never reached")

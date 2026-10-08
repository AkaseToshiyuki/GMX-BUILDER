"""How many cores a task gets, and when that is decided.

Per-task threads used to be fixed at startup: cores divided by a concurrency
ceiling, whether or not anything else was running. One build on an idle
48-core machine therefore used four cores and left forty-four idle.

The budget is now decided when a task starts, from how many are running then,
and never revisited -- GROMACS fixes its thread count when mdrun launches, so
there is nothing to hand back to a task that arrives later.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gmxbuilder.runtime.hardware import (
    current_task_threads,
    task_thread_allocation,
    task_thread_scope,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The rule


@pytest.mark.parametrize(
    ("running", "expected"),
    [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (11, 4)],
)
def test_tasks_share_the_allocation_equally(running, expected):
    assert task_thread_allocation(running, cpu_cores=48, max_parallel=12) == expected


def test_one_task_on_an_idle_machine_gets_everything():
    """The case the old fixed division served worst."""
    assert task_thread_allocation(1, cpu_cores=48, max_parallel=12) == 48


@pytest.mark.parametrize("running", [12, 13, 24, 100])
def test_at_the_concurrency_ceiling_each_task_is_capped(running):
    """Splitting past this buys less than the cost of the split."""
    assert task_thread_allocation(running, cpu_cores=48, max_parallel=12) == 2


def test_every_task_keeps_at_least_one_core():
    for cores in (1, 2, 3):
        for running in (1, 5, 50):
            assert task_thread_allocation(running, cpu_cores=cores, max_parallel=8) >= 1


def test_a_single_core_machine_never_promises_two():
    assert task_thread_allocation(50, cpu_cores=1, max_parallel=8) == 1


@pytest.mark.parametrize("running", [0, -1])
def test_a_nonsensical_running_count_is_treated_as_one(running):
    assert task_thread_allocation(running, cpu_cores=48, max_parallel=12) == 48


def test_the_allocation_never_exceeds_the_cores_available():
    for running in range(1, 30):
        assert task_thread_allocation(running, cpu_cores=16, max_parallel=8) <= 16


# --------------------------------------------------------------------------
# The scope


def test_outside_a_task_the_deployment_maximum_applies(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "4")
    assert current_task_threads() == 4


def test_inside_a_scope_the_task_budget_applies(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "4")
    with task_thread_scope(24):
        assert current_task_threads() == 24
    assert current_task_threads() == 4


def test_a_scope_restores_the_previous_budget(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "4")
    with task_thread_scope(24), task_thread_scope(6):
        assert current_task_threads() == 6
    assert current_task_threads() == 4


def test_a_budget_below_one_is_refused_upward():
    with task_thread_scope(0):
        assert current_task_threads() >= 1


def test_the_budget_reaches_work_the_task_farms_out(monkeypatch):
    """Parallel ligand parameterisation copies the context; the budget rides along."""
    import contextvars

    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "4")
    seen: list[int] = []
    lock = threading.Lock()

    def worker():
        with lock:
            seen.append(current_task_threads())

    with task_thread_scope(24), ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(4):
            pool.submit(contextvars.copy_context().run, worker)
    assert seen == [24, 24, 24, 24]


def test_lipid_equilibration_is_bounded_by_the_task_budget(monkeypatch):
    from gmxbuilder.runtime.hardware import lipid_worker_threads

    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "48")
    monkeypatch.delenv("GMXBUILDER_LIPID_THREADS", raising=False)
    with task_thread_scope(2):
        assert lipid_worker_threads(concurrency=1) <= 2


# --------------------------------------------------------------------------
# The server decides it once, at the start


def test_the_budget_is_decided_when_a_build_acquires_its_slot():
    import inspect

    from gmxbuilder.web import server

    for source in (
        inspect.getsource(server._consume_queue),
        inspect.getsource(server.api_build),
    ):
        if "_building_tasks.add(task_id)" in source:
            block = source[source.index("_building_tasks.add(task_id)") :][:400]
            assert "task_thread_allocation(len(_building_tasks))" in block


def test_the_build_runs_inside_its_own_budget():
    import inspect

    from gmxbuilder.web import server

    source = inspect.getsource(server._run_build_sync)
    assert "task_thread_scope(allocated)" in source


def test_a_finished_build_records_the_cores_it_had():
    """A duration means nothing without the allocation it was measured at."""
    import inspect

    from gmxbuilder.web import server

    source = inspect.getsource(server._run_background_build)
    assert "_build_duration_history.append((duration, int(threads_used)))" in source


# --------------------------------------------------------------------------
# The queue estimate follows


def test_the_estimate_prefers_runs_at_the_same_allocation():
    from gmxbuilder.web import server

    server._build_duration_history.clear()
    try:
        for _ in range(4):
            server._build_duration_history.append((100.0, 24))
        for _ in range(4):
            server._build_duration_history.append((800.0, 2))
        assert server._typical_build_seconds(24) == 100.0
        assert server._typical_build_seconds(2) == 800.0
    finally:
        server._build_duration_history.clear()


def test_too_few_matching_runs_falls_back_to_all_of_them():
    """A weaker estimate, but still an observed one rather than a model."""
    from gmxbuilder.web import server

    server._build_duration_history.clear()
    try:
        server._build_duration_history.append((100.0, 24))
        for _ in range(5):
            server._build_duration_history.append((500.0, 2))
        assert server._typical_build_seconds(24) == 500.0
    finally:
        server._build_duration_history.clear()


def test_a_bare_duration_does_not_break_the_health_endpoint():
    """/api/health reads this; it must not fail over a sample's shape."""
    from gmxbuilder.web import server

    server._build_duration_history.clear()
    try:
        server._build_duration_history.append(30.0)  # type: ignore[arg-type]
        server._build_duration_history.append((50.0, 4))
        assert server._typical_build_seconds(4) == 40.0
    finally:
        server._build_duration_history.clear()


def test_no_history_at_all_uses_the_declared_baseline():
    from gmxbuilder.web import server

    server._build_duration_history.clear()
    assert server._typical_build_seconds(24) == server._baseline_build_seconds()


def test_a_queued_build_is_estimated_at_the_share_it_will_get():
    """Estimating it at an idle machine's speed under-predicts every wait."""
    import inspect

    from gmxbuilder.web import server

    source = inspect.getsource(server._queue_estimate)
    assert "task_thread_allocation(max(1, len(active_ids)))" in source
    assert "_typical_build_seconds(queued_threads)" in source


# --------------------------------------------------------------------------
# The installer default


def test_the_installer_defaults_to_half_the_cores():
    installer = (ROOT / "install-local.sh").read_text(encoding="utf-8")
    block = installer[installer.index("choose_default_slots() {") :][:600]
    assert "cores / 2" in block
    assert "cores / 4" not in block


def test_the_installer_still_refuses_more_slots_than_cores():
    installer = (ROOT / "install-local.sh").read_text(encoding="utf-8")
    assert "Concurrent task slots cannot exceed allocated CPU cores" in installer

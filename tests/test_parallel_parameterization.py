"""Independent molecules are parameterized at the same time.

Each molecule is a separate `sqm` run: single-threaded, minutes long, sharing
nothing with the others -- distinct cache keys, distinct directories, and
`_lock_for` is per key. A system with four ligands was taking four times as
long as it needed to.

The subtle part is not the pool. A worker thread starts with an empty context,
so without copying one it sees neither the task's scoped GAFF cache nor its
custom lipid registry, and would write a task-private molecule into the shared
cache. That is asserted here directly.
"""

from __future__ import annotations

import threading
import time

import pytest

from gmxbuilder.modules.forcefield import gaff_backend as gb
from gmxbuilder.modules.forcefield.gaff_backend import MoleculeJob, parameterize_molecules


@pytest.fixture
def fake_molecules(monkeypatch):
    """Replace the real toolchain with something observable and fast."""
    calls: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def fake_prepare(name, structure, indices, net_charge, *, charge_method=None, target_pH=7.0):
        started = time.monotonic()
        time.sleep(0.30)
        with lock:
            calls.append((name, threading.current_thread().name, started))
        if name == "FAIL":
            raise RuntimeError(f"parameterization failed for {name}")
        return f"template-{name}"

    monkeypatch.setattr(gb, "prepare_gaff_molecule", fake_prepare)
    return calls


def _jobs(*names):
    return [MoleculeJob(name, object(), [0, 1, 2], 0) for name in names]


def test_molecules_are_parameterized_concurrently(fake_molecules, monkeypatch):
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    jobs = _jobs("AAA", "BBB", "CCC", "DDD")

    started = time.monotonic()
    result = parameterize_molecules(jobs)
    elapsed = time.monotonic() - started

    assert set(result) == {"AAA", "BBB", "CCC", "DDD"}
    # Four 0.30 s molecules: serial is 1.2 s, concurrent is about one of them.
    assert elapsed < 0.75, f"took {elapsed:.2f}s, which is not concurrent"
    assert len({thread for _name, thread, _t in fake_molecules}) > 1


def test_concurrency_never_exceeds_the_task_thread_budget(fake_molecules, monkeypatch):
    """One `sqm` is one thread, so the budget is also the useful pool size."""
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 2)
    parameterize_molecules(_jobs("AAA", "BBB", "CCC", "DDD"))
    assert len({thread for _name, thread, _t in fake_molecules}) <= 2


def test_one_molecule_takes_no_pool_at_all(fake_molecules):
    """The common case must stay exactly what it was."""
    caller = threading.current_thread().name
    result = parameterize_molecules(_jobs("AAA"))
    assert result == {"AAA": "template-AAA"}
    assert [thread for _name, thread, _t in fake_molecules] == [caller]


def test_no_molecules_is_not_an_error(fake_molecules):
    assert parameterize_molecules([]) == {}


def test_results_come_back_in_job_order(fake_molecules, monkeypatch):
    """Downstream iteration order is the caller's, not whoever finished first."""
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    names = ["DDD", "AAA", "CCC", "BBB"]
    assert list(parameterize_molecules(_jobs(*names))) == names


def test_progress_is_reported_on_the_calling_thread(fake_molecules, monkeypatch):
    """So a caller may report progress without being thread-safe."""
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    threads: list[str] = []
    seen: list[tuple[int, int]] = []

    def on_progress(completed, total, _name):
        threads.append(threading.current_thread().name)
        seen.append((completed, total))

    parameterize_molecules(_jobs("AAA", "BBB", "CCC"), on_progress=on_progress)
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert set(threads) == {threading.current_thread().name}


def test_a_failure_is_raised_for_the_first_molecule_in_job_order(fake_molecules, monkeypatch):
    """Which molecule is blamed must not depend on scheduling."""
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    with pytest.raises(RuntimeError, match="FAIL"):
        parameterize_molecules(_jobs("AAA", "FAIL", "CCC"))


def test_a_worker_sees_the_task_scoped_cache_root(monkeypatch, tmp_path):
    """The bug this guards: a task-private molecule in the shared cache.

    A pool worker starts with an empty context. Submitting the call directly
    resolves `_cache_root` to the global cache; only a copied context carries
    the task scope across.
    """
    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    roots: dict[str, str] = {}

    def record_root(name, structure, indices, net_charge, *, charge_method=None, target_pH=7.0):
        roots[name] = str(gb._cache_root(name))
        return f"template-{name}"

    monkeypatch.setattr(gb, "prepare_gaff_molecule", record_root)

    scoped = tmp_path / "task-cache"
    with gb.task_gaff_cache(scoped, {"AAA", "BBB"}):
        parameterize_molecules(_jobs("AAA", "BBB"))

    assert roots["AAA"] == str(scoped.resolve())
    assert roots["BBB"] == str(scoped.resolve())


def test_a_worker_sees_the_module_progress_scope(monkeypatch, tmp_path):
    """The same context copy carries progress reporting into the workers."""
    from gmxbuilder.pipeline.progress import module_progress_scope, report_progress

    monkeypatch.setattr("gmxbuilder.runtime.hardware.configured_task_threads", lambda: 4)
    seen: list[str] = []
    lock = threading.Lock()

    def reporting(name, structure, indices, net_charge, *, charge_method=None, target_pH=7.0):
        report_progress(0.5, f"inside {name}")
        return f"template-{name}"

    monkeypatch.setattr(gb, "prepare_gaff_molecule", reporting)

    def sink(fraction, phase):
        with lock:
            seen.append(phase)

    with module_progress_scope(sink):
        parameterize_molecules(_jobs("AAA", "BBB"))

    assert sorted(seen) == ["inside AAA", "inside BBB"]


def test_the_force_field_module_uses_the_parallel_path():
    import inspect

    from gmxbuilder.modules.forcefield.selector import ForceFieldSelector

    source = inspect.getsource(ForceFieldSelector._parameterize_gaff2_ligands)
    assert "parameterize_molecules(" in source
    assert "for name, instances in groups.items():\n            templates[name]" not in source

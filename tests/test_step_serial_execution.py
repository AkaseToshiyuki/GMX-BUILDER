"""Task steps remain serial even when the web executor submits overlapping work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.pipeline.step_executor import StepRunner


class _ObservedModule(BaseModule):
    name = "observed"
    description = "test-only concurrency observer"

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def validate_config(self, config: dict) -> bool:
        return True

    def run(self, system, config: dict) -> ModuleResult:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.05)
            return ModuleResult(success=True, system=system, log=[])
        finally:
            with self._lock:
                self.active -= 1


def test_overlapping_requests_for_one_task_execute_strictly_serial(
    tmp_path, monkeypatch
):
    runner = StepRunner(tmp_path / "task", "membrane-bilayer")
    observed = _ObservedModule()
    monkeypatch.setattr(
        "gmxbuilder.pipeline.step_executor._get_module",
        lambda _step, _pipeline: observed,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(runner.run_step, "input", {"seed": seed})
            for seed in (1, 2)
        ]
        results = [future.result() for future in futures]

    assert [result["status"] for result in results] == ["ok", "ok"]
    assert observed.maximum_active == 1

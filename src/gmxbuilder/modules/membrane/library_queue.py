"""Resource-bounded continuation queue for production lipid libraries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import Iterable

from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.runtime.hardware import (
    configure_runtime_resources,
    configured_cpu_ids,
    configured_gpu_devices,
    configured_task_threads,
)


DEFAULT_FORCE_FIELDS = ("amber14sb", "charmm36m", "charmm36")


def _positive_timeout(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number of seconds") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number of seconds")
    return value


def _run_with_watchdog(
    command: list[str],
    *,
    log,
    environment: dict[str, str],
    cpus: tuple[int, ...],
    log_path: Path,
) -> int:
    """Run one worker with total-runtime and output-activity deadlines."""
    total_timeout = _positive_timeout("GMXBUILDER_LIPID_JOB_TIMEOUT_S", 60 * 60 * 50)
    idle_timeout = _positive_timeout("GMXBUILDER_LIPID_JOB_IDLE_TIMEOUT_S", 60 * 60 * 4)

    def restrict_cpu_affinity() -> None:
        os.sched_setaffinity(0, set(cpus))

    started = time.monotonic()
    last_activity = started
    last_size = -1
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        cwd=Path(__file__).resolve().parents[4],
        preexec_fn=restrict_cpu_affinity,
        start_new_session=True,
    )
    try:
        while process.poll() is None:
            try:
                size = log_path.stat().st_size
            except OSError:
                size = last_size
            if size != last_size:
                last_size = size
                last_activity = time.monotonic()
            now = time.monotonic()
            if now - started > total_timeout:
                raise TimeoutError(
                    f"lipid worker exceeded {total_timeout:.0f}s total runtime"
                )
            if now - last_activity > idle_timeout:
                raise TimeoutError(
                    f"lipid worker produced no log output for {idle_timeout:.0f}s"
                )
            time.sleep(15.0)
        return int(process.returncode or 0)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def _run_job(
    job: dict,
    *,
    gpu: str | None,
    cpus: tuple[int, ...],
    npt_ps: float,
    log_dir: Path,
) -> tuple[dict, bool, str]:
    executable = shutil.which("gmxbuilder")
    if not executable:
        raise RuntimeError("gmxbuilder executable is not available on PATH")
    label = f"{job['force_field']}-{job['lipid_name']}"
    device_label = f"gpu{gpu}" if gpu is not None else "cpu"
    log_path = log_dir / f"{label}-{device_label}.log"
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "GMXBUILDER_LIPID_LIBRARY_GPU": "1" if gpu is not None else "0",
        "GMXBUILDER_GAFF_THREADS": str(len(cpus)),
        "GMXBUILDER_LIPID_THREADS": str(len(cpus)),
        "GMXBUILDER_CPU_CORES": str(len(cpus)),
        "PYTHONUNBUFFERED": "1",
    })
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

    command = [
        executable, "lipid-library", "build",
        "--force-field", job["force_field"],
        "--lipid", job["lipid_name"],
        "--npt-ps", str(float(npt_ps)),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        returncode = _run_with_watchdog(
            command,
            log=log,
            environment=environment,
            cpus=cpus,
            log_path=log_path,
        )
    return job, returncode == 0, str(log_path)


def run_library_queue(
    force_fields: Iterable[str] = DEFAULT_FORCE_FIELDS,
    *,
    npt_ps: float = 1000.0,
    log_dir: str | Path = "output/lipid-library",
) -> list[tuple[dict, bool, str]]:
    """Build missing entries within the configured CPU and GPU budget."""
    if npt_ps < 500.0:
        raise ValueError("Production lipid libraries require at least 500 ps NPT")
    destination = Path(log_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    configure_runtime_resources(apply_affinity=False)
    gpu_devices = configured_gpu_devices()
    cpu_ids = configured_cpu_ids()
    threads = min(configured_task_threads(), len(cpu_ids))
    cpu_slots = max(1, len(cpu_ids) // threads)
    concurrency = min(2, len(gpu_devices), cpu_slots) if gpu_devices else 1
    cpu_sets = tuple(
        tuple(cpu_ids[index * threads:(index + 1) * threads])
        for index in range(concurrency)
    )
    library = EquilibratedLipidLibrary()
    pending = []
    for force_field in force_fields:
        pending.extend(
            job for job in library.coverage([str(force_field).lower()])
            if not job["ready"]
        )

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for offset in range(0, len(pending), concurrency):
            batch = pending[offset:offset + concurrency]
            futures = [
                executor.submit(
                    _run_job,
                    job,
                    gpu=(gpu_devices[index] if gpu_devices else None),
                    cpus=cpu_sets[index],
                    npt_ps=npt_ps,
                    log_dir=destination,
                )
                for index, job in enumerate(batch)
            ]
            for job, future in zip(batch, futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_log = destination / (
                        f"{job['force_field']}-{job['lipid_name']}-queue-error.log"
                    )
                    error_log.write_text(
                        f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                    )
                    results.append((job, False, str(error_log)))
    return results

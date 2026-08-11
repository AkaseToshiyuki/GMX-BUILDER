"""Resource-bounded continuation queue for production lipid libraries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.runtime.hardware import (
    configure_runtime_resources,
    configured_cpu_ids,
    configured_gpu_devices,
    configured_task_threads,
)


DEFAULT_FORCE_FIELDS = ("amber14sb", "charmm36m", "charmm36")


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

    def restrict_cpu_affinity() -> None:
        os.sched_setaffinity(0, set(cpus))

    command = [
        executable, "lipid-library", "build",
        "--force-field", job["force_field"],
        "--lipid", job["lipid_name"],
        "--npt-ps", str(float(npt_ps)),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=Path(__file__).resolve().parents[4],
            preexec_fn=restrict_cpu_affinity,
            check=False,
        )
    return job, result.returncode == 0, str(log_path)


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
            results.extend(future.result() for future in futures)
    return results

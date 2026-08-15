"""Portable GROMACS, CPU and CUDA discovery.

The web process has one explicit resource budget.  ``gmxbuilder serve``
calculates it before importing the server, applies CPU affinity, and exposes
only the requested CUDA devices to all child processes.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_gpu_probe_cache: dict[tuple[str, tuple[str, ...], int], tuple[bool, str | None]] = {}
_COMMAND_TOKEN = re.compile(r"^[A-Za-z0-9_./+-]+$")


@dataclass(frozen=True)
class HardwareCapabilities:
    gmx_path: str | None
    gmx_version: str | None
    gmx_gpu_backend: str | None
    gmx_gpu_compiled: bool
    gmx_gpu_probe_passed: bool
    cuda_runtime_usable: bool
    detected_gpu_count: int
    detected_cpu_threads: int
    allowed_cpu_threads: int
    configured_cpu_cores: int
    configured_task_threads: int
    configured_task_slots: int
    configured_gpu_count: int
    configured_gpu_devices: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def gmx_installed(self) -> bool:
        return self.gmx_path is not None

    @property
    def gmx_gpu_usable(self) -> bool:
        return (
            self.gmx_installed
            and self.gmx_gpu_compiled
            and self.gmx_gpu_probe_passed
            and self.cuda_runtime_usable
            and self.detected_gpu_count > 0
        )

    def as_dict(self) -> dict:
        result = asdict(self)
        result["gmx_installed"] = self.gmx_installed
        result["gmx_gpu_usable"] = self.gmx_gpu_usable
        result["configured_gpu_devices"] = list(self.configured_gpu_devices)
        result["warnings"] = list(self.warnings)
        return result

    def as_public_dict(self) -> dict:
        """Return hardware status without host paths or diagnostic output."""
        return {
            "gmx_installed": self.gmx_installed,
            "gmx_version": self.gmx_version,
            "gmx_gpu_backend": self.gmx_gpu_backend,
            "gmx_gpu_compiled": self.gmx_gpu_compiled,
            "gmx_gpu_probe_passed": self.gmx_gpu_probe_passed,
            "gmx_gpu_usable": self.gmx_gpu_usable,
            "cuda_runtime_usable": self.cuda_runtime_usable,
            "detected_gpu_count": self.detected_gpu_count,
            "detected_cpu_threads": self.detected_cpu_threads,
            "allowed_cpu_threads": self.allowed_cpu_threads,
            "configured_cpu_cores": self.configured_cpu_cores,
            "configured_task_threads": self.configured_task_threads,
            "configured_task_slots": self.configured_task_slots,
            "configured_gpu_count": self.configured_gpu_count,
        }


def available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(max(1, int(os.cpu_count() or 1))))


def _candidate_gromacs_paths() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("GMX_BIN", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    for executable in ("gmx", "gmx_mpi"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(Path(resolved))
    for root in (Path.home() / "Software", Path("/opt"), Path("/usr/local")):
        if not root.is_dir():
            continue
        patterns = (
            "Gromacs*/bin/gmx",
            "gromacs*/bin/gmx",
            "Gromacs*/build/bin/gmx",
            "gromacs*/build/bin/gmx",
        )
        for pattern in patterns:
            candidates.extend(sorted(root.glob(pattern), reverse=True))
    return candidates


def find_gromacs_executable() -> str | None:
    """Find an executable GROMACS frontend without version-specific paths."""
    seen: set[Path] = set()
    for candidate in _candidate_gromacs_paths():
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
    return None


def _gromacs_build_info(executable: str | None) -> tuple[str | None, str | None]:
    if not executable:
        return None, None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    output = result.stdout + "\n" + result.stderr
    version_match = re.search(r"GROMACS version:\s*(?:VERSION\s*)?([^\s]+)", output)
    gpu_match = re.search(r"GPU support:\s*([^\r\n]+)", output, re.IGNORECASE)
    version = version_match.group(1).strip() if version_match else None
    backend = gpu_match.group(1).strip() if gpu_match else None
    return version, backend


def _cuda_device_count() -> tuple[bool, int, str | None]:
    """Use the CUDA driver API; unlike nvidia-smi this does not depend on NVML."""
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return False, 0, f"CUDA driver library is unavailable: {exc}"
    initialized = int(library.cuInit(0))
    if initialized != 0:
        return False, 0, f"CUDA driver initialization failed with code {initialized}"
    count = ctypes.c_int()
    status = int(library.cuDeviceGetCount(ctypes.byref(count)))
    if status != 0:
        return False, 0, f"CUDA device enumeration failed with code {status}"
    return count.value > 0, max(0, int(count.value)), None


def probe_gromacs_gpu(
    executable: str,
    device_id: int = 0,
) -> tuple[bool, str | None]:
    """Run one real short-range GPU step with a temporary argon box."""
    try:
        with tempfile.TemporaryDirectory(prefix="gmxbuilder-gpu-probe-") as temp:
            work = Path(temp)
            coordinates = []
            serial = 1
            for z in (0.6, 1.2):
                for y in (0.6, 1.2, 1.8, 2.4):
                    for x in (0.6, 1.2, 1.8, 2.4):
                        coordinates.append(
                            f"{1:5d}{'ARG':<5}{'AR':>5}{serial:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                        )
                        serial += 1
            (work / "probe.gro").write_text(
                "GMXBUILDER CUDA probe\n"
                f"{len(coordinates):5d}\n"
                + "\n".join(coordinates)
                + "\n   3.00000   3.00000   3.00000\n"
            )
            (work / "probe.top").write_text(
                "[ defaults ]\n1 2 yes 0.5 0.833333\n\n"
                "[ atomtypes ]\nAR 18 39.948 0.0 A 0.340 0.997\n\n"
                "[ moleculetype ]\nARG 1\n\n[ atoms ]\n"
                "1 AR 1 ARG AR 1 0.0 39.948"
                + "\n\n[ system ]\nCUDA probe\n\n[ molecules ]\nARG "
                + str(len(coordinates))
                + "\n"
            )
            (work / "probe.mdp").write_text(
                "integrator = md\nnsteps = 1\ndt = 0.001\n"
                "cutoff-scheme = Verlet\nnstlist = 1\nrlist = 1.0\n"
                "coulombtype = Cut-off\nrcoulomb = 1.0\n"
                "vdwtype = Cut-off\nrvdw = 1.0\n"
                "tcoupl = no\npcoupl = no\nconstraints = none\npbc = xyz\n"
                "gen-vel = yes\ngen-temp = 300\ngen-seed = 20260728\n"
            )
            grompp = subprocess.run(
                [
                    executable,
                    "grompp",
                    "-f",
                    "probe.mdp",
                    "-c",
                    "probe.gro",
                    "-p",
                    "probe.top",
                    "-o",
                    "probe.tpr",
                    "-maxwarn",
                    "1",
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if grompp.returncode != 0:
                detail = (grompp.stdout + "\n" + grompp.stderr)[-1200:]
                return False, f"GROMACS GPU probe input failed: {detail}"
            mdrun = subprocess.run(
                [
                    executable,
                    "mdrun",
                    "-s",
                    "probe.tpr",
                    "-deffnm",
                    "probe",
                    "-ntmpi",
                    "1",
                    "-ntomp",
                    "1",
                    "-nb",
                    "gpu",
                    "-pme",
                    "cpu",
                    "-bonded",
                    "cpu",
                    "-update",
                    "cpu",
                    "-gpu_id",
                    str(int(device_id)),
                    "-noconfout",
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            output = mdrun.stdout + "\n" + mdrun.stderr
            if mdrun.returncode != 0:
                return False, f"GROMACS could not execute on GPU {device_id}: {output[-1200:]}"
            if not re.search(r"\b(?:GPU|CUDA)\b", output, re.IGNORECASE):
                return False, "GROMACS completed the probe without reporting GPU execution"
            return True, None
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"GROMACS GPU probe failed: {exc}"


def _positive_int(value: str | int | None, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact:
        raise ValueError(f"{label} must be an integer")
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _nonnegative_int(value: str | int | None, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact:
        raise ValueError(f"{label} must be an integer")
    if parsed < 0:
        raise ValueError(f"{label} must be zero or a positive integer")
    return parsed


def _default_task_threads(cpu_cores: int, target_slots: int) -> int:
    """Choose the largest exact divisor that can provide *target_slots*."""
    target = max(1, min(int(target_slots), cpu_cores))
    for threads in range(max(1, cpu_cores // target), 0, -1):
        if cpu_cores % threads == 0:
            return threads
    return 1


def configure_runtime_resources(
    *,
    cpu_cores: int | None = None,
    gpu_count: int | None = None,
    task_threads: int | None = None,
    target_task_slots: int = 4,
    apply_affinity: bool = True,
) -> HardwareCapabilities:
    """Validate and apply the process-wide CPU/GPU resource budget."""
    original_cpu_ids = available_cpu_ids()
    detected_cpu_threads = int(
        os.environ.get("GMXBUILDER_DETECTED_CPU_THREADS", len(original_cpu_ids))
    )
    available_cpu_threads = len(original_cpu_ids)
    requested_cpu = _positive_int(
        cpu_cores if cpu_cores is not None else os.environ.get("GMXBUILDER_CPU_CORES"),
        "CPU core count",
    )
    selected_cpu = requested_cpu or max(1, available_cpu_threads // 2)
    if selected_cpu > available_cpu_threads:
        raise ValueError(
            f"Requested {selected_cpu} CPU cores, but only "
            f"{available_cpu_threads} threads are available to this process"
        )
    selected_cpu_ids = original_cpu_ids[:selected_cpu]
    if apply_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(selected_cpu_ids))

    requested_task_threads = _positive_int(
        task_threads if task_threads is not None else os.environ.get("GMXBUILDER_TASK_THREADS"),
        "Per-task thread count",
    )
    selected_task_threads = requested_task_threads or _default_task_threads(
        selected_cpu, target_task_slots
    )
    if selected_cpu % selected_task_threads != 0:
        raise ValueError(
            f"Per-task thread count {selected_task_threads} must divide the "
            f"allocated CPU count {selected_cpu} exactly"
        )
    task_slots = selected_cpu // selected_task_threads

    executable = find_gromacs_executable()
    version, gpu_backend = _gromacs_build_info(executable)
    cuda_usable, visible_gpus, cuda_warning = _cuda_device_count()
    try:
        detected_gpus = max(0, int(os.environ.get("GMXBUILDER_DETECTED_GPU_COUNT", visible_gpus)))
    except ValueError:
        detected_gpus = visible_gpus
    gpu_compiled = bool(
        gpu_backend and gpu_backend.strip().lower() not in {"disabled", "none", "no"}
    )
    gpu_available = bool(executable and gpu_compiled and cuda_usable and visible_gpus)
    requested_gpu = _nonnegative_int(
        gpu_count if gpu_count is not None else os.environ.get("GMXBUILDER_GPU_COUNT"),
        "GPU count",
    )
    selected_gpu_count = requested_gpu if requested_gpu is not None else (1 if gpu_available else 0)
    if selected_gpu_count and not executable:
        raise ValueError("GPU exposure was requested, but GROMACS was not found")
    if selected_gpu_count and not gpu_compiled:
        raise ValueError("GPU exposure was requested, but GROMACS has no GPU backend")
    if selected_gpu_count and not cuda_usable:
        raise ValueError("GPU exposure was requested, but the CUDA driver is unusable")

    preconfigured = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip() and value.strip() != "-1"
    ]
    available_devices = preconfigured or [str(index) for index in range(visible_gpus)]
    if selected_gpu_count > len(available_devices):
        raise ValueError(
            f"Requested {selected_gpu_count} GPUs, but only "
            f"{len(available_devices)} CUDA devices are available"
        )
    selected_devices = tuple(available_devices[:selected_gpu_count])
    warnings: list[str] = []
    gpu_probe_passed = False
    if selected_devices and executable:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
        gpu_probe_passed = True
        probe_error = None
        for logical_device in range(len(selected_devices)):
            cache_key = (executable, selected_devices, logical_device)
            result = _gpu_probe_cache.get(cache_key)
            if result is None:
                result = probe_gromacs_gpu(executable, logical_device)
                _gpu_probe_cache[cache_key] = result
            device_passed, device_error = result
            if not device_passed:
                gpu_probe_passed = False
                probe_error = device_error
                break
        if not gpu_probe_passed:
            if requested_gpu is not None and requested_gpu > 0:
                raise ValueError(probe_error or "GROMACS GPU probe failed")
            warnings.append(probe_error or "GROMACS GPU probe failed")
            selected_gpu_count = 0
            selected_devices = ()

    os.environ["GMXBUILDER_CPU_CORES"] = str(selected_cpu)
    os.environ["GMXBUILDER_TASK_THREADS"] = str(selected_task_threads)
    os.environ["GMXBUILDER_TASK_SLOTS"] = str(task_slots)
    os.environ["GMXBUILDER_DETECTED_CPU_THREADS"] = str(detected_cpu_threads)
    os.environ["GMXBUILDER_DETECTED_GPU_COUNT"] = str(detected_gpus)
    os.environ["GMXBUILDER_CPU_IDS"] = ",".join(map(str, selected_cpu_ids))
    os.environ["GMXBUILDER_GPU_COUNT"] = str(selected_gpu_count)
    os.environ["GMXBUILDER_GPU_IDS"] = ",".join(str(index) for index in range(selected_gpu_count))
    lipid_concurrency = min(2, selected_gpu_count) if selected_gpu_count else 1
    os.environ.setdefault(
        "GMXBUILDER_LIPID_THREADS",
        str(max(1, selected_cpu // lipid_concurrency)),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices) if selected_devices else "-1"
    os.environ["GMXBUILDER_LIPID_LIBRARY_GPU"] = "1" if selected_gpu_count else "0"
    if executable:
        os.environ["GMX_BIN"] = executable
    # Cap BLAS-style numerical libraries used inside each worker. Do not set
    # OMP_NUM_THREADS or OMP_THREAD_LIMIT on the parent process: GROMACS calls
    # use an explicit -ntomp value and reject a conflicting inherited OpenMP
    # environment. Tools that require OpenMP receive a scoped child env.
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(selected_task_threads)

    if not executable:
        warnings.append("GROMACS executable was not found")
    elif version is None:
        warnings.append("GROMACS version output could not be parsed")
    if gpu_compiled and cuda_warning:
        warnings.append(cuda_warning)
    if executable and not gpu_compiled:
        warnings.append("Detected GROMACS build has no GPU backend")

    hardware_capabilities.cache_clear()
    return HardwareCapabilities(
        gmx_path=executable,
        gmx_version=version,
        gmx_gpu_backend=gpu_backend,
        gmx_gpu_compiled=gpu_compiled,
        gmx_gpu_probe_passed=gpu_probe_passed,
        cuda_runtime_usable=cuda_usable,
        detected_gpu_count=detected_gpus,
        detected_cpu_threads=detected_cpu_threads,
        allowed_cpu_threads=len(original_cpu_ids),
        configured_cpu_cores=selected_cpu,
        configured_task_threads=selected_task_threads,
        configured_task_slots=task_slots,
        configured_gpu_count=selected_gpu_count,
        configured_gpu_devices=selected_devices,
        warnings=tuple(warnings),
    )


@lru_cache(maxsize=1)
def hardware_capabilities() -> HardwareCapabilities:
    """Inspect current resources without changing process affinity or visibility."""
    return configure_runtime_resources(apply_affinity=False)


def configured_cpu_cores() -> int:
    return int(os.environ.get("GMXBUILDER_CPU_CORES", max(1, len(available_cpu_ids()) // 2)))


def configured_task_threads() -> int:
    """Return the deployment-wide maximum native threads for one task."""
    return int(os.environ.get("GMXBUILDER_TASK_THREADS", "1"))


def configured_task_slots() -> int:
    """Return how many full per-task CPU budgets fit in the allocation."""
    configured = os.environ.get("GMXBUILDER_TASK_SLOTS", "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, configured_cpu_cores() // configured_task_threads())


def configured_cpu_ids() -> tuple[int, ...]:
    raw = os.environ.get("GMXBUILDER_CPU_IDS", "")
    if raw.strip():
        return tuple(int(value) for value in raw.split(",") if value.strip())
    cpu_ids = available_cpu_ids()
    return cpu_ids[: configured_cpu_cores()]


def configured_gpu_ids() -> tuple[int, ...]:
    raw = os.environ.get("GMXBUILDER_GPU_IDS", "")
    return tuple(int(value) for value in raw.split(",") if value.strip())


def configured_gpu_devices() -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return tuple(
        value.strip() for value in raw.split(",") if value.strip() and value.strip() != "-1"
    )


def lipid_worker_threads(concurrency: int = 1) -> int:
    configured = os.environ.get("GMXBUILDER_LIPID_THREADS", "").strip()
    if configured:
        return min(configured_task_threads(), max(1, int(configured)))
    return min(
        configured_task_threads(),
        max(1, configured_cpu_cores() // max(1, int(concurrency))),
    )


def normalize_simulation_hardware(config: object | None) -> dict[str, object]:
    """Validate portable execution defaults embedded in a generated package."""
    raw = {} if config is None else config
    if not isinstance(raw, dict):
        raise ValueError("simulation hardware settings must be an object")
    allowed = {
        "mode",
        "cpu_threads",
        "mpi_ranks",
        "gpu_count",
        "gpu_ids",
        "use_gpu",
        "gmx_command",
        "mpi_launcher",
        "pin",
        "omp_threads",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown simulation hardware setting(s): " + ", ".join(unknown))

    mode = str(raw.get("mode", "thread-mpi")).strip().lower()
    if mode not in {"thread-mpi", "external-mpi"}:
        raise ValueError("simulation hardware mode must be thread-mpi or external-mpi")
    cpu_threads = _positive_int(raw.get("cpu_threads", 1), "Simulation CPU thread count")
    mpi_ranks = _positive_int(raw.get("mpi_ranks", 1), "Simulation MPI rank count")
    assert cpu_threads is not None and mpi_ranks is not None
    if cpu_threads > 4096 or mpi_ranks > 4096:
        raise ValueError("simulation CPU threads and MPI ranks must not exceed 4096")
    if cpu_threads % mpi_ranks != 0:
        raise ValueError(
            f"Simulation MPI rank count {mpi_ranks} must divide CPU thread "
            f"count {cpu_threads} exactly"
        )
    derived_omp_threads = cpu_threads // mpi_ranks
    if "omp_threads" in raw:
        omp_threads = _positive_int(raw["omp_threads"], "Simulation OpenMP thread count")
        if omp_threads != derived_omp_threads:
            raise ValueError("simulation omp_threads must equal cpu_threads divided by mpi_ranks")

    use_gpu = raw.get("use_gpu", False)
    if not isinstance(use_gpu, bool):
        raise ValueError("simulation use_gpu must be true or false")
    gpu_value = raw.get("gpu_ids", "")
    if isinstance(gpu_value, list):
        gpu_tokens = [str(value).strip() for value in gpu_value]
    else:
        gpu_tokens = [value.strip() for value in str(gpu_value).split(",") if value.strip()]
    if any(not token.isdigit() for token in gpu_tokens):
        raise ValueError("simulation GPU IDs must be non-negative integers")
    gpu_ids = [int(token) for token in gpu_tokens]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("simulation GPU IDs must not contain duplicates")
    if use_gpu and not gpu_ids:
        raise ValueError("select at least one GPU ID when GPU execution is enabled")
    if use_gpu:
        gpu_count = _positive_int(raw.get("gpu_count", len(gpu_ids)), "Simulation GPU count")
        assert gpu_count is not None
        if gpu_count > 256:
            raise ValueError("simulation GPU count must not exceed 256")
        if gpu_count != len(gpu_ids):
            raise ValueError("simulation gpu_count must equal the number of selected GPU IDs")
        if gpu_count > mpi_ranks:
            raise ValueError(
                "simulation MPI rank count must be at least the selected GPU count; "
                "one rank cannot drive multiple GPUs"
            )
    else:
        disabled_count = raw.get("gpu_count", 0)
        if isinstance(disabled_count, bool) or not str(disabled_count).isdigit():
            raise ValueError("Simulation GPU count must be a non-negative integer")
        gpu_count = int(disabled_count)
        if gpu_count != 0:
            raise ValueError("simulation gpu_count must be 0 when GPU execution is disabled")
        gpu_ids = []

    default_command = "gmx_mpi" if mode == "external-mpi" else "gmx"
    gmx_command = str(raw.get("gmx_command", default_command)).strip()
    if not gmx_command or not _COMMAND_TOKEN.fullmatch(gmx_command):
        raise ValueError(
            "simulation GROMACS command must be one executable name or path without shell syntax"
        )
    mpi_launcher = str(raw.get("mpi_launcher", "mpirun")).strip().lower()
    if mpi_launcher not in {"mpirun", "mpiexec", "srun"}:
        raise ValueError("simulation MPI launcher must be mpirun, mpiexec, or srun")
    pin = str(raw.get("pin", "auto")).strip().lower()
    if pin not in {"auto", "on", "off"}:
        raise ValueError("simulation thread pinning must be auto, on, or off")

    return {
        "mode": mode,
        "cpu_threads": cpu_threads,
        "mpi_ranks": mpi_ranks,
        "omp_threads": derived_omp_threads,
        "use_gpu": use_gpu,
        "gpu_count": gpu_count,
        "gpu_ids": gpu_ids,
        "gmx_command": gmx_command,
        "mpi_launcher": mpi_launcher,
        "pin": pin,
    }

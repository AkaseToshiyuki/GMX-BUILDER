"""Portable CPU, CUDA and GROMACS resource configuration."""

import os
from pathlib import Path

import pytest

from gmxbuilder.runtime import hardware


_ENV_KEYS = (
    "GMX_BIN",
    "GMXBUILDER_CPU_CORES",
    "GMXBUILDER_CPU_IDS",
    "GMXBUILDER_TASK_THREADS",
    "GMXBUILDER_TASK_SLOTS",
    "GMXBUILDER_DETECTED_CPU_THREADS",
    "GMXBUILDER_DETECTED_GPU_COUNT",
    "GMXBUILDER_GPU_COUNT",
    "GMXBUILDER_GPU_IDS",
    "GMXBUILDER_LIPID_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "GMXBUILDER_LIPID_LIBRARY_GPU",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@pytest.fixture(autouse=True)
def _isolate_hardware_state():
    """Do not leak fake executable probes or resource settings to other tests."""
    original_environment = {key: os.environ.get(key) for key in _ENV_KEYS}
    hardware.hardware_capabilities.cache_clear()
    hardware._gpu_probe_cache.clear()
    yield
    for key, value in original_environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    hardware.hardware_capabilities.cache_clear()
    hardware._gpu_probe_cache.clear()


def _clean_environment(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    hardware.hardware_capabilities.cache_clear()
    hardware._gpu_probe_cache.clear()


def _fake_gmx(tmp_path: Path) -> Path:
    executable = tmp_path / "gmx"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return executable


def test_defaults_to_half_cpu_threads_and_gpu_zero(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    executable = _fake_gmx(tmp_path)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(8)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [executable])
    monkeypatch.setattr(hardware, "_gromacs_build_info", lambda _path: ("2025.4", "CUDA"))
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (True, 3, None))
    monkeypatch.setattr(hardware, "probe_gromacs_gpu", lambda *_args: (True, None))

    result = hardware.configure_runtime_resources(apply_affinity=False)

    assert result.configured_cpu_cores == 4
    assert result.configured_task_threads == 1
    assert result.configured_task_slots == 4
    assert result.configured_gpu_count == 1
    assert result.configured_gpu_devices == ("0",)
    assert result.gmx_gpu_usable is True
    assert hardware.configured_gpu_ids() == (0,)
    assert hardware.configured_cpu_ids() == (0, 1, 2, 3)
    assert hardware.lipid_worker_threads() == 1


def test_operator_can_limit_cpu_and_expose_multiple_gpus(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    executable = _fake_gmx(tmp_path)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(12)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [executable])
    monkeypatch.setattr(hardware, "_gromacs_build_info", lambda _path: ("2025.4", "CUDA"))
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (True, 3, None))
    monkeypatch.setattr(hardware, "probe_gromacs_gpu", lambda *_args: (True, None))

    result = hardware.configure_runtime_resources(
        cpu_cores=6,
        gpu_count=2,
        task_threads=3,
        apply_affinity=False,
    )

    assert result.configured_cpu_cores == 6
    assert result.configured_task_threads == 3
    assert result.configured_task_slots == 2
    assert result.configured_gpu_devices == ("0", "1")
    assert hardware.configured_gpu_ids() == (0, 1)
    assert hardware.lipid_worker_threads() == 3
    assert hardware.configured_task_threads() == 3


def test_no_gromacs_disables_default_gpu_execution(monkeypatch):
    _clean_environment(monkeypatch)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(4)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [])
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (True, 2, None))

    result = hardware.configure_runtime_resources(apply_affinity=False)

    assert result.gmx_installed is False
    assert result.configured_gpu_count == 0
    assert "GROMACS executable was not found" in result.warnings
    assert hardware.configured_gpu_ids() == ()


def test_invalid_operator_limits_are_rejected(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    executable = _fake_gmx(tmp_path)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(4)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [executable])
    monkeypatch.setattr(hardware, "_gromacs_build_info", lambda _path: ("2025.4", "CUDA"))
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (True, 1, None))

    with pytest.raises(ValueError, match="only 4 threads"):
        hardware.configure_runtime_resources(cpu_cores=5, apply_affinity=False)
    with pytest.raises(ValueError, match="only 1 CUDA"):
        hardware.configure_runtime_resources(gpu_count=2, apply_affinity=False)
    with pytest.raises(ValueError, match="must divide"):
        hardware.configure_runtime_resources(cpu_cores=4, task_threads=3, apply_affinity=False)


def test_every_manually_exposed_gpu_is_probed(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    executable = _fake_gmx(tmp_path)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(8)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [executable])
    monkeypatch.setattr(hardware, "_gromacs_build_info", lambda _path: ("2025.4", "CUDA"))
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (True, 2, None))
    probes = []

    def probe(_executable, device):
        probes.append(device)
        return (device == 0, None if device == 0 else "GPU 1 failed")

    monkeypatch.setattr(hardware, "probe_gromacs_gpu", probe)
    with pytest.raises(ValueError, match="GPU 1 failed"):
        hardware.configure_runtime_resources(
            gpu_count=2,
            apply_affinity=False,
        )
    assert probes == [0, 1]


def test_public_hardware_status_does_not_expose_host_path(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    executable = _fake_gmx(tmp_path)
    monkeypatch.setattr(hardware, "available_cpu_ids", lambda: tuple(range(4)))
    monkeypatch.setattr(hardware, "_candidate_gromacs_paths", lambda: [executable])
    monkeypatch.setattr(hardware, "_gromacs_build_info", lambda _path: ("2025.4", "disabled"))
    monkeypatch.setattr(hardware, "_cuda_device_count", lambda: (False, 0, "detail"))

    public = hardware.configure_runtime_resources(
        gpu_count=0,
        apply_affinity=False,
    ).as_public_dict()

    assert "gmx_path" not in public
    assert "warnings" not in public
    assert str(tmp_path) not in str(public)


def test_simulation_hardware_requires_exact_cpu_rank_partition():
    configured = hardware.normalize_simulation_hardware(
        {
            "mode": "external-mpi",
            "cpu_threads": 24,
            "mpi_ranks": 4,
            "use_gpu": True,
            "gpu_count": 2,
            "gpu_ids": "0,1",
            "gmx_command": "/opt/gromacs/bin/gmx_mpi",
            "mpi_launcher": "srun",
            "pin": "on",
        }
    )

    assert configured["omp_threads"] == 6
    assert configured["gpu_count"] == 2
    assert configured["gpu_ids"] == [0, 1]
    with pytest.raises(ValueError, match="must divide"):
        hardware.normalize_simulation_hardware(
            {
                "cpu_threads": 10,
                "mpi_ranks": 3,
            }
        )
    with pytest.raises(ValueError, match="shell syntax"):
        hardware.normalize_simulation_hardware(
            {
                "gmx_command": "gmx; rm -rf /",
            }
        )


def test_simulation_hardware_rejects_inconsistent_multi_gpu_selection():
    with pytest.raises(ValueError, match="gpu_count must equal"):
        hardware.normalize_simulation_hardware(
            {
                "cpu_threads": 8,
                "mpi_ranks": 2,
                "use_gpu": True,
                "gpu_count": 1,
                "gpu_ids": "0,1",
            }
        )
    with pytest.raises(ValueError, match="at least the selected GPU count"):
        hardware.normalize_simulation_hardware(
            {
                "cpu_threads": 8,
                "mpi_ranks": 1,
                "use_gpu": True,
                "gpu_count": 2,
                "gpu_ids": "0,1",
            }
        )

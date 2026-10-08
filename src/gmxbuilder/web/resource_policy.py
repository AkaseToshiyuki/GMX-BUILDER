"""Configuration shared by the installer, coordinator and isolated workers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


def positive_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True)
class ResourcePolicy:
    task_memory_bytes: int
    task_storage_bytes: int
    total_storage_bytes: int
    lifetime_hours: float
    cpu_cores: int
    slots: int

    @classmethod
    def from_environment(cls) -> ResourcePolicy:
        memory = int(positive_number("GMXBUILDER_TASK_MEMORY_GIB", 16) * GIB)
        task = int(positive_number("GMXBUILDER_TASK_MAX_PERSISTENT_GIB", 2) * GIB)
        total = int(positive_number("GMXBUILDER_STORAGE_MAX_GIB", 100) * GIB)
        lifetime = positive_number("GMXBUILDER_TASK_TTL_HOURS", 24)
        cores = int(positive_number("GMXBUILDER_CPU_CORES", os.cpu_count() or 1))
        slots = int(positive_number("GMXBUILDER_MAX_BUILDS", 4))
        if min(memory, task, total, cores, slots) < 1 or task >= total:
            raise ValueError("Resource limits must be positive; task storage must be below total")
        return cls(memory, task, total, lifetime, cores, min(slots, cores))


def managed_root() -> Path | None:
    raw = os.environ.get("GMXBUILDER_MANAGED_ROOT", "")
    return Path(raw).resolve() if raw else None


def is_worker() -> bool:
    return os.environ.get("GMXBUILDER_RESOURCE_WORKER") == "1"

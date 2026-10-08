"""Bounded checkpoint summaries, published with scientific checkpoint writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

MAX_STATUS_BYTES = 256 * 1024


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def checkpoint_stamp(directory):
    return [file_stamp(Path(directory) / name) for name in ("system.json", "system.npz")]


def atomic_json(path, value):
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_STATUS_BYTES:
        raise ValueError("Checkpoint status exceeds its bounded metadata budget")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    with path.open("rb") as handle:
        data = handle.read(MAX_STATUS_BYTES + 1)
    if len(data) > MAX_STATUS_BYTES:
        raise ValueError("Oversized checkpoint status")
    return json.loads(data)


def write_status(system, directory):
    from gmxbuilder.core.step_metrics import compute_step_metrics

    directory = Path(directory)
    validation = system.metadata.get("input_validation")
    if not isinstance(validation, dict):
        validation = {}
    payload = {
        "schema": 1,
        "stamp": checkpoint_stamp(directory),
        "input_validation": {
            "policy_version": validation.get("policy_version"),
            "can_proceed": validation.get("can_proceed") is True,
            "has_errors": bool(validation.get("errors")),
        },
        "system_confirmed": bool(system.metadata.get("system_confirmed")),
        "membrane_metrics": compute_step_metrics(system, directory.name)
        if directory.name in {"membrane", "cg_environment"}
        else None,
    }
    atomic_json(directory / "status.json", payload)


def read_status(directory):
    directory = Path(directory)
    try:
        value = read_json(directory / "status.json")
        if value.get("schema") == 1 and value.get("stamp") == checkpoint_stamp(directory):
            return value
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None


def refresh_status(directory):
    """Legacy migration: call only from admitted construction/resume work."""
    from gmxbuilder.core.system import System

    directory = Path(directory)
    if all((directory / name).is_file() for name in ("system.json", "system.npz")):
        if read_status(directory) is None:
            write_status(System.load_checkpoint(directory), directory)

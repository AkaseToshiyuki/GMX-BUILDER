"""Task lifecycle manager — create, persist, resume, and expire tasks."""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

TASK_ROOT = Path(os.environ.get("GMXBUILDER_TASK_DIR", "/tmp/gmxbuilder_tasks"))
DEFAULT_TASK_TTL_HOURS = 8.0


def task_ttl_hours() -> float:
    """Return the configured task lifetime, rejecting unsafe values."""
    raw = os.environ.get("GMXBUILDER_TASK_TTL_HOURS", str(DEFAULT_TASK_TTL_HOURS))
    try:
        hours = float(raw)
    except ValueError as exc:
        raise ValueError("GMXBUILDER_TASK_TTL_HOURS must be a positive number") from exc
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("GMXBUILDER_TASK_TTL_HOURS must be a positive number")
    return hours


def task_expiry(now: datetime | None = None) -> str:
    """Return an ISO expiration timestamp using the current deployment TTL."""
    current = now or datetime.now(timezone.utc)
    return (current + timedelta(hours=task_ttl_hours())).isoformat()


# Per-task locks for atomic read-modify-write on state.json
_state_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(task_id: str) -> threading.Lock:
    """Return (or create) the per-task state lock."""
    with _locks_lock:
        if task_id not in _state_locks:
            _state_locks[task_id] = threading.Lock()
        return _state_locks[task_id]


class TaskManager:
    """Manages task directories, state persistence, and expiration."""

    def __init__(self, root: Path | None = None):
        self.root = root or TASK_ROOT
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._active_counts: dict[str, int] = {}
        self._active_lock = threading.Lock()
        # Tighten legacy task permissions on upgrade.  Do not follow symlinks.
        for task_dir in self.root.iterdir():
            if not task_dir.is_dir() or task_dir.is_symlink():
                continue
            task_dir.chmod(0o700)
            state_file = task_dir / "state.json"
            if state_file.is_file() and not state_file.is_symlink():
                state_file.chmod(0o600)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_task(self, filename: str = "") -> dict:
        """Create a new task directory and return task metadata."""
        # The task ID is also the bearer capability for resume/download.
        # Keep the full UUID entropy rather than the former 48-bit prefix.
        task_id = uuid.uuid4().hex
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        task_dir.chmod(0o700)

        now = datetime.now(timezone.utc)
        state = {
            "task_id": task_id,
            "filename": filename,
            "created_at": now.isoformat(),
            "expires_at": task_expiry(now),
            "current_step": "input",
            "steps_completed": [],
            "pdb_info": None,
            "structure": None,
            "orient": None,
            "membrane": None,
            "solvation": None,
            "ions": None,
            "forcefield": None,
            "simparams": None,
        }
        self._write_state(task_dir, state)
        return state

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state(self, task_id: str) -> dict | None:
        """Read the state.json for a task."""
        state_file = self.root / task_id / "state.json"
        if not state_file.exists():
            return None
        with open(state_file) as f:
            return json.load(f)

    def update_state(self, task_id: str, updates: dict) -> dict | None:
        """Merge updates into the task state and return the new state.

        Uses per-task locking to prevent lost-update races when multiple
        concurrent requests modify the same task's state.json.
        """
        with _get_lock(task_id):
            state = self.get_state(task_id)
            if state is None:
                return None
            state.update(updates)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            task_dir = self.root / task_id
            self._write_state(task_dir, state)
        return state

    def save_step_state(self, task_id: str, step_name: str, step_data: dict) -> dict | None:
        """Save non-authoritative browser UI state under a dedicated namespace.

        Scientific completion is represented by durable step checkpoints, not
        by this convenience state.  Keeping UI drafts out of the top-level
        state also prevents a caller from overwriting lifecycle fields.
        """
        if not isinstance(step_name, str) or not step_name:
            raise ValueError("UI step name must be a non-empty string")
        if not isinstance(step_data, dict):
            raise ValueError("UI step data must be an object")
        with _get_lock(task_id):
            state = self.get_state(task_id)
            if state is None:
                return None
            ui_state = state.get("step_ui_state", {})
            if not isinstance(ui_state, dict):
                ui_state = {}
            ui_state[step_name] = step_data
            state["step_ui_state"] = ui_state
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            task_dir = self.root / task_id
            self._write_state(task_dir, state)
        return state

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def get_task_dir(self, task_id: str) -> Path:
        """Return the task directory path."""
        return self.root / task_id

    def get_pdb_path(self, task_id: str) -> Path | None:
        """Return a PDB-formatted structure suitable for legacy consumers.

        Converted/filtered files take precedence over the original upload.
        This is important for mmCIF tasks and for legacy ``*.cif.pdb`` uploads,
        whose original bytes are not actually PDB-formatted.
        """
        task_dir = self.root / task_id
        for preferred in ("filtered.pdb", "converted.pdb"):
            candidate = task_dir / preferred
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        pdb_files = sorted(
            path
            for path in task_dir.glob("*.pdb")
            if path.is_file()
            and not path.is_symlink()
            and not path.name.lower().endswith(".cif.pdb")
            and not path.name.lower().endswith(".mmcif.pdb")
        )
        if pdb_files:
            return pdb_files[0]
        ent_files = sorted(
            path for path in task_dir.glob("*.ent") if path.is_file() and not path.is_symlink()
        )
        return ent_files[0] if ent_files else None

    def save_uploaded_pdb(self, task_id: str, filename: str, content: bytes) -> Path:
        """Save an uploaded PDB/mmCIF structure using a safe basename."""
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        task_dir.chmod(0o700)
        # Use original filename or fallback
        safe_name = Path(filename).name if filename else "upload.pdb"
        suffix = Path(safe_name).suffix.lower()
        if suffix in {".pdb", ".ent", ".cif", ".mmcif"}:
            safe_name = f"{Path(safe_name).stem}{suffix}"
        else:
            safe_name += ".pdb"
        pdb_path = task_dir / safe_name
        pdb_path.write_bytes(content)
        pdb_path.chmod(0o600)
        return pdb_path

    def get_output_dir(self, task_id: str) -> Path:
        """Return (and create) the output directory for a task."""
        out = self.root / task_id / "output"
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.chmod(0o700)
        return out

    def save_build_request(self, task_id: str, request: dict) -> Path:
        """Persist an accepted finalization request for restart-safe queuing."""
        task_dir = self.root / task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task {task_id} does not exist")
        destination = task_dir / "build_request.json"
        temporary = destination.with_suffix(".tmp")
        with open(temporary, "w") as handle:
            json.dump(request, handle, indent=2, default=str)
        temporary.chmod(0o600)
        temporary.replace(destination)
        destination.chmod(0o600)
        return destination

    def load_build_request(self, task_id: str) -> dict | None:
        """Load a previously accepted finalization request."""
        path = self.root / task_id / "build_request.json"
        if not path.is_file() or path.is_symlink():
            return None
        with open(path) as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def is_expired(self, task_id: str) -> bool:
        """Check if a task has exceeded its TTL."""
        state = self.get_state(task_id)
        if state is None:
            return True  # no state = invalid task, treat as expired
        try:
            expires_at = datetime.fromisoformat(state["expires_at"])
            return datetime.now(timezone.utc) > expires_at
        except (KeyError, ValueError):
            return True

    def cleanup_expired(self) -> list[str]:
        """Remove all expired task directories. Returns list of removed IDs."""
        removed = []
        for task_dir in sorted(self.root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            with self._active_lock:
                if self._active_counts.get(task_id, 0) > 0:
                    continue
            if self.is_expired(task_id):
                try:
                    shutil.rmtree(task_dir)
                    removed.append(task_id)
                except OSError:
                    pass
        return removed

    @contextmanager
    def active_task(self, task_id: str) -> Iterator[None]:
        """Prevent expiration cleanup while task-owned files are being written."""
        with self._active_lock:
            self._active_counts[task_id] = self._active_counts.get(task_id, 0) + 1
        try:
            yield
        finally:
            with self._active_lock:
                remaining = self._active_counts.get(task_id, 1) - 1
                if remaining > 0:
                    self._active_counts[task_id] = remaining
                else:
                    self._active_counts.pop(task_id, None)

    def delete_task(self, task_id: str) -> bool:
        """Explicitly delete a task directory."""
        task_dir = self.root / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _write_state(task_dir: Path, state: dict) -> None:
        state_file = task_dir / "state.json"
        # Atomic write via temp file
        tmp = state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.chmod(0o600)
        tmp.replace(state_file)
        state_file.chmod(0o600)


# Global singleton
task_manager = TaskManager()

"""Persistent ownership and source evidence for a V4 queue invocation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def source_manifest(root: Path) -> dict:
    files = {}
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "root": str(root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "files": files,
    }


def verify_source(root: Path, manifest: dict) -> None:
    if source_manifest(root) != manifest:
        raise RuntimeError("Queue source/parameters changed; refusing to launch another replica")


@contextmanager
def queue_ownership(output: Path, root: Path, *, configuration: dict | None = None):
    """Hold the lock through child lifetimes; preserve an immutable run manifest."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another V4 queue owns {output}") from exc
        # Children inherit this descriptor, so a killed scheduler cannot leave
        # running replicas unprotected against a second queue.
        os.set_inheritable(lock.fileno(), True)
        manifest = source_manifest(root)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        runs = output / "runs"
        runs.mkdir(exist_ok=True)
        manifest_path = runs / f"{run_id}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "source": manifest,
                    "configuration": configuration,
                    "python": str(Path(sys.executable).resolve()),
                    "legacy_provenance": (
                        "This manifest does not establish the origin of earlier trajectories."
                    ),
                },
                indent=2,
            )
        )
        yield manifest, manifest_path, lock.fileno()

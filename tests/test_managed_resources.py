"""Construction-free tests of lifecycle, ticket persistence and write boundaries."""

import errno
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gmxbuilder.web.resource_policy import GIB, ResourcePolicy
from gmxbuilder.web.resource_queue import ResourceQueue
from gmxbuilder.web.task_manager import TaskManager


def test_policy_defaults_and_invalid_limits(monkeypatch):
    for key in list(os.environ):
        if key.startswith("GMXBUILDER_"):
            monkeypatch.delenv(key)
    policy = ResourcePolicy.from_environment()
    assert policy.task_memory_bytes == 16 * GIB
    assert policy.task_storage_bytes == 2 * GIB
    assert policy.total_storage_bytes == 100 * GIB
    assert policy.lifetime_hours == 24
    for value in ("nan", "inf", "-1", "0", "no"):
        monkeypatch.setenv("GMXBUILDER_TASK_MEMORY_GIB", value)
        with pytest.raises(ValueError):
            ResourcePolicy.from_environment()


def test_anonymous_public_mode_requires_managed_resources_and_tls(monkeypatch):
    from gmxbuilder.web.security import SecurityConfig, validate_server_bind

    for key in list(os.environ):
        if key.startswith("GMXBUILDER_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "public-anonymous")
    assert SecurityConfig.from_environment().errors
    monkeypatch.setenv("GMXBUILDER_MANAGED_ROOT", "/configured/mount")
    monkeypatch.setenv("GMXBUILDER_TRUSTED_PROXIES", "127.0.0.1/32")
    monkeypatch.setenv("GMXBUILDER_CORS_ORIGINS", "https://example.test")
    config = validate_server_bind("0.0.0.0")
    assert not config.errors
    assert config.require_https
    assert not config.authentication_enabled
    monkeypatch.delenv("GMXBUILDER_MANAGED_ROOT")
    with pytest.raises(ValueError, match="managed resource"):
        validate_server_bind("0.0.0.0")


def test_access_cannot_extend_creation_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TASK_TTL_HOURS", "24")
    manager = TaskManager(tmp_path)
    original = manager.create_task()
    for _ in range(3):
        changed = manager.update_state(
            original["task_id"],
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": "2100-01-01",
            },
        )
        assert changed["created_at"] == original["created_at"]
        assert changed["expires_at"] == original["expires_at"]
    original["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    original["expires_at"] = "2100-01-01T00:00:00+00:00"
    manager._write_state(tmp_path / original["task_id"], original)
    assert manager.get_state(original["task_id"]) is None
    assert manager.cleanup_expired() == [original["task_id"]]


def test_cross_process_file_lease_prevents_deletion(tmp_path):
    manager = TaskManager(tmp_path)
    task = manager.create_task()
    program = (
        "import fcntl,sys,time; "
        "h=open(sys.argv[1],'a');fcntl.flock(h,fcntl.LOCK_SH); "
        "print('leased',flush=True);time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path / task["task_id"] / ".lease")],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "leased"
        assert manager.delete_task(task["task_id"]) is False
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert manager.delete_task(task["task_id"]) is True


def test_fifo_persists_and_deduplicates_without_count_limit(tmp_path):
    path = tmp_path / "queue.db"
    queue = ResourceQueue(path)
    deadline = time.time() + 86400
    for index in range(2500):
        queue.enqueue(f"{index:032x}", "same-input", "check", deadline, now=100 + index)
    first = queue.next()
    duplicate = queue.enqueue(first["task_id"], "same-input", "check", deadline)
    assert duplicate["id"] == first["id"]
    restored = ResourceQueue(path)
    assert restored.next() == first
    assert restored.snapshot(first["id"], now=1000)["queue_length"] == 2500
    assert restored.snapshot(first["id"], now=1000)["waited_seconds"] == 900
    assert restored.snapshot(first["id"])["estimated_wait_seconds"] is None
    restored.update(first["id"], status="running", started=time.time())
    restored.enqueue(first["task_id"], "changed-input", "check", deadline, now=101)
    assert restored.next()["task_id"] != first["task_id"]


@pytest.fixture
def quota_mount(tmp_path):
    pytest.importorskip("pyfuse3")
    if not Path("/dev/fuse").exists():
        pytest.skip("FUSE device is unavailable")
    mount, backing = tmp_path / "mounted", tmp_path / "backing"
    mount.mkdir()
    log = (tmp_path / "mount.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gmxbuilder.web.quota_fs",
            "--backing",
            str(backing),
            "--mount",
            str(mount),
            "--total-bytes",
            str(1024 * 1024),
            "--task-bytes",
            str(128 * 1024),
        ],
        stdout=log,
        stderr=log,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        for _ in range(100):
            if os.path.ismount(mount):
                break
            if process.poll() is not None:
                pytest.fail((tmp_path / "mount.log").read_text())
            time.sleep(0.02)
        assert os.path.ismount(mount)
        yield mount
    finally:
        subprocess.run(["fusermount3", "-u", str(mount)], capture_output=True)
        process.terminate()
        process.wait(timeout=10)
        log.close()


def test_filesystem_enforces_aggregate_sparse_and_unlinked_limits(quota_mount):
    assert json.loads(os.getxattr(quota_mount, "user.gmxbuilder.quota")) == {
        "total_bytes": 1024 * 1024,
        "task_bytes": 128 * 1024,
    }
    task = quota_mount / "tasks" / "a"
    task.mkdir(parents=True)
    first = task / "first"
    first.write_bytes(b"x" * 65536)
    with pytest.raises(OSError) as failure:
        (task / "second").write_bytes(b"x" * 65536)
    assert failure.value.errno == errno.EDQUOT
    with first.open("r+b") as handle:
        with pytest.raises(OSError) as failure:
            handle.truncate(1024 * 1024)
        assert failure.value.errno == errno.EDQUOT
        first.unlink()
        with pytest.raises(OSError):
            (task / "second").write_bytes(b"x" * 65536)
    (task / "second").write_bytes(b"x" * 65536)
    with pytest.raises(OSError) as failure:
        (quota_mount / "global-cache").write_bytes(b"x" * (1024 * 1024))
    assert failure.value.errno == errno.EDQUOT


def test_filesystem_database_and_atomic_replacement(quota_mount):
    queue = ResourceQueue(quota_mount / ".control" / "queue.db")
    task = "a" * 32
    item = queue.enqueue(task, "hash", "check", time.time() + 86400)
    queue.update(item["id"], status="running")
    assert queue.get(item["id"])["status"] == "running"
    directory = quota_mount / "tasks" / task
    directory.mkdir(parents=True)
    manager = TaskManager(directory / "nested")
    state = manager.create_task()
    manager.update_state(state["task_id"], {"hello": "world"})
    assert manager.get_state(state["task_id"])["hello"] == "world"


def test_landlock_covers_child_processes(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux write confinement")
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "forbidden"
    allowed.mkdir()
    code = """
import subprocess,sys
from pathlib import Path
from gmxbuilder.web.write_sandbox import restrict_writes
restrict_writes(Path(sys.argv[1]))
Path(sys.argv[1], 'ok').write_text('allowed')
p = subprocess.run([sys.executable,'-c',
    'from pathlib import Path;import sys;Path(sys.argv[1]).write_text("bad")',sys.argv[2]],
    capture_output=True)
assert p.returncode != 0
assert b'PermissionError' in p.stderr
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(allowed), str(forbidden)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert not forbidden.exists()


def test_migration_preserves_old_checkpoint_paths_and_cleans_expired(tmp_path, monkeypatch):
    from gmxbuilder.web.resource_setup import migrate

    source = tmp_path / "legacy"
    mount = tmp_path / "managed"
    (mount / "tasks").mkdir(parents=True)
    manager = TaskManager(source)
    live = manager.create_task("live")
    expired = manager.create_task("expired")
    expired["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    manager._write_state(source / expired["task_id"], expired)
    input_path = source / live["task_id"] / "input.pdb"
    input_path.write_text("retained input bytes")
    (source / ".metadata").write_text("metadata")
    monkeypatch.setattr(
        "gmxbuilder.web.resource_setup.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 3),
    )
    result = migrate(source, mount, ResourcePolicy.from_environment())
    assert result == {"migrated": [live["task_id"]], "expired_removed": [expired["task_id"]]}
    assert source.is_symlink()
    assert input_path.read_text() == "retained input bytes"
    assert not (source / expired["task_id"]).exists()
    assert (mount / ".control/legacy-task-root/.metadata").read_text() == "metadata"

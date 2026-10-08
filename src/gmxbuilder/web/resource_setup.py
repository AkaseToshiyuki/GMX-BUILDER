"""Install the unprivileged quota volume and migrate a quiescent task directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gmxbuilder.web.resource_policy import ResourcePolicy


def inventory(source: Path, policy: ResourcePolicy):
    items = []
    if not source.is_dir():
        return items
    for directory in source.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        if not re.fullmatch(r"[0-9a-f]{12}(?:[0-9a-f]{20})?", directory.name):
            raise ValueError(
                f"Unrecognized directory must be reviewed before migration: {directory}"
            )
        state = json.loads((directory / "state.json").read_text())
        created = datetime.fromisoformat(state["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        size = 4096
        for path in directory.rglob("*"):
            st = path.lstat()
            size += 4096
            if path.is_file() and not path.is_symlink():
                size += ((st.st_size + 4095) // 4096) * 4096
        expired = created.timestamp() + policy.lifetime_hours * 3600 <= time.time()
        items.append(
            {
                "task_id": directory.name,
                "created_at": created.isoformat(),
                "charged_bytes": size,
                "expired": expired,
            }
        )
    return sorted(items, key=lambda item: item["created_at"])


def systemd_quote(value):
    return '"' + str(value).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"') + '"'


def install_volume(root: Path, policy: ResourcePolicy):
    import pyfuse3  # noqa: F401 -- fail before writing a service if prerequisites are absent

    if not Path("/dev/fuse").exists() or not shutil.which("fusermount3"):
        raise RuntimeError("Managed storage requires FUSE3 and access to /dev/fuse")
    backing, mount = root / "backing", root / "mounted"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    backing.mkdir(exist_ok=True, mode=0o700)
    mount.mkdir(exist_ok=True, mode=0o700)
    if os.path.ismount(mount):
        actual = json.loads(os.getxattr(mount, "user.gmxbuilder.quota"))
        expected = {
            "total_bytes": policy.total_storage_bytes,
            "task_bytes": policy.task_storage_bytes,
        }
        if actual != expected:
            raise RuntimeError(
                "Active storage limits differ. Stop Web and storage services "
                "before changing volume quotas; existing data will be checked."
            )
    unit_directory = Path.home() / ".config/systemd/user"
    unit_directory.mkdir(parents=True, exist_ok=True)
    command = " ".join(
        map(
            systemd_quote,
            [
                sys.executable,
                "-m",
                "gmxbuilder.web.quota_fs",
                "--backing",
                backing,
                "--mount",
                mount,
                "--total-bytes",
                policy.total_storage_bytes,
                "--task-bytes",
                policy.task_storage_bytes,
            ],
        )
    )
    wait_command = " ".join(
        map(
            systemd_quote,
            [
                sys.executable,
                "-m",
                "gmxbuilder.web.resource_setup",
                "--wait-mounted",
                mount,
            ],
        )
    )
    (unit_directory / "gmxbuilder-storage.service").write_text(
        "[Unit]\nDescription=GMXBUILDER bounded Web storage\n"
        "Before=gmxbuilder.service\n\n[Service]\nType=simple\nUMask=0077\n"
        "Environment=PYTHONDONTWRITEBYTECODE=1\n"
        f"ExecStart={command}\nExecStartPost={wait_command}\n"
        f"ExecStop=/usr/bin/fusermount3 -u {systemd_quote(mount)}\n"
        f"ExecStopPost=-/usr/bin/fusermount3 -uz {systemd_quote(mount)}\n"
        "TimeoutStopSec=10\nRestart=on-failure\nRestartSec=3\n"
        # fusermount3 uses its installed setuid helper for an unprivileged
        # user's mount. NoNewPrivileges would prevent that helper working.
        "MemoryMax=1G\n"
        "StandardOutput=null\nStandardError=journal\n\n[Install]\nWantedBy=default.target\n"
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "gmxbuilder-storage.service"], check=True
    )
    for name in ("tasks", "tmp", "cache", "logs", ".control"):
        (mount / name).mkdir(exist_ok=True, mode=0o700)
    return mount


def digest_tree(directory):
    result = {}
    for path in directory.rglob("*"):
        relative = str(path.relative_to(directory))
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[relative] = digest.hexdigest()
    return result


def migrate(source: Path, mount: Path, policy: ResourcePolicy):
    if source.resolve() == (mount / "tasks").resolve():
        return {"migrated": [], "expired_removed": []}
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", "gmxbuilder.service"])
    if active.returncode == 0:
        raise RuntimeError("Stop the Web service after checking its workers before migrating tasks")
    items = inventory(source, policy)
    extras = [path for path in source.iterdir() if not path.is_dir()]
    if any(path.is_symlink() or not path.is_file() for path in extras):
        raise ValueError("Unexpected legacy task-root entry; review it before migration")
    live = [item for item in items if not item["expired"]]
    if any(item["charged_bytes"] > policy.task_storage_bytes for item in live):
        raise ValueError("An unexpired legacy task exceeds the per-task quota; migration stopped")
    if sum(item["charged_bytes"] for item in live) + 1024**2 > shutil.disk_usage(mount).free:
        raise ValueError("Not enough quota space to migrate retained tasks without loss")
    result = {"migrated": [], "expired_removed": []}
    for item in live:
        old, new = source / item["task_id"], mount / "tasks" / item["task_id"]
        shutil.copytree(old, new, symlinks=True, dirs_exist_ok=True)
        if digest_tree(old) != digest_tree(new):
            raise RuntimeError("Task migration verification failed; source data retained")
    for path in extras:
        control = mount / ".control"
        control.mkdir(exist_ok=True, mode=0o700)
        if path.name.startswith(".rate-limits.sqlite3"):
            target = control / path.name
        else:
            (control / "legacy-task-root").mkdir(exist_ok=True, mode=0o700)
            target = control / "legacy-task-root" / path.name
        if not target.exists():
            shutil.copy2(path, target)
        if path.read_bytes() != target.read_bytes():
            raise RuntimeError("Legacy metadata copy verification failed; source retained")
    # Verify all retained tasks before deleting any source task. Interrupted
    # copies are safe to repeat; installed scientific assets are never traversed.
    for item in items:
        shutil.rmtree(source / item["task_id"])
        result["expired_removed" if item["expired"] else "migrated"].append(item["task_id"])
    for path in extras:
        path.unlink()
    # Old checkpoints can contain server-authored absolute input paths. Keep
    # those paths valid through a directory alias; resolved paths still fall
    # inside the same quota volume and pass the task-scope validation.
    source.rmdir()
    source.symlink_to(mount / "tasks", target_is_directory=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / ".local/share/gmxbuilder/web-storage"
    )
    parser.add_argument("--source-tasks", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--wait-mounted", type=Path)
    parser.add_argument("--install-web-recovery", action="store_true")
    args = parser.parse_args()
    if args.wait_mounted:
        from gmxbuilder.web.storage_health import probe_storage

        for _ in range(100):
            if probe_storage(args.wait_mounted)["ready"]:
                return
            time.sleep(0.1)
        raise RuntimeError("Quota filesystem did not mount within 10 seconds")
    if args.install_web_recovery:
        from gmxbuilder.web.recovery import install_recovery

        install_recovery(args.root)
    policy = ResourcePolicy.from_environment()
    if args.inventory:
        if not args.source_tasks:
            parser.error("--inventory requires --source-tasks")
        print(json.dumps(inventory(args.source_tasks, policy), indent=2))
    if args.install:
        print(install_volume(args.root, policy))
    if args.migrate:
        if not args.source_tasks:
            parser.error("--migrate requires --source-tasks")
        print(json.dumps(migrate(args.source_tasks, args.root / "mounted", policy), indent=2))


if __name__ == "__main__":
    main()

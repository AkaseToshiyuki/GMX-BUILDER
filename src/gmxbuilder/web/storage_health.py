"""Read-only quota-volume readiness shared by Web and its recovery supervisor."""

import json
import os
import shutil
from pathlib import Path


def probe_storage(mount: Path, expected: dict | None = None) -> dict:
    try:
        mount = mount.absolute()
        found = False
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, _separator, right = line.partition(" - ")
            fields = left.split()
            if len(fields) > 4 and fields[4].replace("\\040", " ") == str(mount):
                found = right.startswith("fuse") and "gmxbuilder-quota" in right
        if not found:
            return {"ready": False, "reason": "storage-not-mounted"}
        quota = json.loads(os.getxattr(mount, "user.gmxbuilder.quota"))
        if (
            not isinstance(quota, dict)
            or set(quota) != {"total_bytes", "task_bytes"}
            or any(type(value) is not int or value <= 0 for value in quota.values())
            or quota["task_bytes"] > quota["total_bytes"]
        ):
            return {"ready": False, "reason": "storage-policy-invalid"}
        if expected is not None and quota != expected:
            return {"ready": False, "reason": "storage-policy-mismatch"}
        health = json.loads(os.getxattr(mount, "user.gmxbuilder.health"))
        if health != {"write_fault": False}:
            return {"ready": False, "reason": "storage-write-fault"}
        if shutil.disk_usage(mount).free < 65536:
            return {"ready": False, "reason": "storage-full"}
        return {"ready": True, "reason": "ready"}
    except (OSError, ValueError, TypeError):
        return {"ready": False, "reason": "storage-unavailable"}

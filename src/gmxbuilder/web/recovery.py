"""Bounded recovery of a Web service whose explicitly online target remains active."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gmxbuilder.web.resource_setup import systemd_quote
from gmxbuilder.web.storage_health import probe_storage


def recovery_units(root: Path, executable: str = sys.executable) -> dict[str, str]:
    command = " ".join(
        systemd_quote(value)
        for value in (executable, "-m", "gmxbuilder.web.recovery", "--mount", root / "mounted")
    )
    return {
        "gmxbuilder-online.target": (
            "[Unit]\nDescription=GMXBUILDER intended online state\n"
            "DefaultDependencies=no\n"
            "Wants=gmxbuilder-storage.service gmxbuilder.service gmxbuilder-recovery.timer\n"
            "After=gmxbuilder-storage.service\n\n[Install]\nWantedBy=default.target\n"
        ),
        "gmxbuilder.service.d/90-recovery.conf": (
            "[Unit]\nRequisite=gmxbuilder-online.target\n"
            "After=gmxbuilder-online.target gmxbuilder-storage.service\n"
            "PartOf=gmxbuilder-online.target\nBindsTo=gmxbuilder-storage.service\n"
        ),
        "gmxbuilder-recovery.timer": (
            "[Unit]\nDescription=Check GMXBUILDER recovery while intended online\n"
            # A timer normally starts before timers.target (and thus basic.target).
            # This timer starts after the online target, which waits for storage
            # and basic.target; the default ordering would form a boot cycle.
            "DefaultDependencies=no\n"
            "PartOf=gmxbuilder-online.target\nAfter=gmxbuilder-online.target\n"
            "Requisite=gmxbuilder-online.target\n\n"
            "[Timer]\nOnActiveSec=5s\nOnUnitActiveSec=10s\nAccuracySec=1s\n"
            "Unit=gmxbuilder-recovery.service\n"
        ),
        "gmxbuilder-recovery.service": (
            "[Unit]\nDescription=Recover GMXBUILDER after quota storage returns\n"
            "PartOf=gmxbuilder-online.target\nAfter=gmxbuilder-online.target\n"
            "Requisite=gmxbuilder-online.target\n\n[Service]\nType=oneshot\n"
            f"ExecStart={command}\nTimeoutStartSec=8s\nTimeoutStopSec=2s\n"
            "RuntimeDirectory=gmxbuilder-recovery\nRuntimeDirectoryMode=0700\n"
            "RuntimeDirectoryPreserve=yes\nUMask=0077\n"
            "Environment=PYTHONDONTWRITEBYTECODE=1\n"
            "NoNewPrivileges=true\nMemoryMax=128M\n"
            "StandardOutput=journal\nStandardError=journal\n"
        ),
    }


def install_recovery(root: Path, *, unit_directory: Path | None = None, activate=True):
    destination = unit_directory or Path.home() / ".config/systemd/user"
    for name, content in recovery_units(root.absolute()).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if activate:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "gmxbuilder-online.target"], check=True
        )


def _active(unit: str) -> bool:
    return (
        subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], timeout=2).returncode
        == 0
    )


def recover(mount: Path, state_path: Path) -> dict:
    if not _active("gmxbuilder-online.target"):
        return {"action": "none", "reason": "maintenance"}
    if not _active("gmxbuilder-storage.service"):
        return {"action": "none", "reason": "waiting-for-storage"}
    readiness = probe_storage(mount)
    if not readiness["ready"]:
        return {"action": "none", "reason": readiness["reason"]}
    # A live process can still have a latched, unwritable quota volume.
    # Report storage faults even while Web is active; never clear them here.
    if _active("gmxbuilder.service"):
        if state_path.exists():
            state_path.unlink()
            print("GMXBUILDER Web recovery complete", flush=True)
        return {"action": "none", "reason": "web-active"}
    try:
        previous = json.loads(state_path.read_text())
        attempts = max(0, int(previous["attempts"]))
        next_attempt = float(previous["next_attempt"])
    except (OSError, ValueError, KeyError, TypeError):
        attempts, next_attempt = 0, 0
    now = time.time()
    if now < next_attempt:
        return {"action": "none", "reason": "backoff"}
    # Requisite on the Web unit prevents a recovery racing maintenance from
    # bringing the online target back up. Never start that target from here.
    if not _active("gmxbuilder-online.target"):
        return {"action": "none", "reason": "maintenance"}
    state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"attempts": attempts + 1, "next_attempt": now + min(300, 3 * 2 ** min(attempts, 7))}
        )
    )
    temporary.replace(state_path)
    result = subprocess.run(
        ["systemctl", "--user", "start", "--no-block", "gmxbuilder.service"], timeout=2
    )
    print(
        "GMXBUILDER Web recovery requested"
        if result.returncode == 0
        else "GMXBUILDER Web recovery request failed",
        flush=True,
    )
    return {"action": "start-web", "accepted": result.returncode == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount", type=Path, required=True)
    args = parser.parse_args()
    state_dir = Path(
        os.environ.get("RUNTIME_DIRECTORY", f"/run/user/{os.getuid()}/gmxbuilder-recovery")
    )
    result = recover(args.mount, state_dir / "attempt.json")
    status_path = state_dir / "status.json"
    try:
        previous = json.loads(status_path.read_text())
    except (OSError, ValueError):
        previous = None
    if result != previous:
        print(json.dumps(result, sort_keys=True), flush=True)
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        status_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()

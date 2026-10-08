"""Standard-library installer lifecycle checks; never execute an old launcher."""

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def assert_idle(config: Path) -> None:
    active = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=active,activating",
            "gmxbuilder-op-*.service",
            "gmxbuilder-local-v4.service",
            "gmxbuilder-v4-library.service",
            "--no-legend",
            "--plain",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if active.stdout.strip():
        raise RuntimeError("Web or offline library work is active; installation must wait")
    running = (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "gmxbuilder.service"], check=False
        ).returncode
        == 0
    )
    if not running:
        return
    runner = config / "run-local.sh"
    text = runner.read_text() if runner.is_file() else ""
    match = re.search(r"--port\s+(\d+)", text)
    port = match.group(1) if match else "7788"
    url = os.environ.get("GMXBUILDER_HEALTH_URL", f"http://127.0.0.1:{port}/health")
    with urllib.request.urlopen(url, timeout=5) as response:
        health = json.load(response)
    if health.get("installation_lock_supported") is not True:
        raise RuntimeError(
            "The running older Web service lacks installer admission locking. "
            "Stop it after its jobs finish before upgrading with this installer."
        )
    keys = ("builds_active", "builds_queued", "operations_active", "operations_queued")
    if any(health.get(key) != 0 for key in keys):
        raise RuntimeError("Web work is active or health is incomplete; installation must wait")


def admin_token(config: Path, *, rotate: bool = False) -> str:
    config.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = config / "admin-token"
    token = ""
    if not rotate:
        if target.is_file():
            token = target.read_text().strip()
        elif (config / "run-local.sh").is_file():
            for line in (config / "run-local.sh").read_text().splitlines():
                if line.startswith("export GMXBUILDER_ADMIN_TOKEN="):
                    parts = shlex.split(line.partition("=")[2])
                    if len(parts) == 1:
                        token = parts[0]
    if not token:
        token = secrets.token_urlsafe(32)
    fd, temporary = tempfile.mkstemp(prefix=".admin-token-", dir=config)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(token + "\n")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("idle", "token"))
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/gmxbuilder")
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()
    if args.action == "idle":
        assert_idle(args.config)
    else:
        print(admin_token(args.config, rotate=args.rotate))


if __name__ == "__main__":
    main()

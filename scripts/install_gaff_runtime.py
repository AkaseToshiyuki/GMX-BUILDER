#!/usr/bin/env python3
"""Install the optional GAFF2 toolchain into GMXBUILDER's managed user runtime."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request


MICROMAMBA_VERSION = "2.8.1-0"
MICROMAMBA_ASSETS = {
    "x86_64": (
        "micromamba-linux-64",
        "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82",
    ),
    "aarch64": (
        "micromamba-linux-aarch64",
        "e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a",
    ),
}
GAFF_PACKAGES = ("ambertools=26.0", "acpype=2023.10.27", "openbabel=3.1.1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(prefix: Path) -> bool:
    return all((prefix / "bin" / name).is_file() for name in ("acpype", "antechamber", "tleap"))


def install(prefix: Path, runtime_root: Path) -> Path:
    if _is_complete(prefix):
        return prefix
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    if machine == "arm64":
        machine = "aarch64"
    if machine not in MICROMAMBA_ASSETS:
        raise RuntimeError(f"Automatic GAFF2 installation does not support Linux {machine}")

    asset, expected = MICROMAMBA_ASSETS[machine]
    tool_dir = runtime_root / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    micromamba = tool_dir / "micromamba"
    if not micromamba.exists() or _sha256(micromamba) != expected:
        micromamba.unlink(missing_ok=True)
        url = (
            "https://github.com/mamba-org/micromamba-releases/releases/download/"
            f"{MICROMAMBA_VERSION}/{asset}"
        )
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, micromamba.open(
            "wb"
        ) as out:
            shutil.copyfileobj(response, out)
        if _sha256(micromamba) != expected:
            micromamba.unlink(missing_ok=True)
            raise RuntimeError("Micromamba checksum verification failed")
        micromamba.chmod(0o700)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(micromamba),
        "create",
        "--yes",
        "--prefix",
        str(prefix),
        "--channel",
        "conda-forge",
        "--strict-channel-priority",
        *GAFF_PACKAGES,
    ]
    env = dict(os.environ)
    env["MAMBA_ROOT_PREFIX"] = str(runtime_root / "mamba-root")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    if not _is_complete(prefix):
        raise RuntimeError("GAFF2 environment is incomplete after installation")
    return prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path.home() / ".local" / "share" / "gmxbuilder" / "gaff-env",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".local" / "share" / "gmxbuilder" / "runtime",
    )
    args = parser.parse_args()
    try:
        prefix = install(args.prefix.expanduser(), args.runtime_root.expanduser())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

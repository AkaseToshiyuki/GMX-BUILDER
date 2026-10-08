#!/usr/bin/env python3
"""Install the optional GAFF2 toolchain into GMXBUILDER's managed user runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_download_spec = importlib.util.spec_from_file_location(
    "_installer_download", Path(__file__).with_name("installer_download.py")
)
_download = importlib.util.module_from_spec(_download_spec)
_download_spec.loader.exec_module(_download)

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
GAFF_PACKAGES = ("ambertools=24.8", "acpype=2023.10.27", "openbabel=3.1.1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_archive(record: dict, archives: Path) -> str:
    """Verify and atomically publish one locked package independently."""
    filename = record["url"].rsplit("/", 1)[-1]
    archive = archives / filename
    # The common helper raises checksum verification failed without publishing
    # partial or oversized data at the cache's final path.
    _download.download_verified(
        record["url"], archive, record["sha256"], opener=urllib.request.urlopen
    )
    return archive.resolve().as_uri() + "#" + record["sha256"]


def _is_complete(prefix: Path, lock: dict) -> bool:
    if not all(
        (prefix / "bin" / name).is_file()
        for name in ("acpype", "antechamber", "tleap", "obabel", "parmchk2")
    ):
        return False
    try:
        installed = {
            record["name"]: record
            for path in (prefix / "conda-meta").glob("*.json")
            for record in [json.loads(path.read_text())]
        }
        return all(
            all(
                installed.get(record["name"], {}).get(key) == record[key]
                for key in ("version", "build", "sha256")
            )
            for record in lock["packages"]
        ) and len(installed) == len(lock["packages"])
    except (OSError, ValueError, KeyError):
        return False


def install(prefix: Path, runtime_root: Path) -> Path:
    import fcntl

    runtime_root.mkdir(parents=True, exist_ok=True)
    with (runtime_root / ".gaff-install.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _install_locked(prefix, runtime_root)


def _install_locked(prefix: Path, runtime_root: Path) -> Path:
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    if machine != "x86_64":
        raise RuntimeError(
            "No reviewed GAFF artifact lock for this platform; configure an independently "
            "validated GAFF runtime instead of an unpinned automatic installation"
        )
    lock = json.loads(Path(__file__).with_name("gaff-linux-64.lock.json").read_text())
    if _is_complete(prefix, lock):
        return prefix
    if prefix.exists():
        raise RuntimeError(
            "Existing GAFF environment differs from the reviewed artifact lock. "
            "Install into a new --prefix; existing scientific runtimes are never overwritten."
        )

    asset, expected = MICROMAMBA_ASSETS[machine]
    tool_dir = runtime_root / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    micromamba = tool_dir / "micromamba"
    if not micromamba.exists() or _sha256(micromamba) != expected:
        url = (
            "https://github.com/mamba-org/micromamba-releases/releases/download/"
            f"{MICROMAMBA_VERSION}/{asset}"
        )
        print(f"Downloading {url}", flush=True)
        _download.download_verified(
            url, micromamba, expected, max_bytes=128 * 1024**2, opener=urllib.request.urlopen
        )
        micromamba.chmod(0o700)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Conda embeds its prefix: prepare at a new final path, never in place.
    with tempfile.TemporaryDirectory(prefix="gaff-lock-", dir=runtime_root) as work:
        explicit = Path(work) / "explicit.txt"
        archives = runtime_root / "gaff-artifacts"
        archives.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(lock["packages"])))) as pool:
            urls = list(pool.map(lambda record: _fetch_archive(record, archives), lock["packages"]))
        explicit.write_text("@EXPLICIT\n" + "\n".join(urls) + "\n")
        command = [
            str(micromamba),
            "--no-rc",
            "create",
            "--offline",
            "--yes",
            "--prefix",
            str(prefix),
            "--file",
            str(explicit),
        ]
        env = dict(os.environ)
        env["MAMBA_ROOT_PREFIX"] = str(runtime_root / "mamba-root")
        try:
            subprocess.run(command, check=True, env=env)
            if not _is_complete(prefix, lock):
                raise RuntimeError("GAFF environment does not match its artifact lock")
            for binary in ("antechamber", "tleap", "parmchk2", "obabel"):
                if not os.access(prefix / "bin" / binary, os.X_OK):
                    raise RuntimeError(f"GAFF executable is not runnable: {binary}")
        except BaseException:
            if prefix.exists():
                prefix.rename(prefix.with_name(prefix.name + ".failed-" + uuid.uuid4().hex))
            raise
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

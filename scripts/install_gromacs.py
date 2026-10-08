#!/usr/bin/env python3
"""Install a pinned, verified GROMACS runtime from its official source archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid
from pathlib import Path

_download_spec = importlib.util.spec_from_file_location(
    "_installer_download", Path(__file__).with_name("installer_download.py")
)
_download = importlib.util.module_from_spec(_download_spec)
_download_spec.loader.exec_module(_download)

VERSION = "2026.3"
SOURCE_URL = f"https://ftp.gromacs.org/gromacs/gromacs-{VERSION}.tar.gz"
SOURCE_SHA256 = "1094b7bbc6a3960223827114626657110b40096cdf9598a727935fc84ebf8aa0"
FFTW_URL = "https://www.fftw.org/fftw-3.3.10.tar.gz"
# Verified from the official HTTPS archive; retain GROMACS' FFTW version.
FFTW_SHA256 = "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = (destination / member.name).resolve()
            if (
                destination_resolved not in member_path.parents
                and member_path != destination_resolved
            ):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported archive member type: {member.name}")
        kwargs = (
            {"filter": "data"}
            if "filter" in inspect.signature(handle.extractall).parameters
            else {}
        )
        handle.extractall(destination, **kwargs)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _fetch_verified(url: str, path: Path, expected: str) -> None:
    _download.download_verified(url, path, expected, opener=urllib.request.urlopen)


def install(target_root: Path, cache_root: Path, jobs: int, force_cpu: bool) -> Path:
    import fcntl

    target_root = target_root.resolve()
    cache_root = cache_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    with (target_root / ".gromacs-install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _install_locked(target_root, cache_root, jobs, force_cpu)


def _install_locked(target_root: Path, cache_root: Path, jobs: int, force_cpu: bool) -> Path:
    alias = target_root / f"gromacs-{VERSION}"
    executable = alias / "bin/gmx"
    cuda = not force_cpu and shutil.which("nvcc") is not None
    expected = {
        "version": VERSION,
        "source_sha256": SOURCE_SHA256,
        "fftw_sha256": FFTW_SHA256,
        "cuda": cuda,
    }
    try:
        manifest = json.loads((alias / "gmxbuilder-install.json").read_text())
        if all(manifest.get(k) == v for k, v in expected.items()) and (
            executable.is_file()
            and os.access(executable, os.X_OK)
            and _sha256(executable) == manifest.get("executable_sha256")
        ):
            return executable
    except (OSError, ValueError):
        pass
    cmake = shutil.which("cmake")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cmake or not compiler:
        raise RuntimeError("Cannot build GROMACS: cmake and a C++17 compiler are required")
    archive = cache_root / f"gromacs-{VERSION}.tar.gz"
    fftw = cache_root / "fftw-3.3.10.tar.gz"
    _fetch_verified(SOURCE_URL, archive, SOURCE_SHA256)
    _fetch_verified(FFTW_URL, fftw, FFTW_SHA256)
    versions = target_root / "versions"
    versions.mkdir(exist_ok=True)
    # Build with its final immutable prefix; never overwrite a running runtime.
    prefix = versions / f"gromacs-{VERSION}-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-gromacs-", dir=cache_root) as work:
        source_parent = Path(work)
        _safe_extract(archive, source_parent)
        source = source_parent / f"gromacs-{VERSION}"
        build = source_parent / "build"
        configure = [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGMX_BUILD_OWN_FFTW=ON",
            f"-DGMX_BUILD_OWN_FFTW_URL={fftw}",
            "-DGMX_MPI=OFF",
            "-DGMX_THREAD_MPI=ON",
            "-DGMX_OPENMP=ON",
            f"-DGMX_GPU={'CUDA' if cuda else 'OFF'}",
            "-DGMX_BUILD_TESTS=OFF",
        ]
        _run(configure)
        _run([cmake, "--build", str(build), "--parallel", str(jobs)])
        _run([cmake, "--install", str(build)])
    built = prefix / "bin/gmx"
    probe = subprocess.run([str(built), "--version"], capture_output=True, text=True, timeout=30)
    if probe.returncode or f"GROMACS version:    {VERSION}" not in probe.stdout:
        # Whitespace varies across builds; still require the exact version line.
        import re

        if probe.returncode or not re.search(
            rf"GROMACS version:\s+{re.escape(VERSION)}(?:\s|$)", probe.stdout
        ):
            raise RuntimeError("New GROMACS executable failed its version check")
    (prefix / "gmxbuilder-install.json").write_text(
        json.dumps(
            {
                **expected,
                "executable_sha256": _sha256(built),
                "toolchain": {
                    "cmake_sha256": _sha256(Path(cmake).resolve()),
                    "compiler_sha256": _sha256(Path(compiler).resolve()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    link = alias.with_name(alias.name + ".new-" + uuid.uuid4().hex)
    link.symlink_to(prefix, target_is_directory=True)
    retired = None
    if alias.exists() and not alias.is_symlink():
        retired = alias.with_name(alias.name + ".retired-" + uuid.uuid4().hex)
        alias.rename(retired)
    try:
        link.replace(alias)
    except OSError:
        if retired is not None:
            retired.rename(alias)
        raise
    finally:
        link.unlink(missing_ok=True)
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path.home() / ".local" / "share" / "gmxbuilder" / "runtime",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "gmxbuilder" / "downloads",
    )
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        executable = install(
            args.target_root.expanduser(),
            args.cache_root.expanduser(),
            args.jobs,
            args.force_cpu,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

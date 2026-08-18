#!/usr/bin/env python3
"""Install a pinned, verified GROMACS runtime from its official source archive."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request


VERSION = "2026.3"
SOURCE_URL = f"https://ftp.gromacs.org/gromacs/gromacs-{VERSION}.tar.gz"
SOURCE_SHA256 = "1094b7bbc6a3960223827114626657110b40096cdf9598a727935fc84ebf8aa0"


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
            if member.issym() or member.islnk():
                link_path = (member_path.parent / member.linkname).resolve()
                if (
                    destination_resolved not in link_path.parents
                    and link_path != destination_resolved
                ):
                    raise RuntimeError(f"Unsafe archive link: {member.name}")
        handle.extractall(destination)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install(target_root: Path, cache_root: Path, jobs: int, force_cpu: bool) -> Path:
    prefix = target_root / f"gromacs-{VERSION}"
    executable = prefix / "bin" / "gmx"
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable

    cmake = shutil.which("cmake")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cmake or not compiler:
        missing = []
        if not cmake:
            missing.append("cmake")
        if not compiler:
            missing.append("a C++17 compiler")
        raise RuntimeError(
            "Cannot build GROMACS automatically because " + " and ".join(missing) + " is missing"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"gromacs-{VERSION}.tar.gz"
    if not archive.exists() or _sha256(archive) != SOURCE_SHA256:
        archive.unlink(missing_ok=True)
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, archive.open(
            "wb"
        ) as out:
            shutil.copyfileobj(response, out)
    actual = _sha256(archive)
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"GROMACS source checksum mismatch: expected {SOURCE_SHA256}, received {actual}"
        )

    source_parent = target_root / "source"
    source_parent.mkdir(parents=True, exist_ok=True)
    source = source_parent / f"gromacs-{VERSION}"
    if not source.exists():
        _safe_extract(archive, source_parent)
    build = source / "build-gmxbuilder"
    build.mkdir(parents=True, exist_ok=True)

    cuda = not force_cpu and shutil.which("nvcc") is not None
    configure = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGMX_BUILD_OWN_FFTW=ON",
        "-DGMX_MPI=OFF",
        "-DGMX_THREAD_MPI=ON",
        "-DGMX_OPENMP=ON",
        f"-DGMX_GPU={'CUDA' if cuda else 'OFF'}",
        "-DGMX_BUILD_TESTS=OFF",
    ]
    _run(configure)
    _run([cmake, "--build", str(build), "--parallel", str(jobs)])
    _run([cmake, "--install", str(build)])
    if not executable.is_file():
        raise RuntimeError(f"GROMACS installation did not create {executable}")
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

#!/usr/bin/env python3
"""Hydrate the release lipid archive without requiring Git LFS.

Public source archives and Git checkouts can contain a small Git LFS pointer.
This bootstrap reads the release manifest, downloads the immutable-by-digest
payload from the checkout origin LFS (or public media for source archives),
verifies its exact size and
SHA-256 digest, and atomically replaces the pointer.  Runtime installation
performs the same verification again before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "src/gmxbuilder/data/prebuilt_assets/manifest.json"
USER_AGENT = "GMXBUILDER prebuilt-asset bootstrap/1"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_payload(path: Path, *, expected_size: int, expected_digest: str) -> bool:
    return (
        path.is_file() and path.stat().st_size == expected_size and sha256(path) == expected_digest
    )


def _checkout_lfs_object(archive: Path, expected_digest: str, expected_size: int) -> bool:
    """Fetch this checkout's one asset through origin, including private LFS auth.

    Source archives without Git/LFS use the verified public media URL instead.
    Never pull unrelated objects or run a lipid preparation job during install.
    """
    if not shutil.which("git") or not shutil.which("git-lfs"):
        return False
    try:

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(archive.parent), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            ).stdout.strip()

        root = Path(git("rev-parse", "--show-toplevel"))
        relative = archive.resolve().relative_to(root.resolve()).as_posix()
        remote = git("remote", "get-url", "origin")
        # A developer's configured public LFS URL must not redirect a private
        # checkout's asset fetch to another repository.
        override = (
            ["-c", f"lfs.url={remote.rstrip('/')}/info/lfs"]
            if remote.startswith("https://")
            else []
        )
        # Decode only the manifest's exact object. `lfs fetch` scans the whole
        # tree and can trigger unrelated lazy downloads in a partial clone.
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{expected_digest}\nsize {expected_size}\n"
        ).encode()
        environment = dict(os.environ)
        environment.pop("GIT_LFS_SKIP_SMUDGE", None)
        with tempfile.NamedTemporaryFile(
            dir=archive.parent, prefix=".lfs-", delete=False
        ) as output:
            temporary = Path(output.name)
            try:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        *override,
                        "-c",
                        "lfs.fetchinclude=",
                        "-c",
                        "lfs.fetchexclude=",
                        "lfs",
                        "smudge",
                        relative,
                    ],
                    input=pointer,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    env=environment,
                    check=True,
                    timeout=180,
                )
                output.flush()
                if not _is_payload(
                    temporary, expected_size=expected_size, expected_digest=expected_digest
                ):
                    return False
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        return True
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return False


def fetch(manifest_path: Path = DEFAULT_MANIFEST) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_name = str(manifest.get("archive", "")).strip()
    expected_digest = str(manifest.get("archive_sha256", "")).strip().lower()
    expected_size = int(manifest.get("archive_bytes", -1))
    url = str(manifest.get("download_url", "")).strip()
    if not archive_name or Path(archive_name).name != archive_name:
        raise RuntimeError("invalid prebuilt-asset archive name")
    if expected_size <= 0:
        raise RuntimeError("invalid prebuilt-asset size")
    if len(expected_digest) != 64 or any(c not in "0123456789abcdef" for c in expected_digest):
        raise RuntimeError("invalid prebuilt-asset SHA-256 digest")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("prebuilt-asset download URL must use HTTPS")

    archive = manifest_path.parent / archive_name
    if _is_payload(archive, expected_size=expected_size, expected_digest=expected_digest):
        return f"present: {archive_name}"
    if archive.is_file() and archive.stat().st_size > 1024:
        with archive.open("rb") as handle:
            prefix = handle.read(len(LFS_PREFIX))
        if not prefix.startswith(LFS_PREFIX):
            raise RuntimeError("existing prebuilt asset is not the expected payload or LFS pointer")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if _checkout_lfs_object(archive, expected_digest, expected_size):
        return f"downloaded from origin LFS: {archive_name}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_name}.", suffix=".download", dir=archive.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request, timeout=180) as response:
                if getattr(response, "status", 200) != 200:
                    raise RuntimeError(f"download returned HTTP {response.status}")
                remaining = expected_size
                while remaining:
                    block = response.read(min(1024 * 1024, remaining + 1))
                    if not block:
                        break
                    temporary.write(block)
                    remaining -= len(block)
                    if remaining < 0:
                        raise RuntimeError("prebuilt-asset download is larger than its manifest")
                if response.read(1):
                    raise RuntimeError("prebuilt-asset download is larger than its manifest")
            temporary.flush()
            if not _is_payload(
                temporary_path,
                expected_size=expected_size,
                expected_digest=expected_digest,
            ):
                raise RuntimeError("prebuilt-asset download failed size or SHA-256 verification")
            temporary_path.replace(archive)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return f"downloaded: {archive_name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(fetch(args.manifest.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Hydrate the release lipid archive without requiring Git LFS.

Public source archives and Git checkouts can contain a small Git LFS pointer.
This bootstrap reads the release manifest, downloads the immutable-by-digest
payload from the configured public media URL, verifies its exact size and
SHA-256 digest, and atomically replaces the pointer.  Runtime installation
performs the same verification again before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
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

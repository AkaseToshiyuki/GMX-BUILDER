"""Standard-library-only, bounded and atomic installer asset downloads."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

MAX_ASSET_BYTES = 1024**3
MAX_TRANSFER_SECONDS = 900
_NETWORK_OPENER = urllib.request.urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transfer(url, temporary, max_bytes, opener, seconds):
    deadline = time.monotonic() + seconds
    with opener(url, timeout=min(120, seconds)) as response, temporary.open("wb") as out:
        header = getattr(response, "headers", {}).get("Content-Length")
        try:
            announced = int(header)
        except (TypeError, ValueError):
            announced = None
        if announced is not None and (announced < 0 or announced > max_bytes):
            raise RuntimeError("Asset download exceeds its size budget")
        read = getattr(response, "read1", response.read)
        size = 0
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError("Asset download exceeded its total time budget")
            chunk = read(min(64 * 1024, max_bytes - size + 1))
            if time.monotonic() >= deadline:
                raise RuntimeError("Asset download exceeded its total time budget")
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError("Asset download exceeds its size budget")
            out.write(chunk)


def download_verified(url, path, expected, *, max_bytes=MAX_ASSET_BYTES, opener=None):
    """Bound the whole network process, including header/chunk/trailer parsing.

    Four concurrent GAFF transfers hold at most 4 GiB of temporary payload.
    A child timeout also cancels peers that continuously trickle protocol bytes.
    Injected openers are for deterministic offline transfer tests only.
    """
    path = Path(path)
    if path.is_file() and sha256(path) == expected:
        return path
    temporary = path.with_name(path.name + ".download-" + uuid.uuid4().hex)
    try:
        if opener is None or opener is _NETWORK_OPENER:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        url,
                        str(temporary),
                        str(max_bytes),
                        str(MAX_TRANSFER_SECONDS),
                    ],
                    timeout=MAX_TRANSFER_SECONDS,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Asset download exceeded its total time budget") from exc
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "Asset transfer failed")
        else:
            _transfer(url, temporary, max_bytes, opener, MAX_TRANSFER_SECONDS)
        if sha256(temporary) != expected:
            raise RuntimeError(f"Asset checksum verification failed: {url}")
        temporary.replace(path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        _transfer(
            sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), _NETWORK_OPENER, float(sys.argv[4])
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

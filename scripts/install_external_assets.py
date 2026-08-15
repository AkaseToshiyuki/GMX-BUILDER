#!/usr/bin/env python3
"""Install separately distributed force-field data from official sources."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("external_assets.json")
OVERLAYS = ROOT / "src" / "gmxbuilder" / "data" / "forcefield_overlays"
USER_AGENT = "GMXBUILDER external-asset bootstrap/1"
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_directory(path: Path, required: list[str]) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in required)


def validate_member(member: tarfile.TarInfo, expected_root: str) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != expected_root
        or member.issym()
        or member.islnk()
        or member.isdev()
    ):
        raise RuntimeError(f"unsafe archive member: {member.name}")


def download(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("external-asset download URL must use HTTPS")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"download returned HTTP {response.status}")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_ARCHIVE_BYTES:
            raise RuntimeError("external-asset archive exceeds the download limit")
        written = 0
        while block := response.read(1024 * 1024):
            written += len(block)
            if written > MAX_ARCHIVE_BYTES:
                raise RuntimeError("external-asset archive exceeds the download limit")
            output.write(block)


def apply_overlay(staging: Path, overlay_name: str) -> None:
    overlay = OVERLAYS / overlay_name
    if not overlay.is_dir():
        return
    for source in sorted(overlay.iterdir()):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"invalid overlay entry: {source}")
        shutil.copy2(source, staging / source.name)


def install_one(spec: dict, target_root: Path, *, force: bool = False) -> str:
    target = target_root / str(spec["target"])
    required = [str(name) for name in spec["required_files"]]
    if not force and validate_directory(target, required):
        return f"present: {spec['name']}"

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-assets-") as temporary:
        work = Path(temporary)
        archive = work / "asset.tgz"
        download(str(spec["url"]), archive)
        if sha256(archive) != str(spec["sha256"]):
            raise RuntimeError(f"SHA-256 verification failed for {spec['name']}")
        extract_root = work / "extract"
        extract_root.mkdir()
        with tarfile.open(archive, mode="r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                validate_member(member, str(spec["archive_root"]))
            # Every member is restricted above. Use the standard data filter
            # when the running Python provides it, while retaining 3.10
            # compatibility for bootstrap environments without that keyword.
            extract_kwargs = (
                {"filter": "data"}
                if "filter" in inspect.signature(handle.extractall).parameters
                else {}
            )
            handle.extractall(extract_root, members=members, **extract_kwargs)
        source = extract_root / str(spec["archive_root"])
        apply_overlay(source, str(spec.get("overlay", "")))
        if not validate_directory(source, required):
            raise RuntimeError(f"official archive is incomplete for {spec['name']}")

        target_root.mkdir(parents=True, exist_ok=True)
        replacement = target.with_name(target.name + ".installing")
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(source, replacement)
        if target.exists():
            shutil.rmtree(target)
        replacement.replace(target)
    return f"installed: {spec['name']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "src" / "gmxbuilder" / "data" / "forcefields",
        help="force-field directory populated before GMXBUILDER installation",
    )
    parser.add_argument("--force", action="store_true", help="replace existing assets")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported external-asset manifest schema")
    for spec in manifest.get("assets", []):
        print(install_one(spec, args.target.resolve(), force=args.force), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

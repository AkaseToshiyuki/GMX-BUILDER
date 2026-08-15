"""Install release-bundled lipid assets into writable user caches.

The release archive is immutable and Git-LFS managed.  Runtime code extracts
it once, without overwriting newer cache files, so normal simulation jobs do
not repeat GAFF2 parameterization or explicit-solvent lipid equilibration.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Iterator

from gmxbuilder.modules.membrane.equilibrated_library import (
    SCHEMA_VERSION as LIBRARY_SCHEMA_VERSION,
    EquilibratedLipidLibrary,
)


ASSET_SCHEMA_VERSION = 1
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "prebuilt_assets"
_MANIFEST_PATH = _DATA_DIR / "manifest.json"


def _configured_roots() -> tuple[Path, Path]:
    lipid_root = Path(
        os.environ.get(
            "GMXBUILDER_LIPID_LIBRARY",
            Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated",
        )
    ).expanduser()
    gaff_root = Path(
        os.environ.get(
            "GMXBUILDER_GAFF_CACHE",
            Path.home() / ".cache" / "gmxbuilder" / "gaff2",
        )
    ).expanduser()
    return lipid_root, gaff_root


def _read_manifest(path: Path = _MANIFEST_PATH) -> dict:
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The prebuilt lipid asset manifest is missing from this installation"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The prebuilt lipid asset manifest is invalid") from exc
    if manifest.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise RuntimeError("Unsupported prebuilt lipid asset manifest schema")
    if manifest.get("library_schema_version") != LIBRARY_SCHEMA_VERSION:
        raise RuntimeError("Prebuilt lipid assets do not match the current strict-library schema")
    filename = str(manifest.get("archive", "")).strip()
    digest = str(manifest.get("archive_sha256", "")).strip().lower()
    if not filename or Path(filename).name != filename:
        raise RuntimeError("The prebuilt lipid asset archive name is invalid")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("The prebuilt lipid asset checksum is invalid")
    return manifest


def _archive_path(manifest: dict, manifest_path: Path = _MANIFEST_PATH) -> Path:
    return manifest_path.parent / str(manifest["archive"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(path: Path, manifest: dict) -> None:
    if not path.is_file():
        raise RuntimeError(
            "Prebuilt lipid assets are absent. From a source checkout, run "
            "`python3 scripts/fetch_prebuilt_assets.py` before installation."
        )
    with path.open("rb") as handle:
        prefix = handle.read(80)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            "Prebuilt lipid assets are still a Git LFS pointer. Run "
            "`python3 scripts/fetch_prebuilt_assets.py`, then retry."
        )
    expected_size = int(manifest.get("archive_bytes", -1))
    if expected_size >= 0 and path.stat().st_size != expected_size:
        raise RuntimeError("Prebuilt lipid asset archive size does not match its manifest")
    if _sha256(path) != manifest["archive_sha256"]:
        raise RuntimeError("Prebuilt lipid asset archive checksum verification failed")


def _marker_path(root: Path) -> Path:
    return root / ".gmxbuilder-prebuilt-assets.json"


def _marker_matches(root: Path, manifest: dict) -> bool:
    try:
        marker = json.loads(_marker_path(root).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("schema_version") == ASSET_SCHEMA_VERSION
        and marker.get("asset_version") == manifest.get("asset_version")
        and marker.get("archive_sha256") == manifest.get("archive_sha256")
    )


def _write_marker(root: Path, manifest: dict, installed_files: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "asset_version": manifest["asset_version"],
        "archive_sha256": manifest["archive_sha256"],
        "installed_files": int(installed_files),
    }
    temporary = _marker_path(root).with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    temporary.replace(_marker_path(root))


@contextmanager
def _installation_lock(lipid_root: Path, gaff_root: Path) -> Iterator[None]:
    common = Path(
        os.path.commonpath(
            [
                str(lipid_root.resolve(strict=False)),
                str(gaff_root.resolve(strict=False)),
            ]
        )
    )
    if str(common) == os.path.sep:
        common = Path.home() / ".cache" / "gmxbuilder"
    common.mkdir(parents=True, exist_ok=True)
    lock_path = common / ".prebuilt-assets.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in {"lipid_equilibrated", "gaff2"}
        or member.issym()
        or member.islnk()
        or member.isdev()
    ):
        raise RuntimeError(f"Unsafe path in prebuilt lipid asset archive: {member.name}")


def _merge_tree(source: Path, destination: Path) -> int:
    installed = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise RuntimeError(f"Unsupported prebuilt asset entry: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        temporary = target.with_suffix(target.suffix + ".prebuilt.tmp")
        shutil.copy2(path, temporary)
        temporary.replace(target)
        installed += 1
    return installed


def _validate_staging(staging: Path, manifest: dict) -> None:
    """Reject a hash-valid archive whose scientific entries are unusable."""
    lipid_root = staging / "lipid_equilibrated"
    metadata_files = sorted(lipid_root.glob("*/*/metadata.json"))
    expected_lipids = int(manifest["contents"]["strict_library_entries"])
    if len(metadata_files) != expected_lipids:
        raise RuntimeError("Prebuilt lipid asset entry count does not match its manifest")

    library = EquilibratedLipidLibrary([lipid_root])
    for metadata_path in metadata_files:
        try:
            metadata = json.loads(metadata_path.read_text())
            name = metadata_path.parent.name
            force_field = str(metadata["force_field"])
            lipid_ff = str(metadata["lipid_ff"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid strict lipid metadata in release archive: "
                f"{metadata_path.relative_to(staging)}"
            ) from exc
        if library.inspect(name, force_field, lipid_ff) is None:
            raise RuntimeError(
                f"Prebuilt lipid entry is incompatible with the current runtime: "
                f"{metadata_path.relative_to(staging)}"
            )

    gaff_root = staging / "gaff2"
    gaff_entries = [path for path in gaff_root.iterdir()] if gaff_root.is_dir() else []
    expected_gaff = int(manifest["contents"]["gaff2_cache_entries"])
    if len([path for path in gaff_entries if path.is_dir()]) != expected_gaff:
        raise RuntimeError("Prebuilt GAFF2 cache entry count does not match its manifest")


def _remove_stale_lipid_entries(staging: Path, destination: Path) -> int:
    """Replace only existing strict entries that the current runtime rejects."""
    removed = 0
    library = EquilibratedLipidLibrary([destination])
    for metadata_path in sorted((staging / "lipid_equilibrated").glob("*/*/metadata.json")):
        source_entry = metadata_path.parent
        relative = source_entry.relative_to(staging / "lipid_equilibrated")
        target = destination / relative
        if not target.exists() and not target.is_symlink():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
            valid = (
                library.inspect(
                    source_entry.name,
                    str(metadata["force_field"]),
                    str(metadata["lipid_ff"]),
                )
                is not None
            )
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if valid:
            continue
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
        removed += 1
    return removed


def install_prebuilt_assets(
    *,
    manifest_path: str | Path = _MANIFEST_PATH,
    lipid_root: str | Path | None = None,
    gaff_root: str | Path | None = None,
) -> dict:
    """Verify and install the bundled release archive without overwriting caches."""
    manifest_path = Path(manifest_path)
    manifest = _read_manifest(manifest_path)
    archive = _archive_path(manifest, manifest_path)
    default_lipid, default_gaff = _configured_roots()
    lipid_destination = Path(lipid_root).expanduser() if lipid_root else default_lipid
    gaff_destination = Path(gaff_root).expanduser() if gaff_root else default_gaff
    with _installation_lock(lipid_destination, gaff_destination):
        if _marker_matches(lipid_destination, manifest) and _marker_matches(
            gaff_destination, manifest
        ):
            return {
                "status": "ready",
                "asset_version": manifest["asset_version"],
                "installed_files": 0,
                "replaced_lipid_entries": 0,
                "lipid_root": str(lipid_destination),
                "gaff_root": str(gaff_destination),
            }
        _verify_archive(archive, manifest)
        with tempfile.TemporaryDirectory(prefix="gmxbuilder-assets-") as temporary:
            staging = Path(temporary)
            with tarfile.open(archive, mode="r:xz") as bundle:
                members = bundle.getmembers()
                for member in members:
                    _validate_member(member)
                    target = staging.joinpath(*PurePosixPath(member.name).parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise RuntimeError(f"Unsupported prebuilt asset entry: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Could not read prebuilt asset entry: {member.name}")
                    with source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            _validate_staging(staging, manifest)
            replaced_lipids = _remove_stale_lipid_entries(
                staging,
                lipid_destination,
            )
            installed_lipid = _merge_tree(
                staging / "lipid_equilibrated",
                lipid_destination,
            )
            installed_gaff = _merge_tree(staging / "gaff2", gaff_destination)
        _write_marker(lipid_destination, manifest, installed_lipid)
        _write_marker(gaff_destination, manifest, installed_gaff)
    return {
        "status": "installed",
        "asset_version": manifest["asset_version"],
        "installed_files": installed_lipid + installed_gaff,
        "replaced_lipid_entries": replaced_lipids,
        "lipid_root": str(lipid_destination),
        "gaff_root": str(gaff_destination),
    }


def ensure_prebuilt_assets() -> dict:
    """Install release assets on first use unless explicitly disabled."""
    disabled = os.environ.get("GMXBUILDER_PREBUILT_AUTO_INSTALL", "").strip().lower()
    if disabled in {"0", "false", "no", "off"}:
        return {"status": "disabled"}
    return install_prebuilt_assets()


def prebuilt_asset_status(
    *,
    manifest_path: str | Path = _MANIFEST_PATH,
    lipid_root: str | Path | None = None,
    gaff_root: str | Path | None = None,
) -> dict:
    """Report archive and cache-marker state without extracting files."""
    manifest_path = Path(manifest_path)
    manifest = _read_manifest(manifest_path)
    archive = _archive_path(manifest, manifest_path)
    default_lipid, default_gaff = _configured_roots()
    lipid_destination = Path(lipid_root).expanduser() if lipid_root else default_lipid
    gaff_destination = Path(gaff_root).expanduser() if gaff_root else default_gaff
    archive_state = "missing"
    if archive.is_file():
        with archive.open("rb") as handle:
            archive_state = (
                "lfs-pointer"
                if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/")
                else "present"
            )
    return {
        "status": (
            "ready"
            if (
                archive_state == "present"
                and _marker_matches(lipid_destination, manifest)
                and _marker_matches(gaff_destination, manifest)
            )
            else "not-installed"
        ),
        "asset_version": manifest["asset_version"],
        "archive": archive_state,
        "archive_bytes": manifest["archive_bytes"],
        "contents": manifest["contents"],
        "lipid_cache_ready": _marker_matches(lipid_destination, manifest),
        "gaff_cache_ready": _marker_matches(gaff_destination, manifest),
    }

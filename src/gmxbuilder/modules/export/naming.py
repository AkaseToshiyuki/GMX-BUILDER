"""Shared filename boundary for every generated export archive."""

from __future__ import annotations

from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError

MAX_SYSTEM_NAME_LENGTH = 64


def validated_system_name(value: object, *, default: str) -> str:
    """Return a portable archive stem with no path syntax."""
    candidate = default if value is None else value
    if not isinstance(candidate, str) or not candidate:
        raise ModuleConfigError("export.system_name must be a non-empty string")
    if len(candidate) > MAX_SYSTEM_NAME_LENGTH:
        raise ModuleConfigError(
            f"export.system_name must be at most {MAX_SYSTEM_NAME_LENGTH} characters"
        )
    if not all(character.isalnum() or character in "_-" for character in candidate):
        raise ModuleConfigError("export.system_name may contain only letters, numbers, '_' and '-'")
    return candidate


def confined_archive_path(output_dir: Path, system_name: str) -> Path:
    """Resolve an archive path and require it to remain a direct child."""
    root = output_dir.resolve()
    archive = (root / f"{system_name}.zip").resolve()
    if archive.parent != root:
        raise ModuleConfigError("Export archive path escapes the output directory")
    return archive


# Written beside a generated archive to name it as the authoritative one.
# Modification time is not a reliable answer: a workflow may legitimately write
# more than one archive into a directory, and two writes within a filesystem's
# timestamp resolution are indistinguishable.
ARCHIVE_MARKER_NAME = ".authoritative-archive"


def record_authoritative_archive(archive: Path) -> None:
    """Record *archive* as the archive that consumers should serve."""
    marker = archive.parent / ARCHIVE_MARKER_NAME
    marker.write_text(archive.name + "\n", encoding="utf-8")


def read_authoritative_archive(directory: Path) -> Path | None:
    """Return the recorded archive in *directory*, or None when unmarked.

    Returns None for a directory written before markers existed, so callers
    keep their previous behaviour for tasks that predate this.
    """
    marker = directory / ARCHIVE_MARKER_NAME
    try:
        name = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    archive = directory / name
    return archive if archive.is_file() else None

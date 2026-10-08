"""The served archive must be named, not inferred from a timestamp."""

from __future__ import annotations

import logging
import zipfile

import pytest

from gmxbuilder.modules.export.naming import (
    ARCHIVE_MARKER_NAME,
    read_authoritative_archive,
    record_authoritative_archive,
    validated_system_name,
)
from gmxbuilder.web.server_parts.input_limits import StructureInputLimits


def _archive(path):
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("README.txt", "x")
    return path


def test_recorded_archive_wins_over_a_newer_unrelated_one(tmp_path):
    """Two archives in one directory must not be resolved by modification time.

    A workflow may legitimately write more than one archive, and two writes
    within the filesystem's timestamp resolution are indistinguishable.
    """
    authoritative = _archive(tmp_path / "system.zip")
    record_authoritative_archive(authoritative)
    decoy = _archive(tmp_path / "zz-later.zip")
    # Make the decoy unambiguously newer than the recorded archive.
    import os

    stat = authoritative.stat()
    os.utime(decoy, ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))

    assert read_authoritative_archive(tmp_path) == authoritative


def test_unmarked_directory_reports_nothing_so_callers_keep_their_fallback(tmp_path):
    _archive(tmp_path / "system.zip")
    assert read_authoritative_archive(tmp_path) is None


@pytest.mark.parametrize("name", ["", "../escape.zip", "sub/dir.zip", ".hidden.zip"])
def test_marker_never_resolves_outside_its_directory(tmp_path, name):
    (tmp_path / ARCHIVE_MARKER_NAME).write_text(name, encoding="utf-8")
    assert read_authoritative_archive(tmp_path) is None


def test_marker_pointing_at_a_missing_file_is_ignored(tmp_path):
    (tmp_path / ARCHIVE_MARKER_NAME).write_text("gone.zip\n", encoding="utf-8")
    assert read_authoritative_archive(tmp_path) is None


def test_dry_and_solvated_exports_agree_on_the_default_archive_name():
    """The dry-bilayer subclass must not default to a different name.

    It runs after its parent in the same directory; a different default left
    the parent's solvated archive beside the dry one with nothing but the
    timestamp to separate them.
    """
    import inspect

    from gmxbuilder.modules.export.exporter import ExportModule
    from gmxbuilder.modules.pure_membrane.export import PureMembraneExportModule

    parent = inspect.getsource(ExportModule.run)
    child = inspect.getsource(PureMembraneExportModule.run)
    assert 'default="system"' in parent
    assert 'default="system"' in child
    assert validated_system_name(None, default="system") == "system"


def test_out_of_range_limit_is_clamped_and_reported(monkeypatch, caplog):
    """Falling back to the default silently gave neither the request nor the limit."""
    monkeypatch.setenv("GMXBUILDER_MAX_UPLOAD_MB", "128")
    with caplog.at_level(logging.WARNING, logger="gmxbuilder.web"):
        limits = StructureInputLimits.from_environment()

    assert limits.max_bytes == 64 * 1024 * 1024
    assert any("exceeds the supported maximum" in record.message for record in caplog.records)


def test_below_minimum_limit_is_clamped_and_reported(monkeypatch, caplog):
    monkeypatch.setenv("GMXBUILDER_MAX_STRUCTURE_ATOMS", "0")
    with caplog.at_level(logging.WARNING, logger="gmxbuilder.web"):
        limits = StructureInputLimits.from_environment()

    assert limits.max_atoms == 1
    assert any("below the minimum" in record.message for record in caplog.records)


def test_unparsable_limit_reports_and_uses_the_default(monkeypatch, caplog):
    monkeypatch.setenv("GMXBUILDER_MAX_UPLOAD_MB", "plenty")
    with caplog.at_level(logging.WARNING, logger="gmxbuilder.web"):
        limits = StructureInputLimits.from_environment()

    assert limits.max_bytes == 32 * 1024 * 1024
    assert any("is not an integer" in record.message for record in caplog.records)

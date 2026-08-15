"""Regression tests for the pre-environment external-asset bootstrap."""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_external_assets.py"
SPEC = importlib.util.spec_from_file_location("gmxbuilder_external_assets", SCRIPT)
assert SPEC and SPEC.loader
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def test_rejects_archive_path_traversal():
    member = tarfile.TarInfo("bundle/../outside")
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        installer.validate_member(member, "bundle")


def test_installs_verified_archive_and_is_idempotent(tmp_path, monkeypatch):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, content in {
            "bundle/forcefield.itp": b"[ defaults ]\n",
            "bundle/ffbonded.itp": b"[ bondtypes ]\n",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            handle.addfile(member, BytesIO(content))

    monkeypatch.setattr(installer, "OVERLAYS", tmp_path / "no-overlays")
    monkeypatch.setattr(
        installer,
        "download",
        lambda _url, destination: destination.write_bytes(archive.read_bytes()),
    )
    spec = {
        "name": "test asset",
        "target": "testff",
        "archive_root": "bundle",
        "url": "https://example.invalid/test.tgz",
        "sha256": installer.sha256(archive),
        "required_files": ["forcefield.itp", "ffbonded.itp"],
    }
    target = tmp_path / "target"
    assert installer.install_one(spec, target).startswith("installed:")
    assert (target / "testff" / "forcefield.itp").is_file()
    assert installer.install_one(spec, target).startswith("present:")


def test_checksum_mismatch_stops_install(tmp_path, monkeypatch):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        content = b"data"
        member = tarfile.TarInfo("bundle/forcefield.itp")
        member.size = len(content)
        handle.addfile(member, BytesIO(content))
    monkeypatch.setattr(
        installer,
        "download",
        lambda _url, destination: destination.write_bytes(archive.read_bytes()),
    )
    spec = {
        "name": "test asset",
        "target": "testff",
        "archive_root": "bundle",
        "url": "https://example.invalid/test.tgz",
        "sha256": "0" * 64,
        "required_files": ["forcefield.itp"],
    }
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        installer.install_one(spec, tmp_path / "target")

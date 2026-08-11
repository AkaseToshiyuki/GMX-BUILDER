import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from gmxbuilder.runtime.prebuilt_assets import (
    install_prebuilt_assets,
    prebuilt_asset_status,
)


def _fixture_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    lipid = source / "lipid_equilibrated" / "amber-gaff2" / "TEST"
    gaff = source / "gaff2" / "TEST-key"
    lipid.mkdir(parents=True)
    gaff.mkdir(parents=True)
    (lipid / "metadata.json").write_text('{"status":"ready"}')
    (lipid / "conf_0000.npz").write_bytes(b"fixture")
    (gaff / "metadata.json").write_text('{"name":"TEST"}')
    (gaff / "lipid.itp").write_text("[ atoms ]\n")
    archive_dir = tmp_path / "bundle"
    archive_dir.mkdir()
    archive = archive_dir / "assets.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        bundle.add(source / "lipid_equilibrated", arcname="lipid_equilibrated")
        bundle.add(source / "gaff2", arcname="gaff2")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "asset_version": 1,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest,
        "contents": {"strict_library_entries": 1, "gaff2_cache_entries": 1},
    }
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_prebuilt_assets_install_once_without_overwriting(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    lipid_root = tmp_path / "cache" / "lipids"
    gaff_root = tmp_path / "cache" / "gaff"
    existing = gaff_root / "TEST-key" / "lipid.itp"
    existing.parent.mkdir(parents=True)
    existing.write_text("newer user cache")

    first = install_prebuilt_assets(
        manifest_path=manifest, lipid_root=lipid_root, gaff_root=gaff_root,
    )
    second = install_prebuilt_assets(
        manifest_path=manifest, lipid_root=lipid_root, gaff_root=gaff_root,
    )

    assert first["status"] == "installed"
    assert first["installed_files"] == 3
    assert second["status"] == "ready"
    assert second["installed_files"] == 0
    assert existing.read_text() == "newer user cache"
    assert (lipid_root / "amber-gaff2/TEST/conf_0000.npz").read_bytes() == b"fixture"
    assert prebuilt_asset_status(
        manifest_path=manifest, lipid_root=lipid_root, gaff_root=gaff_root,
    )["status"] == "ready"


def test_prebuilt_assets_reject_checksum_mismatch(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    data = json.loads(manifest.read_text())
    data["archive_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="checksum"):
        install_prebuilt_assets(
            manifest_path=manifest,
            lipid_root=tmp_path / "lipids",
            gaff_root=tmp_path / "gaff",
        )


def test_prebuilt_assets_reject_lfs_pointer(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    data = json.loads(manifest.read_text())
    archive = manifest.parent / data["archive"]
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{data['archive_sha256']}\nsize {data['archive_bytes']}\n"
    )
    archive.write_text(pointer)
    data["archive_bytes"] = archive.stat().st_size
    data["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        install_prebuilt_assets(
            manifest_path=manifest,
            lipid_root=tmp_path / "lipids",
            gaff_root=tmp_path / "gaff",
        )

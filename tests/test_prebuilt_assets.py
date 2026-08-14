import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from gmxbuilder.modules.membrane.equilibrated_library import (
    ACCEPTED_METHOD,
    SCHEMA_VERSION,
    topology_signature,
)
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
    atom_names = ["C1"]
    (lipid / "metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "coordinate_handedness": "preserved",
        "leaflet_transform": "proper_rotation",
        "status": "ready",
        "method": ACCEPTED_METHOD,
        "parameter_family": "amber-gaff2",
        "n_conformations": 20,
        "topology_sha256": topology_signature(
            atom_names, "amber14sb", "gaff2",
        ),
        "atom_names": atom_names,
        "force_field": "amber14sb",
        "lipid_ff": "gaff2",
        "quality": {
            "passed": True,
            "orientation": {"passed": True, "n_lipids_checked": 20},
        },
    }))
    for index in range(20):
        (lipid / f"conf_{index:04d}.npz").write_bytes(b"fixture")
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
        "library_schema_version": SCHEMA_VERSION,
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
    assert first["installed_files"] == 22
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


def test_prebuilt_assets_reject_stale_library_schema(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    data = json.loads(manifest.read_text())
    data["library_schema_version"] = SCHEMA_VERSION - 1
    manifest.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="strict-library schema"):
        install_prebuilt_assets(
            manifest_path=manifest,
            lipid_root=tmp_path / "lipids",
            gaff_root=tmp_path / "gaff",
        )


def test_prebuilt_assets_replace_stale_strict_entry_on_upgrade(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    lipid_root = tmp_path / "cache" / "lipids"
    stale = lipid_root / "amber-gaff2" / "TEST"
    stale.mkdir(parents=True)
    (stale / "metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION - 1,
        "status": "ready",
    }))
    (stale / "obsolete.txt").write_text("old release")

    result = install_prebuilt_assets(
        manifest_path=manifest,
        lipid_root=lipid_root,
        gaff_root=tmp_path / "cache" / "gaff",
    )

    assert result["replaced_lipid_entries"] == 1
    assert not (stale / "obsolete.txt").exists()
    assert json.loads((stale / "metadata.json").read_text())["schema_version"] == SCHEMA_VERSION

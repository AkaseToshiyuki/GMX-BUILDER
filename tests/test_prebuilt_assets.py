import hashlib
import json
import tarfile
from pathlib import Path

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
from tests.dry_initial_fixture import dry_initial_fixture


def _fixture_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    lipid = source / "lipid_equilibrated" / "amber-gaff2" / "TEST"
    gaff = source / "gaff2" / "TEST-key"
    lipid.mkdir(parents=True)
    gaff.mkdir(parents=True)
    atom_names = ["C1"]
    (lipid / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "coordinate_handedness": "preserved",
                "leaflet_transform": "proper_rotation",
                "status": "ready",
                "method": ACCEPTED_METHOD,
                "parameter_family": "amber-gaff2",
                "n_conformations": 20,
                "topology_sha256": topology_signature(
                    atom_names,
                    "amber14sb",
                    "gaff2",
                ),
                "atom_names": atom_names,
                "force_field": "amber14sb",
                "lipid_ff": "gaff2",
                "quality": {
                    "initial_water_exclusion": dry_initial_fixture(),
                    "passed": True,
                    "orientation": {"passed": True, "n_lipids_checked": 20},
                },
            }
        )
    )
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
        manifest_path=manifest,
        lipid_root=lipid_root,
        gaff_root=gaff_root,
    )
    second = install_prebuilt_assets(
        manifest_path=manifest,
        lipid_root=lipid_root,
        gaff_root=gaff_root,
    )

    assert first["status"] == "installed"
    assert first["installed_files"] == 22
    assert second["status"] == "ready"
    assert second["installed_files"] == 0
    assert existing.read_text() == "newer user cache"
    assert (lipid_root / "amber-gaff2/TEST/conf_0000.npz").read_bytes() == b"fixture"
    assert (
        prebuilt_asset_status(
            manifest_path=manifest,
            lipid_root=lipid_root,
            gaff_root=gaff_root,
        )["status"]
        == "ready"
    )


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


@pytest.mark.parametrize("bundled_parameters", [True, False])
def test_staging_admission_uses_bundled_target_and_host_parameters(
    tmp_path, monkeypatch, caplog, bundled_parameters
):
    """A local fit must neither hide a missing bundle fit nor replace its source."""
    from gmxbuilder.modules.forcefield.gaff_backend import _cache_root
    from gmxbuilder.runtime.prebuilt_assets import _validate_staging

    staging = tmp_path / "staging"
    entry = staging / "lipid_equilibrated/amber-gaff2/20AHC"
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "force_field": "amber14sb",
                "lipid_ff": "gaff2",
                "equilibration_host": {"lipid_name": "POPC"},
            }
        )
    )
    ambient = tmp_path / "user-cache"
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(ambient))
    for root in (ambient, staging / "gaff2"):
        for name in ("L_20AHC", "POPC"):
            directory = root / name
            directory.mkdir(parents=True)
            if (root == staging / "gaff2") == bundled_parameters:
                (directory / "parameters").write_text("the accepted parameter definitions")

    def inspect(self, name, force_field, lipid_ff):
        for parameter_name in ("L_20AHC", "POPC"):
            path = _cache_root(parameter_name, install=False) / parameter_name / "parameters"
            if not path.is_file() or path.read_text() != "the accepted parameter definitions":
                return None
        return object()

    monkeypatch.setattr(
        "gmxbuilder.runtime.prebuilt_assets.EquilibratedLipidLibrary.inspect", inspect
    )
    _validate_staging(
        staging, {"contents": {"strict_library_entries": 1, "gaff2_cache_entries": 2}}
    )
    assert ("will not be served" in caplog.text) is not bundled_parameters
    assert _cache_root("L_20AHC", install=False) == ambient
    assert _cache_root("POPC", install=False) == ambient


def test_prebuilt_assets_replace_stale_strict_entry_on_upgrade(tmp_path):
    manifest = _fixture_bundle(tmp_path)
    lipid_root = tmp_path / "cache" / "lipids"
    stale = lipid_root / "amber-gaff2" / "TEST"
    stale.mkdir(parents=True)
    (stale / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION - 1,
                "status": "ready",
            }
        )
    )
    (stale / "obsolete.txt").write_text("old release")

    result = install_prebuilt_assets(
        manifest_path=manifest,
        lipid_root=lipid_root,
        gaff_root=tmp_path / "cache" / "gaff",
    )

    assert result["replaced_lipid_entries"] == 1
    assert not (stale / "obsolete.txt").exists()
    assert json.loads((stale / "metadata.json").read_text())["schema_version"] == SCHEMA_VERSION


def test_the_installation_lock_can_be_entered_twice_in_one_process(tmp_path):
    """A lock a process can take against itself is not a lock, it is a trap.

    ``flock`` is held per open file description, so opening the file again and
    asking for it again waits forever -- and the installation does reach back
    into code that comes here, because removing a library entry the runtime
    rejects means asking the library about it. That deadlock stalled a full test
    suite for four hours with two charge fits queued behind it.
    """
    import threading

    from gmxbuilder.runtime.prebuilt_assets import _installation_lock

    finished = threading.Event()

    def nest():
        with _installation_lock(tmp_path / "lipid", tmp_path / "gaff"):
            with _installation_lock(tmp_path / "lipid", tmp_path / "gaff"):
                pass
        finished.set()

    worker = threading.Thread(target=nest, daemon=True)
    worker.start()
    assert finished.wait(30), "the second entry waited for the first one to finish"


def test_reading_a_cached_gaff_template_does_not_install_anything(monkeypatch, tmp_path):
    """The lipid library asks this from inside an installation."""
    from gmxbuilder.modules.forcefield import gaff_backend
    from gmxbuilder.runtime import prebuilt_assets

    def refuse():
        raise AssertionError("a peek at the cache must not install assets")

    monkeypatch.setattr(prebuilt_assets, "ensure_prebuilt_assets", refuse)
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    assert gaff_backend.cached_gaff_template("NOSUCHLIPID", "CCO", 0, charge_method="gas") is None

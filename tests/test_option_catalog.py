"""UI snapshots reuse validation, never grant production admission."""

import threading
import time
from types import SimpleNamespace

from gmxbuilder.web.server_parts import option_catalog as catalog
from tests.test_v4_construction import pair  # noqa: F401
from tests.test_v4_reuse import reusable  # noqa: F401


class Library:
    def __init__(self, root):
        self.roots = [root]
        self.calls = 0

    def _candidate_dirs(self, name, force_field, lipid_ff):
        return [self.roots[0] / "family" / name]

    def inspect(self, name, force_field, lipid_ff):
        self.calls += 1
        directory = self._candidate_dirs(name, force_field, lipid_ff)[0]
        if (directory / "metadata.json").read_text() != "valid":
            return None
        if not (directory / "conf_0001.npz").is_file():
            return None
        return SimpleNamespace(
            path=directory,
            metadata={
                "v4_protocol": {"sha256": "policy"},
                "n_conformations": 20,
                "lipid_ff": "lipid21",
                "equilibration_host": None,
                "huge_evidence": "must not remain in display cache",
            },
        )


def setup(monkeypatch, tmp_path):
    directory = tmp_path / "family" / "POPC"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("valid")
    (directory / "conf_0001.npz").write_bytes(b"coordinate")
    library = Library(tmp_path)
    epoch = [0]

    def refresh(*, library, write):
        assert not write
        entry = library.inspect("POPC", "amber14sb", "lipid21")
        return {"library_version": 4, "entries": [{"ready": entry is not None}]}

    monkeypatch.setattr(catalog, "refresh_availability_list", refresh)
    monkeypatch.setattr(catalog, "build_ui_options", lambda **kw: {"lipids": []})
    service = catalog.OptionCatalog(library, dependencies=lambda: epoch[0])
    return service, library, directory, epoch


def test_unchanged_inventory_reuses_compact_result_and_detects_inplace_edit(monkeypatch, tmp_path):
    service, library, directory, _ = setup(monkeypatch, tmp_path)
    service.refresh()
    service.refresh()
    assert library.calls == 1
    assert "huge_evidence" not in next(iter(service.index.records.values()))[1].metadata
    (directory / "metadata.json").write_text("wrong")
    service.refresh()
    assert library.calls == 2
    assert not service.availability["entries"][0]["ready"]


def test_conformer_deletion_and_policy_change_invalidate(monkeypatch, tmp_path):
    service, library, directory, epoch = setup(monkeypatch, tmp_path)
    service.refresh()
    epoch[0] += 1
    service.refresh()
    assert library.calls == 2
    (directory / "conf_0001.npz").unlink()
    service.refresh()
    assert library.calls == 3
    assert not service.availability["entries"][0]["ready"]


def test_publication_during_validation_is_not_advertised(monkeypatch, tmp_path):
    service, library, directory, _ = setup(monkeypatch, tmp_path)
    original = library.inspect

    def inspect(*args):
        result = original(*args)
        (directory / "metadata.json").write_text("wrong")
        return result

    monkeypatch.setattr(library, "inspect", inspect)
    service.refresh()
    assert service.availability["status"] == "checking"
    assert service.signature is None
    monkeypatch.setattr(library, "inspect", original)
    service.refresh()
    assert not service.availability["entries"][0]["ready"]


def test_many_readers_start_only_one_worker_and_do_not_wait_for_validation(monkeypatch, tmp_path):
    service, _, _, _ = setup(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked():
        calls.append(1)
        entered.set()
        release.wait(5)

    monkeypatch.setattr(service, "refresh", blocked)
    try:
        for _ in range(20):
            assert service.read()["status"] == "checking"
        assert entered.wait(1)
        assert len(calls) == 1
    finally:
        release.set()
        service.close()


def test_stalled_watcher_hides_previously_ready_entries(monkeypatch, tmp_path):
    service, _, _, _ = setup(monkeypatch, tmp_path)
    service.refresh()
    monkeypatch.setattr(service, "start", lambda: None)
    service.checked_at = time.monotonic() - 60
    assert service.read()["entries"] == []


def test_unchanged_entries_are_not_revalidated_when_another_is_published(monkeypatch, tmp_path):
    service, library, _, _ = setup(monkeypatch, tmp_path)
    service.refresh()
    another = tmp_path / "family" / "POPE"
    another.mkdir()
    (another / "metadata.json").write_text("valid")
    service.refresh()
    assert library.calls == 1


def test_same_size_replacement_changes_stamp(tmp_path):
    path = tmp_path / "parameter.itp"
    path.write_text("aaa")
    first = catalog.tree_stamp(tmp_path)
    replacement = tmp_path / "other"
    replacement.write_text("bbb")
    replacement.replace(path)
    assert catalog.tree_stamp(tmp_path) != first


def test_real_strict_reader_revokes_cached_publication(reusable, monkeypatch):  # noqa: F811
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary

    root, _, _ = reusable
    library = EquilibratedLipidLibrary([root / "library"])
    service = catalog.OptionCatalog(library, dependencies=lambda: 0)
    monkeypatch.setattr(catalog, "build_ui_options", lambda **kwargs: {"lipids": []})
    service.refresh()

    def ready():
        return next(
            r["ready"]
            for r in service.availability["entries"]
            if r["lipid_name"] == "POPC" and r["lipid_ff"] == "charmm36m"
        )

    assert ready()
    conformer = next((root / "library/charmm36m-lipid/POPC").glob("conf_*.npz"))
    conformer.unlink()
    service.refresh()
    assert not ready()

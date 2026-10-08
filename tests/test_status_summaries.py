"""Polling must be bounded and revision-aware across worker processes."""

import asyncio
import json

import pytest

from gmxbuilder.core.checkpoint_status import read_status, refresh_status
from gmxbuilder.core.system import System
from gmxbuilder.web import server
from tests.test_equilibrated_lipid_library import _bilayer_structure


def test_checkpoint_summary_is_refreshed_only_by_admitted_work(tmp_path, monkeypatch):
    directory = tmp_path / "membrane"
    system = System(_bilayer_structure(0.0))
    system.save_checkpoint(directory)
    assert read_status(directory)["membrane_metrics"]["num_atoms"] == system.num_atoms
    data = json.loads((directory / "system.json").read_text())
    data["metadata"]["system_confirmed"] = True
    (directory / "system.json").write_text(json.dumps(data))
    assert read_status(directory) is None
    refresh_status(directory)
    assert read_status(directory)["system_confirmed"] is True
    monkeypatch.setattr(System, "load_checkpoint", lambda *a: pytest.fail("Polling loaded atoms"))
    assert read_status(directory)["system_confirmed"] is True


def test_propka_status_never_prepares_or_hashes_input(tmp_path, monkeypatch):
    monkeypatch.setattr(server.task_manager, "root", tmp_path)
    task_id = server.task_manager.create_task("input.pdb")["task_id"]
    server.task_manager.save_uploaded_pdb(
        task_id,
        "input.pdb",
        b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n",
    )
    adapter = server._resolve_propka_pdb_path(task_id)
    server._publish_propka_status(adapter, "ready", 0)
    monkeypatch.setattr(
        server, "_resolve_propka_pdb_path", lambda *a: pytest.fail("Polling prepared input")
    )
    monkeypatch.setattr(server, "_pka_digest", lambda *a: pytest.fail("Polling hashed input"))
    assert asyncio.run(server.api_propka_status(task_id=task_id)) == {
        "status": "ready",
        "residues": 0,
    }
    # This also survives an empty process-local result cache after worker exit.
    monkeypatch.setattr(server, "_pka_cache", {})
    assert asyncio.run(server.api_propka_status(task_id=task_id))["status"] == "ready"
    source = server.task_manager.get_pdb_path(task_id)
    source.write_bytes(source.read_bytes() + b"REMARK changed\n")
    assert asyncio.run(server.api_propka_status(task_id=task_id)) == {"status": "not_started"}


def test_invalid_adapter_status_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(server.task_manager, "root", tmp_path)
    task_id = server.task_manager.create_task("input.pdb")["task_id"]
    assert asyncio.run(server.api_propka_status(task_id=task_id)) == {"status": "not_started"}


def test_single_library_status_uses_bounded_catalog_and_revokes_stale_snapshot(monkeypatch):
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.web.server_parts import option_catalog

    entry = {
        "lipid_name": "POPC",
        "lipid_ff": "lipid21",
        "parameter_family": "amber-lipid21",
        "ready": True,
        "n_conformations": 40,
        "amber_mixed_ready": True,
        "validation_scope": {"kind": "initialization"},
    }
    snapshot = {
        "lipids": [{"name": "POPC", "parameter_sources": ["lipid21"]}],
        "availability": {"status": "ready", "entries": [entry]},
    }

    class Catalog:
        def read(self, **kwargs):
            return snapshot

    monkeypatch.setattr(option_catalog, "get_catalog", lambda: Catalog())
    monkeypatch.setattr(
        EquilibratedLipidLibrary, "inspect", lambda *a, **k: pytest.fail("Poll loaded ensemble")
    )
    result = asyncio.run(server.api_lipid_library_status(lipid_name="popc"))
    assert result["has_library"] and result["n_conformations"] == 40
    assert result["metadata_scope"] == "availability_summary"
    assert result["validation_scope"]["kind"] == "initialization"
    snapshot["availability"]["status"] = "checking"
    result = asyncio.run(server.api_lipid_library_status(lipid_name="popc"))
    assert not result["has_library"] and result["metadata"] is None

"""Identity, display preservation and revision-bound final review contracts."""

import asyncio
import base64
import gzip
import json

import numpy as np
import pytest
from starlette.requests import Request

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.web.server_parts.protonation_identity import map_predictions, prepare_analysis
from gmxbuilder.web.server_parts.viewer_data import build_viewer, cached_viewer, checkpoint_revision


def sample_system():
    structure = Structure(
        coordinates=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.09, 0, 0]], dtype=float),
        box_vectors=np.eye(3) * 4,
        atom_names=["P", "P", "OW", "HW1"],
        resnames=["POPC", "POPE", "SOL", "SOL"],
        resids=[10001, 10002, 10003, 10003],
        chain_ids=["", "", "", ""],
        elements=["P", "P", "O", "H"],
    )
    system = System(structure=structure)
    system.add_component(
        Component(
            "Mixed membrane",
            ComponentKind.MEMBRANE,
            np.array([0, 1]),
            metadata={"n_lipids_upper": 1, "n_lipids_lower": 1},
        )
    )
    system.add_component(
        Component("Water", ComponentKind.SOLVENT, np.array([2, 3]), metadata={"n_molecules": 1})
    )
    return system


def test_compact_viewer_preserves_identity_coordinates_and_component_counts(tmp_path):
    system = sample_system()
    directory = tmp_path / "ions"
    system.save_checkpoint(directory)
    payload = json.loads(gzip.decompress(build_viewer(directory).read_bytes()))
    assert payload["atom_count"] == 4 and payload["display_count"] == 3
    assert payload["resnames"]["labels"] == ["POPC", "POPE", "SOL"]
    positions = np.frombuffer(base64.b64decode(payload["coordinates_A"]), dtype="<f4").reshape(
        -1, 3
    )
    np.testing.assert_allclose(positions, system.coordinates[:3] * 10)
    assert [x["atoms"] for x in payload["components"]] == [2, 2]
    assert [x["display_atoms"] for x in payload["components"]] == [2, 1]
    assert payload["revision"] == checkpoint_revision(directory)
    assert cached_viewer(directory)
    system.structure.coordinates[0, 0] = 0.1
    system.save_checkpoint(directory)
    assert cached_viewer(directory) is None
    new = json.loads(gzip.decompress(build_viewer(directory).read_bytes()))
    assert new["revision"] != payload["revision"]


def test_ambiguous_component_membership_fails(tmp_path):
    system = sample_system()
    system.components[1].atom_indices = np.array([1, 2, 3])
    system.save_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="overlapping"):
        build_viewer(tmp_path)


def protein(chains=("", "")):
    return Structure(
        coordinates=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        box_vectors=np.eye(3) * 4,
        atom_names=["CA", "CA"],
        resnames=["ASP", "HIS"],
        resids=[6, 509],
        chain_ids=list(chains),
        elements=["C", "C"],
    )


def test_propka_blank_chain_and_nonconsecutive_numbers_map_reversibly(tmp_path):
    pdb = tmp_path / "propka-input.pdb"
    prepare_analysis(protein(), pdb)
    mapping = json.loads(pdb.with_suffix(".identities.json").read_text())
    assert [(r["chain"], r["resid"], r["analysis_resid"]) for r in mapping["residues"]] == [
        ("", 6, 1),
        ("", 509, 2),
    ]
    predictions = [
        {"chain": "A", "resid": 1, "residue_name": "ASP", "predicted_pKa": 3.2, "shift": -0.6},
        {"chain": "A", "resid": 2, "residue_name": "HIS", "predicted_pKa": 7.19, "shift": 1.19},
    ]
    residues = [
        {"chain": "A", "resid": 6, "resname": "ASP", "index": 0},
        {"chain": "A", "resid": 509, "resname": "HIS", "index": 1},
    ]
    mapped = map_predictions(predictions, residues, str(pdb))
    assert [(x["resid"], x["shift"]) for x in mapped] == [(6, -0.6), (509, 1.19)]
    from gmxbuilder.modules.modifications.protonation import assign_protonation_with_propka

    assigned = assign_protonation_with_propka(residues, mapped, force_field="amber14sb")
    assert assigned[1]["assigned_name"] == "HIP"
    assert assigned[1]["pKa_shift"] == 1.19
    pdb.write_text(pdb.read_text() + "REMARK changed\n")
    with pytest.raises(ValueError, match="stale"):
        map_predictions(predictions, residues, str(pdb))


def test_real_a_and_unnamed_chain_cannot_merge(tmp_path):
    structure = protein(("", "A"))
    structure.resids[:] = [6, 6]
    pdb = tmp_path / "propka-input.pdb"
    prepare_analysis(structure, pdb)
    with pytest.raises(ValueError, match="Ambiguous"):
        map_predictions(
            [{"chain": "A", "resid": 1, "residue_name": "ASP"}],
            [{"chain": "A", "resid": 6, "resname": "ASP"}],
            str(pdb),
        )


def request(body):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


def test_final_confirmation_rejects_stale_revision_and_binds_exact_checkpoint(
    tmp_path, monkeypatch
):
    from gmxbuilder.web import server
    from gmxbuilder.web.server_parts.viewer_data import confirmed_revision
    from gmxbuilder.web.task_manager import TaskManager

    manager = TaskManager(root=tmp_path)
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_resources", None)
    task = manager.create_task()
    tid = task["task_id"]
    manager.update_state(tid, {"task_type": {"id": "solvator"}})
    directory = manager.get_task_dir(tid) / "steps" / "ions"
    system = sample_system()
    system.save_checkpoint(directory)
    bad = asyncio.run(
        server.api_confirm_final_review(tid, request({"source_step": "ions", "revision": "stale"}))
    )
    assert bad.status_code == 409
    revision = checkpoint_revision(directory)
    good = asyncio.run(
        server.api_confirm_final_review(tid, request({"source_step": "ions", "revision": revision}))
    )
    assert good["confirmed"] is True
    assert confirmed_revision(manager.get_state(tid), directory)
    system.structure.coordinates[0, 0] = 0.3
    system.save_checkpoint(directory)
    assert not confirmed_revision(manager.get_state(tid), directory)


def test_dry_bilayer_final_review_uses_membrane_checkpoint(tmp_path, monkeypatch):
    from gmxbuilder.web import server
    from gmxbuilder.web.task_manager import TaskManager

    manager = TaskManager(root=tmp_path)
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_resources", None)
    tid = manager.create_task()["task_id"]
    manager.update_state(tid, {"task_type": {"id": "pure-membrane"}})
    directory = manager.get_task_dir(tid) / "steps" / "membrane"
    sample_system().save_checkpoint(directory)
    result = asyncio.run(
        server.api_confirm_final_review(
            tid, request({"source_step": "membrane", "revision": checkpoint_revision(directory)})
        )
    )
    assert result["confirmed"]
    assert manager.get_state(tid)["step_solvation_config"]["enabled"] is False


def test_cg_confirmation_preserves_coordinates_and_enables_existing_export_gate(
    tmp_path, monkeypatch
):
    from gmxbuilder.web import server
    from gmxbuilder.web.server_parts.viewer_data import confirmed_revision
    from gmxbuilder.web.task_manager import TaskManager

    manager = TaskManager(root=tmp_path)
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_resources", None)
    tid = manager.create_task()["task_id"]
    manager.update_state(tid, {"task_type": {"id": "martini3-bilayer"}})
    directory = manager.get_task_dir(tid) / "steps" / "cg_system"
    original = sample_system()
    original.metadata["cg_scientific_check"] = {"passed": True}
    original.metadata["system_confirmed"] = False
    original.save_checkpoint(directory)
    revision = checkpoint_revision(directory)
    result = asyncio.run(
        server.api_confirm_final_review(
            tid, request({"source_step": "cg_system", "revision": revision})
        )
    )
    assert result["rendered_revision"] == revision
    assert result["revision"] != revision
    saved = System.load_checkpoint(directory)
    np.testing.assert_array_equal(saved.coordinates, original.coordinates)
    assert saved.structure.resnames == original.structure.resnames
    assert saved.metadata["system_confirmed"] is True
    assert confirmed_revision(manager.get_state(tid), directory)


def test_viewer_api_is_gzipped_and_missing_checkpoint_never_looks_ready(tmp_path, monkeypatch):
    from gmxbuilder.web import server
    from gmxbuilder.web.task_manager import TaskManager

    manager = TaskManager(root=tmp_path)
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_resources", None)
    tid = manager.create_task()["task_id"]
    manager.update_state(tid, {"task_type": {"id": "solvator"}})
    directory = manager.get_task_dir(tid) / "steps" / "ions"
    missing = asyncio.run(server.api_step_viewer_data(tid, "ions", request({})))
    assert missing.status_code == 404
    sample_system().save_checkpoint(directory)
    response = asyncio.run(server.api_step_viewer_data(tid, "ions", request({})))
    assert response.headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(response.path.read_bytes()))["atom_count"] == 4
    conditional = Request(
        {
            "type": "http",
            "method": "GET",
            "headers": [(b"if-none-match", response.headers["etag"].encode())],
        }
    )
    cached = asyncio.run(server.api_step_viewer_data(tid, "ions", conditional))
    assert cached.status_code == 304
    changed = sample_system()
    changed.structure.coordinates[0, 0] = 0.8
    changed.save_checkpoint(directory)
    updated = asyncio.run(server.api_step_viewer_data(tid, "ions", conditional))
    assert updated.status_code == 200 and updated.headers["etag"] != response.headers["etag"]


def test_propka_memory_cache_is_bound_to_analysis_bytes(tmp_path, monkeypatch):
    from gmxbuilder.modules.modifications import protonation
    from gmxbuilder.web import server

    source = tmp_path / "analysis.pdb"
    source.write_text("first")
    monkeypatch.setattr(server, "_durable_pka_cache", lambda *args: None)
    monkeypatch.setattr(
        protonation, "predict_pka_from_pdb", lambda path: [{"input": source.read_text()}]
    )
    first = asyncio.run(server._get_propka_results(str(source)))
    source.write_text("second")
    second = asyncio.run(server._get_propka_results(str(source)))
    assert first == [{"input": "first"}]
    assert second == [{"input": "second"}]


def test_queue_wait_cannot_change_which_checkpoint_was_approved(tmp_path):
    from gmxbuilder.web.server_parts.viewer_data import require_final_review

    system = sample_system()
    system.save_checkpoint(tmp_path)
    revision = checkpoint_revision(tmp_path)
    state = {
        "final_review": {"confirmed": True, "source_step": tmp_path.name, "revision": revision}
    }
    require_final_review(state, tmp_path, revision)
    system.structure.coordinates[0, 0] = 0.5
    system.save_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        require_final_review(state, tmp_path, revision)
    state["final_review"]["revision"] = checkpoint_revision(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        require_final_review(state, tmp_path, revision)

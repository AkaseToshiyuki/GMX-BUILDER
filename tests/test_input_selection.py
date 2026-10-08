"""Selections preserve chain identity and always derive from the intact upload."""

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.web import server
from gmxbuilder.web.server_parts.structure_files import filter_pdb_file
from gmxbuilder.web.task_manager import TaskManager


def atom(serial, chain, resname="ALA"):
    return (
        f"ATOM  {serial:5d}  CA  {resname:3s} {chain:1s}{serial:4d}    "
        f"{serial * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
    )


@pytest.fixture
def selection_task(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_schedule_propka_precompute", lambda *args: None)
    with TestClient(server.app) as client:
        reply = client.post(
            "/api/upload-pdb",
            files={"file": ("mixed.pdb", atom(1, "") + atom(2, "A") + "END\n", "chemical/x-pdb")},
            data={"task_type": "solvator"},
        )
        assert reply.status_code == 200, reply.text
        yield client, manager, reply.json()


def test_blank_and_named_chain_are_independent(selection_task):
    client, manager, uploaded = selection_task
    assert uploaded["chains"] == ["A", "B"]
    assert uploaded["chain_mapping"] == {"": "A", "A": "B"}
    tid = uploaded["task_id"]
    for chain in ("A", "B"):
        reply = client.post(f"/api/filter-pdb/{tid}", json={"include_chains": [chain]})
        assert reply.status_code == 200, reply.text
        selected = PDBParser().parse(manager.get_task_dir(tid) / "filtered.pdb")
        assert selected.chain_ids == [chain]


def test_repeat_selection_is_idempotent_and_reselection_restores_atoms(selection_task):
    client, manager, uploaded = selection_task
    tid = uploaded["task_id"]
    source = manager.get_filter_source(tid)
    original = source.read_bytes()
    destination = manager.get_task_dir(tid) / "filtered.pdb"
    for _ in range(2):
        reply = client.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A"]})
        assert reply.status_code == 200, reply.text
        assert reply.json()["n_kept"] == 1
    reply = client.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A", "B"]})
    assert reply.status_code == 200, reply.text
    assert PDBParser().parse(destination).num_atoms == 2
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        {"include_chains": []},
        {"include_chains": ["absent"]},
        {"exclude_resnames": ["ALA"]},
        {"include_chains": "A"},
    ],
)
def test_invalid_selection_preserves_existing_result(selection_task, payload):
    client, manager, uploaded = selection_task
    tid = uploaded["task_id"]
    assert client.post(f"/api/filter-pdb/{tid}", json={}).status_code == 200
    destination = manager.get_task_dir(tid) / "filtered.pdb"
    previous = destination.read_bytes()
    reply = client.post(f"/api/filter-pdb/{tid}", json=payload)
    assert reply.status_code == 400, reply.text
    assert destination.read_bytes() == previous


def test_filter_cannot_replace_a_running_tasks_input(selection_task):
    client, manager, uploaded = selection_task
    tid = uploaded["task_id"]
    assert server._step_admission.try_acquire(tid) == "accepted"
    try:
        reply = client.post(f"/api/filter-pdb/{tid}", json={})
        assert reply.status_code == 409
        assert server._step_admission.active(tid)
        assert not (manager.get_task_dir(tid) / "filtered.pdb").exists()
    finally:
        server._step_admission.release(tid)


def test_resume_old_empty_selection_uses_original_chain_identity(selection_task):
    client, manager, uploaded = selection_task
    tid = uploaded["task_id"]
    (manager.get_task_dir(tid) / "filtered.pdb").write_text("")
    state = manager.get_state(tid)
    summary = state["structure_summary"]
    summary.pop("chain_identity_version")
    summary["chains"] = ["A"]
    manager.update_state(tid, {"structure_summary": summary})
    reply = client.get(f"/api/task/{tid}/resume")
    assert reply.status_code == 200, reply.text
    assert reply.json()["pdb_info_full"]["chains"] == ["A", "B"]


def test_changed_selection_invalidates_checkpoints(selection_task):
    client, manager, uploaded = selection_task
    tid = uploaded["task_id"]
    root = manager.get_task_dir(tid)
    for stage in ("input", "structure", "solvation", "ions"):
        directory = root / "steps" / stage
        directory.mkdir(parents=True)
        (directory / "system.npz").write_bytes(b"old checkpoint")
    reply = client.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A"]})
    assert reply.status_code == 200, reply.text
    assert not list((root / "steps").glob("*/system.npz"))


def test_same_file_and_hardlink_cannot_be_truncated(tmp_path):
    source = tmp_path / "upload.pdb"
    source.write_text(atom(1, "A") + "END\n")
    alias = tmp_path / "alias.pdb"
    alias.hardlink_to(source)
    original = source.read_bytes()
    for destination in (source, alias):
        with pytest.raises(ValueError, match="distinct"):
            filter_pdb_file(source, destination, ["A"], [])
    assert source.read_bytes() == original


def test_complete_charmm_isoleucine_is_not_missing_a_carbon(tmp_path, empty_system):
    from gmxbuilder.modules.solution.input import SolutionInputModule

    names = ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD"]
    path = tmp_path / "ile.pdb"
    from importlib.metadata import distribution

    # Use a complete L-ILE reference, not collinear synthetic atoms.
    template = distribution("pdbfixer").locate_file("pdbfixer/templates/ILE.pdb").read_text()
    path.write_text(template.replace(" CD1 ", " CD  "))
    result = SolutionInputModule().execute(empty_system, {"pdb": str(path)})
    assert result.success
    assert result.system.num_atoms == 8
    assert set(result.system.structure.atom_names) == set(names[:-1] + ["CD1"])
    assert any("before repair" in line for line in result.log)


def test_cached_resume_preserves_cell_and_input_warnings(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_schedule_propka_precompute", lambda *args: None)
    source = (
        "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
        + atom(1, "A")
        + "END\n"
    )
    with TestClient(server.app) as client:
        uploaded = client.post(
            "/api/upload-pdb",
            files={"file": ("placeholder.pdb", source)},
            data={"task_type": "solvator"},
        )
        assert uploaded.status_code == 200, uploaded.text
        data = uploaded.json()
        assert data["cell_info"]["status"] == "placeholder"
        assert any("placeholder" in warning for warning in data["validation_warnings"])

        def unexpected_summary(*args):
            raise AssertionError("Unchanged input should reuse its current summary")

        monkeypatch.setattr(server, "summarize_resume_structure", unexpected_summary)
        resumed = client.get(f"/api/task/{data['task_id']}/resume").json()
    assert resumed["validation_warnings"] == data["validation_warnings"]
    assert resumed["cell_info"] == data["cell_info"]
    assert resumed["input_status"] == data["input_status"]

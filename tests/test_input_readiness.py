"""A parseable upload must not authorize building an incomplete protein."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.pdb import PDBWriter
from gmxbuilder.modules.input.validation import (
    INPUT_VALIDATION_VERSION,
    assess_input_structure,
    read_polymer_metadata,
)
from gmxbuilder.pipeline.step_executor import StepRunner
from gmxbuilder.web import server
from gmxbuilder.web.task_manager import TaskManager


def peptide(resids=(128, 135), separation=2.0, chains=None):
    coordinates = []
    names = []
    numbers = []
    chain_ids = []
    for i, resid in enumerate(resids):
        for name, xyz in zip(
            ["N", "CA", "C", "O"],
            [[0, 0, 0], [0.145, 0, 0], [0.2, 0.142, 0], [0.145, 0.245, 0]],
            strict=True,
        ):
            coordinates.append(np.array(xyz) + [i * separation, 0, 0])
            names.append(name)
            numbers.append(resid)
            chain_ids.append(chains[i] if chains else "R")
    return Structure(
        coordinates=np.array(coordinates),
        box_vectors=np.eye(3) * 6,
        atom_names=names,
        resnames=["GLY"] * len(names),
        resids=numbers,
        chain_ids=chain_ids,
        elements=[n[0] for n in names],
    )


@pytest.mark.parametrize("resids", [(1, 2), (1, 3), (128, 135), (1, 80)])
def test_geometric_break_is_blocked_regardless_of_numbering(resids):
    report = assess_input_structure(peptide(resids))
    assert not report["can_proceed"]
    assert any(issue["code"] == "peptide_break" for issue in report["errors"])


def test_numbering_jump_with_intact_geometry_is_not_called_missing_sequence():
    structure = peptide((1, 80))
    structure.coordinates[4:] += structure.coordinates[2] + [0.133, 0, 0] - structure.coordinates[4]
    assert assess_input_structure(structure)["can_proceed"]


def test_separate_complete_chains_can_continue():
    assert assess_input_structure(peptide(chains=["A", "B"]))["can_proceed"]


def test_all_breaks_are_reported_together():
    errors = assess_input_structure(peptide((1, 2, 3)))["errors"]
    assert len([e for e in errors if e["code"] == "peptide_break"]) == 2


def test_pdb_missing_residue_records_follow_the_selected_model(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(
        "REMARK 465   2 SER R   129\n"
        "REMARK 465   5 GLY R   130\n"
        "MODEL        5\nENDMDL\nMODEL        2\nENDMDL\n"
    )
    assert [r["resid"] for r in read_polymer_metadata(path)["missing_residues"]] == [130]


def test_deposited_sequence_survives_and_excluded_chains_do_not_block(tmp_path):
    cif = tmp_path / "source.cif"
    cif.write_text("""data_gap
loop_
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
R 128 137 GLU
R 129 138 SER
R 135 144 SER
#
loop_
_pdbx_unobs_or_zero_occ_residues.PDB_model_num
_pdbx_unobs_or_zero_occ_residues.polymer_flag
_pdbx_unobs_or_zero_occ_residues.auth_asym_id
_pdbx_unobs_or_zero_occ_residues.auth_seq_id
_pdbx_unobs_or_zero_occ_residues.auth_comp_id
1 Y R 129 SER
1 Y R 27 GLY
2 Y R 136 LEU
#
""")
    source = read_polymer_metadata(cif)
    assert len(source["sequences"]["R"]) == 3
    assert [r["resid"] for r in source["missing_residues"]] == [129, 27]
    report = assess_input_structure(peptide(), source)
    assert any(e["code"] == "deposited_missing_segment" for e in report["errors"])
    assert any(e["code"] == "deposited_missing_termini" for e in report["warnings"])
    assert assess_input_structure(peptide((1,), chains=["A"]), source)["can_proceed"]


@pytest.fixture
def client(tmp_path, monkeypatch, empty_default_lipid_library):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_schedule_propka_precompute", lambda *a: None)
    monkeypatch.setattr(server.ligand_prep, "start", lambda *a: None)
    server._step_runners.clear()
    with TestClient(server.app) as client:
        yield client, manager
    server._step_runners.clear()


def upload(client, tmp_path, structure, workflow):
    path = tmp_path / "source.pdb"
    PDBWriter.write(structure, path)
    response = client.post(
        "/api/upload-pdb",
        data={"task_type": workflow},
        files={"file": (path.name, path.read_bytes(), "chemical/x-pdb")},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("workflow", ["membrane-bilayer", "solvator"])
def test_upload_discloses_and_check_blocks_without_checkpoint(client, tmp_path, workflow):
    api, manager = client
    data = upload(api, tmp_path, peptide(), workflow)
    assert not data["input_validation"]["can_proceed"]
    tid = data["task_id"]
    checked = api.post(f"/api/step/{tid}/input", json={"config": {}}).json()
    assert checked["status"] == "error"
    assert "Protein chain A is broken" in checked["error"]
    assert not checked["input_validation"]["can_proceed"]
    assert not (manager.get_task_dir(tid) / "steps/input/system.json").exists()
    bypass = api.post(f"/api/step/{tid}/forcefield", json={"config": {}}).json()
    assert bypass["status"] == "error" and bypass["input_check_required"]
    build = api.post("/api/build", json={"task_id": tid, "task_type": workflow, "modules": {}})
    assert build.status_code == 409, build.text
    assert build.json()["input_check_required"]
    steps = api.get(f"/api/steps/{tid}").json()["steps"]
    assert not any(step["has_checkpoint"] for step in steps)


def test_deselecting_damaged_chain_then_checking_can_succeed(client, tmp_path):
    api, manager = client
    data = upload(api, tmp_path, peptide((1, 128, 135), chains=["A", "R", "R"]), "solvator")
    tid = data["task_id"]
    assert not data["input_validation"]["can_proceed"]
    assert api.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A"]}).status_code == 200
    checked = api.post(f"/api/step/{tid}/input", json={"config": {}}).json()
    assert checked["status"] == "ok", checked
    assert checked["metrics"]["input_validation"]["can_proceed"]
    assert StepRunner(manager.get_task_dir(tid), "solvator").has_checkpoint("input")


def test_failed_recheck_revokes_existing_passes_and_keeps_specific_reason(tmp_path):
    runner = StepRunner(tmp_path / "task", "solvator")
    good = tmp_path / "good.pdb"
    PDBWriter.write(peptide((1,)), good)
    assert runner.run_step("input", {}, pdb_path=str(good))["status"] == "ok"
    system = runner.load_system("input")
    system.save_checkpoint(runner.step_dir("forcefield"))
    assert runner.has_checkpoint("forcefield")
    bad = tmp_path / "bad.pdb"
    PDBWriter.write(peptide(), bad)
    result = runner.run_step("input", {}, pdb_path=str(bad))
    assert "Protein chain A is broken" in result["error"]
    assert result["invalidated_steps"] == ["input", "forcefield"]
    assert not runner.has_checkpoint("input")
    assert runner.finalize_from_checkpoint("ions")["input_check_required"]


def test_legacy_checkpoints_require_recheck_without_deleting_preview(tmp_path):
    runner = StepRunner(tmp_path, "solvator")
    System(structure=peptide()).save_checkpoint(runner.step_dir("input"))
    assert not runner.has_checkpoint("input")
    assert (runner.step_dir("input") / "system.npz").exists()
    assert runner.run_step("forcefield", {})["input_check_required"]
    # An old schema pass cannot substitute for this policy either.
    path = runner.step_dir("input") / "system.json"
    data = json.loads(path.read_text())
    data["metadata"]["input_validation"] = {
        "policy_version": INPUT_VALIDATION_VERSION - 1,
        "can_proceed": True,
    }
    path.write_text(json.dumps(data))
    assert not runner.has_checkpoint("input")


def test_source_records_survive_upload_filter_and_legacy_resume(client, tmp_path):
    api, manager = client
    path = tmp_path / "deposited.pdb"
    PDBWriter.write(peptide(), path)
    path.write_text(
        "SEQRES   1 R    3  GLY SER GLY\nREMARK 465     SER R   129\n" + path.read_text()
    )
    data = api.post(
        "/api/upload-pdb",
        data={"task_type": "solvator"},
        files={"file": (path.name, path.read_bytes(), "chemical/x-pdb")},
    ).json()
    tid = data["task_id"]
    assert any(e["code"] == "deposited_missing_segment" for e in data["input_validation"]["errors"])
    assert api.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A"]}).status_code == 200
    result = api.post(f"/api/step/{tid}/input", json={"config": {}}).json()
    assert "SER 129" in result["error"]
    saved = manager.get_state(tid)
    assert len(saved["input_source_metadata"]["sequences"]["R"]) == 3
    saved["structure_summary"].pop("input_validation")
    manager.update_state(
        tid,
        {
            "input_source_metadata": None,
            "structure_summary": saved["structure_summary"],
            "steps_completed": ["input", "forcefield"],
        },
    )
    resumed = api.get(f"/api/task/{tid}/resume").json()
    assert resumed["input_check_required"]
    assert resumed["steps_completed"] == []
    assert "SER 129" in str(resumed["input_validation"])


@pytest.mark.parametrize("workflow", ["membrane-bilayer", "solvator"])
def test_explicit_fragments_and_renumbering_survive_check_and_resume(client, tmp_path, workflow):
    api, manager = client
    uploaded = upload(api, tmp_path, peptide(), workflow)
    tid = uploaded["task_id"]
    original = manager.get_filter_source(tid).read_bytes()
    renumber_only = api.post(
        f"/api/step/{tid}/input", json={"config": {"renumber_residues": True}}
    ).json()
    assert renumber_only["status"] == "error"
    config = {
        "allow_incomplete_protein": True,
        "renumber_residues": True,
        "chain_names": {"A": "Z"},
    }
    checked = api.post(f"/api/step/{tid}/input", json={"config": config}).json()
    assert checked["status"] == "ok", checked
    system = System.load_checkpoint(manager.get_task_dir(tid) / "steps/input")
    assert set(system.structure.chain_ids) == {"Z", "A"}
    assert set(system.structure.resids) == {1}
    assert checked["metrics"]["input_reconstruction"]["splits"]
    summary = checked["metrics"]["input_summary"]
    assert summary["num_atoms"] == system.num_atoms
    assert summary["chains"] == ["Z", "A"]
    assert [c["length"] for c in summary["sequences"]] == [1, 1]
    assert summary["small_molecules"] == []
    resumed = api.get(f"/api/task/{tid}/resume").json()
    assert resumed["step_input_config"]["chain_names"] == {"A": "Z"}
    assert resumed["input_selection_summary"]["chains"] == ["A"]
    assert set(resumed["pdb_info_full"]["chains"]) == {"Z", "A"}
    assert manager.get_filter_source(tid).read_bytes() == original
    assert api.post(f"/api/filter-pdb/{tid}", json={"include_chains": ["A"]}).status_code == 200
    repeated = api.post(f"/api/step/{tid}/input", json={"config": config}).json()
    assert repeated["status"] == "ok", repeated


@pytest.mark.parametrize("workflow", ["membrane-bilayer", "solvator", "martini3-solvent"])
def test_fragment_edit_and_reinclusion_reach_checkpoint_and_resume(client, tmp_path, workflow):
    api, manager = client
    tid = upload(api, tmp_path, peptide(), workflow)["task_id"]
    source = manager.get_filter_source(tid)
    original = source.read_bytes()
    config = {
        "allow_incomplete_protein": True,
        "renumber_residues": True,
        "fragment_names": {"A:1": "X", "A:2": "Y"},
        "exclude_fragments": ["A:1"],
    }
    if workflow == "martini3-solvent":
        config.update(include_protein=True, environment="solution")
    for excluded in (["A:1"], [], ["A:2"]):
        config["exclude_fragments"] = excluded
        response = api.post(f"/api/step/{tid}/input", json={"config": config})
        checked = response.json()
        assert checked["status"] == "ok", checked
        summary = checked["metrics"]["input_summary"]
        expected = [label for key, label in config["fragment_names"].items() if key not in excluded]
        assert summary["chains"] == expected
        assert [row["chain_id"] for row in summary["fragment_choices"]] == ["X", "Y"]
        checkpoint = System.load_checkpoint(manager.get_task_dir(tid) / "steps/input")
        assert set(checkpoint.structure.chain_ids) == set(expected)
        resumed = api.get(f"/api/task/{tid}/resume").json()
        assert resumed["step_input_config"]["exclude_fragments"] == excluded
        assert [row["fragment_key"] for row in resumed["input_fragment_choices"]] == ["A:1", "A:2"]
        assert source.read_bytes() == original

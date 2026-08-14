"""Compatibility and safety regressions for Step 1 structure input."""

from __future__ import annotations

import asyncio
import gzip
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.exceptions import ParseError
from gmxbuilder.core.system import System
from gmxbuilder.io.cif import CIFParser
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager
from gmxbuilder.web.task_types import get_task_type_detail


PDB_TEXT = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
    "END\n"
)

MMCIF_TEXT = """data_model
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N  ALA X 1 N  ALA B 10 0.000 0.000 0.000 1.00 20.0 5
ATOM 2 C CA ALA X 1 CA ALA B 10 1.450 0.000 0.000 1.00 20.0 5
ATOM 3 N N  GLY X 2 N  GLY B 11 9.000 9.000 9.000 1.00 20.0 6
#
"""


def test_pdb_parser_selects_first_model_by_order_and_accepts_bom_crlf(tmp_path):
    path = tmp_path / "ensemble.ent"
    text = (
        "\ufeffMODEL        5\r\n"
        + PDB_TEXT.replace("END\n", "ENDMDL\r\n")
        + "MODEL        9\r\n"
        + PDB_TEXT.replace("  1       0.000", "  1       9.000").replace(
            "END\n", "ENDMDL\r\n"
        )
    )
    path.write_text(text, encoding="utf-8")

    structure = PDBParser().parse(path)

    assert structure.num_atoms == 2
    assert np.allclose(structure.coordinates[0], [0.0, 0.0, 0.0])


def test_mmcif_prefers_author_identifiers_selects_first_model_and_estimates_box(
    tmp_path,
):
    path = tmp_path / "model.mmcif"
    path.write_text(MMCIF_TEXT)

    structure = CIFParser().parse(path)

    assert structure.num_atoms == 2
    assert structure.atom_names == ["N", "CA"]
    assert structure.resnames == ["ALA", "ALA"]
    assert structure.chain_ids == ["B", "B"]
    assert structure.resids == [10, 10]
    assert np.allclose(np.diag(structure.box_vectors), [3.0, 3.0, 3.0])


def test_mmcif_selects_highest_occupancy_altloc_and_rejects_insertion(tmp_path):
    header = """data_alt
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
"""
    alternate = tmp_path / "alternate.cif"
    alternate.write_text(
        header
        + "ATOM 1 C CA A ALA A 10 ? 1.0 0.0 0.0 0.40\n"
        + "ATOM 2 C CA B ALA A 10 ? 2.0 0.0 0.0 0.60\n#\n"
    )
    structure = CIFParser().parse(alternate)
    assert structure.num_atoms == 1
    assert structure.coordinates[0, 0] == pytest.approx(0.2)

    insertion = tmp_path / "insertion.cif"
    insertion.write_text(
        header + "ATOM 1 C CA . ALA A 10 A 1.0 0.0 0.0 1.00\n#\n"
    )
    with pytest.raises(ParseError, match="insertion codes"):
        CIFParser().parse(insertion)


def test_upload_accepts_mmcif_and_preserves_original_for_input_step(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    server._step_runners.clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/upload-pdb",
            files={"file": ("author-model.mmcif", MMCIF_TEXT, "chemical/x-mmcif")},
            data={"task_type": "membrane-bilayer"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["structure_format"] == "mmCIF"
    assert payload["chains"] == ["B"]
    state = manager.get_state(payload["task_id"])
    assert state["uploaded_structure_name"].endswith(".cif")
    assert Path(server._resolve_input_pdb(payload["task_id"])).suffix == ".cif"
    assert manager.get_pdb_path(payload["task_id"]).name == "converted.pdb"


def test_coarse_grained_mmcif_check_writes_canonical_mapping_pdb(
    tmp_path, monkeypatch
):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    server._step_runners.clear()

    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"task_type": "martini3-bilayer"})
        assert created.status_code == 200, created.text
        task_id = created.json()["task_id"]
        uploaded = client.post(
            "/api/upload-pdb",
            files={"file": ("cg-model.mmcif", MMCIF_TEXT, "chemical/x-mmcif")},
            data={"task_type": "martini3-bilayer", "task_id": task_id},
        )
        assert uploaded.status_code == 200, uploaded.text
        checked = client.post(
            f"/api/step/{task_id}/input",
            json={"config": {"include_protein": True, "environment": "bilayer"}},
        )

    assert checked.status_code == 200, checked.text
    assert checked.json()["status"] == "ok", checked.text
    canonical = manager.get_task_dir(task_id) / "steps" / "input" / "cg_input.pdb"
    text = canonical.read_text()
    assert text.startswith("HEADER    Martini 3 atomistic input")
    assert "\nATOM" in text
    assert not text.startswith("data_")


def test_upload_accepts_bounded_gzip_pdb(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)

    with TestClient(app) as client:
        response = client.post(
            "/api/upload-pdb",
            files={"file": ("model.pdb.gz", gzip.compress(PDB_TEXT.encode()), "application/gzip")},
            data={"task_type": "solvator"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    state = manager.get_state(payload["task_id"])
    assert payload["structure_format"] == "PDB"
    assert state["uploaded_structure_name"] == "model.pdb"


def test_upload_rejects_invalid_gzip_with_explicit_error(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)

    with TestClient(app) as client:
        response = client.post(
            "/api/upload-pdb",
            files={"file": ("broken.cif.gz", b"not gzip", "application/gzip")},
            data={"task_type": "solvator"},
        )

    assert response.status_code == 400
    assert "valid gzip" in response.json()["error"]


def test_task_manager_prefers_converted_pdb_over_legacy_cif_dot_pdb(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("legacy.cif")
    task_dir = manager.get_task_dir(task["task_id"])
    (task_dir / "legacy.cif.pdb").write_text(MMCIF_TEXT)
    (task_dir / "converted.pdb").write_text(PDB_TEXT)

    assert manager.get_pdb_path(task["task_id"]).name == "converted.pdb"


def test_completed_task_resume_exposes_existing_package_and_result(tmp_path, monkeypatch):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("complete.pdb")
    task_id = task["task_id"]
    detail = get_task_type_detail("membrane-bilayer")
    manager.update_state(task_id, {
        "task_type": detail,
        "task_type_id": "membrane-bilayer",
        "current_step": "simparams",
        "build_status": {
            "status": "completed",
            "result": {"task_id": task_id, "num_atoms": 2, "components": [], "log": []},
        },
    })
    source = System(Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3) * 3.0,
        atom_names=["CA"], resnames=["ALA"], resids=[1],
        chain_ids=["A"], elements=["C"],
    ))
    source.save_checkpoint(manager.get_task_dir(task_id) / "steps" / "ions")
    export = manager.get_task_dir(task_id) / "steps" / "export"
    export.mkdir(parents=True)
    (export / "complete.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(server, "task_manager", manager)
    server._step_runners.pop(task_id, None)

    resumed = asyncio.run(server.api_task_resume(task_id))

    assert resumed["resume_step"] == "simparams"
    assert resumed["build_status"]["download_available"] is True
    assert resumed["build_status"]["result"]["download_url"] == (
        f"/api/task/{task_id}/download"
    )

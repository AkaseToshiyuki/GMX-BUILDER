"""Compatibility and safety regressions for Step 1 structure input."""

from __future__ import annotations

import asyncio
import gzip
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.cif import CIFParser
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager
from gmxbuilder.web.task_types import get_task_type_detail
from tests.prerequisites import requires_lfs_assets

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
        + PDB_TEXT.replace("  1       0.000", "  1       9.000").replace("END\n", "ENDMDL\r\n")
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


def test_mmcif_selects_highest_occupancy_altloc_and_maps_insertion(tmp_path):
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
    insertion.write_text(header + "ATOM 1 C CA . ALA A 10 A 1.0 0.0 0.0 1.00\n#\n")
    inserted = CIFParser().parse(insertion)
    assert inserted.resids == [1]
    assert inserted.source_info["residue_mapping"][0]["insertion_code"] == "A"
    assert inserted.source_info["residue_mapping"][0]["author_resid"] == "10"


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
    assert payload["chains"] == ["A"]
    assert payload["chain_mapping"] == {"B": "A"}
    state = manager.get_state(payload["task_id"])
    assert state["uploaded_structure_name"].endswith(".cif")
    original = manager.get_task_dir(payload["task_id"]) / state["uploaded_structure_name"]
    assert original.read_text() == MMCIF_TEXT
    assert Path(server._resolve_input_pdb(payload["task_id"])).suffix == ".cif"
    assert manager.get_pdb_path(payload["task_id"]).name == "converted.pdb"


@requires_lfs_assets
def test_coarse_grained_mmcif_check_writes_canonical_mapping_pdb(tmp_path, monkeypatch):
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


@pytest.mark.parametrize("current_input", [False, True])
def test_completed_task_resume_exposes_existing_package_and_result(
    tmp_path, monkeypatch, small_pdb_file, current_input
):
    manager = TaskManager(tmp_path / "tasks")
    task = manager.create_task("complete.pdb")
    task_id = task["task_id"]
    detail = get_task_type_detail("membrane-bilayer")
    manager.update_state(
        task_id,
        {
            "task_type": detail,
            "task_type_id": "membrane-bilayer",
            "current_step": "simparams",
            "build_status": {
                "status": "completed",
                "result": {"task_id": task_id, "num_atoms": 2, "components": [], "log": []},
            },
        },
    )
    source = System(
        Structure(
            coordinates=np.zeros((1, 3)),
            box_vectors=np.eye(3) * 3.0,
            atom_names=["CA"],
            resnames=["ALA"],
            resids=[1],
            chain_ids=["A"],
            elements=["C"],
        )
    )
    if current_input:
        from gmxbuilder.pipeline.step_executor import StepRunner

        runner = StepRunner(manager.get_task_dir(task_id), "membrane-bilayer")
        assert runner.run_step("input", {}, pdb_path=str(small_pdb_file))["status"] == "ok"
    source.save_checkpoint(manager.get_task_dir(task_id) / "steps" / "ions")
    export = manager.get_task_dir(task_id) / "steps" / "export"
    export.mkdir(parents=True)
    (export / "complete.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(server, "task_manager", manager)
    server._step_runners.pop(task_id, None)

    resumed = asyncio.run(server.api_task_resume(task_id))

    assert resumed["resume_step"] == ("simparams" if current_input else "input")
    assert resumed["input_check_required"] is not current_input
    assert resumed["build_status"]["download_available"] is True
    assert resumed["build_status"]["result"]["download_url"] == (f"/api/task/{task_id}/download")


# Column-aligned exactly as RCSB writes mmCIF. The compact fixture above uses
# single-space separators, which is why it never exercised the defect below:
# with one space, line[:6] is "ATOM 1" and does not match the PDB record test,
# while real files pad to "ATOM  " and do.
RCSB_STYLE_MMCIF = """data_7LCK
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1    N N   . ALA A 1 37  ? 140.087 168.862 129.079 1.00 52.89  ? 28  ALA R N   1
ATOM   2    C CA  . ALA A 1 37  ? 138.818 168.358 128.569 1.00 52.89  ? 28  ALA R CA  1
ATOM   3    C C   . ALA A 1 37  ? 138.000 169.500 128.000 1.00 52.89  ? 28  ALA R C   1
HETATM 4    O O   . HOH B 2 .   ? 150.000 150.000 150.000 1.00 30.00  ? 101 HOH R O   1
#
"""


def test_column_aligned_mmcif_is_not_mistaken_for_pdb():
    """An mmCIF atom loop begins each row with ATOM or HETATM.

    That is the group_PDB column, so every real mmCIF also looks like it holds
    PDB records. Detection that required their absence classified any mmCIF
    containing coordinates as PDB and handed it to the PDB parser, which then
    reported "no parseable coordinates". `_atom_site.` category tags are the
    discriminator: a PDB file never has them.
    """
    from gmxbuilder.web.server_parts.structure_processing import prepare_structure_upload

    name, _content, detected, warnings = prepare_structure_upload(
        "7LCK.cif", RCSB_STYLE_MMCIF.encode("utf-8"), 32 * 1024 * 1024
    )
    assert detected == "cif"
    assert name.endswith(".cif")
    assert not warnings, warnings


def test_column_aligned_mmcif_uploads_and_reaches_the_next_step():
    """The report was that a CIF upload could not proceed, so drive it through."""
    from fastapi.testclient import TestClient

    from gmxbuilder.web.server import app

    with TestClient(app) as client:
        upload = client.post(
            "/api/upload-pdb",
            files={"file": ("7LCK.cif", RCSB_STYLE_MMCIF.encode("utf-8"))},
        )
        assert upload.status_code == 200, upload.json()
        task_id = upload.json()["task_id"]

        preview = client.post("/api/preview-pdb", json={"task_id": task_id})
        assert preview.status_code == 200, preview.json()


def test_a_real_pdb_file_is_still_detected_as_pdb():
    """The fix must not make everything look like mmCIF."""
    from gmxbuilder.web.server_parts.structure_processing import prepare_structure_upload

    _name, _content, detected, _warnings = prepare_structure_upload(
        "model.pdb", PDB_TEXT.encode("utf-8"), 32 * 1024 * 1024
    )
    assert detected == "pdb"


def test_mmcif_content_under_a_pdb_filename_is_still_detected_as_mmcif():
    """Content wins over the extension, and the mismatch is reported."""
    from gmxbuilder.web.server_parts.structure_processing import prepare_structure_upload

    name, _content, detected, warnings = prepare_structure_upload(
        "mislabelled.pdb", RCSB_STYLE_MMCIF.encode("utf-8"), 32 * 1024 * 1024
    )
    assert detected == "cif"
    assert name.endswith(".cif")
    assert any("detected as CIF" in warning for warning in warnings)

from fastapi.testclient import TestClient

from pathlib import Path
import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.pdb import PDBParser, PDBValidator
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.modules.membrane.orient_module import OrientModule
from gmxbuilder.web import server
from gmxbuilder.web.server import app, task_manager
from gmxbuilder.web.task_types import get_task_type_detail


def test_frontend_residue_indices_preserve_coordinate_encounter_order():
    structure = Structure(
        coordinates=np.zeros((4, 3)),
        box_vectors=np.eye(3),
        atom_names=["N", "CA", "N", "CA"],
        resnames=["TYR", "TYR", "ALA", "ALA"],
        resids=[20, 20, 10, 10],
        chain_ids=["A"] * 4,
        elements=["N", "C", "N", "C"],
    )

    sequences = server._extract_sequences(structure)

    assert [item["resname"] for item in sequences[0]["residues"]] == ["TYR", "ALA"]


def test_task_resume_uses_standardized_input_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="modified.pdb")
    task_id = task["task_id"]
    raw = (
        "ATOM      1  N   SEP A  10       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  SEP A  10       1.458   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    ).encode()
    uploaded = task_manager.save_uploaded_pdb(task_id, "modified.pdb", raw)
    task_manager.update_state(
        task_id,
        {
            "uploaded_structure_name": uploaded.name,
            "task_type_id": "membrane-bilayer",
            "task_type": get_task_type_detail("membrane-bilayer"),
            "pdb_info": {"filename": "modified.pdb"},
        },
    )
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [0.1458, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["N", "CA"],
        resnames=["SER", "SER"],
        resids=[10, 10],
        chain_ids=["A", "A"],
        elements=["N", "C"],
    )
    system = System(
        structure,
        components=[
            Component(
                "PROTEIN",
                ComponentKind.PROTEIN,
                np.array([0, 1]),
                {},
            )
        ],
    )
    checkpoint = tmp_path / task_id / "steps" / "input"
    system.save_checkpoint(checkpoint)
    system.write_viewer_pdb(checkpoint / "viewer.pdb")

    with TestClient(app) as client:
        response = client.get(f"/api/task/{task_id}/resume")

    assert response.status_code == 200, response.text
    residues = response.json()["sequences"][0]["residues"]
    assert residues == [{"resname": "SER", "resid": 10, "is_protein": True}]


def test_step_one_lists_crystallographic_additives_for_explicit_removal(tmp_path):
    pdb = tmp_path / "additives.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2 NA    NA A   2       1.000   0.000   0.000  1.00  0.00          NA\n"
        "HETATM    3  C1  ACT A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    4  O   HOH A   4       3.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    5  C1  PLC A   5       4.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    molecules = PDBValidator.detect_small_molecules(pdb)

    assert {(item["resname"], item["category"]) for item in molecules} == {
        ("NA", "ion_or_solvent_additive"),
        ("ACT", "ion_or_solvent_additive"),
        ("PLC", "small_molecule"),
    }


def test_step_endpoint_rejects_malformed_json(monkeypatch):
    monkeypatch.setattr(task_manager, "get_state", lambda _task_id: {})
    with TestClient(app) as client:
        response = client.post(
            "/api/step/abc123def456/input",
            content="{bad json",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "Request body must be valid JSON"


def test_step_endpoint_rejects_non_object_config(monkeypatch):
    monkeypatch.setattr(task_manager, "get_state", lambda _task_id: {})
    with TestClient(app) as client:
        response = client.post(
            "/api/step/abc123def456/input",
            json={"config": ["not", "an", "object"]},
        )
    assert response.status_code == 400
    assert "must be an object" in response.json()["error"]


def test_build_accepts_stage_owned_default_simulation_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="default.pdb")
    task_id = task["task_id"]
    task_manager.update_state(
        task_id,
        {
            "task_type_id": "membrane-bilayer",
            "task_type": get_task_type_detail("membrane-bilayer"),
            "step_forcefield_config": {"name": "charmm36m"},
        },
    )
    captured_context = {}
    normalize = MDPWriter.normalize_simulation_config

    def capture_normalization(config, context=None):
        captured_context.update(context or {})
        return normalize(config, context)

    monkeypatch.setattr(
        MDPWriter, "normalize_simulation_config", staticmethod(capture_normalization)
    )
    stage = {
        "enabled": True,
        "bb": 0,
        "sc": 0,
        "lipid": 0,
        "dih": 0,
        "dt": 2.0,
        "dt_unit": "fs",
        "nsteps": 1000,
        "ensemble": "npt",
        "comm_grps": "System",
    }
    payload = {
        "task_id": task_id,
        "task_type": "membrane-bilayer",
        "system_name": "default_system",
        "modules": {
            "simparams": {
                "schema_version": 2,
                "minimization": {
                    "integrator": "steep",
                    "nsteps": 5000,
                    "emtol": 1000.0,
                    "emstep": 0.01,
                    "nstlist": 10,
                    "constraints": "h-bonds",
                    "bb": 0,
                    "sc": 0,
                    "lipid": 0,
                    "dih": 0,
                },
                "eq_stages": [stage],
                "prod_iters": [stage],
            },
            "execution": {
                "cpu_threads": 1,
                "mpi_ranks": 1,
                "use_gpu": False,
                "gpu_count": 0,
                "gpu_ids": "",
            },
        },
    }

    with TestClient(app) as client:
        response = client.post("/api/build", json=payload)

    assert response.status_code == 409, response.text
    assert "Ions Check is missing" in response.json()["error"]
    assert captured_context["force_field"] == "charmm36m"
    assert captured_context["force_field_family"] == "charmm"


def test_propka_uses_input_checkpoint_not_structure_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="input.pdb")
    task_dir = task_manager.get_task_dir(task["task_id"])
    input_viewer = task_dir / "steps" / "input" / "viewer.pdb"
    structure_viewer = task_dir / "steps" / "structure" / "viewer.pdb"
    input_viewer.parent.mkdir(parents=True)
    structure_viewer.parent.mkdir(parents=True)
    input_viewer.write_text("INPUT\n")
    structure_viewer.write_text("STRUCTURE\n")

    resolved = server._resolve_propka_pdb_path(task["task_id"])

    assert Path(resolved) == input_viewer


def test_protonate_endpoint_rejects_out_of_range_ph():
    with TestClient(app) as client:
        response = client.post(
            "/api/protonate",
            json={"residues": ["HIS"], "pH": 0.0, "his_tautomer": "HSE"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "pH must be between 1.0 and 13.0"


def test_protonate_endpoint_recalculates_discrete_states_for_changed_ph():
    payload = {
        "residues": ["ASP", "GLU", "HIS", "LYS", "CYS", "TYR"],
        "his_tautomer": "HSE",
    }
    with TestClient(app) as client:
        acidic = client.post("/api/protonate", json={**payload, "pH": 2.0})
        basic = client.post("/api/protonate", json={**payload, "pH": 12.0})

    assert acidic.status_code == 200
    assert basic.status_code == 200
    acidic_states = {
        item["original"]: (item["assigned_name"], item["charge"])
        for item in acidic.json()["assignments"]
    }
    basic_states = {
        item["original"]: (item["assigned_name"], item["charge"])
        for item in basic.json()["assignments"]
    }
    assert acidic_states["ASP"] == ("ASH", 0)
    assert basic_states["ASP"] == ("ASP", -1)
    assert acidic_states["LYS"] == ("LYS", 1)
    assert basic_states["LYS"] == ("LYN", 0)
    assert acidic_states != basic_states


def test_protonate_endpoint_reports_failed_environment_sensitive_fallback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="input.pdb")
    task_manager.save_uploaded_pdb(task["task_id"], "input.pdb", b"END\n")

    async def empty_propka(_path):
        return []

    monkeypatch.setattr(server, "_get_propka_results", empty_propka)

    with TestClient(app) as client:
        response = client.post(
            "/api/protonate",
            json={
                "residues": ["ASP"],
                "pH": 7.0,
                "his_tautomer": "HSE",
                "task_id": task["task_id"],
            },
        )

    assert response.status_code == 200
    assert response.json()["propka_requested"] is True
    assert "could not produce" in response.json()["propka_warning"]


def test_options_advertise_validated_lipid21_but_reject_gaff2_chol():
    with TestClient(app) as client:
        response = client.get("/api/options")

    assert response.status_code == 200
    chol = next(item for item in response.json()["lipids"] if item["name"] == "CHOL")
    assert "lipid21" in chol["parameterizations"]
    assert "gaff2" not in chol["parameterizations"]
    assert "charmm36m" in chol["parameterizations"]
    assert "Amber Lipid21" in chol["parameterization"]
    assert "93.8% correctly oriented" in chol["gaff2_unavailable_reason"]


def test_options_reject_thin_gaff2_bsm_and_advertise_exact_charmm():
    with TestClient(app) as client:
        response = client.get("/api/options")

    assert response.status_code == 200
    bsm = next(item for item in response.json()["lipids"] if item["name"] == "BSM")
    assert "gaff2" not in bsm["parameterizations"]
    assert set(bsm["parameterizations"]) == {"charmm36m", "charmm36"}
    assert "experimental C24:0 sphingomyelin DHH" in bsm["gaff2_unavailable_reason"]
    assert "use CHARMM36m or CHARMM36" in bsm["gaff2_unavailable_reason"]


def test_options_advertise_exact_lipid21_popc_support():
    with TestClient(app) as client:
        response = client.get("/api/options")

    assert response.status_code == 200
    popc = next(item for item in response.json()["lipids"] if item["name"] == "POPC")
    assert popc["parameterizations"][0] == "lipid21"
    assert "Amber Lipid21 v1.0 (exact)" in popc["parameterization"]


def test_options_mark_validated_gm1_and_oxysterol_alternatives():
    with TestClient(app) as client:
        response = client.get("/api/options")

    assert response.status_code == 200
    lipids = {item["name"]: item for item in response.json()["lipids"]}
    assert set(lipids["GM1"]["parameterizations"]) == {"charmm36m", "charmm36"}
    assert lipids["GM1"]["charge"] == -1
    assert lipids["20AHC"]["parameterizations"] == []
    assert "no validated bundled alternative" in lipids["20AHC"]["gaff2_unavailable_reason"]


def test_crosslink_capabilities_are_force_field_specific_and_scientific():
    with TestClient(app) as client:
        amber = client.get("/api/crosslink-capabilities?force_field=amber14sb")
        charmm = client.get("/api/crosslink-capabilities?force_field=charmm36m")

    assert amber.status_code == 200
    amber_disulfide = amber.json()["disulfide"]
    assert amber_disulfide["supported"] is True
    assert 0.19 < amber_disulfide["target_distance_nm"] < 0.22
    assert charmm.status_code == 200
    assert charmm.json()["disulfide"]["supported"] is False
    assert "force-field-native cross-residue patch" in charmm.json()["disulfide"]["reason"]


def test_ligand_charge_suggestion_endpoint_uses_target_ph(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="ligand.pdb")
    task_id = task["task_id"]
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [0.145, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C01", "N02"],
        resnames=["LIG", "LIG"],
        resids=[1, 1],
        chain_ids=["L", "L"],
        elements=["C", "N"],
    )
    system = System(
        structure,
        components=[
            Component(
                "UNKNOWN",
                ComponentKind.UNKNOWN,
                np.array([0, 1]),
                {},
            )
        ],
    )
    system.save_checkpoint(tmp_path / task_id / "steps" / "input")

    with TestClient(app) as client:
        response = client.post(
            f"/api/ligand-charge-suggestions/{task_id}",
            json={"pH": 7.0},
        )

    assert response.status_code == 200, response.text
    suggestion = response.json()["suggestions"]["LIG"]
    assert suggestion["net_charge"] == 1
    assert suggestion["formula"].endswith("+")
    assert "suggestion" in suggestion["warning"]


def test_orientation_preview_matches_real_step_module(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.orient._find_best_ppm_orientation",
        lambda _structure, half_thickness=None: (
            np.array([0.0, 0.0, 1.0]),
            0.0,
            np.array([1.0, 0.0, 0.0]),
            0.0,
            0.0,
            0.0,
        ),
    )
    task = task_manager.create_task(filename="protein.pdb")
    task_id = task["task_id"]
    structure = Structure(
        coordinates=np.array(
            [
                [-0.3, 0.0, -1.0],
                [0.2, 0.1, -0.4],
                [-0.1, -0.2, 0.3],
                [0.3, 0.0, 1.1],
            ]
        ),
        box_vectors=np.eye(3) * 8.0,
        atom_names=["CA"] * 4,
        resnames=["LEU", "ILE", "VAL", "PHE"],
        resids=[1, 2, 3, 4],
        chain_ids=["A"] * 4,
        elements=["C"] * 4,
    )
    system = System(
        structure=structure,
        components=[Component("PROTEIN", ComponentKind.PROTEIN, np.arange(4))],
        metadata={"seed": 42},
    )
    system.save_checkpoint(tmp_path / task_id / "steps" / "structure")
    config = {"method": "manual", "z_offset": 0.35, "tilt": 17.0, "phi": 65.0}
    expected = OrientModule().run(system.copy(), config).system

    with TestClient(app) as client:
        response = client.post(
            f"/api/orient-preview/{task_id}",
            json={"config": config},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    preview_path = tmp_path / "preview.pdb"
    preview_path.write_text(payload["oriented_pdb"], encoding="utf-8")
    preview = PDBParser().parse(preview_path)
    assert payload["orientation"] == expected.metadata["_orient_params"]
    assert np.allclose(
        preview.coordinates,
        expected.structure.coordinates,
        atol=1.1e-4,
    )

"""Regression coverage for persistent small-molecule display labels."""

import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.web.server import (
    _normalise_small_molecule_labels,
    app,
    task_manager,
)


_LIGAND_PDB = Path(__file__).parent / "fixtures" / "small_molecule_label.pdb"


def _ligand_system() -> System:
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C1", "O1"],
        resnames=["AMP", "AMP"],
        resids=[101, 101],
        chain_ids=["A", "A"],
        elements=["C", "O"],
    )
    return System(
        structure=structure,
        components=[
            Component("SMALL_MOLECULES", ComponentKind.LIGAND, np.array([0, 1]))
        ],
    )


def test_small_molecule_label_validation_rejects_ambiguous_names():
    try:
        _normalise_small_molecule_labels(
            {"AMP": "ligand", "LIG": "LIGAND"}, {"AMP", "LIG"}
        )
    except ValueError as exc:
        assert "used for both" in str(exc)
    else:
        raise AssertionError("Duplicate display labels must be rejected")


def test_label_persists_and_is_returned_by_forcefield_report(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="ligand.pdb")
    task_id = task["task_id"]
    task_manager.save_uploaded_pdb(task_id, "ligand.pdb", _LIGAND_PDB.read_bytes())

    with TestClient(app) as client:
        filtered = client.post(
            f"/api/filter-pdb/{task_id}",
            json={
                "include_chains": ["A"],
                "exclude_resnames": [],
                "small_molecule_labels": {"AMP": "Adenosine ligand"},
            },
        )
        assert filtered.status_code == 200, filtered.text

        checkpoint = tmp_path / task_id / "steps" / "input"
        _ligand_system().save_checkpoint(checkpoint)
        report = client.post(
            f"/api/forcefield-compatibility/{task_id}",
            json={"protein_ff": "amber14sb", "lipid_names": []},
        )

    assert task_manager.get_state(task_id)["small_molecule_labels"] == {
        "AMP": "Adenosine ligand"
    }
    assert report.status_code == 200, report.text
    payload = report.json()
    assert payload["ligand_names"] == ["AMP"]
    assert payload["ligand_labels"] == {"AMP": "Adenosine ligand"}
    assert payload["ligands"][0]["display_name"] == "Adenosine ligand"

"""Security regressions for task-scoped web resources and UI state."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gmxbuilder.web import server
from gmxbuilder.web.server import app, task_manager


def _task(tmp_path, monkeypatch, name: str) -> str:
    monkeypatch.setattr(task_manager, "root", tmp_path)
    created = task_manager.create_task(f"{name}.pdb")
    task_id = created["task_id"]
    task_manager.update_state(task_id, {
        "task_type_id": "membrane-bilayer",
        "task_type": {
            "id": "membrane-bilayer",
            "route_slug": "BilayerBuilder",
            "visible_modules": [
                "input", "forcefield", "structure", "orient", "membrane",
                "solvation", "ions", "simparams",
            ],
        },
    })
    task_manager.save_uploaded_pdb(
        task_id,
        f"{name}.pdb",
        (
            b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            b"  1.00  0.00           C\nEND\n"
        ),
    )
    return task_id


def test_client_supplied_paths_are_rejected_even_inside_another_task(
    tmp_path, monkeypatch
):
    task_a = _task(tmp_path, monkeypatch, "a")
    task_b = _task(tmp_path, monkeypatch, "b")
    other_path = str(task_manager.get_pdb_path(task_b))

    with TestClient(app) as client:
        orient = client.post(
            "/api/orient-ppm",
            json={"task_id": task_a, "tmp_path": other_path, "algorithm": "ppm"},
        )
        protonate = client.post(
            "/api/protonate",
            json={
                "task_id": task_a,
                "tmp_path": other_path,
                "residues": ["ASP"],
                "pH": 7.0,
                "his_tautomer": "HSE",
            },
        )
        modifications = client.post(
            "/api/apply-modifications",
            json={"task_id": task_a, "tmp_path": other_path},
        )

    assert orient.status_code == 400
    assert protonate.status_code == 400
    assert modifications.status_code == 400
    assert "filesystem paths" in orient.json()["error"]
    assert "filesystem paths" in protonate.json()["error"]
    assert "filesystem paths" in modifications.json()["error"]


def test_task_resource_validator_rejects_cross_task_path(tmp_path, monkeypatch):
    task_a = _task(tmp_path, monkeypatch, "a")
    task_b = _task(tmp_path, monkeypatch, "b")

    try:
        server._validate_task_resource(task_a, task_manager.get_pdb_path(task_b))
    except ValueError as exc:
        assert "outside task" in str(exc)
    else:
        raise AssertionError("Cross-task resources must be rejected")


def test_public_task_state_and_filter_response_do_not_expose_server_paths(
    tmp_path, monkeypatch
):
    task_id = _task(tmp_path, monkeypatch, "private")
    task_dir = task_manager.get_task_dir(task_id)
    task_manager.update_state(task_id, {
        "preview_pdb_path": str(task_dir / "preview.pdb"),
        "cgenff_uploads": {
            "LIG": {
                "mol2_path": str(task_dir / "cgenff" / "LIG" / "LIG.mol2"),
                "str_path": str(task_dir / "cgenff" / "LIG" / "LIG.str"),
                "force_field": "charmm36m",
                "cgenff_version": "4.6",
            }
        },
    })

    with TestClient(app) as client:
        status = client.get(f"/api/task/{task_id}")
        resumed = client.get(f"/api/task/{task_id}/resume")
        filtered = client.post(
            f"/api/filter-pdb/{task_id}",
            json={
                "include_chains": ["A"],
                "exclude_resnames": [],
                "small_molecule_labels": {},
            },
        )

    for response in (status, resumed, filtered):
        assert response.status_code == 200, response.text
        assert str(tmp_path) not in response.text
    public_upload = resumed.json()["cgenff_uploads"]["LIG"]
    assert public_upload["ready"] is True
    assert "mol2_path" not in public_upload
    assert "str_path" not in public_upload
    assert filtered.json()["filtered_resource"] == "filtered.pdb"


def test_save_step_only_accepts_visible_ui_steps_and_cannot_complete_science(
    tmp_path, monkeypatch
):
    task_id = _task(tmp_path, monkeypatch, "state")

    with TestClient(app) as client:
        reserved = client.post(
            f"/api/task/{task_id}/save-step",
            json={"step": "expires_at", "data": {"value": "never"}},
        )
        unknown = client.post(
            f"/api/task/{task_id}/save-step",
            json={"step": "made_up", "data": {}},
        )
        non_object = client.post(
            f"/api/task/{task_id}/save-step",
            json={"step": "simparams", "data": []},
        )
        accepted = client.post(
            f"/api/task/{task_id}/save-step",
            json={"step": "simparams", "data": {"temperature": 310.0}},
        )

    assert reserved.status_code == 400
    assert unknown.status_code == 400
    assert non_object.status_code == 400
    assert accepted.status_code == 200
    state = task_manager.get_state(task_id)
    assert state["expires_at"] != {"value": "never"}
    assert state["step_ui_state"]["simparams"] == {"temperature": 310.0}
    assert "simparams" not in state["steps_completed"]


def test_save_step_rejects_oversized_ui_state(tmp_path, monkeypatch):
    task_id = _task(tmp_path, monkeypatch, "large")

    with TestClient(app) as client:
        response = client.post(
            f"/api/task/{task_id}/save-step",
            json={"step": "simparams", "data": {"value": "x" * 300_000}},
        )

    assert response.status_code == 413


def test_step_results_expose_resource_urls_instead_of_host_paths(tmp_path):
    result = server._public_step_result("a" * 32, {
        "status": "ok",
        "step": "ions",
        "viewer_pdb_path": str(tmp_path / "viewer.pdb"),
        "index_path": str(tmp_path / "index.ndx"),
    })

    assert str(tmp_path) not in str(result)
    assert result["viewer_pdb_url"] == (
        f"/api/step/{'a' * 32}/ions/viewer.pdb"
    )
    assert result["index_available"] is True

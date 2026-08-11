"""Regression tests for independently routed v0.7 workflows."""

from pathlib import Path

from fastapi.testclient import TestClient

from gmxbuilder.modules.pure_membrane import (
    PureMembraneBuilder,
    PureMembraneForceFieldSelector,
    PureMembraneSolvationBuilder,
)
from gmxbuilder.modules.solution import (
    SolutionForceFieldSelector,
    SolutionInputModule,
    SolutionSolvationBuilder,
)
from gmxbuilder.pipeline.pipeline import Pipeline
from gmxbuilder.pipeline.step_executor import StepRunner, _get_module, get_pipeline_steps
from gmxbuilder.web.server import app
from gmxbuilder.web.task_types import get_all_task_types, get_task_type


ROOT = Path(__file__).parents[1]


def test_removed_and_new_task_cards_are_authoritative():
    task_ids = {item["id"] for item in get_all_task_types()}
    assert "micelle-builder" not in task_ids
    assert "hex-phase-builder" not in task_ids
    assert "pure-membrane" in task_ids

    pure = get_task_type("pure-membrane")
    assert pure is not None
    assert pure.enabled
    assert pure.pipeline == "pure_membrane"
    assert not pure.requires_input
    assert pure.visible_modules == [
        "forcefield", "membrane", "solvation", "ions", "simparams"
    ]


def test_solvator_and_pure_membrane_use_task_specific_module_classes():
    solution = dict(Pipeline.create_solvator()._modules)
    pure = dict(Pipeline.create_pure_membrane()._modules)

    assert isinstance(solution["input"], SolutionInputModule)
    assert isinstance(solution["forcefield"], SolutionForceFieldSelector)
    assert isinstance(solution["solvation"], SolutionSolvationBuilder)
    assert isinstance(pure["forcefield"], PureMembraneForceFieldSelector)
    assert isinstance(pure["membrane"], PureMembraneBuilder)
    assert isinstance(pure["solvation"], PureMembraneSolvationBuilder)
    assert type(solution["solvation"]) is not type(pure["solvation"])
    assert type(solution["solvation"]).__module__.startswith("gmxbuilder.modules.solution")
    assert type(pure["solvation"]).__module__.startswith("gmxbuilder.modules.pure_membrane")


def test_step_factory_is_scoped_by_pipeline_type():
    assert isinstance(_get_module("solvation", "solvator"), SolutionSolvationBuilder)
    assert isinstance(
        _get_module("solvation", "pure-membrane"), PureMembraneSolvationBuilder
    )
    assert get_pipeline_steps("pure-membrane")[0] == "forcefield"


def test_pure_membrane_first_step_creates_checkpoint_without_upload(tmp_path):
    runner = StepRunner(tmp_path, pipeline_type="pure-membrane")
    result = runner.run_step("forcefield", {
        "name": "amber14sb",
        "lipid_names": ["POPC"],
        "lipid_ff": "lipid21",
        "ligand_ff": "none",
        "water_model": "tip3p",
        "system_name": "pure_test",
    })

    assert result["status"] == "ok"
    assert runner.has_checkpoint("forcefield")
    saved = runner.load_system("forcefield")
    assert saved is not None
    assert saved.num_atoms == 0
    assert saved.metadata["lipid_ff"] == "lipid21"


def test_no_input_task_api_rejects_input_workflow_and_creates_pure_task():
    with TestClient(app) as client:
        rejected = client.post("/api/tasks", json={"task_type": "solvator"})
        assert rejected.status_code == 400
        created = client.post("/api/tasks", json={"task_type": "pure-membrane"})
        assert created.status_code == 200
        payload = created.json()
        state = client.get(f"/api/task/{payload['task_id']}").json()
        assert state["task_type_id"] == "pure-membrane"
        assert state["task_type"]["pipeline"] == "pure_membrane"
        compatibility = client.post(
            f"/api/forcefield-compatibility/{payload['task_id']}",
            json={"protein_ff": "amber14sb", "lipid_names": ["POPC"]},
        )
        assert compatibility.status_code == 200
        assert compatibility.json()["family"] == "amber"


def test_uploaded_task_persists_solution_pipeline_identity():
    fixture = ROOT / "tests/fixtures/small_molecule_label.pdb"
    with TestClient(app) as client, fixture.open("rb") as handle:
        response = client.post(
            "/api/upload-pdb",
            files={"file": (fixture.name, handle, "chemical/x-pdb")},
            data={"task_type": "solvator"},
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        state = client.get(f"/api/task/{task_id}").json()
        assert state["task_type_id"] == "solvator"
        assert state["task_type"]["pipeline"] == "solvator"


def test_frontend_exposes_explicit_optional_solvation_without_touching_bilayer():
    app_js = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()

    assert 'id="pure-membrane-include-solvent" checked' in template
    assert "syncPureMembraneSolvationOption" in app_js
    assert "desired = ['forcefield', 'membrane', 'solvation']" in app_js
    assert "desired.push('ions')" in app_js
    assert "formData.append('task_type'" in app_js

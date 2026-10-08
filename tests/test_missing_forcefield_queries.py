"""A missing installation prerequisite is distinct from invalid user chemistry."""

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.modules.forcefield import rtp_parser
from gmxbuilder.web.server import app


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_missing_forcefield_returns_actionable_unavailable_response(monkeypatch, force_field):
    original = rtp_parser._force_field_path

    def missing(name):
        if name.startswith("charmm"):
            raise FileNotFoundError("/private/example/forcefield")
        return original(name)

    monkeypatch.setattr(rtp_parser, "_rtp_by_force_field", {})
    monkeypatch.setattr(rtp_parser, "_force_field_path", missing)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"task_type": "pure-membrane"})
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        response = client.post(
            f"/api/forcefield-compatibility/{task_id}",
            json={"protein_ff": force_field, "lipid_names": ["POPC"]},
        )
    assert response.status_code == 503
    assert response.json()["code"] == "missing_compatibility_data"
    assert "./install-local.sh" in response.json()["error"]
    assert "/private/example" not in response.text
    assert "lipid_options" not in response.json()


def test_invalid_template_remains_a_validation_error(monkeypatch):
    def malformed(*args):
        raise ValueError("invalid template coefficients")

    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.compatibility.compatibility_report", malformed
    )
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"task_type": "pure-membrane"})
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        response = client.post(
            f"/api/forcefield-compatibility/{task_id}",
            json={"protein_ff": "charmm36m", "lipid_names": ["POPC"]},
        )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid template coefficients"}

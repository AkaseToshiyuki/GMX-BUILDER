"""End-to-end enforcement of the HTTP rate-limit policy.

The buckets in `_rate_policy` were covered only by a test asserting which
bucket a path maps to. Nothing checked that exceeding a bucket actually
returns 429, that one bucket's exhaustion leaves the others usable, or that
the limiter runs before the handler. Those are the properties the limiter
exists for -- it is the service's protection against a client, hostile or
merely looping, that submits work faster than the machine can absorb it.

The limits are read from the environment on every request, so these tests set
them low instead of issuing hundreds of calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager


@pytest.fixture
def unauthenticated_client(monkeypatch, tmp_path):
    for name in (
        "GMXBUILDER_DEPLOYMENT_MODE",
        "GMXBUILDER_AUTH_USER",
        "GMXBUILDER_AUTH_PASSWORD",
        "GMXBUILDER_ACCESS_TOKEN",
        "GMXBUILDER_CORS_ORIGINS",
        "GMXBUILDER_HEAVY_RATE_DISABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GMXBUILDER_RATE_LIMIT_DB", str(tmp_path / "rate.sqlite3"))
    monkeypatch.setattr(server, "task_manager", TaskManager(tmp_path / "tasks"))
    with TestClient(app) as client:
        yield client


def _upload(client):
    return client.post("/api/upload-pdb", files={"file": ("probe.pdb", "ATOM\n")})


def test_exceeding_a_bucket_returns_429_with_retry_after(monkeypatch, unauthenticated_client):
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE", "2")

    codes = [_upload(unauthenticated_client).status_code for _ in range(4)]
    assert codes[:2] == [400, 400], codes
    assert codes[2:] == [429, 429], codes

    refused = _upload(unauthenticated_client)
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) > 0
    assert "rate limit" in refused.json()["error"].lower()


def test_temporary_heavy_override_bypasses_exhaustion_and_can_be_restored(
    monkeypatch, unauthenticated_client
):
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE", "20")
    assert [_upload(unauthenticated_client).status_code for _ in range(20)] == [400] * 20
    assert _upload(unauthenticated_client).status_code == 429
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE_DISABLED", "1")
    assert [_upload(unauthenticated_client).status_code for _ in range(25)] == [400] * 25
    monkeypatch.delenv("GMXBUILDER_HEAVY_RATE_DISABLED")
    assert _upload(unauthenticated_client).status_code == 429


def test_heavy_override_retains_other_budgets(monkeypatch, unauthenticated_client):
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE_DISABLED", "1")
    for path in (
        "/api/upload-pdb",
        "/api/build-lipid-library",
        "/api/cgenff-upload/" + "a" * 32,
        "/api/ligand-chemistry-upload/" + "a" * 32,
        "/api/task/" + "a" * 32 + "/custom-lipids",
    ):
        assert server._rate_policy(path, "POST") is None
    assert server._rate_policy("/api/build", "POST")[0] == "finalize"
    assert server._rate_policy("/api/task-types", "GET")[0] == "api-read"
    monkeypatch.setenv("GMXBUILDER_API_RATE", "1")
    assert unauthenticated_client.post("/api/unrecognised", json={}).status_code == 404
    assert unauthenticated_client.post("/api/unrecognised", json={}).status_code == 429


def test_the_limiter_runs_before_the_handler(monkeypatch, unauthenticated_client):
    """A refused request must not reach the endpoint at all.

    Otherwise a flood would still cost whatever the handler does before its own
    validation rejects it, which is the work the limiter exists to prevent.
    """
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE", "1")
    assert _upload(unauthenticated_client).status_code == 400

    calls: list[str] = []
    monkeypatch.setattr(
        server,
        "process_uploaded_structure",
        lambda *args, **kwargs: calls.append("handler ran"),
    )
    assert _upload(unauthenticated_client).status_code == 429
    assert calls == []


def test_exhausting_one_bucket_leaves_the_others_usable(monkeypatch, unauthenticated_client):
    """Buckets must be independent, or one noisy path would deny the whole API.

    This is the property a separately budgeted endpoint depends on: a new
    expensive route can be given its own allowance without a burst on it
    taking the ordinary API down with it.
    """
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE", "1")
    monkeypatch.setenv("GMXBUILDER_API_READ_RATE", "500")

    assert _upload(unauthenticated_client).status_code == 400
    assert _upload(unauthenticated_client).status_code == 429

    # A read on a different bucket is unaffected.
    assert unauthenticated_client.get("/api/task-types").status_code == 200


def test_health_is_never_rate_limited(monkeypatch, unauthenticated_client):
    """Liveness must answer even while every other bucket is exhausted."""
    monkeypatch.setenv("GMXBUILDER_HEAVY_RATE", "1")
    _upload(unauthenticated_client)
    assert _upload(unauthenticated_client).status_code == 429

    for path in ("/health", "/health/live"):
        assert unauthenticated_client.get(path).status_code == 200

    # Both liveness paths are exempt from every bucket. "/api/health" is
    # exempt too but is not a route; the exemption is harmless and kept so a
    # future move of the endpoint under /api/ does not silently start being
    # throttled.
    for path in ("/health", "/api/health"):
        assert server._rate_policy(path, "GET") is None


@pytest.mark.parametrize(
    "path,method,expected",
    [
        ("/api/upload-pdb", "POST", "heavy"),
        ("/api/build-lipid-library", "POST", "heavy"),
        ("/api/cgenff-upload/" + "a" * 32, "POST", "heavy"),
        ("/api/task/" + "a" * 32 + "/custom-lipids", "POST", "heavy"),
        ("/api/build", "POST", "finalize"),
        ("/api/task-types", "GET", "api-read"),
        ("/api/tasks", "POST", "api-write"),
    ],
)
def test_every_documented_path_lands_in_its_bucket(path, method, expected):
    policy = server._rate_policy(path, method)
    assert policy is not None, path
    assert policy[0] == expected


def test_an_unrecognised_api_path_still_gets_a_budget():
    """A new endpoint must not be unlimited merely because nobody classified it.

    Anything added under /api/ inherits the api-write allowance until it is
    given one of its own, so forgetting to classify it fails safe.
    """
    policy = server._rate_policy("/api/ext/something-added-later", "POST")
    assert policy is not None
    assert policy[0] == "api-write"

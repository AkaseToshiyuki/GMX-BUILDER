"""Delivery and control-plane responsiveness; no molecular computation."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from gmxbuilder import VERSION
from gmxbuilder.web import server


def test_versioned_static_assets_cache_and_compress_without_caching_private_api():
    with TestClient(server.app) as client:
        asset = client.get(f"/static/app.js?v={VERSION}", headers={"Accept-Encoding": "gzip"})
        assert asset.status_code == 200
        assert asset.headers["content-encoding"] == "gzip"
        assert "immutable" in asset.headers["cache-control"]
        assert "private" in asset.headers["cache-control"]
        assert "accept-encoding" in asset.headers["vary"].lower()
        plain = client.get("/static/app.js", headers={"Accept-Encoding": "identity"})
        assert plain.content == asset.content
        assert plain.headers["cache-control"] == "private, no-cache"
        invalid_version = client.get("/static/app.js?v=old")
        assert invalid_version.headers["cache-control"] == "private, no-cache"
        conditional = client.get(
            f"/static/app.js?v={VERSION}", headers={"If-None-Match": asset.headers["etag"]}
        )
        assert conditional.status_code == 304
        assert "immutable" in conditional.headers["cache-control"]
        private = client.get("/api/task/" + "a" * 32)
        assert "immutable" not in private.headers.get("cache-control", "")
        assert "content-encoding" not in private.headers


@pytest.mark.parametrize("worker_count", [1, 2])
def test_api_control_read_does_not_wait_for_blocked_previews(monkeypatch, worker_count):
    monkeypatch.setattr(server, "_MAX_CONCURRENT_BUILDS", worker_count)
    release = threading.Event()
    entered = [threading.Event() for _ in range(worker_count)]

    def blocked(event):
        event.set()
        release.wait(5)

    with TestClient(server.app) as client:
        futures = [server._submit_interactive(blocked, event) for event in entered]
        try:
            assert all(event.wait(1) for event in entered)
            start = time.monotonic()
            response = client.get("/api/task-types")
            assert response.status_code == 200
            assert time.monotonic() - start < 1.0
            assert not any(future.done() for future in futures)
        finally:
            release.set()
            for future in futures:
                future.result(2)

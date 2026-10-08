"""Managed worker responses preserve the representation delivered to browsers."""

import asyncio
import gzip
import json
import time
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from gmxbuilder.web.resource_coordinator import ResourceCoordinator
from gmxbuilder.web.resource_worker import replay


def test_managed_gzip_result_preserves_encoding_and_checkpoint_cache_headers(tmp_path):
    payload = {"revision": "checkpoint-digest", "components": [{"kind": "membrane"}]}
    compressed = gzip.compress(json.dumps(payload).encode())

    async def source(_request):
        return Response(
            compressed,
            media_type="application/json",
            headers={
                "Content-Encoding": "gzip",
                "ETag": '"checkpoint-digest"',
                "Cache-Control": "private, no-cache",
                "Vary": "Accept-Encoding",
                "Set-Cookie": "must-not-be-replayed=1",
            },
        )

    request_body = tmp_path / "request.body"
    request_body.write_bytes(b"")
    asyncio.run(
        replay(
            Starlette(routes=[Route("/fixture", source)]),
            {"method": "GET", "path": "/fixture", "query": "", "headers": []},
            request_body,
            tmp_path / "response.json",
        )
    )
    assert (tmp_path / "response.body").read_bytes() == compressed
    operation = {"expires": time.time() + 60}
    coordinator = SimpleNamespace(
        queue=SimpleNamespace(get=lambda _id: operation),
        job_directory=lambda _operation: tmp_path,
    )

    async def result(_request):
        return ResourceCoordinator.operation_response(coordinator, "a" * 32, result=True)

    with TestClient(Starlette(routes=[Route("/result", result)])) as client:
        response = client.get("/result")
    assert response.status_code == 200
    assert response.json() == payload  # The HTTP client must decode it like fetch does.
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["etag"] == '"checkpoint-digest"'
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["vary"] == "Accept-Encoding"
    assert "set-cookie" not in response.headers

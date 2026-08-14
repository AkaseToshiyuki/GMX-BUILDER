"""Public-deployment, proxy, durable-rate and vendored-asset regressions."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
import tomllib
from fastapi.testclient import TestClient
from starlette.requests import Request

from gmxbuilder.web import server
from gmxbuilder.web.security import (
    DurableRateLimiter,
    SecurityConfig,
    client_identity,
    request_is_https,
    validate_server_bind,
)
from gmxbuilder.web.task_manager import TaskManager

ROOT = Path(__file__).resolve().parents[1]


def _public_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("GMXBUILDER_AUTH_USER", "researcher")
    monkeypatch.setenv("GMXBUILDER_AUTH_PASSWORD", "correct-horse-battery-staple")
    monkeypatch.setenv("GMXBUILDER_TRUSTED_PROXIES", "127.0.0.1/32")
    monkeypatch.setenv("GMXBUILDER_CORS_ORIGINS", "https://gmxbuilder.example.org")
    monkeypatch.setenv("GMXBUILDER_RATE_LIMIT_DB", str(tmp_path / "rate.sqlite3"))


def _basic_header() -> dict[str, str]:
    value = base64.b64encode(b"researcher:correct-horse-battery-staple").decode()
    return {"Authorization": f"Basic {value}"}


def _request(*, peer: str, forwarded: str = "", proto: str = "http") -> Request:
    headers = []
    if forwarded:
        headers.append((b"x-forwarded-for", forwarded.encode()))
        headers.append((b"x-forwarded-proto", proto.encode()))
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": headers,
        "client": (peer, 1234),
        "server": ("127.0.0.1", 7788),
    })


def test_nonloopback_bind_requires_explicit_mode(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_DEPLOYMENT_MODE", raising=False)
    with pytest.raises(ValueError, match="Non-loopback"):
        validate_server_bind("0.0.0.0")
    validate_server_bind("127.0.0.1")

    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "trusted-lan")
    assert validate_server_bind("0.0.0.0").mode == "trusted-lan"


def test_public_mode_requires_complete_auth_tls_proxy_and_https_origins(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "public")
    monkeypatch.delenv("GMXBUILDER_AUTH_USER", raising=False)
    monkeypatch.delenv("GMXBUILDER_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("GMXBUILDER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GMXBUILDER_TRUSTED_PROXIES", raising=False)
    config = SecurityConfig.from_environment()
    assert any("requires a Basic password" in error for error in config.errors)
    assert any("TRUSTED_PROXIES" in error for error in config.errors)
    assert any("https://" in error for error in config.errors)


def test_liveness_survives_invalid_public_security_configuration(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "public")
    monkeypatch.delenv("GMXBUILDER_AUTH_USER", raising=False)
    monkeypatch.delenv("GMXBUILDER_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("GMXBUILDER_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GMXBUILDER_TRUSTED_PROXIES", raising=False)
    with TestClient(server.app, base_url="http://testserver") as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/api/task-types").status_code == 503


def test_public_mode_auth_https_origin_and_liveness(monkeypatch, tmp_path):
    _public_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "task_manager", TaskManager(tmp_path / "tasks"))
    with TestClient(server.app, base_url="http://testserver") as insecure:
        assert insecure.get("/health/live").status_code == 200
        response = insecure.get("/api/task-types", headers=_basic_header())
        assert response.status_code == 426

    with TestClient(server.app, base_url="https://testserver") as client:
        response = client.get("/api/task-types")
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Basic")

        response = client.get("/api/task-types", headers=_basic_header())
        assert response.status_code == 200
        assert response.headers["strict-transport-security"].startswith("max-age=")
        assert "https://3Dmol.org" not in response.headers["content-security-policy"]

        response = client.post(
            "/api/tasks",
            headers={**_basic_header(), "Origin": "https://evil.example"},
            json={"task_type": "pure-membrane"},
        )
        assert response.status_code == 403

        response = client.post(
            "/api/tasks",
            headers={
                **_basic_header(),
                "Origin": "https://gmxbuilder.example.org",
            },
            json={"task_type": "pure-membrane"},
        )
        assert response.status_code == 200


def test_json_body_limit_rejects_declared_and_chunked_requests(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_JSON_BODY_LIMIT", "64")
    with TestClient(server.app) as client:
        declared = client.post(
            "/api/tasks",
            content=b"x" * 65,
            headers={"Content-Type": "application/json"},
        )

        def chunks():
            yield b'{"task_type":"'
            yield b"x" * 80
            yield b'"}'

        chunked = client.post(
            "/api/tasks",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert declared.status_code == 413
    assert chunked.status_code == 413


def test_forwarded_client_and_proto_are_used_only_from_trusted_proxy(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TRUSTED_PROXIES", "127.0.0.1/32,10.0.0.0/8")
    config = SecurityConfig.from_environment()
    trusted = _request(
        peer="127.0.0.1", forwarded="203.0.113.9, 10.1.2.3", proto="https"
    )
    assert client_identity(trusted, config) == "203.0.113.9"
    assert request_is_https(trusted, config) is True

    spoofed = _request(peer="192.0.2.8", forwarded="203.0.113.9", proto="https")
    assert client_identity(spoofed, config) == "192.0.2.8"
    assert request_is_https(spoofed, config) is False


def test_rate_limit_survives_limiter_recreation_and_uses_private_file(tmp_path):
    path = tmp_path / "rate.sqlite3"
    first = DurableRateLimiter(path)
    assert first.allow("198.51.100.4", "heavy", 2, 60, now=1000) == (True, 0)
    assert first.allow("198.51.100.4", "heavy", 2, 60, now=1001) == (True, 0)

    restarted = DurableRateLimiter(path)
    allowed, retry_after = restarted.allow(
        "198.51.100.4", "heavy", 2, 60, now=1002
    )
    assert allowed is False
    assert retry_after == 58
    assert path.stat().st_mode & 0o077 == 0
    assert restarted.allow("198.51.100.4", "heavy", 2, 60, now=1061) == (True, 0)


@pytest.mark.parametrize(
    ("relative_path", "digest"),
    [
        (
            "src/gmxbuilder/web/static/vendor/3dmol-2.5.5/3Dmol-min.js",
            "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016",
        ),
        (
            "src/gmxbuilder/web/static/vendor/smiles-drawer-2.0.3/smiles-drawer.min.js",
            "917c95165fd8af50f76ffbf35eac0b74e7f0c93d715132eaa9bfc5e3374445ec",
        ),
    ],
)
def test_vendored_browser_asset_digest(relative_path, digest):
    assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == digest


def test_template_has_no_runtime_cdn_dependency_or_inline_script_handler():
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    assert "https://3Dmol.org" not in template
    assert "https://unpkg.com" not in template
    assert " onerror=" not in template
    assert "/static/vendor/3dmol-2.5.5/3Dmol-min.js" in template
    assert "/static/vendor/smiles-drawer-2.0.3/smiles-drawer.min.js" in template


def test_uv_lock_pins_registry_artifacts_and_vcs_commit():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = lock["package"]
    registry_packages = [
        package for package in packages if "registry" in package.get("source", {})
    ]
    assert registry_packages
    for package in registry_packages:
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get(
            "wheels", []
        )
        assert artifacts, package["name"]
        assert all(item.get("hash", "").startswith("sha256:") for item in artifacts)
    pdbfixer = next(package for package in packages if package["name"] == "pdbfixer")
    source = pdbfixer["source"]["git"]
    commit = "94cfa4c0ca551cdc5f13320f9a658efd59f2b881"
    assert f"rev={commit}" in source and source.endswith(f"#{commit}")


def test_distribution_excludes_runtime_outputs_and_bytecode():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excluded = project["tool"]["setuptools"]["exclude-package-data"]["gmxbuilder"]
    assert "web/static/output/**/*" in excluded
    assert "**/*.pyc" in excluded
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "prune src/gmxbuilder/web/static/output" in manifest
    assert "__pycache__" in manifest and "*.py[cod]" in manifest


def test_local_installer_emits_hardened_service_and_safe_bind_default():
    installer = (ROOT / "install-local.sh").read_text()
    assert "DEFAULT_HOST=127.0.0.1" in installer
    for directive in (
        "UMask=0077",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "RestrictSUIDSGID=true",
        "SystemCallFilter=",
    ):
        assert directive in installer
    assert "GMXBUILDER_DEPLOYMENT_MODE" in installer
    assert "uv sync" in installer and "--frozen" in installer

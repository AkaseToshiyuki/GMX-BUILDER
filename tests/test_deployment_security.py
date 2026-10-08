"""Public-deployment, proxy, durable-rate and vendored-asset regressions."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

# tomllib joined the standard library in 3.11, and this project supports 3.10.
# An unconditional import here failed collection for the whole module, taking
# every unrelated test in it down on the oldest supported interpreter.
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None
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
    return Request(
        {
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
        }
    )


def test_nonloopback_bind_requires_explicit_unsafe_opt_in(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError, match="unauthenticated non-loopback"):
        validate_server_bind("192.0.2.10")
    validate_server_bind("127.0.0.1")

    # A wildcard address is the broadest exposure, so it does not authorize
    # itself; it needs the same opt-in as any other non-loopback address.
    for wildcard in ("0.0.0.0", "::"):
        with pytest.raises(ValueError, match="unauthenticated non-loopback"):
            validate_server_bind(wildcard)
        assert validate_server_bind(wildcard, allow_unsafe_deployment=True).allow_unsafe_deployment

    assert (
        validate_server_bind("192.0.2.10", allow_unsafe_deployment=True).allow_unsafe_deployment
        is True
    )

    monkeypatch.setenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", "1")
    assert validate_server_bind("192.0.2.10").mode == "local"


def test_trusted_lan_mode_is_not_a_way_around_the_unsafe_opt_in(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", raising=False)
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "trusted-lan")
    monkeypatch.delenv("GMXBUILDER_AUTH_USER", raising=False)
    monkeypatch.delenv("GMXBUILDER_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("GMXBUILDER_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="unauthenticated non-loopback"):
        validate_server_bind("0.0.0.0")
    assert validate_server_bind("0.0.0.0", allow_unsafe_deployment=True).mode == "trusted-lan"


def test_configured_authentication_removes_the_need_for_the_unsafe_opt_in(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", raising=False)
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "trusted-lan")
    monkeypatch.setenv("GMXBUILDER_ACCESS_TOKEN", "x" * 32)
    assert validate_server_bind("0.0.0.0").authentication_enabled is True


def test_invalid_unsafe_deployment_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", "sometimes")
    with pytest.raises(ValueError, match="GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT"):
        validate_server_bind("127.0.0.1")


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
    with pytest.raises(ValueError, match="public mode requires"):
        validate_server_bind("0.0.0.0", allow_unsafe_deployment=True)


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
    trusted = _request(peer="127.0.0.1", forwarded="203.0.113.9, 10.1.2.3", proto="https")
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
    allowed, retry_after = restarted.allow("198.51.100.4", "heavy", 2, 60, now=1002)
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
    assets = (ROOT / "src/gmxbuilder/web/static/assets.js").read_text()
    assert "/static/assets.js?v={{ version }}" in template
    assert "https://" not in assets
    assert "/static/vendor/3dmol-2.5.5/3Dmol-min.js" in assets
    assert "/static/vendor/smiles-drawer-2.0.3/smiles-drawer.min.js" in assets


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11 or newer")
def test_uv_lock_pins_registry_and_direct_url_artifacts():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = lock["package"]
    registry_packages = [package for package in packages if "registry" in package.get("source", {})]
    assert registry_packages
    for package in registry_packages:
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get("wheels", [])
        assert artifacts, package["name"]
        assert all(item.get("hash", "").startswith("sha256:") for item in artifacts)
    pdbfixer = next(package for package in packages if package["name"] == "pdbfixer")
    commit = "94cfa4c0ca551cdc5f13320f9a658efd59f2b881"
    assert pdbfixer["source"]["url"].endswith(f"/{commit}.tar.gz")
    assert pdbfixer["sdist"]["hash"].startswith("sha256:")


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11 or newer")
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
    assert "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT" in installer
    assert "--allow-unsafe-deployment" in installer
    assert '"$UV_BIN" sync' in installer and "--frozen" in installer

    nginx = (ROOT / "deploy/nginx-gmxbuilder.conf.example").read_text()
    assert "access_log off;" in nginx


def _unauthenticated_local_environment(monkeypatch, tmp_path):
    """The default local deployment: no credentials, loopback, no CORS override."""
    for name in (
        "GMXBUILDER_DEPLOYMENT_MODE",
        "GMXBUILDER_AUTH_USER",
        "GMXBUILDER_AUTH_PASSWORD",
        "GMXBUILDER_ACCESS_TOKEN",
        "GMXBUILDER_CORS_ORIGINS",
        "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GMXBUILDER_RATE_LIMIT_DB", str(tmp_path / "rate.sqlite3"))
    monkeypatch.setattr(server, "task_manager", TaskManager(tmp_path / "tasks"))


def test_cross_site_write_is_refused_without_authentication(monkeypatch, tmp_path):
    """A page on another origin must not be able to drive the local instance.

    Authentication is not what makes a cross-site write dangerous; reachability
    is. Request.json() also ignores Content-Type, so a cross-site request needs
    no preflight and would otherwise be processed normally.
    """
    _unauthenticated_local_environment(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/tasks",
            headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
            content='{"task_type":"pure-membrane"}',
        )
        assert response.status_code == 403
        assert response.json()["error"] == "Request Origin is not allowed"


def test_non_browser_clients_without_origin_are_unaffected(monkeypatch, tmp_path):
    _unauthenticated_local_environment(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        response = client.post("/api/tasks", json={"task_type": "pure-membrane"})
        assert response.status_code == 200


def test_same_origin_write_is_allowed_on_a_non_default_port(monkeypatch, tmp_path):
    """The allowlist hardcodes the default port; the real bind port may differ.

    Nothing derives GMXBUILDER_CORS_ORIGINS from the port the server was
    actually started on, so the check has to accept the listener's own origin
    or every non-default-port deployment would reject its own browser client.
    """
    _unauthenticated_local_environment(monkeypatch, tmp_path)
    with TestClient(server.app, base_url="http://127.0.0.1:9999") as client:
        response = client.post(
            "/api/tasks",
            headers={"Origin": "http://127.0.0.1:9999"},
            json={"task_type": "pure-membrane"},
        )
        assert response.status_code == 200

        response = client.post(
            "/api/tasks",
            headers={"Origin": "http://127.0.0.1:4321"},
            json={"task_type": "pure-membrane"},
        )
        assert response.status_code == 403

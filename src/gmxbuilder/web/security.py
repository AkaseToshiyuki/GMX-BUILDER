"""Deployment security and durable request-rate controls for the Web service."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from starlette.responses import Response

_VALID_MODES = {"local", "trusted-lan", "public"}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _split_environment(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class SecurityConfig:
    """Validated security settings read from the current process environment."""

    mode: str
    allowed_origins: tuple[str, ...]
    trusted_proxies: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    auth_user: str
    auth_password: str
    access_token: str
    errors: tuple[str, ...]

    @property
    def authentication_enabled(self) -> bool:
        return bool((self.auth_user and self.auth_password) or self.access_token)

    @property
    def require_https(self) -> bool:
        return self.mode == "public"

    @classmethod
    def from_environment(cls) -> SecurityConfig:
        mode = os.environ.get("GMXBUILDER_DEPLOYMENT_MODE", "local").strip().lower()
        errors: list[str] = []
        if mode not in _VALID_MODES:
            errors.append("GMXBUILDER_DEPLOYMENT_MODE must be local, trusted-lan, or public")

        allowed_origins = _split_environment(
            "GMXBUILDER_CORS_ORIGINS",
            "http://127.0.0.1:7788,http://localhost:7788",
        )
        trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for item in _split_environment("GMXBUILDER_TRUSTED_PROXIES"):
            try:
                trusted.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                errors.append(f"Invalid trusted proxy address/network: {item}")

        auth_user = os.environ.get("GMXBUILDER_AUTH_USER", "").strip()
        auth_password = os.environ.get("GMXBUILDER_AUTH_PASSWORD", "")
        access_token = os.environ.get("GMXBUILDER_ACCESS_TOKEN", "")
        if bool(auth_user) != bool(auth_password):
            errors.append("GMXBUILDER_AUTH_USER and GMXBUILDER_AUTH_PASSWORD must be set together")

        if mode == "public":
            if not ((auth_user and len(auth_password) >= 16) or len(access_token) >= 32):
                errors.append(
                    "public mode requires a Basic password of at least 16 characters "
                    "or GMXBUILDER_ACCESS_TOKEN of at least 32 characters"
                )
            if not trusted:
                errors.append(
                    "public mode requires GMXBUILDER_TRUSTED_PROXIES for the TLS reverse proxy"
                )
            if not allowed_origins or any(
                not origin.lower().startswith("https://") for origin in allowed_origins
            ):
                errors.append("public mode requires explicit https:// GMXBUILDER_CORS_ORIGINS")

        return cls(
            mode=mode,
            allowed_origins=allowed_origins,
            trusted_proxies=tuple(trusted),
            auth_user=auth_user,
            auth_password=auth_password,
            access_token=access_token,
            errors=tuple(errors),
        )


def _address_in_networks(
    address: str,
    networks: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(parsed.version == network.version and parsed in network for network in networks)


def validate_server_bind(host: str, config: SecurityConfig | None = None) -> SecurityConfig:
    """Reject accidental non-loopback exposure without an explicit deployment mode."""
    config = config or SecurityConfig.from_environment()
    if config.errors:
        raise ValueError("; ".join(config.errors))
    try:
        address = ipaddress.ip_address(host)
        loopback = address.is_loopback
    except ValueError:
        loopback = host.strip().lower() == "localhost"
    if not loopback and config.mode == "local":
        raise ValueError(
            "Non-loopback binding requires GMXBUILDER_DEPLOYMENT_MODE=trusted-lan "
            "or public; local mode is intentionally loopback-only"
        )
    return config


def _direct_peer(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _trusted_forward_chain(request: Request, config: SecurityConfig) -> list[str] | None:
    peer = _direct_peer(request)
    if not _address_in_networks(peer, config.trusted_proxies):
        return None
    raw = request.headers.get("X-Forwarded-For", "")
    if not raw:
        return None
    addresses = [item.strip() for item in raw.split(",") if item.strip()]
    try:
        for address in addresses:
            ipaddress.ip_address(address)
    except ValueError:
        return None
    return addresses


def client_identity(request: Request, config: SecurityConfig) -> str:
    """Return the first untrusted client in a trusted proxy chain."""
    peer = _direct_peer(request)
    forwarded = _trusted_forward_chain(request, config)
    if not forwarded:
        return peer
    chain = forwarded + [peer]
    for address in reversed(chain):
        if not _address_in_networks(address, config.trusted_proxies):
            return address
    return chain[0]


def request_is_https(request: Request, config: SecurityConfig) -> bool:
    if request.url.scheme.lower() == "https":
        return True
    if _trusted_forward_chain(request, config) is None:
        return False
    return request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip().lower() == "https"


def request_origin_allowed(request: Request, config: SecurityConfig) -> bool:
    if request.method not in _UNSAFE_METHODS:
        return True
    origin = request.headers.get("Origin")
    if not origin:
        # Non-browser API clients generally do not send Origin.
        return True
    return origin in config.allowed_origins


def _basic_credentials(header: str) -> tuple[str, str] | None:
    if not header.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(header.split(None, 1)[1], validate=True).decode("utf-8")
        return tuple(decoded.split(":", 1)) if ":" in decoded else None
    except (ValueError, UnicodeDecodeError):
        return None


def request_authenticated(request: Request, config: SecurityConfig) -> bool:
    authorization = request.headers.get("Authorization", "")
    basic = _basic_credentials(authorization)
    if basic and config.auth_user and config.auth_password:
        return secrets.compare_digest(basic[0], config.auth_user) and secrets.compare_digest(
            basic[1], config.auth_password
        )

    supplied_token = ""
    if authorization.lower().startswith("bearer "):
        supplied_token = authorization.split(None, 1)[1]
    elif request.headers.get("X-GMXBUILDER-Token"):
        supplied_token = request.headers["X-GMXBUILDER-Token"]
    return bool(
        config.access_token
        and supplied_token
        and secrets.compare_digest(config.access_token, supplied_token)
    )


def apply_security_headers(response: Response, *, public_mode: bool = False) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; connect-src 'self'; worker-src 'self' blob:; "
        "font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
        "frame-ancestors 'none'",
    )
    if public_mode:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        response.headers.setdefault("Cache-Control", "no-store")
    return response


class DurableRateLimiter:
    """SQLite-backed fixed-window limiter shared across restarts and workers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=NORMAL")
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS rate_hits ("
                        "client_hash TEXT NOT NULL, bucket TEXT NOT NULL, ts REAL NOT NULL)"
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS rate_hits_lookup "
                        "ON rate_hits(client_hash, bucket, ts)"
                    )
                    self._initialized = True
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return connection

    def allow(
        self,
        client: str,
        bucket: str,
        limit: int,
        window_seconds: float,
        *,
        now: float | None = None,
    ) -> tuple[bool, int]:
        now = time.time() if now is None else float(now)
        threshold = now - window_seconds
        client_hash = hashlib.sha256(client.encode("utf-8")).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rate_hits WHERE client_hash=? AND bucket=? AND ts<=?",
                (client_hash, bucket, threshold),
            )
            row = connection.execute(
                "SELECT COUNT(*), MIN(ts) FROM rate_hits WHERE client_hash=? AND bucket=?",
                (client_hash, bucket),
            ).fetchone()
            count = int(row[0] or 0)
            oldest = float(row[1] or now)
            if limit <= 0 or count >= limit:
                connection.execute("COMMIT")
                retry_after = max(1, int(oldest + window_seconds - now + 0.999))
                return False, retry_after
            connection.execute(
                "INSERT INTO rate_hits(client_hash, bucket, ts) VALUES (?, ?, ?)",
                (client_hash, bucket, now),
            )
            connection.execute("COMMIT")
            return True, 0
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

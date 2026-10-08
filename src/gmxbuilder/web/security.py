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
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Request
from starlette.responses import Response

_VALID_MODES = {"local", "trusted-lan", "public", "public-anonymous"}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _split_environment(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


def _environment_flag(name: str, errors: list[str]) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    errors.append(f"{name} must be one of: 1, 0, true, false, yes, no, on, off")
    return False


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
    allow_unsafe_deployment: bool = False
    allowed_hosts: tuple[str, ...] = ()
    lan_origins: tuple[str, ...] = ()
    lan_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    @property
    def authentication_enabled(self) -> bool:
        return bool((self.auth_user and self.auth_password) or self.access_token)

    @property
    def require_https(self) -> bool:
        return self.mode in {"public", "public-anonymous"}

    @classmethod
    def from_environment(cls) -> SecurityConfig:
        mode = os.environ.get("GMXBUILDER_DEPLOYMENT_MODE", "local").strip().lower()
        errors: list[str] = []
        if mode not in _VALID_MODES:
            errors.append(
                "GMXBUILDER_DEPLOYMENT_MODE must be local, trusted-lan, public, or public-anonymous"
            )
        allow_unsafe_deployment = _environment_flag("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", errors)

        allowed_origins = _split_environment(
            "GMXBUILDER_CORS_ORIGINS",
            "http://127.0.0.1:7788,http://localhost:7788",
        )
        allowed_hosts = _split_environment("GMXBUILDER_ALLOWED_HOSTS")
        if any(_authority(host) is None or "*" in host for host in allowed_hosts):
            errors.append("GMXBUILDER_ALLOWED_HOSTS requires exact host names or IP addresses")
        trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for item in _split_environment("GMXBUILDER_TRUSTED_PROXIES"):
            try:
                trusted.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                errors.append(f"Invalid trusted proxy address/network: {item}")
        lan_origins = _split_environment("GMXBUILDER_LAN_ORIGINS")
        lan_networks = []
        for item in _split_environment("GMXBUILDER_LAN_NETWORKS"):
            try:
                network = ipaddress.ip_network(item, strict=False)
                if not network.is_private:
                    raise ValueError
                lan_networks.append(network)
            except ValueError:
                errors.append("LAN networks must be explicit private address ranges")
        if bool(lan_origins) != bool(lan_networks):
            errors.append("LAN HTTP access requires both LAN_ORIGINS and LAN_NETWORKS")
        for origin in lan_origins:
            try:
                parsed = urlsplit(origin)
                address = ipaddress.ip_address(parsed.hostname or "")
                if (
                    parsed.scheme != "http"
                    or parsed.path
                    or parsed.query
                    or parsed.fragment
                    or parsed.username
                    or not any(address in network for network in lan_networks)
                ):
                    raise ValueError
                # Access validates nonnumeric and out-of-range ports lazily.
                _validated_port = parsed.port
            except ValueError:
                errors.append("LAN origins must be exact http://private-address[:port] origins")

        auth_user = os.environ.get("GMXBUILDER_AUTH_USER", "").strip()
        auth_password = os.environ.get("GMXBUILDER_AUTH_PASSWORD", "")
        access_token = os.environ.get("GMXBUILDER_ACCESS_TOKEN", "")
        if bool(auth_user) != bool(auth_password):
            errors.append("GMXBUILDER_AUTH_USER and GMXBUILDER_AUTH_PASSWORD must be set together")

        if mode in {"public", "public-anonymous"}:
            if mode == "public" and not (
                (auth_user and len(auth_password) >= 16) or len(access_token) >= 32
            ):
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
            if mode == "public-anonymous" and not os.environ.get("GMXBUILDER_MANAGED_ROOT"):
                errors.append("public-anonymous mode requires managed resource isolation")

        return cls(
            mode=mode,
            allowed_origins=allowed_origins,
            trusted_proxies=tuple(trusted),
            auth_user=auth_user,
            auth_password=auth_password,
            access_token=access_token,
            errors=tuple(errors),
            allow_unsafe_deployment=allow_unsafe_deployment,
            allowed_hosts=allowed_hosts,
            lan_origins=lan_origins,
            lan_networks=tuple(lan_networks),
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


def _bind_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip())
    except ValueError:
        return None


def bind_is_loopback(host: str) -> bool:
    """Return whether *host* unambiguously denotes a loopback listener."""
    address = _bind_address(host)
    return address.is_loopback if address is not None else host.strip().lower() == "localhost"


def bind_is_wildcard(host: str) -> bool:
    """Return whether *host* selects every IPv4 or IPv6 interface."""
    address = _bind_address(host)
    return bool(address is not None and address.is_unspecified)


def validate_server_bind(
    host: str,
    config: SecurityConfig | None = None,
    *,
    allow_unsafe_deployment: bool = False,
) -> SecurityConfig:
    """Reject an unauthenticated non-loopback listener without an explicit opt-in.

    ``trusted-lan`` does not bypass the explicit opt-in. ``public`` requires
    credentials; ``public-anonymous`` is the explicit alternative requiring
    managed resources, HTTPS and trusted proxy configuration. Configuration
    errors are validated first and cannot be bypassed by a bind flag.
    """
    config = config or SecurityConfig.from_environment()
    if config.errors:
        raise ValueError("; ".join(config.errors))
    explicit_unsafe = config.allow_unsafe_deployment or allow_unsafe_deployment
    if (
        not bind_is_loopback(host)
        and not config.authentication_enabled
        and not explicit_unsafe
        and config.mode != "public-anonymous"
    ):
        raise ValueError(
            "An unauthenticated non-loopback listener requires "
            "--allow-unsafe-deployment or GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=1, "
            "or configured authentication. A wildcard host (0.0.0.0 or ::) is "
            "the broadest exposure and needs the same explicit opt-in"
        )
    if explicit_unsafe and not config.allow_unsafe_deployment:
        config = replace(config, allow_unsafe_deployment=True)
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


def _authority(value: str):
    if (
        not value
        or any(char.isspace() for char in value)
        or any(char in value for char in "/\\@?#")
    ):
        return None
    try:
        parsed = urlsplit("//" + value)
        if not parsed.hostname:
            return None
        return parsed.hostname.lower().rstrip("."), parsed.port
    except ValueError:
        return None


def request_host_allowed(request: Request, config: SecurityConfig) -> bool:
    values = request.headers.getlist("host")
    if len(values) != 1 or (authority := _authority(values[0])) is None:
        return False
    allowed = set()
    for host in config.allowed_hosts:
        parsed = _authority(host)
        if parsed:
            allowed.add(parsed[0])
    for origin in config.allowed_origins + config.lan_origins:
        try:
            parsed = _authority(urlsplit(origin).netloc)
        except ValueError:
            parsed = None
        if parsed:
            allowed.add(parsed[0])
    server = request.scope.get("server")
    if server and not bind_is_wildcard(server[0]):
        allowed.add(server[0].lower().rstrip("."))
        if bind_is_loopback(server[0]):
            allowed.update({"localhost", "127.0.0.1", "::1"})
    return authority[0] in allowed


def request_is_direct_lan(request: Request, config: SecurityConfig) -> bool:
    """A narrow HTTP exception for explicit local addresses and direct LAN peers."""
    peer = _direct_peer(request)
    if _address_in_networks(peer, config.trusted_proxies):
        return False
    if not _address_in_networks(peer, config.lan_networks):
        return False
    return f"http://{request.headers.get('host', '').lower()}" in config.lan_origins


def _listener_origins(request: Request, config: SecurityConfig) -> set[str]:
    """Allow a nondefault direct listener port using server-owned bind state."""
    if config.require_https or not request_host_allowed(request, config):
        return set()
    server = request.scope.get("server")
    if not server or bind_is_wildcard(server[0]):
        return set()
    host = request.headers["host"]
    parsed = _authority(host)
    port = parsed[1] or (443 if request.url.scheme == "https" else 80)
    if port != server[1]:
        return set()
    # An explicitly configured public hostname is not a license to invent
    # extra origins for it. Automatic origins are only direct bind addresses.
    direct = {server[0].lower().rstrip(".")}
    if bind_is_loopback(server[0]):
        direct.update({"localhost", "127.0.0.1", "::1"})
    if parsed[0] not in direct:
        return set()
    return {f"{request.url.scheme}://{host.lower()}"}


def request_origin_allowed(request: Request, config: SecurityConfig) -> bool:
    """Reject cross-site state-changing requests.

    This runs for every deployment mode, not only authenticated ones: an
    unauthenticated loopback listener is still reachable from any page the
    user happens to open, and ``Request.json()`` parses a body regardless of
    its ``Content-Type``, so a simple cross-site request needs no preflight.
    """
    if request.method not in _UNSAFE_METHODS:
        return True
    origin = request.headers.get("Origin")
    if not origin:
        # Non-browser API clients generally do not send Origin.
        return True
    return (
        origin in config.allowed_origins
        or origin in _listener_origins(request, config)
        or (request_is_direct_lan(request, config) and origin in config.lan_origins)
    )


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
    """SQLite-backed fixed-window limiter shared across restarts and workers.

    The database defaults to ``<task root>/.rate-limits.sqlite3``, so the
    budget is scoped to the task root rather than to a process. Two GMXBUILDER
    instances pointed at one ``GMXBUILDER_TASK_DIR`` therefore share a single
    budget, and restarting an instance does not reset it -- durability across
    restarts is the point. Give each instance its own ``GMXBUILDER_TASK_DIR``,
    or set ``GMXBUILDER_RATE_LIMIT_DB`` explicitly, when they should be
    counted separately.
    """

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
            if connection.in_transaction:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass  # Preserve the original database error.
            raise
        finally:
            connection.close()

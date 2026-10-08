"""Serialize installer changes with HTTP admission; active work is checked separately."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from starlette.responses import JSONResponse


def installation_lock() -> Path:
    configured = os.environ.get("GMXBUILDER_INSTALL_LOCK")
    if configured:
        return Path(configured)
    state = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
    return state / "gmxbuilder/install.lock"


def prepare_installation_lock() -> None:
    """Create the shared lock before managed Web writes are confined."""
    path = installation_lock()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("ab"):
        pass


class InstallationGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from gmxbuilder.web.resource_policy import is_worker

        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"} or is_worker():
            return await self.app(scope, receive, send)
        path = installation_lock()
        try:
            # Managed Web can read this outside its write sandbox; LOCK_SH needs no write fd.
            handle = path.open("rb")
        except FileNotFoundError:
            try:
                prepare_installation_lock()
                handle = path.open("rb")
            except OSError:
                handle = None
        except OSError:
            handle = None
        if handle is None:
            response = JSONResponse(
                {"error": "Submissions are temporarily unavailable; please retry"},
                status_code=503,
                headers={"Retry-After": "30"},
            )
            return await response(scope, receive, send)
        with handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                response = JSONResponse(
                    {"error": "Installation is in progress; retry after it completes"},
                    status_code=503,
                    headers={"Retry-After": "30"},
                )
                return await response(scope, receive, send)
            try:
                # Keep admission protected until the operation has been registered.
                return await self.app(scope, receive, send)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

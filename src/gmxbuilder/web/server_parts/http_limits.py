"""Bounded HTTP request-body handling for the web application."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass

from fastapi.responses import JSONResponse

logger = logging.getLogger("gmxbuilder.web")
_UPLOAD = re.compile(r"/api/(?:upload-pdb|(?:ligand-chemistry-upload|cgenff-upload)/[^/]+)/?\Z")


class BodyLimitError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def _body_failure(error):
    if isinstance(error, BodyLimitError):
        return error
    nested = getattr(error, "exceptions", ())
    if nested:
        failures = [_body_failure(item) for item in nested]
        if all(failures):
            return failures[0]
    return None


def positive_seconds(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return value if 0 < value <= 3600 else default
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class BodyContract:
    limit: int
    multipart: bool
    media_allowed: bool
    declared: int | None


def body_contract(scope) -> BodyContract:
    headers = {}
    for key, value in scope.get("headers", []):
        key = key.lower()
        if key in {b"content-type", b"content-length"} and key in headers:
            raise BodyLimitError(400, "Duplicate request body header")
        headers[key] = value
    declared = headers.get(b"content-length")
    if declared is not None:
        try:
            if not declared.isdigit():
                raise ValueError
            declared = int(declared)
        except ValueError as exc:
            raise BodyLimitError(400, "Invalid Content-Length header") from exc
    multipart = scope.get("method") == "POST" and bool(_UPLOAD.fullmatch(scope.get("path", "")))
    media = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
    if multipart:
        limit = positive_body_limit("GMXBUILDER_UPLOAD_BODY_LIMIT", 128 * 1024**2)
        allowed = media == b"multipart/form-data"
    else:
        limit = positive_body_limit("GMXBUILDER_JSON_BODY_LIMIT", 2 * 1024**2)
        allowed = media == b"application/json" or (
            media.startswith(b"application/") and media.endswith(b"+json")
        )
    return BodyContract(limit, multipart, allowed, declared)


def validate_body_media(scope) -> None:
    contract = body_contract(scope)
    # Empty body-less operations remain compatible. Missing headers on a
    # streamed nonempty body are checked at the first received chunk as well.
    has_body = contract.declared not in {None, 0} or any(
        key.lower() in {b"content-type", b"transfer-encoding"}
        for key, _value in scope.get("headers", [])
    )
    if has_body and not contract.media_allowed:
        raise BodyLimitError(415, "Unsupported media type for this endpoint")


class BodyMemoryBudget:
    """One process-wide reservation pool, independent of asyncio loop lifetimes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.used = 0
        self.upload_used = 0

    def acquire(self, contract):
        try:
            budget = int(os.environ.get("GMXBUILDER_HTTP_BODY_MEMORY_MIB", "2048"))
            if not 1 <= budget <= 65536:
                raise ValueError
        except ValueError:
            budget = 2048
        budget *= 1024**2
        # Include raw-body copies and JSON object expansion. Reserve a quarter
        # for control requests so uploads cannot consume the entire pool.
        cost = contract.limit * (4 if contract.multipart else 32)
        with self.lock:
            if self.used + cost > budget or (
                contract.multipart and self.upload_used + cost > budget * 3 // 4
            ):
                raise BodyLimitError(503, "Request body capacity is busy; retry shortly")
            self.used += cost
            self.upload_used += cost if contract.multipart else 0
        return cost

    def release(self, cost, multipart):
        with self.lock:
            self.used -= cost
            self.upload_used -= cost if multipart else 0


BODY_MEMORY = BodyMemoryBudget()


def positive_body_limit(name: str, default: int) -> int:
    """Read a bounded positive request-body limit from the environment."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default
    return value if 1 <= value <= 1024 * 1024 * 1024 else default


class RequestBodyLimitMiddleware:
    """Reject oversized fixed-length and chunked HTTP request bodies."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.application(scope, receive, send)
            return
        consumed = 0
        response_started = False
        finished = False
        reservation = 0
        started = None
        contract = None

        async def limited_receive():
            nonlocal consumed, finished, reservation, started
            if finished:
                return await receive()
            if not reservation:
                reservation = BODY_MEMORY.acquire(contract)
            loop = asyncio.get_running_loop()
            if started is None:
                started = loop.time()
            remaining = positive_seconds("GMXBUILDER_BODY_TOTAL_TIMEOUT", 300) - (
                loop.time() - started
            )
            timeout = min(positive_seconds("GMXBUILDER_BODY_IDLE_TIMEOUT", 30), remaining)
            if timeout <= 0:
                raise BodyLimitError(408, "Request body deadline exceeded")
            try:
                message = await asyncio.wait_for(receive(), timeout)
            except TimeoutError as exc:
                raise BodyLimitError(408, "Request body deadline exceeded") from exc
            if message.get("type") == "http.request":
                chunk = message.get("body", b"")
                if chunk and not contract.media_allowed:
                    raise BodyLimitError(415, "Unsupported media type for this endpoint")
                consumed += len(chunk)
                if consumed > contract.limit:
                    raise BodyLimitError(413, "Request body exceeds the endpoint byte limit")
                finished = not message.get("more_body", False)
            elif message.get("type") == "http.disconnect":
                finished = True
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            contract = body_contract(scope)
            if contract.declared is not None and contract.declared > contract.limit:
                raise BodyLimitError(413, "Request body exceeds the endpoint byte limit")
            await self.application(scope, limited_receive, tracked_send)
        except Exception as error:
            exc = _body_failure(error)
            if exc is None:
                raise
            if not response_started:
                from gmxbuilder.web.security import apply_security_headers

                response = JSONResponse(
                    {"error": str(exc)},
                    status_code=exc.status,
                    headers={"Retry-After": "1"} if exc.status == 503 else None,
                )
                apply_security_headers(
                    response,
                    public_mode=os.environ.get("GMXBUILDER_DEPLOYMENT_MODE")
                    in {"public", "public-anonymous"},
                )
                await response(scope, receive, send)
        finally:
            if reservation:
                BODY_MEMORY.release(reservation, contract.multipart)

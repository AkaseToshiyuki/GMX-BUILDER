"""FastAPI web server for GMXBUILDER."""

from __future__ import annotations

import asyncio
import copy
import heapq
import json
import logging
import math
import os
import re
import secrets
import tempfile
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware

from gmxbuilder import VERSION
from gmxbuilder.core.exceptions import ModuleConfigError, ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.io.pdb import PDBValidator
from gmxbuilder.modules.export.naming import read_authoritative_archive
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.runtime.hardware import (
    configured_task_slots,
    configured_task_threads,
    hardware_capabilities,
    task_thread_allocation,
    task_thread_scope,
)
from gmxbuilder.web import ligand_prep
from gmxbuilder.web.custom_lipids import (
    CustomLipidStore,
    task_custom_lipid_scope,
)
from gmxbuilder.web.resource_policy import is_worker, managed_root
from gmxbuilder.web.security import (
    DurableRateLimiter,
    SecurityConfig,
    apply_security_headers,
    client_identity,
    request_authenticated,
    request_host_allowed,
    request_is_direct_lan,
    request_is_https,
    request_origin_allowed,
)
from gmxbuilder.web.server_parts import http_limits as _http_limits
from gmxbuilder.web.server_parts import modification_preview as _modification_preview
from gmxbuilder.web.server_parts import orientation_preview as _orientation_preview
from gmxbuilder.web.server_parts import structure_files as _structure_files
from gmxbuilder.web.server_parts import structure_processing as _structure_processing
from gmxbuilder.web.server_parts import task_resources as _task_resource_support
from gmxbuilder.web.server_parts.admission import (
    BoundedAdmission,
    TaskAdmission,
    WorkQueueFull,
)
from gmxbuilder.web.server_parts.input_limits import (
    StructureInputLimitError,
    StructureInputLimits,
)
from gmxbuilder.web.server_parts.static_delivery import CatalogCompression, VersionedStaticFiles
from gmxbuilder.web.server_parts.task_security import (
    InvalidTaskId,
    install_capability_log_filter,
    task_log_reference,
)
from gmxbuilder.web.server_parts.task_security import (
    validate_task_id as _validate_task_id,
)
from gmxbuilder.web.task_manager import task_manager

# Compatibility aliases keep server.py's legacy private helper surface intact
# while the implementations live together under server_parts/.
RequestBodyLimitMiddleware = _http_limits.RequestBodyLimitMiddleware
_positive_body_limit = _http_limits.positive_body_limit
_NONSTANDARD_AA_MAP = _structure_processing.NONSTANDARD_AA_MAP
_STRUCTURE_SUFFIX_FORMATS = _structure_processing.STRUCTURE_SUFFIX_FORMATS
_WATER_RESNAMES = _structure_processing.WATER_RESNAMES
_is_hydrogen = _structure_processing.is_hydrogen
_apply_disulfide_rename = _structure_processing.apply_disulfide_rename
_auto_clean_pdb = _structure_processing.auto_clean_pdb
_detect_disulfides = _structure_processing.detect_disulfides
_extract_sequences = _structure_processing.extract_sequences
_filter_pdb_for_display = _structure_processing.filter_pdb_for_display
_prepare_and_inspect_structure_upload = _structure_processing.prepare_and_inspect_structure_upload
_prepare_structure_upload = _structure_processing.prepare_structure_upload
process_uploaded_structure = _structure_processing.process_uploaded_structure
read_bounded_pdb_display = _structure_processing.read_bounded_pdb_display
_structure_upload_suffix = _structure_processing.structure_upload_suffix
summarize_resume_structure = _structure_processing.summarize_resume_structure
_normalise_small_molecule_labels = _structure_files.normalise_small_molecule_labels
_build_modification_preview = _modification_preview.build_modification_preview
_legacy_orientation_payload = _orientation_preview.legacy_orientation_payload
_task_resource_helpers = _task_resource_support.TaskResourceHelpers(
    lambda: task_manager,
    _validate_task_id,
)
_SERVER_PATH_PATTERN = _task_resource_support.SERVER_PATH_PATTERN
_validate_task_resource = _task_resource_helpers.validate_task_resource
_resolve_input_pdb = _task_resource_helpers.resolve_input_pdb
_resolve_pdb_path = _task_resource_helpers.resolve_pdb_path
_resolve_propka_pdb_path = _task_resource_helpers.resolve_propka_pdb_path
_redact_server_paths = _task_resource_helpers.redact_server_paths
_sanitize_public_value = _task_resource_helpers.sanitize_public_value
_public_task_state = _task_resource_helpers.public_task_state
_public_step_result = _task_resource_helpers.public_step_result
_authoritative_task_zip = _task_resource_helpers.authoritative_task_zip

# ---------------------------------------------------------------------------
# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gmxbuilder.web")
install_capability_log_filter()

_MARTINI_TASK_TYPES = frozenset(
    {
        "martini3-bilayer",
        "martini3-solvent",
    }
)


def _is_martini_task_type(task_type: str | None) -> bool:
    return task_type in _MARTINI_TASK_TYPES


# ---------------------------------------------------------------------------
# App setup

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Configurable task root (production: set GMXBUILDER_TASK_DIR to persistent location)
#
# Production deployment notes:
#   - Use a reverse proxy with TLS (nginx + Let's Encrypt) to protect
#     X-Admin-Token header from cleartext interception.
#   - Set GMXBUILDER_ADMIN_TOKEN to a long random string for /api/tasks access.
#   - Set GMXBUILDER_CORS_ORIGINS to your actual domain (comma-separated).
#   - Rate-limiting is recommended (e.g. via nginx limit_req_zone).
#   - Example:  gmxbuilder serve --host 127.0.0.1 --port 8000
#              nginx proxy_pass https://your-domain → http://127.0.0.1:8000
_INITIAL_SECURITY = SecurityConfig.from_environment()
_ALLOWED_ORIGINS = _INITIAL_SECURITY.allowed_origins


@asynccontextmanager
async def _app_lifespan(_application: FastAPI):
    await startup_background_tasks()
    try:
        yield
    finally:
        await shutdown_event()


app = FastAPI(title="GMXBUILDER", version=VERSION, lifespan=_app_lifespan)
app.add_middleware(CatalogCompression)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-GMXBUILDER-Token"],
)
app.mount(
    "/static", VersionedStaticFiles(directory=str(_STATIC_DIR), follow_symlink=False), name="static"
)


class InvalidJsonBody(ValueError):
    """Expected client error for malformed or non-object JSON input."""


@app.exception_handler(InvalidTaskId)
async def _invalid_task_id_handler(_request: Request, _exc: InvalidTaskId):
    return JSONResponse({"error": "Invalid task ID format"}, status_code=400)


@app.exception_handler(InvalidJsonBody)
async def _invalid_json_handler(_request: Request, exc: InvalidJsonBody):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(WorkQueueFull)
async def _work_queue_full_handler(_request: Request, _exc: WorkQueueFull):
    return JSONResponse(
        {"error": "The interactive work queue is full; retry shortly."},
        status_code=503,
        headers={"Retry-After": "5"},
    )


async def _json_object(request: Request) -> dict[str, Any]:
    """Decode one bounded JSON object without turning client errors into 500s."""
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidJsonBody("Request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise InvalidJsonBody("Request body must be a JSON object")
    return value


_RATE_LIMITERS: dict[Path, DurableRateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _rate_limiter() -> DurableRateLimiter:
    configured = os.environ.get("GMXBUILDER_RATE_LIMIT_DB", "").strip()
    default_root = managed_root() / ".control" if managed_root() else task_manager.root
    path = Path(configured) if configured else default_root / ".rate-limits.sqlite3"
    path = path.expanduser().resolve()
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(path)
        if limiter is None:
            limiter = DurableRateLimiter(path)
            _RATE_LIMITERS[path] = limiter
        return limiter


def _rate_policy(path: str, method: str) -> tuple[str, int, float] | None:
    if method == "GET" and (path.startswith("/api/") or path == "/health"):
        if path in {"/api/health", "/health"}:
            return None
        return ("api-read", int(os.environ.get("GMXBUILDER_API_READ_RATE", "600")), 60.0)
    if path == "/api/build":
        return ("finalize", int(os.environ.get("GMXBUILDER_FINALIZE_RATE", "30")), 3600.0)
    if (
        path in {"/api/upload-pdb", "/api/build-lipid-library"}
        or path.startswith("/api/cgenff-upload/")
        or path.startswith("/api/ligand-chemistry-upload/")
        or (path.startswith("/api/task/") and path.endswith("/custom-lipids"))
    ):
        # Temporary deployment override; removing it restores the durable budget.
        if os.environ.get("GMXBUILDER_HEAVY_RATE_DISABLED", "") == "1":
            return None
        return ("heavy", int(os.environ.get("GMXBUILDER_HEAVY_RATE", "20")), 3600.0)
    if path.startswith("/api/") and path not in {"/api/health"}:
        return ("api-write", int(os.environ.get("GMXBUILDER_API_RATE", "240")), 60.0)
    return None


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforce deployment authentication, trusted-proxy and durable rate policy."""
    # Only the local executable sets this flag; the worker has no network
    # listener. Its request was authenticated and admitted by this process.
    if is_worker():
        return await call_next(request)
    security = SecurityConfig.from_environment()
    public_mode = security.require_https
    live_probe = request.url.path in {"/health/live", "/health/ready"}

    def secured(response):
        return apply_security_headers(response, public_mode=public_mode)

    if not request_host_allowed(request, security):
        return secured(JSONResponse({"error": "Request Host is not allowed"}, status_code=400))

    if security.errors and not live_probe:
        return secured(
            JSONResponse(
                {
                    "error": "Invalid server security configuration",
                    "details": list(security.errors),
                },
                status_code=503,
            )
        )

    if (
        not live_probe
        and public_mode
        and not request_is_https(request, security)
        and not request_is_direct_lan(request, security)
    ):
        return secured(
            JSONResponse(
                {"error": "HTTPS is required in public deployment mode"},
                status_code=426,
            )
        )

    authentication_required = security.mode == "public" or security.authentication_enabled
    if not live_probe and authentication_required and not request_authenticated(request, security):
        headers = {
            "WWW-Authenticate": (
                'Basic realm="GMXBUILDER", charset="UTF-8"' if security.auth_user else "Bearer"
            )
        }
        return secured(
            JSONResponse({"error": "Authentication is required"}, status_code=401, headers=headers)
        )

    # Enforced in every mode. Authentication is not what makes a cross-site
    # write dangerous; reachability is, and a loopback listener is reachable
    # from any page the operator happens to have open.
    if not live_probe and not request_origin_allowed(request, security):
        return secured(JSONResponse({"error": "Request Origin is not allowed"}, status_code=403))

    try:
        _http_limits.validate_body_media(request.scope)
    except _http_limits.BodyLimitError as exc:
        return secured(JSONResponse({"error": str(exc)}, status_code=exc.status))
    request.state.gmxbuilder_client_identity = client_identity(request, security)

    policy = _rate_policy(request.url.path, request.method)
    if policy is not None:
        bucket_name, limit, window = policy
        try:
            allowed, retry_after = await _run_control(
                _rate_limiter().allow,
                client_identity(request, security),
                bucket_name,
                limit,
                window,
            )
        except WorkQueueFull:
            return secured(
                JSONResponse({"error": "Server is busy; retry shortly"}, status_code=503)
            )
        if not allowed:
            return secured(
                JSONResponse(
                    {
                        "error": (
                            "Request rate limit exceeded. Wait before retrying; "
                            "running or queued work has not been cancelled."
                        )
                    },
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            )

    if _resources is not None:
        response = await _managed_request(request, call_next)
    else:
        response = await call_next(request)
    return secured(response)


# The managed branch returns before call_next: its network reads must also
# traverse the outer ASGI byte/deadline/memory guard.
app.add_middleware(RequestBodyLimitMiddleware)
from gmxbuilder.web.installation_guard import InstallationGuardMiddleware  # noqa: E402

app.add_middleware(InstallationGuardMiddleware)


_resources = None


async def _managed_request(request, call_next):
    from gmxbuilder.web.resource_coordinator import expensive_request, task_from_path

    cached_display = False
    if request.method == "GET" and request.url.path.endswith("/viewer.json"):
        from gmxbuilder.web.server_parts.viewer_data import cached_viewer

        parts = request.url.path.strip("/").split("/")
        if len(parts) == 5 and parts[1] == "step":
            try:
                task_id = _validate_task_id(parts[2])
                directory = _validate_task_resource(task_id, Path("steps") / parts[3])
                cached_display = cached_viewer(directory) is not None
            except (ValueError, OSError):
                pass
    if expensive_request(request.method, request.url.path) and not cached_display:
        return await _resources.enqueue_request(request)
    task_id = task_from_path(request.url.path)
    if request.url.path.startswith("/api/operations/"):
        parts = request.url.path.strip("/").split("/")
        operation = _resources.queue.get(parts[2])
        task_id = operation["task_id"] if operation else None
        # Status remains readable after pressure cleanup so the reason is visible.
        if not request.url.path.endswith("/result"):
            task_id = None
        elif operation and operation["status"] in {"failed", "cancelled"}:
            return _resources.operation_response(parts[2], result=True)
    leased_state = task_manager.get_state(task_id) if task_id else None
    if task_id and leased_state is None:
        return JSONResponse(
            {"error": "Task expired or was removed to release storage."}, status_code=410
        )
    if request.method == "POST" and request.url.path in {"/api/tasks"}:
        if not _resources.make_room(65536):
            return JSONResponse(
                {"error": "Storage is full; new writes are paused."}, status_code=507
            )
    if not task_id:
        return await call_next(request)
    lease = task_manager.active_task(task_id)
    lease.__enter__()
    try:
        response = await call_next(request)
    except BaseException:
        lease.__exit__(None, None, None)
        raise
    original = response.body_iterator

    async def leased_body():
        from datetime import datetime

        deadline = datetime.fromisoformat(leased_state["expires_at"]).timestamp()
        transfer = asyncio.current_task()
        expiry_timer = asyncio.get_running_loop().call_later(
            max(0, deadline - time.time()), transfer.cancel
        )
        try:
            async for chunk in original:
                yield chunk
        finally:
            expiry_timer.cancel()
            lease.__exit__(None, None, None)

    response.body_iterator = leased_body()
    return response


@app.get("/api/operations/{operation_id}")
async def api_operation_status(operation_id: str):
    if _resources is None:
        return JSONResponse({"error": "Managed resource service is not enabled"}, status_code=404)
    return _resources.operation_response(operation_id)


@app.get("/api/operations/{operation_id}/result")
async def api_operation_result(operation_id: str):
    if _resources is None:
        return JSONResponse({"error": "Managed resource service is not enabled"}, status_code=404)
    return _resources.operation_response(operation_id, result=True)


@app.get("/api/resource-policy")
async def api_resource_policy():
    from gmxbuilder.web.resource_policy import ResourcePolicy

    policy = ResourcePolicy.from_environment()
    return {
        "managed": _resources is not None,
        "task_lifetime_hours": policy.lifetime_hours,
        "task_storage_bytes": policy.task_storage_bytes,
        "total_storage_bytes": policy.total_storage_bytes,
        "task_memory_bytes": policy.task_memory_bytes,
        "pause_reason": _resources.pause_reason if _resources else None,
    }


def _is_admin_request(request: Request) -> bool:
    configured = os.environ.get("GMXBUILDER_ADMIN_TOKEN", "")
    supplied = request.headers.get("X-Admin-Token", "")
    return bool(configured and supplied and secrets.compare_digest(configured, supplied))


# Task store (in-memory — survives as long as the server runs)
_tasks: dict[str, dict] = {}
# Live progress for the Check currently running on a task. Deliberately in
# memory only: it is meaningless after a restart, and the step itself is
# already checkpointed on disk.
_step_progress: dict[str, dict] = {}
_step_progress_lock = threading.Lock()
_build_logs: dict[str, list[str]] = {}  # task_id → list of log lines
_tasks_lock = threading.Lock()
_build_logs_lock = threading.Lock()


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


# Limit concurrent builds to prevent memory/CPU exhaustion.
# Each build can use several GB of RAM; 4 concurrent builds is a safe ceiling.
_MAX_CONCURRENT_BUILDS = min(
    _positive_environment_integer("GMXBUILDER_MAX_BUILDS", 4),
    configured_task_slots(),
)
_build_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_BUILDS)
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_MAX_STEP_WORKERS = min(2, _MAX_CONCURRENT_BUILDS)
_MAX_STEP_SUBMISSIONS = min(
    _positive_environment_integer("GMXBUILDER_MAX_STEP_SUBMISSIONS", _MAX_STEP_WORKERS * 2),
    64,
)
_step_admission = TaskAdmission(_MAX_STEP_SUBMISSIONS)
_step_executor: ThreadPoolExecutor | None = None
_step_executor_lock = threading.Lock()
_custom_lipid_executor: ThreadPoolExecutor | None = None
_custom_lipid_executor_lock = threading.Lock()
_MAX_INTERACTIVE_SUBMISSIONS = min(
    _positive_environment_integer("GMXBUILDER_MAX_INTERACTIVE_SUBMISSIONS", 6),
    64,
)
_interactive_admission = BoundedAdmission(_MAX_INTERACTIVE_SUBMISSIONS)
_interactive_executor: ThreadPoolExecutor | None = None
_interactive_executor_lock = threading.Lock()
_control_executor: ThreadPoolExecutor | None = None
_control_executor_lock = threading.Lock()
_control_admission = BoundedAdmission(32)
_event_loop: asyncio.AbstractEventLoop | None = None
_lifespan_tasks: list[asyncio.Task] = []
_pka_cache: dict[str, list[dict]] = {}  # task_id → pKa predictions
_pka_cache_digest: dict[str, str] = {}
_pka_cache_lock = threading.Lock()
_pka_running: set[str] = set()  # task_ids currently being computed
# Track which tasks are currently building (prevent duplicate builds)
_building_tasks: set[str] = set()


def _configured_custom_gpu_ids() -> tuple[int, ...]:
    """Read the already-established deployment allocation without probing CUDA."""
    configured = os.environ.get("GMXBUILDER_GPU_IDS", "").strip()
    if not configured:
        return ()
    try:
        identifiers = tuple(int(value.strip()) for value in configured.split(","))
    except ValueError:
        logger.error("Ignoring invalid GMXBUILDER_GPU_IDS=%r", configured)
        return ()
    if any(value < 0 for value in identifiers) or len(set(identifiers)) != len(identifiers):
        logger.error("Ignoring invalid GMXBUILDER_GPU_IDS=%r", configured)
        return ()
    return identifiers


_CUSTOM_GPU_IDS = _configured_custom_gpu_ids()
_CUSTOM_GPU_CONCURRENCY = min(
    2,
    max(1, int(os.environ.get("GMXBUILDER_CUSTOM_LIPID_CONCURRENCY", "2"))),
    max(1, len(_CUSTOM_GPU_IDS)),
)

# Cleanup expired tasks on startup
_startup_removed = [] if managed_root() else task_manager.cleanup_expired()
if _startup_removed:
    logger.info("Cleaned up %d expired task(s)", len(_startup_removed))


# ---------------------------------------------------------------------------
# Lifespan: periodic cleanup + graceful shutdown


async def startup_background_tasks():
    """Start background tasks: periodic cleanup + build-queue consumer."""
    global _queue_event, _event_loop
    global _resources
    if managed_root() and not is_worker():
        from gmxbuilder.web.resource_coordinator import ResourceCoordinator

        _resources = ResourceCoordinator(managed_root(), task_manager)
        await _resources.start()
        _resources.confine_web_writes()
        return
    _get_executor()
    _get_step_executor()
    _get_custom_lipid_executor()
    _get_interactive_executor()
    _event_loop = asyncio.get_running_loop()
    _queue_event = asyncio.Event()
    # Finalization requests are task-owned and restart-safe.  A service
    # restart converts both formerly running and queued jobs back to FIFO
    # queue entries; checkpoint finalization is deterministic and idempotent.
    recovered: list[tuple[str, int]] = []
    with _queue_lock:
        queued_ids = {task_id for task_id, _data in _build_queue}
        for task_dir in sorted(task_manager.root.iterdir()):
            if not task_dir.is_dir() or task_manager.is_expired(task_dir.name):
                continue
            state = task_manager.get_state(task_dir.name) or {}
            build_status = state.get("build_status") or {}
            if build_status.get("status") not in {"queued", "running"}:
                continue
            request = task_manager.load_build_request(task_dir.name)
            if request is None or task_dir.name in queued_ids:
                continue
            _build_queue.append((task_dir.name, request))
            _queue_enqueued_at[task_dir.name] = time.time()
            queued_ids.add(task_dir.name)
            recovered.append((task_dir.name, len(_build_queue)))
    for task_id, position in recovered:
        with _tasks_lock:
            _tasks[task_id] = {
                "status": "queued",
                "progress": 0,
                "result": None,
                "error": None,
                "queue_position": position,
            }
        _persist_build_status(
            task_id,
            "queued",
            queue_position=position,
            recovered_after_restart=True,
        )
    if _build_queue:
        _queue_event.set()
    # New lipids are reviewed and built offline by the administrator. Never
    # restart legacy task-local pre-equilibration jobs during Web startup.
    # Keep their records intact so users can read the previous status/results.

    # ---- Periodic cleanup ----
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(1800)
            try:
                removed = task_manager.cleanup_expired()
                if removed:
                    logger.info("Periodic cleanup: removed %d expired task(s)", len(removed))
                    with _tasks_lock:
                        for tid in removed:
                            _tasks.pop(tid, None)
                            _building_tasks.discard(tid)
                    with _build_logs_lock:
                        for tid in removed:
                            _build_logs.pop(tid, None)
                    for tid in removed:
                        ligand_prep.cancel(tid)
                    # pKa cache is keyed by file path, not task ID —
                    # reconstruct paths from removed task IDs
                    with _pka_cache_lock:
                        keys_to_drop = []
                        for cache_path in list(_pka_cache.keys()):
                            for tid in removed:
                                if tid in cache_path:
                                    keys_to_drop.append(cache_path)
                                    break
                        for k in keys_to_drop:
                            _pka_cache.pop(k, None)
                            _pka_cache_digest.pop(k, None)
                        for tid in removed:
                            _pka_running.discard(tid)
                    # StepRunner cache cleanup
                    with _step_runners_lock:
                        for tid in removed:
                            _step_runners.pop(tid, None)
                    # Also remove expired tasks from queue
                    with _queue_lock:
                        _build_queue[:] = [
                            (tid, d) for tid, d in _build_queue if tid not in removed
                        ]
                        for tid in removed:
                            _queue_enqueued_at.pop(tid, None)
                            _build_started_at.pop(tid, None)
            except Exception:
                logger.exception("Periodic cleanup failed")

    _lifespan_tasks.append(asyncio.create_task(_cleanup_loop()))

    # ---- Build-queue consumer ----
    _lifespan_tasks.append(asyncio.create_task(_consume_queue()))


async def shutdown_event():
    """Graceful shutdown: wait for in-flight builds to complete."""
    global _executor, _step_executor, _custom_lipid_executor, _interactive_executor, _event_loop
    global _resources, _control_executor
    if _resources is not None:
        await _resources.close()
        _resources = None
    from gmxbuilder.web.server_parts.option_catalog import close_catalog

    await asyncio.to_thread(close_catalog)
    logger.info("Shutting down — waiting for in-flight builds...")
    for task in _lifespan_tasks:
        task.cancel()
    if _lifespan_tasks:
        await asyncio.gather(*_lifespan_tasks, return_exceptions=True)
        _lifespan_tasks.clear()
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=False)
    with _step_executor_lock:
        step_executor = _step_executor
        _step_executor = None
    if step_executor is not None:
        step_executor.shutdown(wait=True, cancel_futures=False)
    with _custom_lipid_executor_lock:
        custom_executor = _custom_lipid_executor
        _custom_lipid_executor = None
    if custom_executor is not None:
        custom_executor.shutdown(wait=True, cancel_futures=False)
    with _interactive_executor_lock:
        interactive_executor = _interactive_executor
        _interactive_executor = None
    if interactive_executor is not None:
        interactive_executor.shutdown(wait=True, cancel_futures=False)
    with _control_executor_lock:
        control_executor, _control_executor = _control_executor, None
    if control_executor is not None:
        control_executor.shutdown(wait=True, cancel_futures=True)
    _event_loop = None
    logger.info("Shutdown complete")


def _get_executor() -> ThreadPoolExecutor:
    """Return a live executor, recreating it after a TestClient/app restart."""
    global _executor
    with _executor_lock:
        if _executor is None or getattr(_executor, "_shutdown", False):
            _executor = ThreadPoolExecutor(
                max_workers=_MAX_CONCURRENT_BUILDS,
                thread_name_prefix="gmxbuilder",
            )
        return _executor


def _get_custom_lipid_executor() -> ThreadPoolExecutor:
    """Background ligand parameterization with the existing worker allocation.

    The legacy name is retained for callers; Web lipid pre-equilibration has
    been retired and no longer uses this pool.
    """
    global _custom_lipid_executor
    with _custom_lipid_executor_lock:
        if _custom_lipid_executor is None or getattr(_custom_lipid_executor, "_shutdown", False):
            _custom_lipid_executor = ThreadPoolExecutor(
                max_workers=_CUSTOM_GPU_CONCURRENCY,
                thread_name_prefix="gmxbuilder-custom-lipid",
            )
        return _custom_lipid_executor


def _get_step_executor() -> ThreadPoolExecutor:
    """Dedicated bounded pool so interactive Checks cannot starve finalization."""
    global _step_executor
    with _step_executor_lock:
        if _step_executor is None or getattr(_step_executor, "_shutdown", False):
            _step_executor = ThreadPoolExecutor(
                max_workers=_MAX_STEP_WORKERS,
                thread_name_prefix="gmxbuilder-step",
            )
        return _step_executor


def _get_interactive_executor() -> ThreadPoolExecutor:
    """Small bounded pool for previews and scientific helper calculations."""
    global _interactive_executor
    with _interactive_executor_lock:
        if _interactive_executor is None or getattr(_interactive_executor, "_shutdown", False):
            _interactive_executor = ThreadPoolExecutor(
                max_workers=min(2, _MAX_CONCURRENT_BUILDS),
                thread_name_prefix="gmxbuilder-interactive",
            )
        return _interactive_executor


def _submit_interactive(function, /, *args, **kwargs):
    """Release admission after completion or cancellation before execution."""
    admission = _interactive_admission
    if not admission.try_acquire():
        raise WorkQueueFull("interactive work queue is full")
    try:
        future = _get_interactive_executor().submit(function, *args, **kwargs)
    except BaseException:
        admission.release()
        raise
    # A queued future may be cancelled without ever entering the callable. A
    # running future cannot be cancelled and keeps its slot until it finishes.
    future.add_done_callback(lambda _finished: admission.release())
    return future


async def _run_control(function, /, *args, **kwargs):
    """Keep bounded rate-limit/control reads independent of scientific previews."""
    global _control_executor
    admission = _control_admission
    if not admission.try_acquire():
        raise WorkQueueFull("control work queue is full")
    try:
        with _control_executor_lock:
            if _control_executor is None:
                _control_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="gmxbuilder-control"
                )
            future = _control_executor.submit(function, *args, **kwargs)
    except BaseException:
        admission.release()
        raise
    future.add_done_callback(lambda _finished: admission.release())
    return await asyncio.wrap_future(future)


async def _run_interactive(function, /, *args, **kwargs):
    return await asyncio.wrap_future(_submit_interactive(function, *args, **kwargs))


def _signal_queue() -> None:
    """Wake the asyncio queue consumer safely from worker threads."""
    event = _queue_event
    loop = _event_loop
    if event is None:
        return
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(event.set)
    else:
        event.set()


# ---------------------------------------------------------------------------
# Page routes


def _wizard_html() -> HTMLResponse:
    """Render the single frontend shell used by history-based task routes."""
    from gmxbuilder import VERSION

    template_path = _TEMPLATE_DIR / "index.html"
    if not template_path.is_file():
        return HTMLResponse("<h1>GMXBUILDER Web</h1><p>Template not found.</p>")
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("{{ version }}", VERSION)
    # Optional deployment extensions. A build with none -- every public build --
    # substitutes an empty string, so the region collapses rather than leaving
    # an empty panel. Failures inside an extension are contained there; the page
    # must render either way.
    try:
        from gmxbuilder.extensions import render_announcements_html

        announcements = render_announcements_html()
    except Exception:
        logger.warning("Homepage announcements were skipped", exc_info=True)
        announcements = ""
    html = html.replace("{{ announcements }}", announcements)
    return HTMLResponse(html)


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index(request: Request):
    """Serve the workflow selection page."""
    return _wizard_html()


_WORKFLOW_ROUTES = {
    "BilayerBuilder",
    "PureBilayerSystem",
    "Solvator",
    "CoarseGrainedBuilder",
    "Martini3BilayerBuilder",
    "Martini3SolventBuilder",
}


@app.get("/{workflow}/Step{step}", response_class=HTMLResponse)
async def workflow_step_page(workflow: str, step: int):
    """Serve a workflow URL before it owns a persistent task."""
    if workflow not in _WORKFLOW_ROUTES or step < 1 or step > 20:
        return HTMLResponse("Not found", status_code=404)
    return _wizard_html()


@app.get("/{workflow}/{task_id}/Step{step}", response_class=HTMLResponse)
async def task_step_page(workflow: str, task_id: str, step: int):
    """Retire legacy task-bearing links without rendering their identifiers."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/", status_code=307)


# ---------------------------------------------------------------------------
# Health check


@app.api_route("/health/live", methods=["GET", "HEAD"])
async def liveness_check():
    """Minimal unauthenticated liveness probe without host fingerprinting."""
    return {"status": "ok"}


@app.api_route("/health/ready", methods=["GET", "HEAD"])
async def readiness_check():
    """Report ability to use managed storage, without exposing task details."""
    result = await _resources.readiness() if _resources else {"ready": True, "reason": "ready"}
    return JSONResponse(result, status_code=200 if result["ready"] else 503)


@app.get("/health")
async def health_check():
    """Authenticated detailed health in public mode; unchanged on trusted networks."""
    from gmxbuilder import VERSION as _ver

    security = SecurityConfig.from_environment()
    operations = _resources.queue.counts() if _resources is not None else {}
    return {
        "status": "ok",
        "version": _ver,
        "builds_active": len(_building_tasks),
        "builds_max": _MAX_CONCURRENT_BUILDS,
        "builds_queued": len(_build_queue),
        "operations_active": operations.get("running", 0),
        "operations_queued": operations.get("queued", 0),
        "resource_isolation_enabled": _resources is not None,
        "installation_lock_supported": True,
        "typical_build_seconds": int(round(_typical_build_seconds())),
        "hardware": hardware_capabilities().as_public_dict(),
        "security": {
            "deployment_mode": security.mode,
            "authentication_enabled": security.authentication_enabled,
            "unsafe_deployment_allowed": security.allow_unsafe_deployment,
            "trusted_proxy_count": len(security.trusted_proxies),
        },
    }


@app.get("/api/hardware")
async def api_hardware():
    """Report detected and operator-configured compute resources."""
    return hardware_capabilities().as_public_dict()


# ---------------------------------------------------------------------------
# API: task types


@app.get("/api/task-types")
async def api_task_types():
    """Return all available task types for the selection wizard."""
    from gmxbuilder.web.task_types import get_all_task_types

    return {"task_types": get_all_task_types()}


@app.get("/api/task-type/{task_id}")
async def api_task_type_detail(task_id: str):
    """Return full detail for a specific task type."""
    from gmxbuilder.web.task_types import get_task_type_detail

    detail = get_task_type_detail(task_id)
    if detail is None:
        return JSONResponse({"error": f"Unknown task type: {task_id}"}, status_code=404)
    return detail


@app.post("/api/tasks")
async def api_create_task(request: Request):
    """Create a task for a workflow that does not require an uploaded structure."""
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    task_type_id = str(data.get("task_type", "")).strip()
    from gmxbuilder.web.task_types import get_task_type_detail

    detail = get_task_type_detail(task_type_id)
    if detail is None or not detail.get("enabled"):
        return JSONResponse({"error": "Unknown or unavailable task type"}, status_code=400)
    if detail.get("requires_input", True):
        return JSONResponse(
            {"error": f"{detail['title']} requires a structure upload"}, status_code=400
        )
    task = task_manager.create_task(f"{task_type_id}_system")
    task_manager.update_state(
        task["task_id"],
        {
            "task_type": detail,
            "task_type_id": task_type_id,
            "current_step": detail["visible_modules"][0],
        },
    )
    return {"task_id": task["task_id"], "task_type": detail}


# ---------------------------------------------------------------------------
# API: PPM orientation


@app.post("/api/orient-ppm")
async def api_orient_ppm(request: Request):
    """Compute orientation for a previously uploaded PDB."""
    data = await _json_object(request)
    if data.get("tmp_path"):
        return JSONResponse(
            {"error": ("Client-supplied filesystem paths are not accepted; provide task_id")},
            status_code=400,
        )
    task_id_val = data.get("task_id", "")
    algorithm = data.get("algorithm", "ppm")
    half_thickness = data.get("half_thickness")  # nm, lipid-specific; None → use default
    if not task_id_val:
        return JSONResponse({"error": "task_id is required"}, status_code=400)
    try:
        task_id_val = _validate_task_id(str(task_id_val))
        if task_manager.get_state(task_id_val) is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
        tmp_path = str(_validate_task_resource(task_id_val, _resolve_pdb_path(task_id_val)))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not Path(tmp_path).exists():
        return JSONResponse({"error": "PDB file not found"}, status_code=400)
    if algorithm not in {"ppm", "hmoment", "tmd", "com"}:
        return JSONResponse(
            {"error": "algorithm must be ppm, hmoment, tmd, or com"},
            status_code=400,
        )

    try:
        return await _run_interactive(
            _legacy_orientation_payload, tmp_path, algorithm, half_thickness
        )
    except Exception:
        logger.exception("Unhandled error in orient-ppm")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def _generate_orientation_preview(task_id: str, config: dict) -> dict:
    """Run the real Step 4 module without writing or invalidating checkpoints.

    Task-directory resolution stays here because it depends on the module-level
    ``task_manager`` that tests replace; the scientific work itself lives in
    :mod:`gmxbuilder.web.server_parts.orientation_preview`.
    """
    return _orientation_preview.generate_orientation_preview(
        task_manager.get_task_dir(task_id), config
    )


@app.post("/api/orient-preview/{task_id}")
async def api_orient_preview(task_id: str, request: Request):
    """Preview exactly the coordinates that Step 4 Check would persist."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be an object"}, status_code=400)
    config = data.get("config", data)
    if not isinstance(config, dict):
        return JSONResponse({"error": "Orientation config must be an object"}, status_code=400)

    try:
        # Preview is read-only and short-lived. Keep it off the persistent
        # build executor so browser slider traffic cannot occupy build slots,
        # and so application/TestClient restarts cannot reuse a shut-down pool.
        payload = await _run_interactive(_generate_orientation_preview, task_id, config)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": f"Invalid orientation config: {exc}"}, status_code=400)
    except Exception as exc:
        from gmxbuilder.core.exceptions import ModuleConfigError

        if isinstance(exc, ModuleConfigError):
            return JSONResponse({"error": str(exc)}, status_code=400)
        logger.exception("Failed to generate orientation preview")
        return JSONResponse({"error": "Orientation preview failed"}, status_code=500)
    return JSONResponse(payload)


_cg_orientation_preview_cache: dict[tuple[str, int, float], tuple[System, list[str]]] = {}
_cg_orientation_preview_lock = threading.Lock()


def _cg_orientation_cache_key(task_id: str, half_thickness: float) -> tuple[str, int, float]:
    mapping = task_manager.get_task_dir(task_id) / "steps" / "cg_mapping" / "system.npz"
    if not mapping.is_file():
        raise FileNotFoundError("Martini mapping checkpoint is missing; run Map Protein first.")
    return task_id, mapping.stat().st_mtime_ns, round(float(half_thickness), 6)


def _cache_cg_automatic_pose(key: tuple[str, int, float], system: System, log: list[str]) -> None:
    with _cg_orientation_preview_lock:
        for old_key in list(_cg_orientation_preview_cache):
            if old_key[0] == key[0] and old_key != key:
                _cg_orientation_preview_cache.pop(old_key, None)
        _cg_orientation_preview_cache[key] = (system.copy(), list(log))
        while len(_cg_orientation_preview_cache) > 16:
            _cg_orientation_preview_cache.pop(next(iter(_cg_orientation_preview_cache)))


def _generate_cg_orientation_preview(task_id: str, config: dict) -> dict:
    """Preview the exact independent Martini orientation implementation.

    The expensive deterministic PPM-like base pose is cached per mapping
    checkpoint. Manual slider requests then apply only their inexpensive
    adjustment, while the Check module recomputes the same deterministic base.
    """
    from gmxbuilder.modules.martini3_bilayer.orientation import (
        CGOrientationModule,
        apply_manual_adjustment,
    )

    half_thickness = float(config.get("half_thickness", 1.4))
    key = _cg_orientation_cache_key(task_id, half_thickness)
    runner = _get_step_runner(task_id, "martini3-bilayer")
    mapped = runner.load_system("cg_mapping")
    if mapped is None:
        raise FileNotFoundError("Martini mapping checkpoint is missing; run Map Protein first.")
    module = CGOrientationModule()
    module.validate_config(config)
    method = str(config.get("method", "ppm")).lower()

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-cg-orient-preview-") as tmp:
        root = Path(tmp)
        if method == "ppm":
            preview_config = dict(config)
            preview_config.update(
                {
                    "_task_dir": str(root),
                    "_step_dir": str(root / "steps" / "cg_orientation"),
                }
            )
            result = module.execute(mapped, preview_config)
            if not result.success:
                raise RuntimeError("Martini orientation module reported failure")
            oriented = result.system
            log = list(result.log)
            _cache_cg_automatic_pose(key, oriented, log)
        else:
            with _cg_orientation_preview_lock:
                cached = _cg_orientation_preview_cache.get(key)
                cached = (cached[0].copy(), list(cached[1])) if cached else None
            if cached is None:
                automatic_config = {
                    "method": "ppm",
                    "half_thickness": half_thickness,
                    "_task_dir": str(root),
                    "_step_dir": str(root / "steps" / "cg_orientation"),
                }
                automatic = module.execute(mapped, automatic_config)
                if not automatic.success:
                    raise RuntimeError("Martini orientation module reported failure")
                oriented = automatic.system
                log = list(automatic.log)
                _cache_cg_automatic_pose(key, oriented, log)
            else:
                oriented, log = cached
            base_metrics = dict(oriented.metadata.get("cg_orientation") or {})
            metrics = apply_manual_adjustment(oriented, config, base_metrics)
            oriented.metadata["cg_orientation"] = metrics
            oriented.metadata["cg_orientation_method"] = "manual"
            log.extend(
                [
                    "Applied interactive manual Martini orientation adjustment",
                    f"Protein Z adjustment: {metrics['z_adjustment_nm']:.2f} nm",
                    f"Protein tilt adjustment: {metrics['tilt_degrees']:.1f} degrees",
                ]
            )

        from gmxbuilder.modules.coarse_grained.common import write_cg_viewer_pdb

        preview_path = root / "viewer.pdb"
        write_cg_viewer_pdb(oriented, preview_path, task_dir=runner.task_dir)
        pdb = preview_path.read_text(encoding="utf-8")

    return {
        "status": "ok",
        "method": method,
        "orientation": dict(oriented.metadata.get("cg_orientation") or {}),
        "oriented_pdb": pdb,
        "log": log,
    }


@app.post("/api/cg-orient-preview/{task_id}")
async def api_cg_orient_preview(task_id: str, request: Request):
    """Return exact Martini Step 4 coordinates without changing checkpoints."""
    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    task_type = ((task_state or {}).get("task_type") or {}).get("id") or (task_state or {}).get(
        "task_type_id"
    )
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    if task_type != "martini3-bilayer":
        return JSONResponse(
            {"error": "Martini orientation preview requires a Martini 3 Bilayer task"},
            status_code=409,
        )
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be an object"}, status_code=400)
    config = data.get("config", data)
    if not isinstance(config, dict):
        return JSONResponse({"error": "Orientation config must be an object"}, status_code=400)
    try:
        payload = await _run_interactive(_generate_cg_orientation_preview, task_id, config)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ModuleConfigError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Failed to generate Martini orientation preview")
        return JSONResponse({"error": "Martini orientation preview failed"}, status_code=500)
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# API: System Verification — preview PDB & geometry comparison
# ---------------------------------------------------------------------------


@app.post("/api/preview-pdb")
async def api_preview_pdb(request: Request):
    """Generate a preview PDB from the frontend 3D viewer state.

    The frontend sends its computed box dimensions, membrane parameters,
    and the oriented protein PDB.  The backend writes a ``preview.pdb``
    and stores the configuration for later comparison during the build.
    """
    data = await _json_object(request)
    task_id = data.get("task_id", "")

    # Validate task_id
    if task_id:
        try:
            task_id = _validate_task_id(task_id)
        except ValueError:
            return JSONResponse({"error": "Invalid task ID"}, status_code=400)
        state = task_manager.get_state(task_id)
        if state is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    else:
        return JSONResponse({"error": "task_id is required"}, status_code=400)

    oriented_pdb = data.get("oriented_pdb", "")
    box_dimensions_nm = data.get("box_dimensions_nm")  # [x, y, z]
    protein_metrics = data.get("protein")  # {center_of_mass_nm, extent_nm, ...}
    membrane_metrics = data.get("membrane")  # {midplane_z_nm, half_thickness_nm, ...}

    preview_config = {
        "box_dimensions_nm": box_dimensions_nm,
        "protein": protein_metrics,
        "membrane": membrane_metrics,
    }

    # ---- Write preview PDB (oriented protein with box/membrane CRYST1) ----
    preview_path = None
    try:
        task_dir = task_manager.get_task_dir(task_id)
        preview_path = task_dir / "preview.pdb"
        if oriented_pdb:
            _structure_files.write_preview_pdb(
                oriented_pdb,
                preview_path,
                box_dimensions_nm,
            )
    except Exception:
        logger.exception("Failed to write preview PDB")
        # Non-fatal — preview_config is still stored

    # ---- Store preview_config in task state ----
    task_manager.update_state(
        task_id,
        {
            "preview_config": preview_config,
            "preview_pdb_resource": "preview.pdb"
            if preview_path and preview_path.exists()
            else None,
        },
    )

    return {
        "status": "ok",
        "preview_config": preview_config,
        "preview_saved": bool(preview_path and preview_path.exists()),
        "preview_resource": "preview.pdb" if preview_path and preview_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# API: Filter PDB by chain/molecule selection
# ---------------------------------------------------------------------------


@app.post("/api/filter-pdb/{task_id}")
async def api_filter_pdb(task_id: str, request: Request):
    task_id = _validate_task_id(task_id)
    data = await _json_object(request)
    with _build_admission_lock:
        with _queue_lock:
            queued = any(tid == task_id for tid, _data in _build_queue)
        if (
            task_id in _building_tasks
            or queued
            or _step_admission.try_acquire(task_id) != "accepted"
        ):
            return JSONResponse(
                {"error": "Task has work in progress; retry after it finishes"}, status_code=409
            )
    try:
        return _filter_task_structure(task_id, data)
    finally:
        _step_admission.release(task_id)


def _filter_task_structure(task_id: str, data: dict) -> dict | JSONResponse:
    """Apply an input selection while the caller holds task admission."""
    include_chains = data.get("include_chains")
    excluded = data.get("exclude_resnames", [])
    for label, values in (("include_chains", include_chains), ("exclude_resnames", excluded)):
        if values is None and label == "include_chains":
            continue
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            return JSONResponse({"error": f"{label} must be an array of strings"}, status_code=400)
    exclude_resnames = set(excluded)

    task_dir = task_manager.get_task_dir(task_id)
    src = task_manager.get_filter_source(task_id)
    if not src or not src.exists():
        return JSONResponse({"error": "No PDB file found"}, status_code=400)

    from gmxbuilder.io.input_document import canonical_path, read_input

    detected_small_molecules = PDBValidator.detect_small_molecules(
        read_input(canonical_path(src) if canonical_path(src).exists() else src)
    )
    allowed_labels = {str(item["resname"]).strip().upper() for item in detected_small_molecules}
    try:
        small_molecule_labels = _normalise_small_molecule_labels(
            data.get("small_molecule_labels", {}), allowed_labels
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    filtered_path = task_dir / "filtered.pdb"
    previous = filtered_path.read_bytes() if filtered_path.is_file() else None
    try:
        n_kept, n_removed = _structure_files.filter_pdb_file(
            src,
            filtered_path,
            include_chains,
            exclude_resnames,
        )
    except (ValueError, ParseError) as exc:
        return JSONResponse({"error": _redact_server_paths(exc)}, status_code=400)

    if filtered_path.read_bytes() != previous:
        state = task_manager.get_state(task_id) or {}
        pipeline = (state.get("task_type") or {}).get("id") or state.get("task_type_id")
        runner = _get_step_runner(task_id, pipeline or "membrane-bilayer")
        runner.invalidate_downstream("input", include_current=True)
        task_manager.update_state(task_id, {"steps_completed": [], "current_step": "input"})

    task_manager.update_state(
        task_id,
        {
            # These are UI labels only.  The original residue key remains stable
            # in coordinates and force-field parameterization.
            "small_molecule_labels": small_molecule_labels,
            "input_selection": {
                "include_chains": include_chains,
                "exclude_resnames": sorted(exclude_resnames),
                "source_name": src.name,
            },
        },
    )

    return {
        "status": "ok",
        "filtered_resource": "filtered.pdb",
        "n_kept": n_kept,
        "n_removed": n_removed,
        "small_molecule_labels": small_molecule_labels,
    }


# ---------------------------------------------------------------------------
# Task management API
# ---------------------------------------------------------------------------


@app.get("/api/task/{task_id}")
async def api_task_status(task_id: str):
    """Get the full state of a task."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    return _public_task_state(state)


@app.post("/api/task/{task_id}/save-step")
async def api_task_save_step(task_id: str, request: Request):
    """Save bounded, non-authoritative browser state for one visible step."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be an object"}, status_code=400)
    step_name = data.get("step", "")
    step_data = data.get("data", {})
    if not isinstance(step_name, str) or not step_name:
        return JSONResponse({"error": "step must be a non-empty string"}, status_code=400)
    task_type = state.get("task_type") or {}
    task_type_id = task_type.get("id") or state.get("task_type_id") or "membrane-bilayer"
    visible_steps = set(task_type.get("visible_modules") or [])
    if not visible_steps:
        try:
            visible_steps = set(get_pipeline_steps(task_type_id)) - {"topology", "export"}
        except ValueError:
            visible_steps = set()
    if step_name not in visible_steps:
        return JSONResponse(
            {"error": f"step {step_name!r} is not a visible task step"},
            status_code=400,
        )
    if not isinstance(step_data, dict):
        return JSONResponse({"error": "data must be an object"}, status_code=400)
    encoded_size = len(
        json.dumps(step_data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    if encoded_size > 256 * 1024:
        return JSONResponse(
            {"error": "UI step state must be 256 KiB or smaller"},
            status_code=413,
        )
    saved = task_manager.save_step_state(task_id, step_name, step_data)
    if saved is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return {
        "status": "ok",
        "saved_step": step_name,
        "scientific_checkpoint_created": False,
    }


@app.get("/api/task/{task_id}/resume")
async def api_task_resume(task_id: str):
    """Return resumable state with a cached or bounded structure summary."""
    from gmxbuilder.modules.input.validation import INPUT_VALIDATION_VERSION

    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)

    task_type = state.get("task_type") or {}
    task_type_id = task_type.get("id") or state.get("task_type_id") or "membrane-bilayer"
    protein_free_cg = (
        _is_martini_task_type(task_type_id)
        and (state.get("step_input_config") or {}).get("include_protein") is False
    )

    # Resume is already an admitted managed operation. Upgrade legacy summaries
    # here so subsequent status polls never deserialize scientific systems.
    from gmxbuilder.core.checkpoint_status import refresh_status

    for step in get_pipeline_steps(task_type_id):
        await _run_interactive(refresh_status, task_manager.get_task_dir(task_id) / "steps" / step)

    # Reuse the immutable upload summary where possible.  Changed input-step
    # checkpoints and legacy tasks are parsed only in the bounded interactive
    # executor, never on the event-loop thread.
    pdb_path = task_manager.get_filter_source(task_id)
    input_viewer = task_manager.get_task_dir(task_id) / "steps" / "input" / "viewer.pdb"
    using_checkpoint = bool(
        not protein_free_cg
        and (input_viewer.parent / "system.npz").is_file()
        and (input_viewer.parent / "system.json").is_file()
        and not input_viewer.is_symlink()
    )
    if using_checkpoint:
        pdb_path = input_viewer
    if (
        not protein_free_cg
        and pdb_path
        and (using_checkpoint or pdb_path.is_file())
        and not pdb_path.is_symlink()
    ):
        try:
            limits = StructureInputLimits.from_environment()
            cached = state.get("structure_summary")
            cached_preview = task_manager.get_task_dir(task_id) / "converted.pdb"
            if (
                not using_checkpoint
                and isinstance(cached, dict)
                and cached.get("source_name") == pdb_path.name
                and cached.get("chain_identity_version") == 4
                and (
                    (
                        _is_martini_task_type(task_type_id)
                        and cached.get("input_status", {}).get("scope") == "coarse_grained"
                    )
                    or (
                        isinstance(cached.get("input_validation"), dict)
                        and cached["input_validation"].get("policy_version")
                        == INPUT_VALIDATION_VERSION
                    )
                )
                and cached_preview.is_file()
                and not cached_preview.is_symlink()
            ):
                summary = dict(cached)
                summary["pdb_content"] = await _run_interactive(
                    read_bounded_pdb_display,
                    cached_preview,
                    limits,
                    _filter_pdb_for_display,
                )
            else:
                original_source = None
                uploaded_name = state.get("uploaded_structure_name")
                if uploaded_name:
                    original_source = _task_resource_helpers.validate_task_resource(
                        task_id, uploaded_name
                    )
                summary = await _run_interactive(
                    summarize_resume_structure,
                    pdb_path,
                    limits,
                    _filter_pdb_for_display,
                    _extract_sequences,
                    PDBInputModule._PROTEIN_RESNAMES,
                    state.get("input_source_metadata"),
                    original_source,
                    not _is_martini_task_type(task_type_id),
                )
            selection_summary = summary
            selection_source = task_manager.get_filter_source(task_id)
            if using_checkpoint and selection_source:
                selection_summary = await _run_interactive(
                    summarize_resume_structure,
                    selection_source,
                    limits,
                    _filter_pdb_for_display,
                    _extract_sequences,
                    PDBInputModule._PROTEIN_RESNAMES,
                    None,
                    None,
                    not _is_martini_task_type(task_type_id),
                )
            state["input_selection_summary"] = {
                key: selection_summary.get(key)
                for key in (
                    "num_atoms",
                    "box_nm",
                    "sequences",
                    "chains",
                    "chain_mapping",
                    "pdb_content",
                    "small_molecules",
                )
            }
            state["chain_mapping"] = summary.get("chain_mapping", {})
            state["pdb_content"] = summary["pdb_content"]
            state["sequences"] = summary["sequences"]
            state["small_molecules"] = summary["small_molecules"]
            state["input_validation"] = (
                None if _is_martini_task_type(task_type_id) else summary.get("input_validation")
            )
            state["input_status"] = summary.get("input_status")
            state["cell_info"] = summary.get("cell_info")
            state["validation_warnings"] = summary.get("validation", {}).get("warnings", [])
            state["pdb_info_full"] = {
                "filename": (state.get("pdb_info") or {}).get("filename", pdb_path.name),
                "num_atoms": summary["num_atoms"],
                "chains": summary["chains"],
                "box_nm": summary["box_nm"],
                "small_molecules": summary["small_molecules"],
            }
        except WorkQueueFull:
            raise
        except Exception:
            logger.exception(
                "Failed to prepare resume structure for %s", task_log_reference(task_id)
            )

    from gmxbuilder.web.task_types import get_task_type_detail

    current_type = get_task_type_detail(task_type_id)
    visible_steps = list((current_type or task_type).get("visible_modules") or [])
    if (
        task_type_id == "pure-membrane"
        and (state.get("step_solvation_config") or {}).get("enabled") is False
    ):
        visible_steps = [step for step in visible_steps if step != "ions"]
    if current_type:
        state["task_type"] = {**current_type, "visible_modules": visible_steps}
    try:
        runner = _get_step_runner(task_id, task_type_id)
        checkpoint_steps = {
            name for name in get_pipeline_steps(task_type_id) if runner.has_checkpoint(name)
        }
    except ValueError:
        checkpoint_steps = set(state.get("steps_completed") or [])
        state["input_check_required"] = True
    else:
        state["input_check_required"] = not runner.input_validation_current()
    state["steps_completed"] = [name for name in visible_steps if name in checkpoint_steps]
    existing_zip = _authoritative_task_zip(task_id)
    build_status = state.get("build_status")
    if not isinstance(build_status, dict):
        build_status = {}
    if existing_zip is not None and build_status.get("status") not in {
        "queued",
        "running",
        "failed",
    }:
        build_status["status"] = "completed"
    if build_status.get("status") == "completed":
        build_status["download_available"] = existing_zip is not None
    if build_status.get("status") == "completed" and existing_zip is not None:
        result = build_status.get("result")
        if not isinstance(result, dict):
            result = {
                "task_id": task_id,
                "num_atoms": None,
                "components": [],
                "log": ["Existing completed package restored for download."],
            }
            build_status["result"] = result
        result["download_url"] = f"/api/task/{task_id}/download"
        build_status["download_available"] = True
    state["build_status"] = build_status
    resume_step = visible_steps[-1] if visible_steps else state.get("current_step", "input")
    if (
        build_status.get("status") == "completed"
        and build_status.get("download_available")
        and "simparams" in visible_steps
        and not state["input_check_required"]
    ):
        resume_step = "simparams"
    else:
        for candidate in visible_steps:
            if candidate == "simparams" or candidate not in checkpoint_steps:
                resume_step = candidate
                break
    resume_index = visible_steps.index(resume_step) if resume_step in visible_steps else 0
    route_slug = task_type.get("route_slug") or {
        "membrane-bilayer": "BilayerBuilder",
        "pure-membrane": "PureBilayerSystem",
        "solvator": "Solvator",
        "martini3-bilayer": "Martini3BilayerBuilder",
        "martini3-solvent": "Martini3SolventBuilder",
    }.get(task_type_id, "BilayerBuilder")
    state["resume_step"] = resume_step
    state["resume_step_number"] = resume_index + 1
    state["resume_url"] = f"/{route_slug}/Step{resume_index + 1}"
    return _public_task_state(state)


@app.get("/api/task/{task_id}/download")
async def api_task_download(task_id: str):
    """Download the build output ZIP for a task (works post-restart too)."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    output_dir = task_manager.get_output_dir(task_id)

    # The checked build is authoritative. Never let a larger legacy archive
    # shadow the current steps/export package.
    zip_path = _authoritative_task_zip(task_id)
    if zip_path is not None:
        filename = f"gmxbuilder_{task_id}.zip"
        return FileResponse(str(zip_path), media_type="application/zip", filename=filename)

    # Fallback: create ZIP from output files recursively
    files = list(output_dir.rglob("*"))
    if not files:
        return JSONResponse({"error": "No output files available"}, status_code=404)
    fallback = output_dir / "gmxbuilder_output.zip"
    with zipfile.ZipFile(fallback, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.is_file() and f.suffix != ".zip":
                z.write(f, f.relative_to(output_dir))
    return FileResponse(
        str(fallback), media_type="application/zip", filename=f"gmxbuilder_{task_id}.zip"
    )


@app.get("/api/build/{task_id}/log")
async def api_build_log(task_id: str, since: int = 0):
    """Return build log lines since the given index (for polling)."""
    task_id = _validate_task_id(task_id)
    with _build_logs_lock:
        lines = list(_build_logs.get(task_id, []))
    new_lines = [_redact_server_paths(line) for line in lines[since:]]
    with _tasks_lock:
        task_status = _tasks.get(task_id, {}).get("status")
    return {"lines": new_lines, "total": len(lines), "done": task_status in ("completed", "failed")}


@app.get("/api/tasks")
async def api_task_list(request: Request):
    """List active tasks (protected — requires X-Admin-Token header)."""
    if not _is_admin_request(request):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    tasks = []
    for d in sorted(task_manager.root.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            state = task_manager.get_state(d.name)
            if state:
                tasks.append(
                    {
                        "task_id": state["task_id"],
                        "filename": state.get("filename", ""),
                        "current_step": state.get("current_step", ""),
                        "created_at": state.get("created_at", ""),
                        "expires_at": state.get("expires_at", ""),
                    }
                )
    return tasks


def _offline_lipid_request_response() -> JSONResponse:
    """Retire every Web entry point that accepted a new lipid calculation."""
    return JSONResponse(
        {
            "error": (
                "New lipid submission and pre-equilibration are not available on the website. "
                "Contact the administrator by email using the address on the homepage or "
                "announcement board. Include the lipid name, SMILES and requested force field."
            ),
            "code": "lipid_submission_offline_only",
            "contact_url": "/",
        },
        status_code=403,
    )


@app.post("/api/build-lipid-library")
async def api_build_lipid_library(request: Request):
    """Retired: administrators maintain lipid libraries through the offline CLI."""
    return _offline_lipid_request_response()


@app.get("/api/lipid-library-list")
async def api_lipid_library_list():
    from gmxbuilder.web.server_parts.option_catalog import get_catalog

    return JSONResponse(get_catalog().read(), headers={"Cache-Control": "no-store"})


@app.get("/api/lipid-library-status")
async def api_lipid_library_status(
    lipid_name: str = "",
    force_field: str = "amber14sb",
    lipid_ff: str = "",
):
    """Check if a lipid's conformation library is built."""
    if not lipid_name:
        return JSONResponse({"error": "lipid_name query parameter required"}, status_code=400)
    from gmxbuilder.modules.forcefield.catalog import get_force_field_profile
    from gmxbuilder.modules.membrane.equilibrated_library import lipid_parameter_family
    from gmxbuilder.web.server_parts.option_catalog import get_catalog

    try:
        force_field = get_force_field_profile(force_field.strip().lower().removesuffix(".ff")).name
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    lipid_name = lipid_name.strip().upper()
    snapshot = get_catalog().read(options=True)
    option = next((row for row in snapshot.get("lipids", []) if row["name"] == lipid_name), {})
    effective_lipid_ff = lipid_ff or (
        ("lipid21" if "lipid21" in option.get("parameter_sources", []) else "gaff2")
        if force_field.startswith("amber")
        else force_field
    )
    if effective_lipid_ff not in {
        "lipid21",
        "gaff2",
        "amber-mixed",
        "charmm36",
        "charmm36m",
        "oplsaa",
    }:
        return JSONResponse({"error": "Unknown lipid parameter source"}, status_code=400)
    source = (
        ("lipid21" if "lipid21" in option.get("parameter_sources", []) else "gaff2")
        if effective_lipid_ff == "amber-mixed"
        else effective_lipid_ff
    )
    family = lipid_parameter_family(force_field, source)
    availability = snapshot.get("availability", {})
    row = next(
        (
            item
            for item in availability.get("entries", [])
            if item["lipid_name"] == lipid_name
            and item["lipid_ff"] == source
            and item["parameter_family"] == family
        ),
        None,
    )
    ready = bool(availability.get("status") == "ready" and row and row["ready"])
    if effective_lipid_ff == "amber-mixed":
        ready = ready and bool(row.get("amber_mixed_ready"))
    return {
        "lipid_name": lipid_name,
        "force_field": force_field,
        "lipid_ff": effective_lipid_ff,
        "status": availability.get("status", "checking"),
        "has_library": ready,
        "n_conformations": row["n_conformations"] if ready else 0,
        "validation_scope": row.get("validation_scope") if ready else None,
        "metadata_scope": "availability_summary",
        "metadata": row if ready else None,
    }


@app.get("/api/orient-algorithms")
async def api_orient_algorithms():
    """Return the list of available orientation algorithms."""
    from gmxbuilder.modules.membrane.orient import list_orientation_algorithms

    return list_orientation_algorithms()


# ---------------------------------------------------------------------------
# API: custom lipid from SMILES


@app.post("/api/custom-lipid")
async def api_custom_lipid(request: Request):
    """Retired: email SMILES to the administrator instead of submitting online."""
    return _offline_lipid_request_response()


@app.post("/api/task/{task_id}/custom-lipids")
async def api_submit_task_custom_lipid(task_id: str, request: Request):
    """Reject legacy clients before parsing, persisting or scheduling a molecule."""
    _validate_task_id(task_id)
    return _offline_lipid_request_response()


@app.get("/api/task/{task_id}/custom-lipids")
async def api_list_task_custom_lipids(task_id: str):
    """List only this task's custom molecules and calculation states."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    store = CustomLipidStore(task_manager.get_task_dir(task_id))
    records = store.list_public()
    for record in records:
        record["message"] = _redact_server_paths(record.get("message", ""))
    return {"task_id": task_id, "lipids": records}


@app.get("/api/task/{task_id}/custom-lipids/{lipid_name}")
async def api_task_custom_lipid_status(task_id: str, lipid_name: str):
    """Return task-private status without exposing any server paths."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    try:
        record = CustomLipidStore(task_manager.get_task_dir(task_id)).public_record(lipid_name)
        record["message"] = _redact_server_paths(record.get("message", ""))
        return record
    except (KeyError, ValueError):
        return JSONResponse({"error": "Custom lipid not found for this task"}, status_code=404)


@app.post("/api/task/{task_id}/custom-lipids/{lipid_name}/retry")
async def api_retry_task_custom_lipid(task_id: str, lipid_name: str):
    """Retired retries must not restart an existing expensive calculation."""
    _validate_task_id(task_id)
    return _offline_lipid_request_response()


# ---------------------------------------------------------------------------
# API: protonation & modifications


def _pka_digest(tmp_path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()


def _durable_pka_cache(tmp_path: str, predictions=None, expected_digest=None):
    """Keep precomputed pKa values useful across isolated operation processes."""
    if not managed_root():
        return None
    import hashlib
    from importlib.metadata import version

    if expected_digest is not None and _pka_digest(tmp_path) != expected_digest:
        raise ValueError("PROPKA input changed during analysis; recompute protonation")
    source = Path(tmp_path).resolve()
    if task_manager.root.resolve() not in source.parents or not source.is_file():
        return None
    relative = source.relative_to(task_manager.root.resolve())
    directory = task_manager.root / relative.parts[0]
    digest = (
        VERSION + ":" + version("propka") + ":" + hashlib.sha256(source.read_bytes()).hexdigest()
    )
    cache = directory / ".propka-cache.json"
    if predictions is not None:
        temp = cache.with_suffix(".tmp")
        temp.write_text(json.dumps({"sha256": digest, "predictions": predictions}))
        temp.replace(cache)
        return predictions
    try:
        saved = json.loads(cache.read_text())
        if saved["sha256"] == digest:
            return saved["predictions"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def _publish_propka_status(tmp_path, status, residues=0, expected=None):
    from gmxbuilder.core.checkpoint_status import atomic_json, file_stamp

    source = Path(tmp_path)
    try:
        stamp = file_stamp(source)
        if expected is not None and expected != stamp:
            return
        atomic_json(
            source.parent / ".propka-status.json",
            {
                "adapter": stamp,
                "status": status,
                "residues": residues,
                "pid": os.getpid(),
            },
        )
    except OSError:
        logger.debug("Unable to persist optional PROPKA status", exc_info=True)


def _schedule_propka_precompute(tmp_path: str) -> None:
    """Kick off PROPKA in a background thread so results are ready later."""
    from gmxbuilder.core.checkpoint_status import file_stamp

    stamp = file_stamp(tmp_path)
    digest = _pka_digest(tmp_path)
    with _pka_cache_lock:
        if tmp_path in _pka_cache and _pka_cache_digest.get(tmp_path) == digest:
            _publish_propka_status(tmp_path, "ready", len(_pka_cache[tmp_path]), stamp)
            return
        if tmp_path in _pka_running:
            return
        _pka_running.add(tmp_path)
    _publish_propka_status(tmp_path, "computing", expected=stamp)

    def _run():
        try:
            from gmxbuilder.modules.modifications.protonation import predict_pka_from_pdb

            preds = predict_pka_from_pdb(tmp_path)
            if _pka_digest(tmp_path) != digest:
                return
            _durable_pka_cache(tmp_path, preds, digest)
            with _pka_cache_lock:
                _pka_cache[tmp_path] = preds
                _pka_cache_digest[tmp_path] = digest
            _publish_propka_status(tmp_path, "ready", len(preds), stamp)
        except Exception as exc:
            _publish_propka_status(tmp_path, "not_started", expected=stamp)
            logger.warning("PROPKA background calculation failed: %s", exc)
        finally:
            with _pka_cache_lock:
                _pka_running.discard(tmp_path)

    try:
        _submit_interactive(_run)
    except WorkQueueFull:
        with _pka_cache_lock:
            _pka_running.discard(tmp_path)
        _publish_propka_status(tmp_path, "not_started", expected=stamp)
        logger.info("Deferred PROPKA precompute because the interactive queue is full")


async def _get_propka_results(tmp_path: str) -> list[dict]:
    """Return cached PROPKA results, or compute asynchronously if not cached.

    Uses the bounded interactive executor so repeated polling cannot bypass
    the configured process resource budget.
    """
    from gmxbuilder.core.checkpoint_status import file_stamp

    stamp = file_stamp(tmp_path)
    digest = _pka_digest(tmp_path)
    durable = _durable_pka_cache(tmp_path)
    if durable is not None:
        _publish_propka_status(tmp_path, "ready", len(durable), stamp)
        return durable
    with _pka_cache_lock:
        if tmp_path in _pka_cache and _pka_cache_digest.get(tmp_path) == digest:
            _publish_propka_status(tmp_path, "ready", len(_pka_cache[tmp_path]), stamp)
            return _pka_cache[tmp_path]

    # Not cached — run in thread pool to keep event loop free
    try:
        from gmxbuilder.modules.modifications.protonation import predict_pka_from_pdb

        preds = await _run_interactive(predict_pka_from_pdb, tmp_path)
        if _pka_digest(tmp_path) != digest:
            raise ValueError("PROPKA input changed during analysis; recompute protonation")
        _durable_pka_cache(tmp_path, preds, digest)
        with _pka_cache_lock:
            _pka_cache[tmp_path] = preds
            _pka_cache_digest[tmp_path] = digest
            _pka_running.discard(tmp_path)
        _publish_propka_status(tmp_path, "ready", len(preds), stamp)
        return preds
    except Exception as exc:
        logger.warning("PROPKA calculation failed; using model pKa values: %s", exc)
        with _pka_cache_lock:
            _pka_running.discard(tmp_path)
        return []


@app.get("/api/propka-status")
async def api_propka_status(tmp_path: str = "", task_id: str = ""):
    """Check whether PROPKA has finished precomputing for a given PDB."""
    if tmp_path:
        return JSONResponse(
            {"error": ("Client-supplied filesystem paths are not accepted; provide task_id")},
            status_code=400,
        )
    if not task_id:
        return JSONResponse({"error": "task_id is required"}, status_code=400)
    try:
        task_id = _validate_task_id(task_id)
        if task_manager.get_state(task_id) is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
        manifest = _task_resource_helpers.propka_manifest(task_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if manifest is not None:
        from gmxbuilder.core.checkpoint_status import read_json

        try:
            saved = read_json(task_manager.get_task_dir(task_id) / ".propka-status.json")
            if saved.get("adapter") == manifest["adapter"]:
                if saved.get("status") == "ready":
                    return {"status": "ready", "residues": saved["residues"]}
                if saved.get("status") == "computing":
                    os.kill(int(saved["pid"]), 0)
                    return {"status": "computing"}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    return {"status": "not_started"}


@app.post("/api/protonate")
async def api_protonate(request: Request):
    """Compute protonation states for a list of residues at a given pH.

    If task_id is provided, runs PROPKA for environment-sensitive pKa
    prediction. Otherwise falls back to standard model-pKa values.
    """
    data = await _json_object(request)
    residues = data.get("residues", [])
    pH = data.get("pH", 7.0)
    his_tautomer = data.get("his_tautomer", "HSE")
    # Residue names for the non-default states differ between Amber and
    # CHARMM, so the preview must name them for the force field this task has
    # actually chosen. Without it the panel offered ASH to a CHARMM build.
    force_field = str(data.get("force_field", "") or "").strip()
    if data.get("tmp_path"):
        return JSONResponse(
            {"error": ("Client-supplied filesystem paths are not accepted; provide task_id")},
            status_code=400,
        )
    tmp_path = ""
    task_id_val = data.get("task_id", "")
    if task_id_val:
        try:
            task_id_val = _validate_task_id(str(task_id_val))
            task_state = task_manager.get_state(task_id_val)
            if task_state is None:
                return JSONResponse({"error": "Task not found or expired"}, status_code=404)
            # The task's own choice wins over anything the client says, and
            # covers a client that does not send one at all.
            saved_force_field = str(
                (task_state.get("step_forcefield_config") or {}).get("name", "") or ""
            ).strip()
            if saved_force_field:
                force_field = saved_force_field
            tmp_path = str(
                _validate_task_resource(task_id_val, _resolve_propka_pdb_path(task_id_val))
            )
        except (ValueError, ParseError) as exc:
            return JSONResponse({"error": _redact_server_paths(exc)}, status_code=400)
    structure_residues = data.get("structure_residues", [])  # [{resname, chain, resid, index}]

    if not residues:
        return JSONResponse({"error": "No residues provided"}, status_code=400)

    try:
        pH = float(pH)
    except (TypeError, ValueError):
        return JSONResponse({"error": "pH must be a number between 1.0 and 13.0"}, status_code=400)
    if not np.isfinite(pH) or not 1.0 <= pH <= 13.0:
        return JSONResponse({"error": "pH must be between 1.0 and 13.0"}, status_code=400)
    if his_tautomer not in {"HSD", "HSE"}:
        return JSONResponse({"error": "his_tautomer must be HSD or HSE"}, status_code=400)

    try:
        from gmxbuilder.modules.modifications.protonation import (
            assign_all_protonations,
            assign_protonation_with_propka,
        )

        used_propka = False
        pka_predictions = []
        propka_requested = bool(tmp_path and Path(tmp_path).exists())
        propka_warning = ""

        # Try PROPKA if PDB file is available
        if tmp_path and Path(tmp_path).exists():
            try:
                pka_predictions = await _get_propka_results(tmp_path)
                used_propka = len(pka_predictions) > 0
            except Exception:
                logger.warning("PROPKA calculation failed for task input", exc_info=True)
            if not used_propka:
                propka_warning = (
                    "PROPKA could not produce environment-sensitive pKa values for this "
                    "structure; model pKa values were used. Check structure completeness "
                    "before production simulation."
                )

        if used_propka and structure_residues:
            from gmxbuilder.web.server_parts.protonation_identity import map_predictions

            pka_predictions = map_predictions(pka_predictions, structure_residues, tmp_path)
            assignments = assign_protonation_with_propka(
                structure_residues,
                pka_predictions,
                pH=float(pH),
                his_tautomer=his_tautomer,
                force_field=force_field,
            )
        else:
            assignments = assign_all_protonations(
                residues,
                pH=float(pH),
                his_tautomer=his_tautomer,
                force_field=force_field,
            )

        modified = [a for a in assignments if a["is_titratable"]]
        matched = sum("predicted_pKa" in assignment for assignment in modified)
        used_propka = matched > 0
        if propka_requested and matched < len(modified):
            propka_warning = (propka_warning + " " if propka_warning else "") + (
                f"PROPKA matched {matched}/{len(modified)} titratable residues. "
                "Unmatched residues use model pKa values; inspect their identities and states."
            )
        for assignment in assignments:
            assignment["prediction_source"] = (
                "PROPKA" if "predicted_pKa" in assignment else "Model pKa"
            )
        method = (
            "PROPKA 3.5 (environment-sensitive)"
            if used_propka
            else ("Model pKa fallback" if propka_requested else "Model pKa (no 3D context)")
        )
        return {
            "pH": pH,
            "assignments": assignments,
            "titratable_count": len(modified),
            "prediction_matched": matched,
            "prediction_expected": len(modified),
            "used_propka": used_propka,
            "method": method,
            "propka_requested": propka_requested,
            "propka_warning": propka_warning,
            "titratable_residues": [
                {
                    "index": a["index"],
                    "original": a["original"],
                    "assigned": a["assigned_name"],
                    "charge": a["charge"],
                    "state": a["state_label"],
                    "pKa": a.get("predicted_pKa", a["pKa"]),
                    "pKa_shift": a.get("pKa_shift"),
                    "alternatives": a.get("alternatives", []),
                }
                for a in modified
            ],
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Unhandled error in protonate")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/patches")
async def api_patches(force_field: str | None = None):
    """Return the list of available PTM patches."""
    from gmxbuilder.modules.modifications.patches import list_patches

    return list_patches(force_field)


@app.get("/api/patches/{resname}")
async def api_patches_for_residue(resname: str, force_field: str | None = None):
    """Return patches applicable to a specific residue."""
    from gmxbuilder.modules.modifications.patches import list_patches_for_residue

    return list_patches_for_residue(resname, force_field)


@app.get("/api/terminal-capabilities")
async def api_terminal_capabilities(force_field: str = "charmm36"):
    """Return explicit cap support for the selected bundled force field."""
    from gmxbuilder.modules.modifications.processor import terminal_capabilities

    try:
        return terminal_capabilities(force_field)
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/crosslink-capabilities")
async def api_crosslink_capabilities(force_field: str = "charmm36"):
    """Return force-field-specific support for dedicated cross-residue chemistry."""
    from gmxbuilder.modules.modifications.patches import disulfide_capability

    try:
        supported, reason, target = disulfide_capability(force_field)
        return {
            "disulfide": {
                "supported": supported,
                "reason": reason,
                "target_distance_nm": target,
            }
        }
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/coarse-grained/capabilities")
async def api_coarse_grained_capabilities():
    """Return the immutable Martini 3 bundle and explicit support boundary."""
    from gmxbuilder.modules.coarse_grained.assets import public_capabilities

    return public_capabilities()


@app.post("/api/apply-modifications")
async def api_apply_modifications(request: Request):
    """Apply a set of modifications and protonation to the uploaded structure."""
    data = await _json_object(request)
    if data.get("tmp_path"):
        return JSONResponse(
            {"error": ("Client-supplied filesystem paths are not accepted; provide task_id")},
            status_code=400,
        )
    task_id_val = data.get("task_id", "")
    if not task_id_val:
        return JSONResponse({"error": "task_id is required"}, status_code=400)
    try:
        task_id_val = _validate_task_id(str(task_id_val))
        if task_manager.get_state(task_id_val) is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
        tmp_path = str(_validate_task_resource(task_id_val, _resolve_pdb_path(task_id_val)))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    pH = data.get("pH", 7.0)
    his_tautomer = data.get("his_tautomer", "HSE")
    modifications = data.get("modifications", [])  # [{index, patch_id}]
    force_field = str(data.get("force_field", "charmm36m"))
    nter_patch = data.get("nter_patch")  # e.g. "ACE" or null
    cter_patch = data.get("cter_patch")  # e.g. "NME" or null

    if not Path(tmp_path).exists():
        return JSONResponse({"error": "PDB file not found"}, status_code=400)

    try:
        return _build_modification_preview(
            tmp_path,
            pH,
            his_tautomer,
            modifications,
            force_field,
            nter_patch,
            cter_patch,
        )
    except (ValueError, ModuleConfigError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Unhandled error in apply-modifications")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# API: options


@app.get("/api/options")
async def api_options():
    """Return all available choices for the UI dropdowns."""
    from gmxbuilder.web.server_parts.option_catalog import get_catalog

    return get_catalog().read(options=True)


@app.post("/api/forcefield-compatibility/{task_id}")
async def api_forcefield_compatibility(task_id: str, request: Request):
    """Return only force-field combinations executable by this installation."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    pipeline_type = (state.get("task_type") or {}).get("id") or state.get("task_type_id")
    checkpoint = task_manager.get_task_dir(task_id) / "steps" / "input"
    if pipeline_type != "pure-membrane" and not (checkpoint / "system.npz").is_file():
        return JSONResponse(
            {"error": "Run Step 1 Check Upload before selecting force fields"},
            status_code=409,
        )
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
    protein_ff = str(data.get("protein_ff", "amber14sb")).strip().lower()
    lipid_names = data.get("lipid_names", [])
    if not isinstance(lipid_names, list):
        return JSONResponse({"error": "lipid_names must be a list"}, status_code=400)
    try:
        if pipeline_type == "pure-membrane":
            system = System(
                structure=Structure(
                    coordinates=np.empty((0, 3)),
                    box_vectors=np.eye(3) * 10.0,
                ),
                metadata={"seed": state.get("seed", 42)},
            )
        else:
            system = System.load_checkpoint(checkpoint)
        from gmxbuilder.modules.forcefield.compatibility import compatibility_report

        report = compatibility_report(system, protein_ff, lipid_names)
        nucleic_report = report.get("nucleic_acid", {})
        if nucleic_report.get("present") and pipeline_type != "solvator":
            nucleic_report["enabled"] = False
            nucleic_report["reason"] = (
                "DNA/RNA polymers are currently supported only by the Solution Solvator workflow"
            )
        saved_labels = state.get("small_molecule_labels", {})
        labels = {
            name: str(saved_labels.get(name, name)).strip() or name
            for name in report.get("ligand_names", [])
        }
        report["ligand_labels"] = labels
        for ligand in report.get("ligands", []):
            name = str(ligand.get("name", ""))
            ligand["display_name"] = labels.get(name, name)
        return report
    except FileNotFoundError:
        return JSONResponse(
            {
                "error": (
                    "Required force-field or checkpoint data is unavailable. "
                    "Complete installation with ./install-local.sh and rerun the input Check."
                ),
                "code": "missing_compatibility_data",
            },
            status_code=503,
        )
    except (ValueError, KeyError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/ligand-charge-suggestions/{task_id}")
async def api_ligand_charge_suggestions(task_id: str, request: Request):
    """Compute pH-dependent GAFF2 integer-charge suggestions for ligands."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
    try:
        target_pH = float(data.get("pH", 7.0))
    except (TypeError, ValueError):
        return JSONResponse({"error": "pH must be numeric"}, status_code=400)
    if not 1.0 <= target_pH <= 13.0:
        return JSONResponse({"error": "pH must be between 1.0 and 13.0"}, status_code=400)
    checkpoint = task_manager.get_task_dir(task_id) / "steps" / "input"
    if not (checkpoint / "system.npz").is_file():
        return JSONResponse({"error": "Run Step 1 Check Upload first"}, status_code=409)

    try:
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups
        from gmxbuilder.modules.forcefield.gaff_backend import estimate_gaff_net_charge

        system = System.load_checkpoint(checkpoint)
        groups = molecule_groups(system)

        def compute() -> dict:
            suggestions = {}
            for name, instances in groups.items():
                estimates = [
                    estimate_gaff_net_charge(name, system.structure, indices, target_pH)
                    for indices in instances
                ]
                charges = {estimate.net_charge for estimate in estimates}
                if len(charges) != 1:
                    suggestions[name] = {
                        "status": "ambiguous",
                        "error": "Molecule instances produced different charge estimates",
                    }
                    continue
                estimate = estimates[0]
                suggestions[name] = {
                    "status": "ok",
                    "net_charge": estimate.net_charge,
                    "pH": estimate.pH,
                    "formula": estimate.formula,
                    "atom_count": estimate.atom_count,
                    "method": estimate.method,
                    "warning": (
                        "Coordinate-derived bond orders and protonation are a suggestion; "
                        "verify unusual chemistry, metals, and covalent cofactors manually."
                    ),
                }
            return suggestions

        suggestions = await _run_interactive(compute)
        return {"status": "ok", "pH": target_pH, "suggestions": suggestions}
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/ligand-chemistry/{task_id}")
async def api_ligand_chemistry(task_id: str, request: Request):
    """Identify ligands independently of parameter generation or simulation."""
    from gmxbuilder.web.server_parts.ligand_chemistry import preview, trusted_config

    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    checkpoint = task_manager.get_task_dir(task_id) / "steps" / "input"
    if not (checkpoint / "system.npz").is_file():
        return JSONResponse({"error": "Run Step 1 Check Upload first"}, status_code=409)
    try:
        data = await _json_object(request)
        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object")
        config = trusted_config(task_id, data, task_manager, _validate_task_resource)
        records = await _run_interactive(preview, checkpoint, config)
        return {"ligands": records, "pH": config.get("ligand_pH", 7.0)}
    except (ValueError, OSError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/ligand-chemistry-upload/{task_id}")
async def api_ligand_chemistry_upload(
    task_id: str,
    ligand_name: str = Form(...),
    mol2_file: UploadFile = File(...),
):
    """Validate a MOL2 identity against every retained instance before saving it."""
    import hashlib

    from gmxbuilder.modules.forcefield.ligand_identity import MAX_IDENTITY_BYTES

    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    name = ligand_name.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_]{1,8}", name):
        return JSONResponse({"error": "Invalid ligand name"}, status_code=400)
    if not (mol2_file.filename or "").lower().endswith(".mol2"):
        return JSONResponse({"error": "Use a .mol2 file"}, status_code=400)
    checkpoint = task_manager.get_task_dir(task_id) / "steps" / "input"
    if not (checkpoint / "system.npz").is_file():
        return JSONResponse({"error": "Run Step 1 Check Upload first"}, status_code=409)
    payload = await mol2_file.read(MAX_IDENTITY_BYTES + 1)
    if len(payload) > MAX_IDENTITY_BYTES:
        return JSONResponse({"error": "MOL2 must be 2 MiB or smaller"}, status_code=413)
    root = task_manager.get_task_dir(task_id) / "ligand_chemistry"
    root.mkdir(parents=True, exist_ok=True)
    # Content-addressed files do not invalidate a concurrent build's input.
    digest = hashlib.sha256(payload).hexdigest()
    path = root / f"{name}-{digest}.mol2"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".mol2", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        from gmxbuilder.web.server_parts.ligand_chemistry import identify_groups

        def validate():
            return identify_groups(
                System.load_checkpoint(checkpoint),
                {"charmm_compat_mol2": {name: str(temporary)}},
                only=name,
            )

        records = await _run_interactive(validate)
        record = records.get(name)
        if not record or record["status"] != "ok":
            raise ValueError((record or {}).get("error", "No retained ligand with that name"))
        temporary.replace(path)
        state = task_manager.get_state(task_id) or {}
        uploads = dict(state.get("ligand_chemistry_uploads") or {})
        uploads[name] = {"file": path.name, "sha256": digest, "smiles": record["smiles"]}
        task_manager.update_state(task_id, {"ligand_chemistry_uploads": uploads})
        return {"ready": True, "smiles": record["smiles"], "sha256": digest}
    except (ValueError, OSError, KeyError, ModuleConfigError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        temporary.unlink(missing_ok=True)


def _inspect_cgenff_checkpoint(checkpoint: Path, ligand_name: str) -> bool:
    """Inspect task-owned ligand groups outside the event-loop thread."""
    from gmxbuilder.modules.forcefield.compatibility import molecule_groups

    instances = molecule_groups(System.load_checkpoint(checkpoint)).get(ligand_name, [])
    if any(len(instance) > 2048 for instance in instances):
        raise ValueError("CGenFF small molecules may contain at most 2,048 atoms")
    return bool(instances)


@app.post("/api/cgenff-upload/{task_id}")
async def api_cgenff_upload(
    task_id: str,
    ligand_name: str = Form(...),
    force_field: str = Form(...),
    mol2_file: UploadFile = File(...),
    str_file: UploadFile = File(...),
):
    """Validate and persist one matching ParamChem MOL2/STR package."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    name = ligand_name.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_]{1,8}", name):
        return JSONResponse({"error": "Invalid small-molecule residue name"}, status_code=400)
    selected_ff = force_field.strip().lower()
    if selected_ff not in {"charmm36", "charmm36m"}:
        return JSONResponse(
            {"error": "CGenFF packages can only be uploaded for CHARMM36/CHARMM36m"},
            status_code=400,
        )
    checkpoint = task_manager.get_task_dir(task_id) / "steps" / "input"
    if not (checkpoint / "system.npz").is_file():
        return JSONResponse({"error": "Run Step 1 Check Upload first"}, status_code=409)
    try:
        ligand_present = await _run_interactive(_inspect_cgenff_checkpoint, checkpoint, name)
    except WorkQueueFull:
        raise
    except (OSError, ValueError, KeyError) as exc:
        return JSONResponse(
            {"error": f"Could not inspect input checkpoint: {exc}"}, status_code=400
        )
    if not ligand_present:
        return JSONResponse(
            {"error": f"The retained input system has no small molecule named {name}"},
            status_code=400,
        )
    if not (mol2_file.filename or "").lower().endswith(".mol2"):
        return JSONResponse({"error": "ParamChem coordinate file must use .mol2"}, status_code=400)
    if not (str_file.filename or "").lower().endswith(".str"):
        return JSONResponse({"error": "ParamChem parameter file must use .str"}, status_code=400)
    maximum = 10 * 1024 * 1024
    mol2_content = await mol2_file.read(maximum + 1)
    str_content = await str_file.read(maximum + 1)
    if len(mol2_content) > maximum or len(str_content) > maximum:
        return JSONResponse({"error": "Each CGenFF file must be 10 MB or smaller"}, status_code=413)
    package_dir = task_manager.get_task_dir(task_id) / "cgenff" / name
    package_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = package_dir / f"{name}.mol2"
    stream_path = package_dir / f"{name}.str"
    mol2_path.write_bytes(mol2_content)
    stream_path.write_bytes(str_content)
    try:
        from gmxbuilder.modules.forcefield.cgenff_import import prepare_cgenff_molecule

        template = await _run_interactive(
            prepare_cgenff_molecule,
            name,
            mol2_path,
            stream_path,
            selected_ff,
            package_dir / "generated",
        )
    except WorkQueueFull:
        mol2_path.unlink(missing_ok=True)
        stream_path.unlink(missing_ok=True)
        raise
    except (ModuleConfigError, OSError, ValueError) as exc:
        mol2_path.unlink(missing_ok=True)
        stream_path.unlink(missing_ok=True)
        return JSONResponse({"error": str(exc)}, status_code=400)
    uploads = dict(state.get("cgenff_uploads", {}))
    uploads[name] = {
        "mol2_file": mol2_path.name,
        "str_file": stream_path.name,
        "force_field": selected_ff,
        "cgenff_version": template.cgenff_version,
        "maximum_penalty": template.maximum_penalty,
    }
    task_manager.update_state(task_id, {"cgenff_uploads": uploads})
    warning = None
    if template.maximum_penalty is not None:
        if template.maximum_penalty >= 50:
            warning = (
                f"Maximum CGenFF penalty is {template.maximum_penalty:.1f}; manual parameter "
                "validation or quantum-chemical refinement is strongly recommended."
            )
        elif template.maximum_penalty >= 10:
            warning = (
                f"Maximum CGenFF penalty is {template.maximum_penalty:.1f}; review the "
                "assigned charges and parameters before production MD."
            )
    return {
        "status": "ok",
        "ligand_name": name,
        "force_field": selected_ff,
        "ready": True,
        "cgenff_version": template.cgenff_version,
        "maximum_penalty": template.maximum_penalty,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# API: upload PDB


@app.post("/api/upload-pdb")
async def api_upload_pdb(
    request: Request,
    file: UploadFile = File(...),
    task_type: str = Form("membrane-bilayer"),
    task_id: str = Form(""),
):
    """Upload a PDB/mmCIF structure and return its parsed summary."""
    original_filename = Path(file.filename or "upload.pdb").name
    try:
        _structure_upload_suffix(original_filename)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    limits = StructureInputLimits.from_environment()
    max_upload_mb = limits.max_bytes // 1024 // 1024
    content_length = request.headers.get("content-length")
    try:
        declared_size = int(content_length) if content_length else None
    except ValueError:
        return JSONResponse({"error": "Invalid Content-Length header"}, status_code=400)
    # Multipart framing contributes to Content-Length. The streamed file read
    # below remains the authoritative per-file limit.
    if declared_size is not None and declared_size > limits.max_bytes + 1024 * 1024:
        return JSONResponse(
            {
                "error": (
                    f"File too large ({declared_size / 1024 / 1024:.0f} MB). "
                    f"Maximum is {max_upload_mb} MB."
                )
            },
            status_code=413,
        )

    # Read with size cap — read max_bytes+1 so we can detect overage
    content = await file.read(limits.max_bytes + 1)
    if len(content) > limits.max_bytes:
        return JSONResponse(
            {"error": (f"File too large (>{max_upload_mb} MB). Maximum is {max_upload_mb} MB.")},
            status_code=413,
        )

    try:
        stored_name, content, structure_format, format_warnings = await _run_interactive(
            _prepare_and_inspect_structure_upload,
            original_filename,
            content,
            limits,
        )
    except WorkQueueFull:
        raise
    except (StructureInputLimitError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    is_cif = structure_format == "cif"

    from gmxbuilder.web.task_types import get_task_type_detail

    task_type_detail = get_task_type_detail(task_type)
    if (
        task_type_detail is None
        or not task_type_detail.get("enabled")
        or (
            not task_type_detail.get("requires_input", True)
            and not _is_martini_task_type(task_type)
        )
    ):
        return JSONResponse(
            {"error": "Selected workflow does not accept a structure upload"},
            status_code=400,
        )

    # Coarse-grained tasks exist before an optional protein upload so the same
    # task can also represent a protein-free bilayer.  All other workflows keep
    # the existing create-on-upload contract.
    if task_id:
        try:
            task_id = _validate_task_id(task_id)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        existing_state = task_manager.get_state(task_id)
        existing_type = ((existing_state or {}).get("task_type") or {}).get("id") or (
            existing_state or {}
        ).get("task_type_id")
        if not _is_martini_task_type(task_type) or existing_type != task_type:
            return JSONResponse(
                {"error": "Existing task does not accept this structure upload"},
                status_code=409,
            )
    else:
        task = task_manager.create_task(original_filename)
        task_id = task["task_id"]
    uploaded_path = task_manager.save_uploaded_pdb(task_id, stored_name, content)
    try:
        summary = await _run_interactive(
            process_uploaded_structure,
            uploaded_path,
            task_manager.get_task_dir(task_id),
            structure_format,
            format_warnings,
            limits,
            _filter_pdb_for_display,
            _extract_sequences,
            PDBInputModule._PROTEIN_RESNAMES,
            not _is_martini_task_type(task_type),
        )
        validation = summary["validation"]
        if not validation["valid"]:
            return JSONResponse(
                {
                    "error": "PDB validation failed",
                    "validation_errors": validation["errors"],
                    "validation_warnings": validation["warnings"],
                },
                status_code=400,
            )

        cached_summary = {
            key: summary[key]
            for key in (
                "chain_identity_version",
                "input_status",
                "cell_info",
                "atom_counts_by_resname",
                "residue_counts_by_resname",
                "chain_mapping",
                "num_atoms",
                "residues",
                "protein_residues",
                "chains",
                "box_nm",
                "sequences",
                "small_molecules",
                "input_source_metadata",
                "input_validation",
            )
        }
        cached_summary["source_name"] = uploaded_path.name
        cached_summary["validation"] = validation

        task_manager.update_state(
            task_id,
            {
                "task_type": task_type_detail,
                "task_type_id": task_type,
                "uploaded_structure_name": uploaded_path.name,
                "uploaded_structure_format": structure_format,
                "pdb_info": {
                    "filename": original_filename,
                    "num_atoms": summary["num_atoms"],
                    "chains": summary["chains"],
                    "box_nm": summary["box_nm"],
                    "small_molecules": summary["small_molecules"],
                },
                "structure_summary": cached_summary,
                "input_source_metadata": summary["input_source_metadata"],
                "current_step": "input",
            },
        )

        if not _is_martini_task_type(task_type):
            try:
                analysis_path = _resolve_propka_pdb_path(task_id)
            except ValueError:
                logger.info("PROPKA precompute deferred: input needs explicit identity review")
            else:
                _schedule_propka_precompute(analysis_path)

        return {
            "task_id": task_id,
            "filename": original_filename,
            "structure_format": "mmCIF" if is_cif else "PDB",
            "input_validation": summary["input_validation"],
            "input_status": summary["input_status"],
            "cell_info": summary["cell_info"],
            "atom_counts_by_resname": summary["atom_counts_by_resname"],
            "residue_counts_by_resname": summary["residue_counts_by_resname"],
            "num_atoms": summary["num_atoms"],
            "residues": summary["residues"],
            "protein_residues": summary["protein_residues"],
            "chains": summary["chains"],
            "chain_mapping": summary.get("chain_mapping", {}),
            "box_nm": summary["box_nm"],
            "pdb_content": summary["pdb_content"],
            "sequences": summary["sequences"],
            "validation_warnings": validation["warnings"],
            "small_molecules": summary["small_molecules"],
        }
    except WorkQueueFull:
        raise
    except (ParseError, StructureInputLimitError, UnicodeError, ValueError) as exc:
        logger.warning("Structure upload rejected for %s: %s", task_log_reference(task_id), exc)
        message = _redact_server_paths(str(exc)).replace("\n", " ")[:400]
        return JSONResponse(
            {
                "error": f"Failed to parse structure file: {message}",
                "parse_issues": getattr(exc, "issues", [])[:100],
                "parse_issue_count": len(getattr(exc, "issues", [])),
            },
            status_code=400,
        )
    except Exception:
        logger.exception("Upload PDB failed")
        return JSONResponse({"error": "Failed to process structure file"}, status_code=400)
    # NOTE: task-owned upload artifacts are cleaned up by task TTL expiry.


# ---------------------------------------------------------------------------
# API: build

# Build queue
# ---------------------------------------------------------------------------
_build_queue: list[tuple[str, dict]] = []  # [(task_id, data), ...]
_queue_lock = threading.Lock()
_build_admission_lock = threading.Lock()
_queue_event: asyncio.Event | None = None  # created in the active app event loop
_MAX_QUEUED_BUILDS = _positive_environment_integer("GMXBUILDER_MAX_QUEUED_BUILDS", 32)
_queue_enqueued_at: dict[str, float] = {}
_build_started_at: dict[str, float] = {}
# Cores allocated to each running build, decided when it started.
_build_threads: dict[str, int] = {}
# (duration, threads). A build that had 24 cores says nothing about how long
# the same work takes on 2, so the estimate compares like with like.
_build_duration_history: deque[tuple[float, int]] = deque(maxlen=50)


def _baseline_build_seconds() -> float:
    try:
        value = float(os.environ.get("GMXBUILDER_EXPECTED_BUILD_SECONDS", "45"))
    except ValueError:
        value = 45.0
    return value if math.isfinite(value) and value > 0 else 45.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _typical_build_seconds(threads: int | None = None) -> float:
    """Return a robust recent finalization duration for queue estimates.

    Per-task cores vary with how many builds are running, and the same work
    takes longer on fewer of them. Where enough runs at the same allocation
    have been observed, those are used; otherwise all of them are, which is a
    weaker estimate but an observed one. Nothing here models a speed-up curve:
    a made-up scaling constant would look more precise than it is.
    """
    if not _build_duration_history:
        return _baseline_build_seconds()
    # Entries are (duration, cores). A bare duration is tolerated because
    # /api/health reads this, and a health endpoint should not fail over the
    # shape of a statistics sample.
    samples: list[tuple[float, int | None]] = []
    for entry in list(_build_duration_history):
        if isinstance(entry, tuple) and len(entry) == 2:
            samples.append((float(entry[0]), int(entry[1])))
        else:
            samples.append((float(entry), None))
    if not samples:
        return _baseline_build_seconds()
    if threads is not None:
        matched = [duration for duration, used in samples if used == int(threads)]
        if len(matched) >= 3:
            return _median(matched)
    return _median([duration for duration, _used in samples])


def _queue_estimate(position: int) -> dict[str, object]:
    """Estimate when a one-based queued position can acquire a task slot."""
    now = time.time()
    with _tasks_lock:
        active_ids = tuple(_building_tasks)
        active_threads = {task_id: _build_threads.get(task_id) for task_id in active_ids}

    # A build already running keeps the cores it started with, so its
    # remaining time is judged against runs that had the same allocation.
    remaining = [
        max(
            1.0,
            _typical_build_seconds(active_threads.get(task_id))
            - max(0.0, now - _build_started_at.get(task_id, now)),
        )
        for task_id in active_ids
    ]
    # A queued build starts once a slot frees, which under the allocation rule
    # means it will be sharing with whatever is running then. Estimating it at
    # a full machine's speed would systematically under-predict the wait.
    queued_threads = task_thread_allocation(max(1, len(active_ids)))
    typical = _typical_build_seconds(queued_threads)

    # Simulate FIFO assignment to the first available slot. Free slots start
    # at t=0; occupied slots start at their estimated remaining duration.
    availability = remaining + [0.0 for _ in range(max(0, _MAX_CONCURRENT_BUILDS - len(remaining)))]
    if not availability:
        availability = [0.0]
    heapq.heapify(availability)
    wait = 0.0
    for _ in range(max(1, int(position))):
        wait = heapq.heappop(availability)
        heapq.heappush(availability, wait + typical)
    wait_seconds = max(0, int(math.ceil(wait)))
    start = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
    return {
        "estimated_wait_seconds": wait_seconds,
        "estimated_start_at": start.isoformat(),
        "estimate_basis_seconds": int(round(typical)),
        "estimate_basis_threads": int(queued_threads),
    }


def _persist_build_status(task_id: str, status: str, **details: object) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    task_manager.update_state(task_id, {"build_status": payload})


async def _consume_queue():
    """Background coroutine: drain the build queue as slots free up.

    Lock ordering: never hold _queue_lock and _tasks_lock simultaneously.
    _queue_lock is released before _tasks_lock is acquired to avoid ABBA
    deadlock with api_build which acquires _tasks_lock → _queue_lock.
    """
    while True:
        if _queue_event is None:
            await asyncio.sleep(0)
            continue
        await _queue_event.wait()
        _queue_event.clear()
        while True:
            # Pop the next queued build under _queue_lock only
            task_id = None
            data = None
            # Keep the queue-to-running transition atomic with interactive
            # step admission for the same task capability.
            with _build_admission_lock:
                with _queue_lock:
                    if not _build_queue:
                        break
                    # Try to acquire a slot
                    acquired = _build_semaphore.acquire(blocking=False)
                    if not acquired:
                        break  # no slot yet — wait for next finish signal
                    task_id, data = _build_queue.pop(0)
                    enqueued_at = _queue_enqueued_at.pop(task_id, time.time())

                if task_id is None:
                    break

                # Now update shared state — _queue_lock already released.
                with _tasks_lock:
                    if task_id in _building_tasks:
                        # Already started by api_build — release slot and skip
                        _build_semaphore.release()
                        continue
                    _building_tasks.add(task_id)
                    _build_started_at[task_id] = time.time()
                    _build_threads[task_id] = task_thread_allocation(len(_building_tasks))
                    active_count = len(_building_tasks)
            waited_seconds = max(0, int(time.time() - enqueued_at))
            _persist_build_status(
                task_id,
                "running",
                started_at=datetime.now(timezone.utc).isoformat(),
                waited_seconds=waited_seconds,
            )
            with _build_logs_lock:
                _build_logs.setdefault(task_id, []).append(
                    f"Build starting from queue after {waited_seconds}s "
                    f"({active_count}/{_MAX_CONCURRENT_BUILDS} slots used)..."
                )

            # Rebuild queue positions for remaining waiters
            _update_queue_positions()

            # Dispatch build (don't await — let it run in background)
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(_get_executor(), _run_queued_build, data, task_id)
            # Schedule queue recheck when this build finishes
            asyncio.create_task(_on_build_done(task))


async def _on_build_done(task_future):
    """Called when a queued build completes — signals queue to process next."""
    try:
        await task_future
    except Exception:
        pass
    # Signal queue consumer that a slot may be free
    _signal_queue()


def _update_queue_positions():
    """Update _tasks for queued builds with their current position.

    Snapshots the queue under _queue_lock, then updates _tasks under
    _tasks_lock — avoids holding both locks simultaneously to prevent
    ABBA deadlock with api_build (_tasks_lock → _queue_lock).
    """
    with _queue_lock:
        snapshot = list(_build_queue)
    with _tasks_lock:
        for pos, (tid, _) in enumerate(snapshot):
            if tid in _tasks:
                _tasks[tid]["queue_position"] = pos + 1
    for pos, (tid, _) in enumerate(snapshot, 1):
        estimate = _queue_estimate(pos)
        _persist_build_status(
            tid,
            "queued",
            queue_position=pos,
            enqueued_at=datetime.fromtimestamp(
                _queue_enqueued_at.get(tid, time.time()), timezone.utc
            ).isoformat(),
            **estimate,
        )


@app.get("/api/build/{task_id}/queue-status")
async def api_build_queue_status(task_id: str):
    """Return the queue position for a waiting build (or null if not queued)."""
    task_id = _validate_task_id(task_id)
    queue_position = None
    with _queue_lock:
        for pos, (tid, _) in enumerate(_build_queue):
            if tid == task_id:
                queue_position = pos + 1
                break
    if queue_position is not None:
        return {
            "status": "queued",
            "queue_position": queue_position,
            "task_id": task_id,
            **_queue_estimate(queue_position),
        }
    # Check if actively building
    with _tasks_lock:
        t = _tasks.get(task_id, {})
        if t.get("status") == "completed":
            return {"status": "completed", "result": t.get("result")}
        if t.get("status") == "failed":
            return {"status": "failed", "error": _redact_server_paths(t.get("error"))}
        if t.get("status") == "running":
            return {
                "status": "running",
                "progress": t.get("progress"),
                "task_id": task_id,
            }
    state = task_manager.get_state(task_id) or {}
    persisted = state.get("build_status") or {}
    if persisted.get("status") in {"queued", "running", "completed", "failed"}:
        return {"task_id": task_id, **persisted}
    return {"status": "not_queued"}


@app.post("/api/build")
async def api_build(request: Request):
    """Finalize a checked system and package it.

    Coordinate-building modules are deliberately not run here.  The exact
    checkpoint shown after Ion Check (or Membrane Check for a dry pure
    bilayer) is the sole coordinate source.
    """
    data = await _json_object(request)
    task_id = data.get("task_id", "")

    # Validate task_id format to prevent path traversal
    task_id = _validate_task_id(task_id)

    # Bind the request to the workflow persisted when this task was created.
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    persisted_task_type = (
        (state.get("task_type") or {}).get("id") or state.get("task_type_id") or "membrane-bilayer"
    )
    requested_task_type = str(data.get("task_type", persisted_task_type))
    if requested_task_type != persisted_task_type:
        return JSONResponse(
            {
                "error": (
                    f"Task type mismatch: task {task_id} is {persisted_task_type}, "
                    f"not {requested_task_type}"
                )
            },
            status_code=409,
        )
    if persisted_task_type not in {
        "membrane-bilayer",
        "pure-membrane",
        "solvator",
        "martini3-bilayer",
        "martini3-solvent",
    }:
        return JSONResponse(
            {"error": f"Workflow {persisted_task_type!r} is not available for finalization"},
            status_code=400,
        )

    # Reject malformed expert MDP input before the task consumes a build slot.
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        return JSONResponse({"error": "Build modules must be an object"}, status_code=400)
    simparams = modules.get("simparams", {})
    execution = modules.get("execution", {})
    runner = _get_step_runner(task_id, persisted_task_type)
    if not runner.input_validation_current():
        return JSONResponse(
            {
                "error": "Run Check Upload again before building: the input has not passed "
                "the current structure validation policy.",
                "input_check_required": True,
            },
            status_code=409,
        )
    try:
        from gmxbuilder.runtime.hardware import normalize_simulation_hardware

        execution = normalize_simulation_hardware(execution)
        if _is_martini_task_type(persisted_task_type):
            from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol

            checked = runner.load_system("cg_system")
            if checked is None:
                raise ValueError("Final CG System Check is missing")
            has_membrane = checked.metadata.get("cg_environment") == "bilayer"
            simparams = normalize_protocol(simparams, has_membrane=has_membrane)
            include_solvent = bool(
                (checked.metadata.get("cg_solvation_config") or {}).get("include_solvent", True)
            )
            export_config = modules.setdefault("export", {})
            if not isinstance(export_config, dict):
                raise ValueError("export settings must be an object")
            export_config["write_mdp"] = include_solvent
            export_config["execution_hardware"] = execution
        else:
            forcefield_config = modules.get("forcefield", {})
            if not isinstance(forcefield_config, dict):
                forcefield_config = {}
            persisted_forcefield = state.get("step_forcefield_config", {})
            if not isinstance(persisted_forcefield, dict):
                persisted_forcefield = {}
            ff_name = str(
                forcefield_config.get("name") or persisted_forcefield.get("name") or "amber14sb"
            )
            simulation_context = {
                "force_field": ff_name,
                "force_field_family": (
                    "charmm"
                    if ff_name.lower().startswith("charmm")
                    else "opls"
                    if ff_name.lower().startswith("opls")
                    else "amber"
                ),
                "has_membrane": persisted_task_type in {"membrane-bilayer", "pure-membrane"},
            }
            simparams = MDPWriter.normalize_simulation_config(simparams, simulation_context)
            export_config = modules.setdefault("export", {})
            if not isinstance(export_config, dict):
                raise ValueError("export settings must be an object")
            export_config["execution_hardware"] = execution
    except (ModuleConfigError, TypeError, ValueError) as exc:
        return JSONResponse({"error": f"Invalid simulation parameters: {exc}"}, status_code=400)
    modules["simparams"] = simparams
    modules["execution"] = execution

    source_step = "cg_system" if _is_martini_task_type(persisted_task_type) else "ions"
    if persisted_task_type == "pure-membrane":
        solvation_config = modules.get("solvation")
        include_solvent = (
            isinstance(solvation_config, dict)
            and solvation_config.get("enabled", True) is not False
        )
        if "solvation" not in modules:
            include_solvent = False
        source_step = "ions" if include_solvent else "membrane"
    if not runner.has_checkpoint(source_step):
        return JSONResponse(
            {
                "error": (
                    f"Required {source_step.title()} Check is missing. "
                    "Return to that step, click Check, and confirm the displayed "
                    "system before finalizing."
                )
            },
            status_code=409,
        )
    from gmxbuilder.web.server_parts.viewer_data import confirmed_revision

    if not await _run_interactive(confirmed_revision, state, runner.step_dir(source_step)):
        return JSONResponse(
            {"error": "Confirm this checkpoint in Final Structure Review before building."},
            status_code=409,
        )
    data["task_type"] = persisted_task_type
    data["source_step"] = source_step
    data["confirmed_revision"] = state["final_review"]["revision"]
    task_manager.update_state(
        task_id,
        {
            "simparams": simparams,
            "step_simparams_config": simparams,
            "execution": execution,
            "step_execution_config": execution,
            "current_step": "simparams",
        },
    )

    queue_pos = None
    already_queued = False
    queue_full = False
    active_count = 0
    with _build_admission_lock:
        if _step_admission.active(task_id):
            return JSONResponse(
                {"error": "An interactive Check is already running for this task."},
                status_code=409,
            )
        with _tasks_lock:
            already_building = task_id in _building_tasks
        if already_building:
            return JSONResponse({"error": "This task is already being built."}, status_code=409)

        with _queue_lock:
            for pos, (tid, _) in enumerate(_build_queue):
                if tid == task_id:
                    already_queued = True
                    queue_pos = pos + 1
                    break

        if not already_queued:
            acquired = _build_semaphore.acquire(blocking=False)
            if acquired:
                try:
                    task_manager.save_build_request(task_id, data)
                    with _tasks_lock:
                        _building_tasks.add(task_id)
                        _build_started_at[task_id] = time.time()
                        _build_threads[task_id] = task_thread_allocation(len(_building_tasks))
                        _tasks[task_id] = {
                            "status": "running",
                            "progress": 0,
                            "result": None,
                            "error": None,
                        }
                        active_count = len(_building_tasks)
                except Exception:
                    _build_semaphore.release()
                    raise
            else:
                with _queue_lock:
                    if len(_build_queue) >= _MAX_QUEUED_BUILDS:
                        queue_full = True
                    else:
                        task_manager.save_build_request(task_id, data)
                        _build_queue.append((task_id, data))
                        _queue_enqueued_at[task_id] = time.time()
                        queue_pos = len(_build_queue)
                if not queue_full:
                    with _tasks_lock:
                        _tasks[task_id] = {
                            "status": "queued",
                            "progress": 0,
                            "result": None,
                            "error": None,
                            "queue_position": queue_pos,
                        }

    if queue_full:
        return JSONResponse(
            {
                "error": (
                    "Finalization queue is full. No work was accepted; "
                    "retry after an existing task completes."
                )
            },
            status_code=503,
            headers={"Retry-After": "30"},
        )

    if queue_pos is not None:
        estimate = _queue_estimate(queue_pos)
        _persist_build_status(
            task_id,
            "queued",
            queue_position=queue_pos,
            enqueued_at=datetime.fromtimestamp(
                _queue_enqueued_at.get(task_id, time.time()), timezone.utc
            ).isoformat(),
            **estimate,
        )
        with _build_logs_lock:
            _build_logs.setdefault(task_id, []).append(
                f"Position {queue_pos} in build queue; estimated wait "
                f"{estimate['estimated_wait_seconds']}s."
            )
        return JSONResponse(
            {
                "status": "queued",
                "task_id": task_id,
                "queue_position": queue_pos,
                **estimate,
                "message": (
                    f"You are position {queue_pos} in the build queue. "
                    f"Estimated start: {estimate['estimated_start_at']}. "
                    f"Your build will start automatically. Save task ID {task_id}; "
                    "it restores this workflow and its queue status."
                ),
            }
        )

    _persist_build_status(
        task_id,
        "running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    # Dispatch build in background — return immediately so HTTP doesn't block
    with _build_logs_lock:
        _build_logs[task_id] = [
            f"Build starting ({active_count}/{_MAX_CONCURRENT_BUILDS} slots used)..."
        ]
    logger.info(
        "Build %s started immediately (%d/%d slots)",
        task_log_reference(task_id),
        active_count,
        _MAX_CONCURRENT_BUILDS,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_get_executor(), _run_background_build, data, task_id)

    return JSONResponse(
        {
            "status": "started",
            "task_id": task_id,
            "message": "Build started. Poll /api/build/{task_id}/log for progress.",
        }
    )


def _run_background_build(data: dict, task_id: str, retry: bool = False) -> None:
    """Finalize a checked build in the background and update task state."""
    try:
        summary = _run_build_sync(data, task_id)
        with _tasks_lock:
            _tasks[task_id] = {
                "status": "completed",
                "progress": 100,
                "result": summary,
                "error": None,
            }
        with _build_logs_lock:
            _build_logs.setdefault(task_id, []).append(
                f"✓ Build complete: {summary['num_atoms']:,} atoms"
            )
        _persist_build_status(
            task_id,
            "completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            result=summary,
        )
    except Exception as exc:
        logger.exception("Build %s failed", task_log_reference(task_id))
        public_error = _redact_server_paths(exc)
        with _tasks_lock:
            _tasks[task_id] = {
                "status": "failed",
                "progress": 0,
                "result": None,
                "error": public_error,
            }
        with _build_logs_lock:
            _build_logs.setdefault(task_id, []).append(f"✗ Finalization failed: {public_error}")
        _persist_build_status(
            task_id,
            "failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=public_error,
        )
    finally:
        with _tasks_lock:
            _building_tasks.discard(task_id)
            started_at = _build_started_at.pop(task_id, None)
            threads_used = _build_threads.pop(task_id, configured_task_threads())
        if started_at is not None:
            duration = max(0.001, time.time() - started_at)
            _build_duration_history.append((duration, int(threads_used)))
        _build_semaphore.release()
        _update_queue_positions()
        _signal_queue()


def _run_queued_build(data: dict, task_id: str):
    """Run a build dispatched from the queue (no HTTP response).

    Must be a regular (non-async) function — it is passed to
    loop.run_in_executor which treats async defs as callables that
    return coroutine objects without executing their body.
    """
    _run_background_build(data, task_id)


def _run_build_sync(data: dict[str, Any], task_id: str) -> dict:
    """Finalize the exact checked checkpoint without rebuilding coordinates."""
    with _tasks_lock:
        allocated = _build_threads.get(task_id, configured_task_threads())
    # The budget decided when this build acquired its slot, made visible to
    # everything it calls. GROMACS fixes its thread count at launch, so there
    # is no point revisiting this while the build runs.
    with task_thread_scope(allocated):
        return _run_build_body(data, task_id)


def _run_build_body(data: dict[str, Any], task_id: str) -> dict:
    # Build state is already initialized by api_build via _tasks_lock
    with _build_logs_lock:
        _build_logs[task_id] = ["Build starting..."]
    try:
        modules_config = data.get("modules", {})
        task_type_id = data["task_type"]
        source_step = data["source_step"]
        runner = _get_step_runner(task_id, task_type_id)
        simparams = dict(modules_config.get("simparams") or {})
        export_config = dict(modules_config.get("export") or {})
        export_config["system_name"] = data.get("system_name", "system")

        with _tasks_lock:
            _tasks[task_id]["status"] = "running"
            _tasks[task_id]["progress"] = 20
        with _build_logs_lock:
            _build_logs[task_id] = [
                f"Finalizing exact {source_step} Check checkpoint...",
                "Coordinate-building modules will not be re-run.",
            ]

        if not _is_martini_task_type(task_type_id):
            _require_task_custom_lipids_ready(task_id)
        lipid_scope = (
            nullcontext()
            if _is_martini_task_type(task_type_id)
            else task_custom_lipid_scope(task_manager.get_task_dir(task_id))
        )
        with task_manager.active_task(task_id), lipid_scope:
            from gmxbuilder.web.server_parts.viewer_data import require_final_review

            require_final_review(
                task_manager.get_state(task_id) or {},
                runner.step_dir(source_step),
                data.get("confirmed_revision"),
            )
            result = runner.finalize_from_checkpoint(
                source_step,
                topology_config=dict(modules_config.get("topology") or {}),
                structure_config=modules_config.get("structure"),
                export_config=export_config,
                simparams=simparams,
            )
        if result["status"] != "ok":
            raise RuntimeError(result.get("error", "Finalization failed"))
        system = result.pop("system")

        with _tasks_lock:
            _tasks[task_id]["progress"] = 95
        with _build_logs_lock:
            _build_logs[task_id].extend(result["log"])
            _build_logs[task_id].append(f"✓ Package complete: {system.num_atoms:,} atoms")
            build_log = [_redact_server_paths(line) for line in _build_logs.get(task_id, [])]
        summary = {
            "task_id": task_id,
            "num_atoms": system.num_atoms,
            "components": [
                {
                    "name": c.name,
                    "atoms": len(c.atom_indices),
                    "kind": c.kind.name,
                    "n_molecules": c.metadata.get("n_molecules"),
                }
                for c in system.components
            ],
            "log": build_log,
            "source_checkpoint": source_step,
            "coordinate_rebuild": False,
            "package_contents": result["package_contents"],
            "download_url": f"/api/task/{task_id}/download",
        }
        return summary

    except Exception:
        logger.exception("Build %s failed", task_log_reference(task_id))
        raise


# ---------------------------------------------------------------------------
# API: status / download


@app.get("/api/status/{task_id}")
async def api_status(task_id: str):
    task_id = _validate_task_id(task_id)
    t = _tasks.get(task_id)
    if t is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(
        {
            "task_id": task_id,
            "status": t["status"],
            "progress": t["progress"],
            "error": (_redact_server_paths(t.get("error")) if t.get("error") is not None else None),
        }
    )


@app.get("/api/download/{task_id}")
async def api_download(task_id: str):
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    t = _tasks.get(task_id)
    ready = t is not None and t.get("status") == "completed"
    if not ready:
        persisted = state.get("build_status") or {}
        ready = isinstance(persisted, dict) and persisted.get("status") == "completed"
    if not ready:
        return JSONResponse({"error": "Task not ready or not found"}, status_code=404)
    zip_path = _authoritative_task_zip(task_id)
    if zip_path is None or not zip_path.exists():
        return JSONResponse({"error": "ZIP file not found on disk"}, status_code=404)
    return FileResponse(
        str(zip_path), media_type="application/zip", filename=f"gmxbuilder_{task_id}.zip"
    )


# ---------------------------------------------------------------------------
# API: Step-based incremental checkpoint build
# ---------------------------------------------------------------------------

_step_runners: dict[str, StepRunner] = {}
_step_runners_lock = threading.Lock()


def _get_step_runner(task_id: str, pipeline_type: str = "membrane-bilayer") -> StepRunner:
    """Get or create a StepRunner for the given task."""
    from gmxbuilder.pipeline.step_executor import StepRunner

    if task_manager.get_state(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    with _step_runners_lock:
        if task_id not in _step_runners:
            if len(_step_runners) >= 256:
                for cached in list(_step_runners):
                    if not _step_admission.active(cached) and cached not in _building_tasks:
                        if _resources is None or not _resources.queue.task_active(cached):
                            del _step_runners[cached]
                            break
                else:
                    raise HTTPException(status_code=503, detail="All cached tasks are active")
            task_dir = task_manager.get_task_dir(task_id)
            _step_runners[task_id] = StepRunner(task_dir, pipeline_type)
        return _step_runners[task_id]


def _require_task_custom_lipids_ready(task_id: str) -> list[dict]:
    """Reject progression while any submitted molecule is not validated."""
    records = CustomLipidStore(task_manager.get_task_dir(task_id)).list_public()
    unavailable = [record for record in records if record.get("state") != "ready"]
    if unavailable:
        details = ", ".join(
            f"{record.get('name')} ({record.get('state')}: {record.get('phase')})"
            for record in unavailable
        )
        raise ValueError(
            "Custom lipid calculation must finish successfully before this "
            f"task can proceed: {details}. Online calculation and retry are no longer "
            "available. Contact the administrator using the email address on the homepage "
            "or announcement board, or start a new task using installed lipids."
        )
    return records


def _trusted_membrane_config(task_id: str, config: dict) -> dict:
    """Bind every non-built-in membrane entry to this task's READY record."""
    trusted = copy.deepcopy(config)
    definitions = CustomLipidStore(task_manager.get_task_dir(task_id)).load_definitions()
    composition = trusted.get("lipid_composition")
    entries: list[dict] = []
    if isinstance(composition, dict):
        for leaflet in ("upper", "lower"):
            value = composition.get(leaflet)
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
    elif isinstance(trusted.get("lipid_type"), str):
        entries = [{"name": trusted["lipid_type"]}]

    builtins = set(LipidRegistry.list_builtin())
    for entry in entries:
        name = str(entry.get("name", "")).strip().upper()
        if not name or name in builtins:
            continue
        definition = definitions.get(name)
        if definition is None:
            raise ValueError(
                f"Lipid {name!r} is not in the standard library and does not belong to this task"
            )
        status = CustomLipidStore(task_manager.get_task_dir(task_id)).load_status(name)
        if status.get("state") != "ready":
            raise ValueError(f"Task-scoped lipid {name} is not ready ({status.get('state')})")
        # Replace all scientific metadata supplied by the browser with the
        # immutable server-side definition while preserving only the ratio.
        ratio = entry.get("ratio")
        entry.clear()
        entry.update(definition)
        if ratio is not None:
            entry["ratio"] = ratio
    return trusted


@app.get("/api/steps/{task_id}")
async def api_steps_status(task_id: str):
    """Return pipeline step list with checkpoint status for each step."""
    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    if task_state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    pipeline_type = (task_state.get("task_type") or {}).get("id") or "membrane-bilayer"
    try:
        steps = get_pipeline_steps(pipeline_type)
    except ValueError:
        return JSONResponse({"error": f"Unknown pipeline: {pipeline_type}"}, status_code=400)

    runner = _get_step_runner(task_id, pipeline_type)
    step_status = []
    for s in steps:
        from gmxbuilder.core.checkpoint_status import read_status

        has_checkpoint = runner.has_checkpoint(s)
        summary = read_status(runner.step_dir(s)) if has_checkpoint else None
        preview_available = has_checkpoint
        confirmed = None
        membrane_metrics = None
        if s in {"membrane", "cg_environment"} and has_checkpoint:
            membrane_metrics = summary.get("membrane_metrics") if summary else None
        if _is_martini_task_type(pipeline_type) and s == "cg_system" and has_checkpoint:
            confirmed = summary.get("system_confirmed") if summary else None
            # Construction completion and final visual confirmation are separate steps.
        step_status.append(
            {
                "name": s,
                "has_checkpoint": has_checkpoint,
                "preview_available": preview_available,
                "confirmed": confirmed,
                "membrane_metrics": membrane_metrics,
                "summary_pending": has_checkpoint and summary is None,
            }
        )

    return {
        "task_id": task_id,
        "pipeline_type": pipeline_type,
        "steps": step_status,
    }


@app.get("/api/step/{task_id}/progress")
async def api_step_progress(task_id: str):
    """Report how far the Check currently running on this task has got.

    A Check can run for minutes with nothing to show but a spinner. This gives
    the browser a phase name, a fraction and an elapsed time. Absence of an
    entry means nothing is running, which is a normal answer rather than an
    error -- the step may have finished between two polls.
    """
    try:
        task_id = _validate_task_id(task_id)
    except InvalidTaskId as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)

    with _step_progress_lock:
        entry = dict(_step_progress.get(task_id, {}))
    if _resources is not None:
        for operation in _resources.queue.active():
            if operation["task_id"] == task_id and operation.get("progress"):
                entry = json.loads(operation["progress"])
                break
    if not entry:
        return JSONResponse({"running": False})
    started = entry.get("started_at", time.time())
    return JSONResponse(
        {
            "running": True,
            "step": entry.get("step"),
            "phase": entry.get("phase"),
            "fraction": entry.get("fraction", 0.0),
            "elapsed_s": round(max(0.0, time.time() - started), 1),
        }
    )


@app.get("/api/ligand-prep/{task_id}")
async def api_ligand_prep_status(task_id: str):
    """Report the background ligand parameterization started after Step 1.

    Purely informational: the forcefield step runs the same calculation and
    does not consult this. It exists so the Force Field panel can say that a
    wait is already being worked on, rather than presenting it as new.
    """
    try:
        task_id = _validate_task_id(task_id)
    except InvalidTaskId as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    if _resources is not None:
        try:
            snapshot = json.loads(
                (task_manager.get_task_dir(task_id) / ".ligand-prep.json").read_text()
            )
            if snapshot.get("state") == "running" and not _resources.queue.task_active(task_id):
                snapshot["state"] = "interrupted"
            return JSONResponse(snapshot)
        except (OSError, ValueError):
            pass
    return JSONResponse(ligand_prep.status(task_id))


@app.post("/api/step/{task_id}/{step_name}")
async def api_run_step(task_id: str, step_name: str, request: Request):
    """Execute a single pipeline step.

    The request body must contain the module configuration for this step.
    The step reads the previous step's checkpoint, runs the module, and
    saves its own checkpoint.
    """
    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)

    try:
        data = await _json_object(request)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
    if "config" in data:
        config = data["config"]
    else:
        modules = data.get("modules", {})
        if not isinstance(modules, dict):
            return JSONResponse({"error": "modules must be a JSON object"}, status_code=400)
        config = modules.get(step_name, {})
    if not isinstance(config, dict):
        return JSONResponse(
            {"error": f"Configuration for step {step_name!r} must be an object"},
            status_code=400,
        )

    if step_name == "membrane":
        try:
            _require_task_custom_lipids_ready(task_id)
            config = _trusted_membrane_config(task_id, config)
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    if step_name == "forcefield":
        from gmxbuilder.web.server_parts.ligand_chemistry import trusted_config

        config = dict(config)
        config.pop("_ligand_source_path", None)
        if str(config.get("ligand_ff", "")).lower() == "charmm_compat":
            try:
                config = trusted_config(task_id, config, task_manager, _validate_task_resource)
            except (ValueError, OSError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        else:
            config.pop("charmm_compat_mol2", None)

    # CGenFF paths are server-owned upload artifacts.  Never trust arbitrary
    # filesystem paths supplied in a step request.
    if step_name == "forcefield" and str(config.get("ligand_ff", "")).lower() == "cgenff":
        selected_ff = str(config.get("name", "")).lower()
        trusted_packages = {}
        expected_root = (task_manager.get_task_dir(task_id) / "cgenff").resolve()
        for name, package in (task_state.get("cgenff_uploads", {}) or {}).items():
            if not isinstance(package, dict) or package.get("force_field") != selected_ff:
                continue
            package_root = (expected_root / str(name).upper()).resolve()
            mol2_value = package.get("mol2_file") or package.get("mol2_path", "")
            str_value = package.get("str_file") or package.get("str_path", "")
            mol2_candidate = Path(str(mol2_value))
            str_candidate = Path(str(str_value))
            if not mol2_candidate.is_absolute():
                mol2_candidate = package_root / mol2_candidate.name
            if not str_candidate.is_absolute():
                str_candidate = package_root / str_candidate.name
            try:
                mol2_path = _validate_task_resource(task_id, mol2_candidate)
                str_path = _validate_task_resource(task_id, str_candidate)
            except ValueError:
                continue
            if package_root not in mol2_path.parents or package_root not in str_path.parents:
                continue
            if mol2_path.is_file() and str_path.is_file():
                trusted_packages[str(name).upper()] = {
                    "mol2_path": str(mol2_path),
                    "str_path": str(str_path),
                }
        config = dict(config)
        config["cgenff_parameters"] = trusted_packages

    pipeline_type = (
        (task_state.get("task_type") or {}).get("id")
        or task_state.get("task_type_id")
        or "membrane-bilayer"
    )
    requested_pipeline = data.get("pipeline_type")
    if requested_pipeline and requested_pipeline != pipeline_type:
        return JSONResponse(
            {
                "error": (
                    f"Pipeline mismatch: task {task_id} is bound to "
                    f"{pipeline_type}, not {requested_pipeline}"
                )
            },
            status_code=409,
        )
    try:
        allowed_steps = get_pipeline_steps(pipeline_type)
    except ValueError:
        return JSONResponse(
            {"error": f"Unknown persisted pipeline: {pipeline_type}"}, status_code=400
        )
    if step_name not in allowed_steps or step_name in {"topology", "export"}:
        return JSONResponse(
            {
                "error": (
                    f"Step {step_name!r} is not an interactive Check step for "
                    f"{pipeline_type}; topology and export are generated only by "
                    "finalization."
                )
            },
            status_code=400,
        )

    runner = _get_step_runner(task_id, pipeline_type)

    # Find the PDB path for the input step — uses the same resolution
    # order as the rest of the pipeline: structure checkpoint > filtered
    # (chain selections applied by frontend Check button) > cleaned > uploaded
    pdb_path = None
    protein_free_cg = (
        _is_martini_task_type(pipeline_type) and config.get("include_protein") is False
    )
    if step_name == "input" and not protein_free_cg:
        try:
            pdb_path = _resolve_input_pdb(task_id)
        except ValueError as exc:
            return JSONResponse({"error": _redact_server_paths(exc)}, status_code=400)

    # Build initial system for first step
    import numpy as np

    from gmxbuilder.core.structure import Structure
    from gmxbuilder.core.system import System

    seed = data.get("seed", task_state.get("seed", 42))
    initial = System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
        metadata={"seed": seed, "input_source_metadata": task_state.get("input_source_metadata")},
    )

    # Admit before submitting so the executor's internal queue is never the
    # first (unbounded) line of defence.  Finalization and Check operations for
    # one task are mutually exclusive to protect checkpoint integrity.
    with _build_admission_lock:
        with _tasks_lock:
            finalizing = task_id in _building_tasks
        with _queue_lock:
            queued_for_finalization = any(tid == task_id for tid, _data in _build_queue)
        if finalizing or queued_for_finalization:
            return JSONResponse(
                {"error": "This task is queued or running finalization."}, status_code=409
            )
        admission = _step_admission.try_acquire(task_id)
    if admission == "duplicate":
        return JSONResponse(
            {"error": "A Check operation is already running for this task."}, status_code=409
        )
    if admission == "full":
        return JSONResponse(
            {"error": "The Check queue is full; retry shortly."},
            status_code=503,
            headers={"Retry-After": "5"},
        )

    # Run step in its dedicated thread pool (all steps are synchronous).
    loop = asyncio.get_running_loop()

    def _publish(fraction: float, phase: str) -> None:
        with _step_progress_lock:
            _step_progress[task_id] = {
                "step": step_name,
                "fraction": round(float(fraction), 3),
                "phase": phase,
                "started_at": _step_progress.get(task_id, {}).get("started_at", time.time()),
                "updated_at": time.time(),
            }
            published = dict(_step_progress[task_id])
        if is_worker():
            from gmxbuilder.web.resource_queue import ResourceQueue

            ResourceQueue(managed_root() / ".control" / "queue.sqlite3").update(
                os.environ["GMXBUILDER_OPERATION_ID"], progress=published
            )

    def _run_scoped_step():
        try:
            with (
                task_manager.active_task(task_id),
                task_custom_lipid_scope(task_manager.get_task_dir(task_id)),
            ):
                if (
                    step_name == "input"
                    and not _is_martini_task_type(pipeline_type)
                    and initial.metadata.get("input_source_metadata") is None
                ):
                    from gmxbuilder.modules.input.validation import read_polymer_metadata

                    uploaded_name = task_state.get("uploaded_structure_name")
                    if uploaded_name:
                        original = _task_resource_helpers.validate_task_resource(
                            task_id, uploaded_name
                        )
                        if original.is_file() and not original.is_symlink():
                            initial.metadata["input_source_metadata"] = read_polymer_metadata(
                                original
                            )
                step_result = runner.run_step(
                    step_name,
                    config,
                    initial_system=initial,
                    pdb_path=pdb_path,
                    on_progress=_publish,
                )
                if step_name == "input" and step_result.get("status") == "ok":
                    # Summaries use canonical coordinates, never wrapped viewer identities.
                    checked = System.load_checkpoint(runner.step_dir("input")).structure
                    sequences = _extract_sequences(checked)
                    # Retain excluded fragments as selectable choices on recheck/resume.
                    # This is a coordinate-only preview, without repair or dynamics.
                    from gmxbuilder.io.input_document import read_input
                    from gmxbuilder.modules.input.reconstruction import reconstruct_input

                    fragment_choices = []
                    if pdb_path:
                        choices = read_input(pdb_path)
                        reconstruct_input(choices, {**config, "exclude_fragments": []})
                        fragment_choices = _extract_sequences(choices)
                    checked_by_key = {row.get("fragment_key"): row for row in sequences}
                    fragment_choices = [
                        checked_by_key.get(row.get("fragment_key"), row) for row in fragment_choices
                    ]
                    step_result.setdefault("metrics", {})["input_summary"] = {
                        "num_atoms": checked.num_atoms,
                        "box_nm": checked.dimensions().tolist(),
                        "chains": [chain["chain_id"] for chain in sequences],
                        "sequences": sequences,
                        "fragment_choices": fragment_choices,
                        "small_molecules": PDBValidator.detect_small_molecules(checked),
                    }
                return step_result
        finally:
            _step_admission.release(task_id)
            with _step_progress_lock:
                _step_progress.pop(task_id, None)

    try:
        step_future = _get_step_executor().submit(_run_scoped_step)
    except BaseException:
        _step_admission.release(task_id)
        with _step_progress_lock:
            _step_progress.pop(task_id, None)
        raise
    admission = _step_admission

    def _release_cancelled_step(completed):
        if completed.cancelled():
            admission.release(task_id)
            with _step_progress_lock:
                _step_progress.pop(task_id, None)

    step_future.add_done_callback(_release_cancelled_step)
    # Client cancellation must not release task admission while the worker is
    # still mutating checkpoints. Only an unstarted cancellation uses the callback;
    # a running worker's finally block owns its release.
    result = await asyncio.wrap_future(step_future, loop=loop)

    if result["status"] == "ok":
        # Update task state
        state_update = {
            "current_step": step_name,
            f"step_{step_name}_config": config,
        }
        if step_name == "input":
            state_update["input_fragment_choices"] = list(
                (result.get("metrics", {}).get("input_summary") or {}).get("fragment_choices", [])
            )
            state_update["input_modifications"] = dict(
                (result.get("metrics") or {}).get("input_modifications") or {}
            )
            state_update["input_sequences"] = list(
                (result.get("metrics") or {}).get("input_sequences") or []
            )
            # Ligand parameterization is the slowest thing a Check does and
            # depends on nothing the user has yet to choose. Starting it here
            # overlaps it with the Force Field panel instead of making the
            # user wait through it. Speculative, so it runs on the custom
            # lipid pool and never takes a build slot.
            ligand_prep.start(
                task_id, task_manager.get_task_dir(task_id), _get_custom_lipid_executor()
            )
        if step_name == "structure":
            state_update["modification_geometry"] = list(
                (result.get("metrics") or {}).get("modification_geometry") or []
            )
        if step_name == "orient":
            saved_orientation = dict((result.get("metrics") or {}).get("orientation") or {})
            saved_orientation["method"] = config.get("method", "ppm")
            state_update["orient"] = saved_orientation
        if step_name == "cg_system":
            state_update["cg_system_confirmed"] = False
        state_update["final_review"] = None
        task_manager.update_state(task_id, state_update)

    return JSONResponse(_public_step_result(task_id, result))


@app.get("/api/step/{task_id}/{step_name}/viewer.json")
async def api_step_viewer_data(task_id: str, step_name: str, request: Request):
    from gmxbuilder.web.server_parts.viewer_data import build_viewer

    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    task_type = (task_state.get("task_type") or {}).get("id") or task_state.get("task_type_id")
    if step_name not in get_pipeline_steps(task_type):
        return JSONResponse({"error": "Unknown checkpoint"}, status_code=400)
    directory = _validate_task_resource(task_id, Path("steps") / step_name)
    try:
        path = await _run_interactive(build_viewer, directory)
    except FileNotFoundError:
        return JSONResponse(
            {"error": "Run this step's Check before loading its viewer"}, status_code=404
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    from gmxbuilder.web.server_parts.viewer_data import SCHEMA

    etag = f'"viewer-{SCHEMA}-{path.name}"'
    headers = {"Content-Encoding": "gzip", "Cache-Control": "private, no-cache", "ETag": etag}
    previous = {
        value.strip().removeprefix("W/")
        for value in request.headers.get("if-none-match", "").split(",")
    }
    if etag in previous or "*" in previous:
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type="application/json", headers=headers)


@app.post("/api/task/{task_id}/final-review")
async def api_confirm_final_review(task_id: str, request: Request):
    from gmxbuilder.web.server_parts.viewer_data import checkpoint_revision, final_source

    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    data = await _json_object(request)
    source = data.get("source_step")
    task_type = (task_state.get("task_type") or {}).get("id") or task_state.get("task_type_id")
    permitted = {final_source(task_state)}
    if task_type == "pure-membrane":
        permitted = {"membrane", "ions"}
    if not isinstance(source, str) or source not in permitted:
        return JSONResponse({"error": "Wrong final checkpoint for this workflow"}, status_code=409)
    if _resources is not None and _resources.queue.task_active(task_id):
        return JSONResponse(
            {"error": "Wait for the current operation before confirming"}, status_code=409
        )
    with _build_admission_lock:
        with _queue_lock:
            queued = any(tid == task_id for tid, _data in _build_queue)
        if (
            queued
            or task_id in _building_tasks
            or _step_admission.try_acquire(task_id) != "accepted"
        ):
            return JSONResponse({"error": "Task has work in progress"}, status_code=409)
    try:
        directory = _validate_task_resource(task_id, Path("steps") / source)
        try:
            revision = await _run_interactive(checkpoint_revision, directory)
        except FileNotFoundError:
            return JSONResponse({"error": "Final checkpoint is missing"}, status_code=409)
        if data.get("revision") != revision:
            return JSONResponse(
                {"error": "The checkpoint changed. Reload and inspect the current system."},
                status_code=409,
            )
        rendered_revision = revision
        if source == "cg_system":
            checked = await _run_interactive(System.load_checkpoint, directory)
            if (checked.metadata.get("cg_scientific_check") or {}).get("passed") is not True:
                return JSONResponse(
                    {"error": "The final CG scientific quality gate has not passed"},
                    status_code=409,
                )
            # The existing CG exporter requires this bookkeeping flag. Record both
            # digests; confirmation changes metadata only, never viewed coordinates.
            if checked.metadata.get("system_confirmed") is not True:
                checked.metadata["system_confirmed"] = True
                await _run_interactive(checked.save_checkpoint, directory)
                revision = await _run_interactive(checkpoint_revision, directory)
        record = {
            "confirmed": True,
            "source_step": source,
            "revision": revision,
            "rendered_revision": rendered_revision,
        }
        updates = {"final_review": record, "current_step": "final_review"}
        if task_type == "pure-membrane":
            updates["step_solvation_config"] = {
                **(task_state.get("step_solvation_config") or {}),
                "enabled": source == "ions",
            }
        task_manager.update_state(task_id, updates)
        return {"status": "ok", **record}
    finally:
        _step_admission.release(task_id)


@app.post("/api/step/{task_id}/cg_system/confirm")
async def api_confirm_cg_system(task_id: str):
    """Retired: confirmations must identify the exact inspected revision."""
    return JSONResponse(
        {"error": "Use /api/task/{task_id}/final-review with source_step and checkpoint revision"},
        status_code=410,
    )


def _render_step_viewer(runner, step_name: str, pipeline_type: str) -> bool:
    """Write a step's viewer PDB from its saved checkpoint. Returns success."""
    system = runner.load_system(step_name)
    if system is None:
        return False
    target = runner.step_dir(step_name) / "viewer.pdb"
    try:
        if _is_martini_task_type(pipeline_type):
            from gmxbuilder.modules.coarse_grained.common import write_cg_viewer_pdb

            write_cg_viewer_pdb(system, target, task_dir=runner.task_dir)
        else:
            system.write_viewer_pdb(target)
    except Exception:
        logger.debug("Viewer PDB could not be rendered for %s", step_name, exc_info=True)
        return False
    return target.exists()


@app.get("/api/step/{task_id}/{step_name}/viewer.pdb")
async def api_step_viewer_pdb(task_id: str, step_name: str):
    """Return the viewer PDB for a completed step."""
    task_id = _validate_task_id(task_id)
    pipeline_type = "membrane-bilayer"  # default
    task_state = task_manager.get_state(task_id)
    if task_state:
        pipeline_type = (task_state.get("task_type") or {}).get("id") or "membrane-bilayer"

    try:
        allowed_steps = get_pipeline_steps(pipeline_type)
    except ValueError:
        return JSONResponse({"error": "Unknown persisted pipeline"}, status_code=400)
    if step_name not in allowed_steps:
        return JSONResponse({"error": "Unknown pipeline step"}, status_code=400)

    runner = _get_step_runner(task_id, pipeline_type)
    pdb_path = runner.step_dir(step_name) / "viewer.pdb"
    # Martinize2 writes authoritative CONECT records for mapped beads.  Keep
    # those bonds in the mapping preview instead of the generic checkpoint PDB,
    # which intentionally stores coordinates only.
    if _is_martini_task_type(pipeline_type) and step_name == "cg_mapping":
        mapped_path = runner.step_dir(step_name) / "martinize" / "cg_protein.pdb"
        step_root = runner.step_dir(step_name).resolve()
        resolved = mapped_path.resolve()
        if step_root in resolved.parents and resolved.is_file() and not resolved.is_symlink():
            pdb_path = resolved
    if not pdb_path.exists():
        # Rendered on demand from the step's checkpoint and then kept. Writing
        # one after every step spent most of its effort on steps the interface
        # never shows.
        rendered = await asyncio.get_running_loop().run_in_executor(
            _get_interactive_executor(), _render_step_viewer, runner, step_name, pipeline_type
        )
        if not rendered:
            return JSONResponse({"error": f"No viewer PDB for step '{step_name}'"}, status_code=404)

    # Enrich an already checked Martini orientation from the immutable mapped
    # protein graph. This also repairs the active task after a service update
    # without changing its scientific checkpoint or requiring a re-run.
    if _is_martini_task_type(pipeline_type) and step_name == "cg_orientation":
        original = pdb_path.read_text(encoding="utf-8", errors="replace")
        if "\nCONECT" not in original:
            system = runner.load_system(step_name)
            if system is not None:
                from gmxbuilder.modules.coarse_grained.common import write_cg_viewer_pdb

                with tempfile.TemporaryDirectory(prefix="gmxbuilder-cg-viewer-") as tmp_dir:
                    enriched_path = Path(tmp_dir) / "viewer.pdb"
                    write_cg_viewer_pdb(system, enriched_path, task_dir=runner.task_dir)
                    enriched = enriched_path.read_text(encoding="utf-8")
                if "\nCONECT" in enriched:
                    return Response(
                        enriched,
                        media_type="chemical/x-pdb",
                        headers={
                            "Content-Disposition": (f'inline; filename="{step_name}_viewer.pdb"')
                        },
                    )

    return FileResponse(
        str(pdb_path), media_type="chemical/x-pdb", filename=f"{step_name}_viewer.pdb"
    )


@app.get("/api/step/{task_id}/export/download")
async def api_step_export_download(task_id: str):
    """Download the final system ZIP after the export step."""
    task_id = _validate_task_id(task_id)
    pipeline_type = "membrane-bilayer"
    task_state = task_manager.get_state(task_id)
    if task_state:
        pipeline_type = (task_state.get("task_type") or {}).get("id") or "membrane-bilayer"

    runner = _get_step_runner(task_id, pipeline_type)
    export_dir = runner.step_dir("export")
    if not export_dir.exists():
        return JSONResponse({"error": "Export step not yet run"}, status_code=404)

    # Find ZIP file
    zip_files = list(export_dir.glob("*.zip"))
    if not zip_files:
        for d in export_dir.rglob("*.zip"):
            zip_files.append(d)

    if not zip_files:
        return JSONResponse({"error": "No ZIP file in export directory"}, status_code=404)

    # Prefer the archive the exporter marked; fall back to the timestamp only
    # for tasks written before that marker existed.
    zip_path = read_authoritative_archive(export_dir) or max(
        zip_files, key=lambda p: p.stat().st_mtime_ns
    )
    return FileResponse(
        str(zip_path), media_type="application/zip", filename=f"gmxbuilder_{task_id}.zip"
    )


# Imported at the end of the module on purpose: the step runner and the PDB
# input module both import this module, so a top-of-file import would be
# circular.  E402 is suppressed rather than silenced globally.
from gmxbuilder.modules.input.pdb_input import PDBInputModule  # noqa: E402
from gmxbuilder.pipeline.step_executor import (  # noqa: E402
    StepRunner,
    get_pipeline_steps,
)

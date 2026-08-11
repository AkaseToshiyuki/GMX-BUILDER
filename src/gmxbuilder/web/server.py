"""FastAPI web server for GMXBUILDER."""

from __future__ import annotations

import asyncio
import copy
import gzip
import io
from collections import deque
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from gmxbuilder import VERSION
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.exceptions import ModuleConfigError, ParseError
from gmxbuilder.io.pdb import PDBParser, PDBValidator
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.modules.membrane.lipids import LipidRegistry, CATEGORY_NAMES
from gmxbuilder.modules.solvation.water_models import WaterRegistry
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry
from gmxbuilder.web.task_manager import task_expiry, task_manager
from gmxbuilder.runtime.hardware import configured_task_slots, hardware_capabilities
from gmxbuilder.web.custom_lipids import (
    CustomLipidStore,
    run_custom_lipid_build,
    task_custom_lipid_scope,
)
from gmxbuilder.web.security import (
    DurableRateLimiter,
    SecurityConfig,
    apply_security_headers,
    client_identity,
    request_authenticated,
    request_is_https,
    request_origin_allowed,
)

# ---------------------------------------------------------------------------
# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gmxbuilder.web")

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Authorization", "Content-Type", "X-Admin-Token", "X-GMXBUILDER-Token"
    ],
)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), follow_symlink=False), name="static")

# New tasks use a full 128-bit UUID.  Twelve-character IDs remain readable so
# existing persisted tasks survive the upgrade.
_TASK_ID_PATTERN = re.compile(r"^(?:[a-f0-9]{12}|[a-f0-9]{32})$")

_RATE_LIMITERS: dict[Path, DurableRateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _rate_limiter() -> DurableRateLimiter:
    configured = os.environ.get("GMXBUILDER_RATE_LIMIT_DB", "").strip()
    path = Path(configured) if configured else task_manager.root / ".rate-limits.sqlite3"
    path = path.expanduser().resolve()
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(path)
        if limiter is None:
            limiter = DurableRateLimiter(path)
            _RATE_LIMITERS[path] = limiter
        return limiter


def _rate_policy(path: str) -> tuple[str, int, float] | None:
    if path == "/api/build":
        return ("finalize", int(os.environ.get("GMXBUILDER_FINALIZE_RATE", "30")), 3600.0)
    if path in {"/api/upload-pdb", "/api/build-lipid-library"} or (
        path.startswith("/api/task/") and path.endswith("/custom-lipids")
    ):
        return ("heavy", int(os.environ.get("GMXBUILDER_HEAVY_RATE", "20")), 3600.0)
    if path.startswith("/api/") and path not in {"/api/health"}:
        return ("api-write", int(os.environ.get("GMXBUILDER_API_RATE", "240")), 60.0)
    return None


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforce deployment authentication, trusted-proxy and durable rate policy."""
    security = SecurityConfig.from_environment()
    public_mode = security.mode == "public"
    live_probe = request.url.path == "/health/live"

    def secured(response):
        return apply_security_headers(response, public_mode=public_mode)

    if security.errors:
        return secured(JSONResponse(
            {"error": "Invalid server security configuration", "details": list(security.errors)},
            status_code=503,
        ))

    if not live_probe and public_mode and not request_is_https(request, security):
        return secured(JSONResponse(
            {"error": "HTTPS is required in public deployment mode"},
            status_code=426,
        ))

    authentication_required = public_mode or security.authentication_enabled
    if not live_probe and authentication_required and not request_authenticated(request, security):
        headers = {
            "WWW-Authenticate": (
                'Basic realm="GMXBUILDER", charset="UTF-8"'
                if security.auth_user else "Bearer"
            )
        }
        return secured(JSONResponse(
            {"error": "Authentication is required"}, status_code=401, headers=headers
        ))

    if authentication_required and not request_origin_allowed(request, security):
        return secured(JSONResponse(
            {"error": "Request Origin is not allowed"}, status_code=403
        ))

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        policy = _rate_policy(request.url.path)
        if policy is not None:
            bucket_name, limit, window = policy
            allowed, retry_after = _rate_limiter().allow(
                client_identity(request, security), bucket_name, limit, window
            )
            if not allowed:
                return secured(JSONResponse(
                    {
                        "error": (
                            "Request rate limit exceeded. Wait before retrying; "
                            "running or queued work has not been cancelled."
                        )
                    },
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                ))

    response = await call_next(request)
    return secured(response)

def _validate_task_resource(task_id: str, resource: str | Path) -> Path:
    """Resolve a server-owned file and confine it to exactly one task."""
    task_id = _validate_task_id(task_id)
    task_dir = task_manager.get_task_dir(task_id).resolve()
    candidate = Path(resource)
    if not candidate.is_absolute():
        candidate = task_dir / candidate
    resolved = candidate.resolve()
    if resolved == task_dir or task_dir not in resolved.parents:
        raise ValueError(f"Resource is outside task {task_id}")
    return resolved

def _resolve_input_pdb(task_id: str) -> str:
    """Structure for the input step — filtered selection > original upload.

    Structure cleaning is now handled inside PDBInputModule, so there is
    no intermediate cleaned.pdb.
    """
    task_id = _validate_task_id(task_id)
    task_dir = task_manager.get_task_dir(task_id)

    filtered = task_dir / "filtered.pdb"
    if filtered.exists():
        return str(filtered)

    state = task_manager.get_state(task_id) or {}
    uploaded_name = state.get("uploaded_structure_name")
    if isinstance(uploaded_name, str) and uploaded_name:
        uploaded = _validate_task_resource(task_id, uploaded_name)
        if uploaded.is_file() and not uploaded.is_symlink():
            return str(uploaded)

    pdb_path = task_manager.get_pdb_path(task_id)
    if pdb_path and pdb_path.exists():
        return str(pdb_path)

    raise ValueError(f"No structure file found for task {task_id}")


def _resolve_pdb_path(task_id: str) -> str:
    """Best-effort PDB for *preview/propka* endpoints — may use structure checkpoint.

    Priority: structure checkpoint > filtered.pdb > uploaded PDB.
    Not used by pipeline steps (they use System.load_checkpoint, which has no fallbacks).
    """
    task_id = _validate_task_id(task_id)
    task_dir = task_manager.get_task_dir(task_id)

    structure_pdb = task_dir / "steps" / "structure" / "viewer.pdb"
    if structure_pdb.exists():
        return str(structure_pdb)

    filtered = task_dir / "filtered.pdb"
    if filtered.exists():
        return str(filtered)

    pdb_path = task_manager.get_pdb_path(task_id)
    if pdb_path and pdb_path.exists():
        return str(pdb_path)

    raise ValueError(f"No PDB file found for task {task_id}")


def _resolve_propka_pdb_path(task_id: str) -> str:
    """Return the stable, pre-protonation structure used by PROPKA.

    A Structure-step checkpoint already contains a previous pH decision. Using
    it as the input for a later Compute request makes the result depend on which
    pH was checked first. Always prefer the repaired Step-1 checkpoint instead.
    """
    task_id = _validate_task_id(task_id)
    task_dir = task_manager.get_task_dir(task_id)

    input_pdb = task_dir / "steps" / "input" / "viewer.pdb"
    if input_pdb.exists():
        return str(input_pdb)

    filtered = task_dir / "filtered.pdb"
    if filtered.exists():
        return str(filtered)

    pdb_path = task_manager.get_pdb_path(task_id)
    if pdb_path and pdb_path.exists():
        return str(pdb_path)

    raise ValueError(f"No PDB file found for task {task_id}")

def _validate_task_id(task_id: str) -> str:
    """Validate task_id format and path safety. Returns sanitized task_id."""
    if not _TASK_ID_PATTERN.match(task_id):
        raise ValueError(f"Invalid task ID format: {task_id}")
    return task_id


def _is_admin_request(request: Request) -> bool:
    configured = os.environ.get("GMXBUILDER_ADMIN_TOKEN", "")
    supplied = request.headers.get("X-Admin-Token", "")
    return bool(configured and supplied and secrets.compare_digest(configured, supplied))


_SERVER_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|root|tmp|var|opt|srv|mnt|media)"
    r"(?:/[^\s,;:)\]}]+)+"
)


def _redact_server_paths(value: object) -> str:
    """Remove host filesystem locations from browser-visible build messages."""
    return _SERVER_PATH_PATTERN.sub("<server-path>", str(value))


def _sanitize_public_value(value: object) -> object:
    """Recursively remove internal path fields and redact path-like strings."""
    if isinstance(value, dict):
        return {
            str(key): _sanitize_public_value(item)
            for key, item in value.items()
            if not str(key).lower().endswith("_path")
        }
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        return _redact_server_paths(value)
    return value


def _public_task_state(state: dict) -> dict:
    """Return resumable task state without exposing host filesystem paths."""
    public = copy.deepcopy(state)
    uploads = public.get("cgenff_uploads")
    if isinstance(uploads, dict):
        public["cgenff_uploads"] = {
            str(name): {
                "force_field": package.get("force_field"),
                "cgenff_version": package.get("cgenff_version"),
                "maximum_penalty": package.get("maximum_penalty"),
                "ready": True,
            }
            for name, package in uploads.items()
            if isinstance(package, dict)
        }
    sanitized = _sanitize_public_value(public)
    assert isinstance(sanitized, dict)
    return sanitized


def _public_step_result(task_id: str, result: dict) -> dict:
    """Replace internal StepRunner paths with task-scoped resource URLs."""
    public = dict(result)
    viewer_available = bool(public.pop("viewer_pdb_path", None))
    index_available = bool(public.pop("index_path", None))
    public.pop("zip_path", None)
    step_name = str(public.get("step", ""))
    if viewer_available and step_name:
        public["viewer_pdb_url"] = (
            f"/api/step/{task_id}/{step_name}/viewer.pdb"
        )
    if index_available:
        public["index_available"] = True
    sanitized = _sanitize_public_value(public)
    assert isinstance(sanitized, dict)
    return sanitized


def _authoritative_task_zip(task_id: str) -> Path | None:
    """Return the current export ZIP, preferring it over legacy output bundles."""
    task_dir = task_manager.get_task_dir(task_id)
    export_dir = task_dir / "steps" / "export"
    current = list(export_dir.glob("*.zip")) if export_dir.is_dir() else []
    if current:
        return max(current, key=lambda path: path.stat().st_mtime_ns)

    output_dir = task_manager.get_output_dir(task_id)
    legacy = list(output_dir.glob("*.zip"))
    if legacy:
        return max(legacy, key=lambda path: path.stat().st_mtime_ns)
    return None


# Task store (in-memory — survives as long as the server runs)
_tasks: dict[str, dict] = {}
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
_custom_lipid_executor: ThreadPoolExecutor | None = None
_custom_lipid_executor_lock = threading.Lock()
_event_loop: asyncio.AbstractEventLoop | None = None
_lifespan_tasks: list[asyncio.Task] = []
_pka_cache: dict[str, list[dict]] = {}  # task_id → pKa predictions
_pka_cache_lock = threading.Lock()
_pka_running: set[str] = set()  # task_ids currently being computed
# Track which tasks are currently building (prevent duplicate builds)
_building_tasks: set[str] = set()
_custom_lipid_jobs: set[tuple[str, str]] = set()
_custom_lipid_jobs_lock = threading.Lock()
_custom_gpu_condition = threading.Condition()
_custom_gpu_in_use: set[int] = set()
_custom_gpu_cursor = 0
_RUNTIME_HARDWARE = hardware_capabilities()
_CUSTOM_GPU_IDS = tuple(range(_RUNTIME_HARDWARE.configured_gpu_count))
_CUSTOM_GPU_CONCURRENCY = min(
    2,
    max(1, int(os.environ.get("GMXBUILDER_CUSTOM_LIPID_CONCURRENCY", "2"))),
    max(1, len(_CUSTOM_GPU_IDS)),
)

# Cleanup expired tasks on startup
_startup_removed = task_manager.cleanup_expired()
if _startup_removed:
    logger.info("Cleaned up %d expired task(s)", len(_startup_removed))


# ---------------------------------------------------------------------------
# Lifespan: periodic cleanup + graceful shutdown

async def startup_background_tasks():
    """Start background tasks: periodic cleanup + build-queue consumer."""
    global _queue_event, _event_loop
    _get_executor()
    _get_custom_lipid_executor()
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
    # Resume task-owned calculations that were queued or interrupted by a
    # service restart.  Definitions and states are disk-backed.
    for task_dir in sorted(task_manager.root.iterdir()):
        if not task_dir.is_dir() or task_manager.is_expired(task_dir.name):
            continue
        store = CustomLipidStore(task_dir)
        for lipid_name in store.pending_names():
            _schedule_custom_lipid_build(task_dir.name, lipid_name)
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
                        for tid in removed:
                            _pka_running.discard(tid)
                    # StepRunner cache cleanup
                    with _step_runners_lock:
                        for tid in removed:
                            _step_runners.pop(tid, None)
                    # Also remove expired tasks from queue
                    with _queue_lock:
                        _build_queue[:] = [(tid, d) for tid, d in _build_queue if tid not in removed]
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
    global _executor, _custom_lipid_executor, _event_loop
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
    with _custom_lipid_executor_lock:
        custom_executor = _custom_lipid_executor
        _custom_lipid_executor = None
    if custom_executor is not None:
        custom_executor.shutdown(wait=True, cancel_futures=False)
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
    """Dedicated two-worker queue so long NPT jobs cannot starve web Checks."""
    global _custom_lipid_executor
    with _custom_lipid_executor_lock:
        if (
            _custom_lipid_executor is None
            or getattr(_custom_lipid_executor, "_shutdown", False)
        ):
            _custom_lipid_executor = ThreadPoolExecutor(
                max_workers=_CUSTOM_GPU_CONCURRENCY,
                thread_name_prefix="gmxbuilder-custom-lipid",
            )
        return _custom_lipid_executor


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
    # Simple template variable replacement (version is the only dynamic value)
    html = html.replace("{{ version }}", VERSION)
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the workflow selection page."""
    return _wizard_html()


_WORKFLOW_ROUTES = {"BilayerBuilder", "PureBilayerSystem", "Solvator", "CoarseGrainedBuilder"}


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

@app.get("/health/live")
async def liveness_check():
    """Minimal unauthenticated liveness probe without host fingerprinting."""
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    """Authenticated detailed health in public mode; unchanged on trusted networks."""
    from gmxbuilder import VERSION as _ver
    security = SecurityConfig.from_environment()
    return {
        "status": "ok",
        "version": _ver,
        "builds_active": len(_building_tasks),
        "builds_max": _MAX_CONCURRENT_BUILDS,
        "builds_queued": len(_build_queue),
        "typical_build_seconds": int(round(_typical_build_seconds())),
        "hardware": hardware_capabilities().as_public_dict(),
        "security": {
            "deployment_mode": security.mode,
            "authentication_enabled": security.authentication_enabled,
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
        data = await request.json()
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
    task_manager.update_state(task["task_id"], {
        "task_type": detail,
        "task_type_id": task_type_id,
        "current_step": detail["visible_modules"][0],
    })
    return {"task_id": task["task_id"], "task_type": detail}


# ---------------------------------------------------------------------------
# API: PPM orientation

@app.post("/api/orient-ppm")
async def api_orient_ppm(request: Request):
    """Compute orientation for a previously uploaded PDB."""
    data = await request.json()
    if data.get("tmp_path"):
        return JSONResponse(
            {
                "error": (
                    "Client-supplied filesystem paths are not accepted; "
                    "provide task_id"
                )
            },
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
        tmp_path = str(
            _validate_task_resource(task_id_val, _resolve_pdb_path(task_id_val))
        )
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
        parser = PDBParser()
        structure = parser.parse(tmp_path)

        from gmxbuilder.modules.membrane.orient import (
            compute_orientation,
            orient_protein,
        )

        z_offset, _, tilt_rad = compute_orientation(
            structure, algorithm=algorithm, half_thickness=half_thickness)

        # ---- Apply full PPM orientation to a copy for 3D preview ----
        oriented_pdb = None
        tmp_name = None
        try:
            from gmxbuilder.io.pdb import PDBWriter
            oriented = parser.parse(tmp_path)
            orient_protein(
                oriented,
                method=algorithm,
                half_thickness=half_thickness,
            )
            # Write oriented structure to temp PDB and read back
            fd, tmp_name = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            PDBWriter.write(oriented, tmp_name)
            oriented_pdb = Path(tmp_name).read_text()
        except Exception:
            logger.exception("Failed to generate oriented PDB for preview")
            oriented_pdb = None  # non-critical — frontend uses raw PDB as fallback
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

        return {
            "algorithm": algorithm,
            "z_offset": round(z_offset, 2),
            "tilt_degrees": round(np.degrees(tilt_rad), 1),
            "oriented_pdb": oriented_pdb,
        }
    except Exception:
        logger.exception("Unhandled error in orient-ppm")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def _generate_orientation_preview(task_id: str, config: dict) -> dict:
    """Run the real Step 4 module without writing or invalidating checkpoints."""
    task_dir = task_manager.get_task_dir(task_id)
    structure_checkpoint = task_dir / "steps" / "structure"
    if not (structure_checkpoint / "system.npz").exists():
        raise FileNotFoundError(
            "Structure checkpoint is missing; run Check Structure Processing first."
        )

    from gmxbuilder.modules.membrane.orient_module import OrientModule

    system = System.load_checkpoint(structure_checkpoint)
    preview_config = dict(config)
    preview_config.setdefault("seed", system.metadata.get("seed", 42))
    module = OrientModule()
    module.validate_config(preview_config)
    result = module.execute(system, preview_config)
    if not result.success:
        raise RuntimeError("Orientation module reported failure")

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-orient-preview-") as tmp_dir:
        preview_path = Path(tmp_dir) / "viewer.pdb"
        result.system.write_viewer_pdb(preview_path)
        oriented_pdb = preview_path.read_text(encoding="utf-8")

    return {
        "status": "ok",
        "method": result.system.metadata.get(
            "_orientation_method", preview_config.get("method", "ppm")
        ),
        "orientation": result.system.metadata.get("_orient_params", {}),
        "orientation_quality": result.system.metadata.get(
            "_orientation_quality", {}
        ),
        "oriented_pdb": oriented_pdb,
        "log": result.log,
    }


@app.post("/api/orient-preview/{task_id}")
async def api_orient_preview(task_id: str, request: Request):
    """Preview exactly the coordinates that Step 4 Check would persist."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"error": "Request body must be valid JSON"}, status_code=400
        )
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be an object"}, status_code=400)
    config = data.get("config", data)
    if not isinstance(config, dict):
        return JSONResponse({"error": "Orientation config must be an object"}, status_code=400)

    try:
        # Preview is read-only and short-lived. Keep it off the persistent
        # build executor so browser slider traffic cannot occupy build slots,
        # and so application/TestClient restarts cannot reuse a shut-down pool.
        payload = await asyncio.to_thread(
            _generate_orientation_preview, task_id, config
        )
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
    data = await request.json()
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
    protein_metrics = data.get("protein")               # {center_of_mass_nm, extent_nm, ...}
    membrane_metrics = data.get("membrane")             # {midplane_z_nm, half_thickness_nm, ...}

    preview_config = {
        "box_dimensions_nm": box_dimensions_nm,
        "protein": protein_metrics,
        "membrane": membrane_metrics,
    }

    # ---- Write preview PDB (oriented protein with box/membrane CRYST1) ----
    preview_path = None
    tmp_name = None
    try:
        task_dir = task_manager.get_task_dir(task_id)
        preview_path = task_dir / "preview.pdb"

        if oriented_pdb:
            # Use the oriented PDB from the frontend, update CRYST1 to match computed box
            from gmxbuilder.io.pdb import PDBParser, PDBWriter

            # Parse the oriented PDB to get structure
            fd, tmp_name = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            Path(tmp_name).write_text(oriented_pdb)
            structure = PDBParser().parse(tmp_name)

            # Update box to match frontend-computed dimensions
            if box_dimensions_nm and len(box_dimensions_nm) == 3:
                structure.box_vectors = np.diag([
                    float(box_dimensions_nm[0]),
                    float(box_dimensions_nm[1]),
                    float(box_dimensions_nm[2]),
                ])

            PDBWriter.write(structure, preview_path, title="GMXBUILDER Preview")
    except Exception:
        logger.exception("Failed to write preview PDB")
        # Non-fatal — preview_config is still stored
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    # ---- Store preview_config in task state ----
    task_manager.update_state(task_id, {
        "preview_config": preview_config,
        "preview_pdb_resource": "preview.pdb" if preview_path and preview_path.exists() else None,
    })

    return {
        "status": "ok",
        "preview_config": preview_config,
        "preview_saved": bool(preview_path and preview_path.exists()),
        "preview_resource": "preview.pdb" if preview_path and preview_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# API: Filter PDB by chain/molecule selection
# ---------------------------------------------------------------------------

def _normalise_small_molecule_labels(
    raw_labels: object, allowed_resnames: set[str]
) -> dict[str, str]:
    """Validate user-facing molecule labels without changing structural IDs."""
    if not isinstance(raw_labels, dict):
        raise ValueError("small_molecule_labels must be an object")

    labels: dict[str, str] = {}
    for raw_key, raw_label in raw_labels.items():
        if not isinstance(raw_key, str) or not isinstance(raw_label, str):
            raise ValueError("Small-molecule label keys and values must be strings")
        key = raw_key.strip().upper()
        label = raw_label.strip()
        if key not in allowed_resnames:
            raise ValueError(f"Unknown small-molecule key {key!r}")
        if not label:
            raise ValueError(f"Display label for {key} must not be empty")
        if len(label) > 64:
            raise ValueError(f"Display label for {key} must be at most 64 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in label):
            raise ValueError(f"Display label for {key} contains control characters")
        labels[key] = label

    effective: dict[str, str] = {
        key: labels.get(key, key) for key in sorted(allowed_resnames)
    }
    seen: dict[str, str] = {}
    for key, label in effective.items():
        folded = label.casefold()
        if folded in seen:
            raise ValueError(
                f"Small-molecule display label {label!r} is used for both "
                f"{seen[folded]} and {key}"
            )
        seen[folded] = key
    return labels

@app.post("/api/filter-pdb/{task_id}")
async def api_filter_pdb(task_id: str, request: Request):
    task_id = _validate_task_id(task_id)
    data = await request.json()
    include_chains = data.get("include_chains", [])
    exclude_resnames = set(data.get("exclude_resnames", []))

    task_dir = task_manager.get_task_dir(task_id)
    pdb_path = task_manager.get_pdb_path(task_id)
    # For CIF uploads the converted PDB is used as filter source
    converted = task_dir / "converted.pdb"
    src = converted if converted.exists() else pdb_path
    if not src or not src.exists():
        return JSONResponse({"error": "No PDB file found"}, status_code=400)

    detected_small_molecules = PDBValidator.detect_small_molecules(src)
    allowed_labels = {
        str(item["resname"]).strip().upper() for item in detected_small_molecules
    }
    try:
        small_molecule_labels = _normalise_small_molecule_labels(
            data.get("small_molecule_labels", {}), allowed_labels
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    filtered_path = task_dir / "filtered.pdb"
    n_kept, n_removed = 0, 0
    with open(src) as fh_in, open(filtered_path, "w") as fh_out:
        for line in fh_in:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21:22].strip() if len(line) > 21 else ""
                resname = line[17:20].strip()
                if include_chains and chain not in include_chains:
                    n_removed += 1
                    continue
                if resname in exclude_resnames:
                    n_removed += 1
                    continue
                n_kept += 1
            fh_out.write(line)

    task_manager.update_state(task_id, {
        # These are UI labels only.  The original residue key remains stable
        # in coordinates and force-field parameterization.
        "small_molecule_labels": small_molecule_labels,
    })

    return {
        "status": "ok",
        "filtered_resource": "filtered.pdb",
        "n_kept": n_kept, "n_removed": n_removed,
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
    # Extend TTL on access (throttled: only writes to disk if >15 min since last write)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    new_expiry = task_expiry(now)
    state["expires_at"] = new_expiry
    # Only persist the extension if it's been >15 min since last write
    last_write = state.get("_last_ttl_write", "")
    try:
        write_age = (
            (now - datetime.fromisoformat(last_write)).total_seconds()
            if last_write else None
        )
    except (TypeError, ValueError):
        write_age = None
    if write_age is None or write_age > 900:
        task_manager.update_state(task_id, {"expires_at": new_expiry, "_last_ttl_write": now.isoformat()})
    return _public_task_state(state)


@app.post("/api/task/{task_id}/save-step")
async def api_task_save_step(task_id: str, request: Request):
    """Save bounded, non-authoritative browser state for one visible step."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    try:
        data = await request.json()
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
    """Return the full state for frontend resume, including re-parsed PDB info."""
    task_id = _validate_task_id(task_id)
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)

    task_type = state.get("task_type") or {}
    task_type_id = task_type.get("id") or state.get("task_type_id") or "membrane-bilayer"
    protein_free_cg = (
        task_type_id == "coarse-grained"
        and (state.get("step_input_config") or {}).get("include_protein") is False
    )

    # Re-parse the PDB to get full pdb_info
    pdb_path = task_manager.get_pdb_path(task_id)
    input_viewer = task_manager.get_task_dir(task_id) / "steps" / "input" / "viewer.pdb"
    if not protein_free_cg and input_viewer.is_file() and not input_viewer.is_symlink():
        pdb_path = input_viewer
    if not protein_free_cg and pdb_path and pdb_path.exists():
        try:
            parser = PDBParser()
            structure = parser.parse(pdb_path)
            pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
            sequences = _extract_sequences(structure)
            # Only include chains that contain at least one protein residue
            # (small-molecule-only chains appear in the Small Molecules section)
            chains = [s["chain_id"] for s in sequences if s.get("chain_id", "").strip()]
            # Fallback: if the PDB has no explicit chain IDs (all blank), use "A"
            if not chains and sequences:
                chains = ["A"]
                for s in sequences:
                    s["chain_id"] = "A"

            # Detect small molecules so the frontend can style them
            small_molecules = PDBValidator.detect_small_molecules(pdb_path)

            state["pdb_content"] = pdb_text
            state["sequences"] = sequences
            state["small_molecules"] = small_molecules
            state["pdb_info_full"] = {
                "filename": (state.get("pdb_info") or {}).get(
                    "filename", pdb_path.name
                ),
                "num_atoms": structure.num_atoms,
                "chains": sorted(chains),
                "box_nm": [round(v, 3) for v in structure.dimensions().tolist()],
                "small_molecules": small_molecules,
            }
        except Exception:
            logger.exception("Failed to re-parse PDB while resuming task %s", task_id)

    visible_steps = list(task_type.get("visible_modules") or [])
    try:
        runner = _get_step_runner(task_id, task_type_id)
        checkpoint_steps = {
            name for name in get_pipeline_steps(task_type_id)
            if runner.has_checkpoint(name)
        }
    except ValueError:
        checkpoint_steps = set(state.get("steps_completed") or [])
    existing_zip = _authoritative_task_zip(task_id)
    build_status = state.get("build_status")
    if not isinstance(build_status, dict):
        build_status = {}
    if existing_zip is not None and build_status.get("status") not in {
        "queued", "running", "failed"
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
    ):
        resume_step = "simparams"
    else:
        for candidate in visible_steps:
            if candidate == "simparams" or candidate not in checkpoint_steps:
                resume_step = candidate
                break
    resume_index = (
        visible_steps.index(resume_step) if resume_step in visible_steps else 0
    )
    route_slug = task_type.get("route_slug") or {
        "membrane-bilayer": "BilayerBuilder",
        "pure-membrane": "PureBilayerSystem",
        "solvator": "Solvator",
        "coarse-grained": "CoarseGrainedBuilder",
    }.get(task_type_id, "BilayerBuilder")
    state["resume_step"] = resume_step
    state["resume_step_number"] = resume_index + 1
    state["resume_url"] = f"/{route_slug}/Step{resume_index + 1}"
    return _public_task_state(state)


@app.get("/api/task/{task_id}/download")
async def api_task_download(task_id: str):
    """Download the build output ZIP for a task (works post-restart too)."""
    task_id = _validate_task_id(task_id)
    output_dir = task_manager.get_output_dir(task_id)

    # The checked build is authoritative. Never let a larger legacy archive
    # shadow the current steps/export package.
    zip_path = _authoritative_task_zip(task_id)
    if zip_path is not None:
        filename = f"gmxbuilder_{task_id}.zip"
        return FileResponse(str(zip_path), media_type="application/zip",
                            filename=filename)

    # Fallback: create ZIP from output files recursively
    files = list(output_dir.rglob("*"))
    if not files:
        return JSONResponse({"error": "No output files available"}, status_code=404)
    fallback = output_dir / "gmxbuilder_output.zip"
    with zipfile.ZipFile(fallback, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.is_file() and f.suffix != ".zip":
                z.write(f, f.relative_to(output_dir))
    return FileResponse(str(fallback), media_type="application/zip",
                        filename=f"gmxbuilder_{task_id}.zip")


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
                tasks.append({
                    "task_id": state["task_id"],
                    "filename": state.get("filename", ""),
                    "current_step": state.get("current_step", ""),
                    "created_at": state.get("created_at", ""),
                    "expires_at": state.get("expires_at", ""),
                })
    return tasks


@app.post("/api/build-lipid-library")
async def api_build_lipid_library(request: Request):
    """Build a genuinely new custom lipid by explicit-solvent NPT.

    Built-in entries are intentionally generated only by the offline CLI;
    this endpoint is not a user-facing way to rebuild the packaged matrix.
    """
    if os.environ.get("GMXBUILDER_ALLOW_ONLINE_LIPID_BUILD", "0") != "1":
        return JSONResponse(
            {
                "error": (
                    "Online lipid pre-equilibration is disabled because it is a "
                    "long-running administrator operation. Ask the administrator "
                    "to build and validate this lipid with the offline lipid-library command."
                )
            },
            status_code=403,
        )
    if not _is_admin_request(request):
        return JSONResponse(
            {"error": "Administrator authorization is required"}, status_code=403
        )
    data = await request.json()
    lipid_name = data.get("lipid_name", "").strip().upper()
    if not lipid_name:
        return JSONResponse({"error": "lipid_name is required"}, status_code=400)
    if data.get("is_custom") is not True:
        return JSONResponse(
            {"error": "Built-in lipid libraries are generated by the offline coverage command"},
            status_code=403,
        )
    force_field = str(data.get("force_field", "")).strip().lower()
    lipid_ff = str(data.get("lipid_ff", "gaff2")).strip().lower()
    if not force_field.startswith("amber") or lipid_ff != "gaff2":
        return JSONResponse(
            {"error": "New custom lipids currently require an Amber + GAFF2 selection"},
            status_code=400,
        )
    try:
        from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder
        from gmxbuilder.modules.membrane.lipids import (
            LipidRegistry,
            find_registered_lipid_matches,
        )

        properties = data.get("properties") or {}
        matches = find_registered_lipid_matches(str(properties.get("smiles", "")))
        exact = [match for match in matches if match["match"] == "exact"]
        if exact:
            return JSONResponse(
                {"error": f"Structure already exists as {exact[0]['name']}; use that entry"},
                status_code=409,
            )
        try:
            LipidRegistry.get(lipid_name)
        except KeyError:
            LipidRegistry.register_custom(lipid_name, properties)
        npt_ps = float(data.get("npt_ps", 1000.0))
        if not math.isfinite(npt_ps) or not 500.0 <= npt_ps <= 5000.0:
            return JSONResponse(
                {"error": "npt_ps must be between 500 and 5000 ps"}, status_code=400
            )
        await asyncio.to_thread(
            LipidEquilibrationBuilder().build,
            lipid_name,
            force_field,
            lipid_ff,
            npt_ps=npt_ps,
            force=bool(data.get("force", False)),
        )
        return {
            "status": "ok",
            "lipid_name": lipid_name,
            "force_field": force_field,
            "lipid_ff": lipid_ff,
            "library_ready": True,
            "message": "Validated explicit-solvent NPT library is ready",
        }
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/lipid-library-status")
async def api_lipid_library_status(
    lipid_name: str = "", force_field: str = "amber14sb", lipid_ff: str = "",
):
    """Check if a lipid's conformation library is built."""
    if not lipid_name:
        return JSONResponse({"error": "lipid_name query parameter required"}, status_code=400)
    from gmxbuilder.modules.membrane.equilibrated_library import (
        get_equilibrated_lipid_library,
    )
    lib = get_equilibrated_lipid_library()
    if lipid_ff:
        effective_lipid_ff = lipid_ff
    elif force_field.startswith("amber"):
        from gmxbuilder.modules.forcefield.lipid_policy import amber_lipid_backend

        effective_lipid_ff, reason = amber_lipid_backend([lipid_name])
        if effective_lipid_ff is None:
            return JSONResponse({"error": reason}, status_code=400)
    else:
        effective_lipid_ff = force_field
    entry = lib.inspect(lipid_name, force_field, effective_lipid_ff)
    return {
        "lipid_name": lipid_name.upper(),
        "force_field": force_field,
        "lipid_ff": effective_lipid_ff,
        "has_library": entry is not None,
        "n_conformations": len(entry.conformer_files) if entry else 0,
        "metadata": entry.metadata if entry else None,
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
    """Parse a SMILES string and estimate lipid physical properties."""
    data = await request.json()
    smiles = data.get("smiles", "").strip()
    name = data.get("name", "").strip()

    try:
        from gmxbuilder.modules.membrane.lipids import parse_custom_lipid
        result = parse_custom_lipid(smiles, name)
        return result
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to parse SMILES: {exc}"}, status_code=500)


def _run_custom_lipid_job(task_id: str, lipid_name: str) -> None:
    """Worker entry point with task cleanup protection and public-safe errors."""
    key = (task_id, lipid_name)
    task_dir = task_manager.get_task_dir(task_id)
    gpu_id = _acquire_custom_gpu()
    try:
        with task_manager.active_task(task_id):
            if task_manager.get_state(task_id) is None:
                return
            from gmxbuilder.modules.membrane.lipid_equilibration import (
                lipid_gpu_device,
            )
            with lipid_gpu_device(gpu_id):
                run_custom_lipid_build(task_dir, lipid_name)
    except Exception as exc:
        logger.exception("Task %s custom lipid %s failed", task_id, lipid_name)
        try:
            CustomLipidStore(task_dir).update_status(
                lipid_name,
                state="failed",
                phase="failed",
                progress=0,
                message=_redact_server_paths(exc),
            )
        except Exception:
            logger.exception("Could not persist custom lipid failure state")
    finally:
        _release_custom_gpu(gpu_id)
        with _custom_lipid_jobs_lock:
            _custom_lipid_jobs.discard(key)


def _acquire_custom_gpu() -> int | None:
    """Allocate at most two distinct GPUs in round-robin order."""
    global _custom_gpu_cursor
    if not _CUSTOM_GPU_IDS:
        return None
    with _custom_gpu_condition:
        while len(_custom_gpu_in_use) >= _CUSTOM_GPU_CONCURRENCY or all(
            gpu in _custom_gpu_in_use for gpu in _CUSTOM_GPU_IDS
        ):
            _custom_gpu_condition.wait()
        for offset in range(len(_CUSTOM_GPU_IDS)):
            index = (_custom_gpu_cursor + offset) % len(_CUSTOM_GPU_IDS)
            gpu = _CUSTOM_GPU_IDS[index]
            if gpu not in _custom_gpu_in_use:
                _custom_gpu_in_use.add(gpu)
                _custom_gpu_cursor = (index + 1) % len(_CUSTOM_GPU_IDS)
                return gpu
    return None


def _release_custom_gpu(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    with _custom_gpu_condition:
        _custom_gpu_in_use.discard(gpu_id)
        _custom_gpu_condition.notify_all()


def _schedule_custom_lipid_build(task_id: str, lipid_name: str) -> bool:
    """Schedule one idempotent task-owned calculation."""
    key = (task_id, str(lipid_name).upper())
    with _custom_lipid_jobs_lock:
        if key in _custom_lipid_jobs:
            return False
        _custom_lipid_jobs.add(key)
    _get_custom_lipid_executor().submit(_run_custom_lipid_job, *key)
    return True


@app.post("/api/task/{task_id}/custom-lipids")
async def api_submit_task_custom_lipid(task_id: str, request: Request):
    """Submit a genuinely new lipid to one task and start its calculation."""
    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
    hardware = hardware_capabilities()
    if not hardware.gmx_installed:
        return JSONResponse(
            {
                "error": (
                    "GROMACS is not installed or was not detected. Custom lipid "
                    "pre-equilibration cannot start."
                )
            },
            status_code=503,
        )

    force_field = str(data.get("force_field", "")).strip().lower()
    lipid_ff = str(data.get("lipid_ff", "gaff2")).strip().lower()
    if not force_field.startswith("amber") or lipid_ff != "gaff2":
        return JSONResponse(
            {"error": "Custom lipids currently require an Amber + GAFF2 selection"},
            status_code=400,
        )
    try:
        from gmxbuilder.modules.membrane.lipids import (
            find_registered_lipid_matches,
            parse_custom_lipid,
        )

        properties = parse_custom_lipid(
            str(data.get("smiles", "")).strip(),
            str(data.get("name", "")).strip(),
        )
        exact = [
            match for match in find_registered_lipid_matches(properties["canonical_smiles"])
            if match["match"] == "exact"
        ]
        if exact:
            return JSONResponse(
                {
                    "error": (
                        f"This molecule already exists in the standard lipid "
                        f"library as {exact[0]['name']}; duplicate custom submission "
                        "is not permitted."
                    ),
                    "existing_lipid": exact[0]["name"],
                },
                status_code=409,
            )
        store = CustomLipidStore(task_manager.get_task_dir(task_id))
        record = store.save_submission(properties, force_field)
        _schedule_custom_lipid_build(task_id, record["name"])
        return JSONResponse(record, status_code=202)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except Exception as exc:
        logger.exception("Custom lipid submission failed")
        return JSONResponse(
            {"error": _redact_server_paths(exc)}, status_code=500
        )


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
        record = CustomLipidStore(
            task_manager.get_task_dir(task_id)
        ).public_record(lipid_name)
        record["message"] = _redact_server_paths(record.get("message", ""))
        return record
    except (KeyError, ValueError):
        return JSONResponse({"error": "Custom lipid not found for this task"}, status_code=404)


@app.post("/api/task/{task_id}/custom-lipids/{lipid_name}/retry")
async def api_retry_task_custom_lipid(task_id: str, lipid_name: str):
    """Retry a failed task-private parameterization without duplicating data."""
    task_id = _validate_task_id(task_id)
    if task_manager.get_state(task_id) is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    store = CustomLipidStore(task_manager.get_task_dir(task_id))
    try:
        record = store.public_record(lipid_name)
        if record.get("state") != "failed":
            return JSONResponse(
                {"error": "Only failed custom lipid calculations can be retried"},
                status_code=409,
            )
        store.update_status(
            record["name"], state="queued", phase="queued", progress=0,
            message="Retry queued",
        )
        _schedule_custom_lipid_build(task_id, record["name"])
        return JSONResponse(store.public_record(record["name"]), status_code=202)
    except (KeyError, ValueError):
        return JSONResponse({"error": "Custom lipid not found for this task"}, status_code=404)


# ---------------------------------------------------------------------------
# API: protonation & modifications

def _schedule_propka_precompute(tmp_path: str) -> None:
    """Kick off PROPKA in a background thread so results are ready later."""
    with _pka_cache_lock:
        if tmp_path in _pka_cache or tmp_path in _pka_running:
            return
        _pka_running.add(tmp_path)

    def _run():
        try:
            from gmxbuilder.modules.modifications.protonation import predict_pka_from_pdb
            preds = predict_pka_from_pdb(tmp_path)
            with _pka_cache_lock:
                _pka_cache[tmp_path] = preds
        except Exception:
            pass
        finally:
            with _pka_cache_lock:
                _pka_running.discard(tmp_path)

    _get_executor().submit(_run)


async def _get_propka_results(tmp_path: str) -> list[dict]:
    """Return cached PROPKA results, or compute asynchronously if not cached.

    Uses asyncio.to_thread to avoid blocking the event loop during
    the (potentially slow, 3-10 s) PROPKA calculation.
    """
    with _pka_cache_lock:
        if tmp_path in _pka_cache:
            return _pka_cache[tmp_path]

    # Not cached — run in thread pool to keep event loop free
    try:
        from gmxbuilder.modules.modifications.protonation import predict_pka_from_pdb
        preds = await asyncio.to_thread(predict_pka_from_pdb, tmp_path)
        with _pka_cache_lock:
            _pka_cache[tmp_path] = preds
            _pka_running.discard(tmp_path)
        return preds
    except Exception:
        with _pka_cache_lock:
            _pka_running.discard(tmp_path)
        return []


@app.get("/api/propka-status")
async def api_propka_status(tmp_path: str = "", task_id: str = ""):
    """Check whether PROPKA has finished precomputing for a given PDB."""
    if tmp_path:
        return JSONResponse(
            {
                "error": (
                    "Client-supplied filesystem paths are not accepted; "
                    "provide task_id"
                )
            },
            status_code=400,
        )
    if not task_id:
        return JSONResponse({"error": "task_id is required"}, status_code=400)
    try:
        task_id = _validate_task_id(task_id)
        if task_manager.get_state(task_id) is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
        tmp_path = str(
            _validate_task_resource(task_id, _resolve_propka_pdb_path(task_id))
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    with _pka_cache_lock:
        if tmp_path in _pka_cache:
            return {"status": "ready", "residues": len(_pka_cache[tmp_path])}
        if tmp_path in _pka_running:
            return {"status": "computing"}
    return {"status": "not_started"}


@app.post("/api/protonate")
async def api_protonate(request: Request):
    """Compute protonation states for a list of residues at a given pH.

    If task_id is provided, runs PROPKA for environment-sensitive pKa
    prediction. Otherwise falls back to standard model-pKa values.
    """
    data = await request.json()
    residues = data.get("residues", [])
    pH = data.get("pH", 7.0)
    his_tautomer = data.get("his_tautomer", "HSE")
    if data.get("tmp_path"):
        return JSONResponse(
            {
                "error": (
                    "Client-supplied filesystem paths are not accepted; "
                    "provide task_id"
                )
            },
            status_code=400,
        )
    tmp_path = ""
    task_id_val = data.get("task_id", "")
    if task_id_val:
        try:
            task_id_val = _validate_task_id(str(task_id_val))
            if task_manager.get_state(task_id_val) is None:
                return JSONResponse(
                    {"error": "Task not found or expired"}, status_code=404
                )
            tmp_path = str(_validate_task_resource(
                task_id_val, _resolve_propka_pdb_path(task_id_val)
            ))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
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
            assignments = assign_protonation_with_propka(
                structure_residues, pka_predictions, pH=float(pH), his_tautomer=his_tautomer
            )
        else:
            assignments = assign_all_protonations(residues, pH=float(pH), his_tautomer=his_tautomer)

        modified = [a for a in assignments if a["is_titratable"]]
        method = (
            "PROPKA 3.5 (environment-sensitive)"
            if used_propka
            else ("Model pKa fallback" if propka_requested else "Model pKa (no 3D context)")
        )
        return {
            "pH": pH,
            "assignments": assignments,
            "titratable_count": len(modified),
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
    data = await request.json()
    if data.get("tmp_path"):
        return JSONResponse(
            {
                "error": (
                    "Client-supplied filesystem paths are not accepted; "
                    "provide task_id"
                )
            },
            status_code=400,
        )
    task_id_val = data.get("task_id", "")
    if not task_id_val:
        return JSONResponse({"error": "task_id is required"}, status_code=400)
    try:
        task_id_val = _validate_task_id(str(task_id_val))
        if task_manager.get_state(task_id_val) is None:
            return JSONResponse({"error": "Task not found or expired"}, status_code=404)
        tmp_path = str(
            _validate_task_resource(task_id_val, _resolve_pdb_path(task_id_val))
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    pH = data.get("pH", 7.0)
    his_tautomer = data.get("his_tautomer", "HSE")
    modifications = data.get("modifications", [])  # [{index, patch_id}]
    force_field = str(data.get("force_field", "charmm36m"))
    nter_patch = data.get("nter_patch")   # e.g. "ACE" or null
    cter_patch = data.get("cter_patch")   # e.g. "NME" or null

    if not Path(tmp_path).exists():
        return JSONResponse({"error": "PDB file not found"}, status_code=400)

    try:
        from gmxbuilder.io.pdb import PDBParser
        from gmxbuilder.modules.modifications.protonation import assign_all_protonations

        parser = PDBParser()
        structure = parser.parse(tmp_path)

        # Build residue list from structure
        residues = structure.resnames

        # Protonation
        prot_assignments = assign_all_protonations(
            list(residues), pH=float(pH), his_tautomer=his_tautomer
        )

        # Collect applied modifications
        applied: list[dict] = []
        for mod in modifications:
            idx = mod.get("index")
            patch_id = mod.get("patch_id")
            if idx is not None and idx < len(residues):
                applied.append({
                    "index": idx,
                    "original_resname": residues[idx],
                    "patch_id": patch_id,
                })

        # Convert protonation assignments to residue-level summary
        residue_changes = []
        for a in prot_assignments:
            if a["is_titratable"]:
                residue_changes.append({
                    "index": a["index"],
                    "original": a["original"],
                    "new_name": a["assigned_name"],
                    "charge": a["charge"],
                    "pKa": a["pKa"],
                    "state": a["state_label"],
                })

        from gmxbuilder.modules.modifications.protonation import compute_net_charge_from_protonation
        net_charge = compute_net_charge_from_protonation(prot_assignments)

        # Add modification charge shifts (side-chain PTMs + N/C-terminal patches)
        from gmxbuilder.modules.modifications.patches import (
            effective_patch_charge_shift,
            get_patch,
        )
        for mod in applied:
            patch = get_patch(mod["patch_id"])
            if patch:
                net_charge += effective_patch_charge_shift(
                    mod["patch_id"], force_field
                )

        # Apply N-terminal patch charge (e.g. ACE: 0, standard NH3+: +1)
        if nter_patch:
            nter_p = get_patch(nter_patch)
            if nter_p:
                net_charge += nter_p.charge_shift
                applied.append({"index": 0, "original_resname": residues[0] if residues else "?",
                               "patch_id": nter_patch, "term": "N"})

        # Apply C-terminal patch charge (e.g. NME: 0, standard COO-: -1)
        if cter_patch:
            cter_p = get_patch(cter_patch)
            if cter_p:
                net_charge += cter_p.charge_shift
                applied.append({"index": len(residues) - 1 if residues else 0,
                               "original_resname": residues[-1] if residues else "?",
                               "patch_id": cter_patch, "term": "C"})

        return {
            "pH": pH,
            "protonation_count": len(residue_changes),
            "residue_changes": residue_changes,
            "modifications_applied": len(applied),
            "modifications": applied,
            "nter_patch": nter_patch,
            "cter_patch": cter_patch,
            "net_charge": net_charge,
        }
    except Exception:
        logger.exception("Unhandled error in apply-modifications")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# API: options

@app.get("/api/options")
async def api_options():
    """Return all available choices for the UI dropdowns."""
    from gmxbuilder.modules.forcefield.gaff_backend import gaff_available
    from gmxbuilder.modules.forcefield.catalog import get_force_field_profile
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability
    from gmxbuilder.modules.forcefield.lipid_policy import (
        gaff_lipid_capability,
        lipid_has_rtp,
    )
    from gmxbuilder.modules.solvation.water_models import supported_force_fields

    gaff_ready = gaff_available()

    def _lipid_option(name: str) -> dict:
        lipid = LipidRegistry.get(name)
        sources = []
        lipid21_supported, _lipid21_reason = lipid21_capability(name)
        if lipid21_supported:
            sources.append("lipid21")
        gaff_supported, gaff_reason = gaff_lipid_capability(name)
        if gaff_ready and gaff_supported:
            sources.append("gaff2")
        if lipid_has_rtp(name, "charmm36m"):
            sources.append("charmm36m")
        if lipid_has_rtp(name, "charmm36"):
            sources.append("charmm36")
        source_labels = {
            "lipid21": "Amber Lipid21 v1.0 (exact)",
            "gaff2": "Amber/GAFF2",
            "charmm36m": "CHARMM36m RTP",
            "charmm36": "CHARMM36 RTP",
        }
        return {
             "name": name, "common_name": lipid.common_name,
             "category": lipid.category,
             "formula": lipid.formula,
             "headgroup": lipid.headgroup,
             "tail1": list(lipid.tail1),
             "tail2": list(lipid.tail2),
             "area_per_lipid": lipid.area_per_lipid,
             "charge": lipid.charge,
             "thickness": lipid.bilayer_thickness,
             "smiles": lipid.smiles,
             "parameterizations": sources,
             "parameterization": " + ".join(source_labels[item] for item in sources)
                 if sources else "Unavailable",
             "gaff2_unavailable_reason": gaff_reason if not gaff_supported else "",
        }

    return {
        "lipids": [_lipid_option(name) for name in LipidRegistry.list()],
        "lipid_categories": {
            cat: {"label": CATEGORY_NAMES.get(cat, cat), "lipids": names}
            for cat, names in LipidRegistry.list_by_category().items()
        },
        "water_models": [
            {"name": n, "full_name": WaterRegistry.get(n).full_name,
             "n_atoms": WaterRegistry.get(n).n_atoms,
             "supported_force_fields": supported_force_fields(n)}
            for n in WaterRegistry.list()
        ],
        "solvents": [
            # Water models (full molecular generation supported)
            {"name": "tip3p", "label": "TIP3P Water", "category": "water", "density": 0.998},
            {"name": "spc",   "label": "SPC Water",   "category": "water", "density": 0.978},
            {"name": "spce",  "label": "SPC/E Water",  "category": "water", "density": 0.998},
            {"name": "tip4p", "label": "TIP4P Water",  "category": "water", "density": 0.997},
            # Organic solvents (ITP bundled in OPLS-AA, geometry TBD)
            {"name": "methanol",    "label": "Methanol (MeOH)",       "category": "organic", "density": 0.791},
            {"name": "ethanol",     "label": "Ethanol (EtOH)",        "category": "organic", "density": 0.789},
            {"name": "1propanol",   "label": "1-Propanol (PrOH)",     "category": "organic", "density": 0.803},
        ],
        "force_fields": [
            {
                **get_force_field_profile(n).as_dict(),
                "version": ForceFieldRegistry.get(n).version,
                "water_model": ForceFieldRegistry.get(n).water_model,
            }
            for n in ForceFieldRegistry.list()
        ],
        "gaff2_available": gaff_ready,
    }


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
        data = await request.json()
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
                "DNA/RNA polymers are currently supported only by the Solution "
                "Solvator workflow"
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
        data = await request.json()
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

        suggestions = await asyncio.to_thread(compute)
        return {"status": "ok", "pH": target_pH, "suggestions": suggestions}
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


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
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups
        groups = molecule_groups(System.load_checkpoint(checkpoint))
    except (OSError, ValueError, KeyError) as exc:
        return JSONResponse({"error": f"Could not inspect input checkpoint: {exc}"}, status_code=400)
    if name not in groups:
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
        template = prepare_cgenff_molecule(
            name, mol2_path, stream_path, selected_ff, package_dir / "generated",
        )
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


# ---- PDB auto-cleanup ----

# Map non-standard amino acid names to standard ones
_NONSTANDARD_AA_MAP = {
    # Histidine protonation states
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    # Cysteine variants
    "CYM": "CYS", "CYX": "CYS",
    # Aspartate / Glutamate protonation
    "ASH": "ASP", "GLH": "GLU",
    # Lysine neutral
    "LYN": "LYS",
    # Selenomethionine
    "MSE": "MET",
    # Phosphorylated (keep as-is for PTM detection)
    # "SEP": "SER", "TPO": "THR", "PTR": "TYR",
}

# Residue names for water and common solvents
_WATER_RESNAMES = {"HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD"}

# Hydrogen atom names (start with H or are pure H)
def _is_hydrogen(atom_name: str, element: str) -> bool:
    name = atom_name.strip()
    if not name:
        return False
    if element.strip().upper() == "H":
        return True
    # Common hydrogen naming: H, HA, HB, HG*, HD*, HE*, HZ*, HH*, 1H, 2H, etc.
    if len(name) >= 1 and name[0] == "H" and len(name) <= 4:
        # H, HA, HB1, HG21 etc. — but not HE (helium) or HG (mercury)
        if name.upper() not in ("HE", "HG", "HF", "HO", "HS"):
            return True
    return False


def _filter_pdb_for_display(pdb_text: str) -> str:
    """Lightweight text filter — strip water and hydrogen lines for the 3D viewer.

    This is a *display* filter only.  Actual structure cleaning is done
    inside PDBInputModule.run() so the checkpoint receives the same
    treatment regardless of whether the user came through the web UI or
    the CLI.
    """
    lines: list[str] = []
    for line in pdb_text.split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            lines.append(line)
            continue
        resname = line[17:20].strip() if len(line) > 20 else ""
        atom_name = line[12:16].strip() if len(line) > 16 else ""
        element = line[76:78].strip() if len(line) >= 78 else ""
        # Skip water
        if resname.upper() in _WATER_RESNAMES:
            continue
        # Skip hydrogens
        if _is_hydrogen(atom_name, element):
            continue
        lines.append(line)
    return "\n".join(lines)



def _detect_disulfides(pdb_path: Path) -> list[tuple[str, int, str, str, int, str]]:
    """Find CYS SG-SG pairs within disulfide bond distance (< 2.5 Å)."""
    cys_atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            if resname == "CYS" and atom_name == "SG":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chain = line[21:22].strip()
                    resid = int(line[22:26].strip())
                    cys_atoms.append((x, y, z, chain, resid))
                except (ValueError, IndexError):
                    continue

    pairs = []
    n = len(cys_atoms)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, zi, chi, ridi = cys_atoms[i]
            xj, yj, zj, chj, ridj = cys_atoms[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
            if dist < 2.5:
                pairs.append((chi, ridi, "CYS", chj, ridj, "CYS"))
    return pairs


def _apply_disulfide_rename(pdb_path: Path, pairs: list) -> None:
    """Rename paired CYS residues to CYX in the PDB file."""
    rename_set = set()
    for ch, rid, _, ch2, rid2, _ in pairs:
        rename_set.add((ch, rid))
        rename_set.add((ch2, rid2))

    with open(pdb_path) as fh:
        lines = fh.readlines()

    with open(pdb_path, "w") as fh:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                try:
                    resid = int(line[22:26].strip())
                except ValueError:
                    resid = 0
                if resname == "CYS" and (chain, resid) in rename_set:
                    line = line[:17] + "CYX" + line[20:]
            fh.write(line)


def _auto_clean_pdb(input_path: Path, output_path: Path) -> None:
    """Clean a PDB file for downstream processing.
    
    - Renames non-standard amino acids to standard names
    - Removes water molecules and common solvents
    - Removes hydrogen atoms (will be added back by topology builder)
    - Preserves all other HETATM records (ligands, cofactors)
    """
    renamed = 0
    stripped_water = 0
    stripped_h = 0
    kept = 0
    
    with open(input_path) as fh_in, open(output_path, "w") as fh_out:
        for line in fh_in:
            if not line.startswith(("ATOM", "HETATM")):
                # Pass through non-atom records (REMARK, CRYST1, TER, etc.)
                fh_out.write(line)
                continue
            
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            
            # Strip water
            if resname.upper() in _WATER_RESNAMES:
                stripped_water += 1
                continue
            
            # Strip hydrogens
            if _is_hydrogen(atom_name, element):
                stripped_h += 1
                continue
            
            # Rename non-standard residues
            new_name = _NONSTANDARD_AA_MAP.get(resname.upper())
            if new_name and line.startswith("ATOM"):
                # Replace resname in columns 18-20
                line = line[:17] + f"{new_name:>3s}" + line[20:]
                renamed += 1
            
            fh_out.write(line)
            kept += 1
    
    # If nothing was kept, don't create empty file
    if kept == 0:
        output_path.unlink(missing_ok=True)
        return

    # ---- Center protein at origin ----
    # Read back the cleaned file, compute centroid of ATOM records, translate to origin
    atom_coords = []
    with open(output_path) as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_coords.append((x, y, z))
            except (ValueError, IndexError):
                pass
    if atom_coords:
        coords = np.array(atom_coords)
        centroid = coords.mean(axis=0)
        # Rewrite with centered coordinates
        with open(output_path, "w") as fh:
            for line in lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        x = float(line[30:38]) - centroid[0]
                        y = float(line[38:46]) - centroid[1]
                        z = float(line[46:54]) - centroid[2]
                        line = (line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
                    except (ValueError, IndexError):
                        pass
                fh.write(line)

    # ---- Detect disulfide bonds ----
    disulfide_pairs = _detect_disulfides(output_path)
    if disulfide_pairs:
        _apply_disulfide_rename(output_path, disulfide_pairs)
        logger.info("PDB cleanup: %d residues renamed, %d waters removed, "
                     "%d hydrogens removed, %d atoms kept, centered at origin, "
                     "%d disulfide bond(s) detected",
                     renamed, stripped_water, stripped_h, kept, len(disulfide_pairs))
    else:
        logger.info("PDB cleanup: %d residues renamed, %d waters removed, "
                     "%d hydrogens removed, %d atoms kept, centered at origin",
                     renamed, stripped_water, stripped_h, kept)

_STRUCTURE_SUFFIX_FORMATS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
}


def _structure_upload_suffix(filename: str) -> tuple[str, bool]:
    """Return the declared structure format and whether it is gzip-compressed."""
    safe_name = Path(filename or "").name
    lower = safe_name.lower()
    compressed = lower.endswith(".gz")
    inner_name = safe_name[:-3] if compressed else safe_name
    declared = _STRUCTURE_SUFFIX_FORMATS.get(Path(inner_name).suffix.lower())
    if declared is None:
        raise ValueError(
            "Accepted structure formats are .pdb, .ent, .cif, .mmcif, "
            "and their .gz variants"
        )
    return declared, compressed


def _prepare_structure_upload(
    filename: str,
    content: bytes,
    max_bytes: int,
) -> tuple[str, bytes, str, list[str]]:
    """Bound decompression, detect content format, and choose a canonical name."""
    declared, compressed = _structure_upload_suffix(filename)
    if compressed:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as stream:
                content = stream.read(max_bytes + 1)
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise ValueError("The uploaded .gz file is not a valid gzip stream") from exc
        if len(content) > max_bytes:
            raise ValueError("Decompressed structure exceeds the upload size limit")
    if not content:
        raise ValueError("The uploaded structure file is empty")
    if b"\x00" in content[: 1024 * 1024]:
        raise ValueError("The uploaded structure contains binary data")

    sample = content[: 2 * 1024 * 1024].decode("utf-8-sig", errors="replace")
    normalized = sample.replace("\r\n", "\n").replace("\r", "\n")
    pdb_records = any(
        line[:6].strip().upper() in {"ATOM", "HETATM", "MODEL", "CRYST1"}
        for line in normalized.splitlines()
    )
    cif_records = bool(
        re.search(r"(?m)^\s*data_\S*", normalized)
        and re.search(r"(?m)^\s*_atom_site\.", normalized)
    )
    detected = "cif" if cif_records and not pdb_records else "pdb" if pdb_records else declared
    warnings: list[str] = []
    if detected != declared:
        warnings.append(
            f"File content was detected as {detected.upper()} despite its filename; "
            "content detection was used."
        )

    original = Path(filename[:-3] if compressed else filename).name
    stem = Path(original).stem or "structure"
    canonical_name = f"{stem}.{'cif' if detected == 'cif' else 'pdb'}"
    return canonical_name, content, detected, warnings


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

    # Check Content-Length header first to avoid OOM from oversized uploads
    max_upload_mb = int(os.environ.get("GMXBUILDER_MAX_UPLOAD_MB", "100"))
    if not 1 <= max_upload_mb <= 500:
        max_upload_mb = 100
    MAX_UPLOAD_BYTES = max_upload_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    try:
        declared_size = int(content_length) if content_length else None
    except ValueError:
        return JSONResponse({"error": "Invalid Content-Length header"}, status_code=400)
    # Multipart framing contributes to Content-Length. The streamed file read
    # below remains the authoritative per-file limit.
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES + 1024 * 1024:
        return JSONResponse({
            "error": (
                f"File too large ({declared_size/1024/1024:.0f} MB). "
                f"Maximum is {max_upload_mb} MB."
            )
        }, status_code=413)

    # Read with size cap — read max_bytes+1 so we can detect overage
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse({
            "error": (
                f"File too large (>{max_upload_mb} MB). "
                f"Maximum is {max_upload_mb} MB."
            )
        }, status_code=413)

    try:
        stored_name, content, structure_format, format_warnings = _prepare_structure_upload(
            original_filename, content, MAX_UPLOAD_BYTES
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    is_cif = structure_format == "cif"

    from gmxbuilder.web.task_types import get_task_type_detail

    task_type_detail = get_task_type_detail(task_type)
    if (
        task_type_detail is None
        or not task_type_detail.get("enabled")
        or (
            not task_type_detail.get("requires_input", True)
            and task_type != "coarse-grained"
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
        existing_type = (
            ((existing_state or {}).get("task_type") or {}).get("id")
            or (existing_state or {}).get("task_type_id")
        )
        if task_type != "coarse-grained" or existing_type != task_type:
            return JSONResponse(
                {"error": "Existing task does not accept this structure upload"},
                status_code=409,
            )
    else:
        task = task_manager.create_task(original_filename)
        task_id = task["task_id"]
    uploaded_path = task_manager.save_uploaded_pdb(task_id, stored_name, content)
    tmp_path = uploaded_path

    # ---- CIF → PDB conversion (for display / filtering only) ----
    # The module handles its own format detection and parsing independently.
    # This converted PDB enables the chain filter, small-molecule detection,
    # and 3D viewer to work with a standard PDB-format file.
    if is_cif:
        try:
            from gmxbuilder.io.cif import CIFParser
            from gmxbuilder.io.pdb import PDBWriter
            cif_structure = CIFParser().parse(tmp_path)
            cif_pdb_path = task_manager.get_task_dir(task_id) / "converted.pdb"
            PDBWriter.write(
                cif_structure,
                cif_pdb_path,
                title="Converted from mmCIF",
                wrap_ids_for_viewer=True,
            )
            tmp_path = cif_pdb_path
        except (ParseError, UnicodeError, ValueError) as exc:
            logger.warning("mmCIF upload rejected for task %s: %s", task_id, exc)
            message = _redact_server_paths(str(exc)).replace("\n", " ")[:400]
            return JSONResponse(
                {"error": f"Failed to parse structure file: {message}"},
                status_code=400,
            )

    # Structure cleaning is now handled inside PDBInputModule — the server
    # layer only does transport.  We apply a lightweight text filter to the
    # PDB content returned to the 3D viewer so it renders cleanly.

    try:
        # ---- Step 1: Validate ----
        from gmxbuilder.io.pdb import PDBValidator
        validation = PDBValidator.validate(tmp_path)
        validation["warnings"] = format_warnings + validation["warnings"]
        if not validation["valid"]:
            return JSONResponse({
                "error": "PDB validation failed",
                "validation_errors": validation["errors"],
                "validation_warnings": validation["warnings"],
            }, status_code=400)

        # ---- Step 2: Parse ----
        parser = PDBParser()
        structure = parser.parse(tmp_path)
        # tmp_path is always PDB-format at this point (converted from CIF
        # above if needed).  Read the file text for display filtering.
        pdb_text = tmp_path.read_text(encoding="utf-8", errors="replace")
        # Lightweight text filter for cleaner 3D viewer display.
        # Actual structure cleaning is done inside PDBInputModule.run().
        display_text = _filter_pdb_for_display(pdb_text)

        from collections import Counter
        res_counts = Counter(structure.resnames)
        protein_res = [r for r in res_counts if r in PDBInputModule._PROTEIN_RESNAMES]

        # Build per-chain sequences with residue numbering
        sequences = _extract_sequences(structure)
        # Only include chains that contain protein residues (small-molecule-only
        # chains appear in the Small Molecules section, not as protein chains)
        chains = [s["chain_id"] for s in sequences if s.get("chain_id", "").strip()]
        # Fallback: if the PDB has no explicit chain IDs (all blank), use "A"
        if not chains and sequences:
            chains = ["A"]
            for s in sequences:
                s["chain_id"] = "A"

        # ---- Step 3: Detect small molecules ----
        small_molecules = PDBValidator.detect_small_molecules(tmp_path)

        # ---- Step 4: Precompute PROPKA in background ----
        if task_type != "coarse-grained":
            _schedule_propka_precompute(str(tmp_path))

        # Update task state
        task_manager.update_state(task_id, {
            "task_type": task_type_detail,
            "task_type_id": task_type,
            "uploaded_structure_name": uploaded_path.name,
            "uploaded_structure_format": structure_format,
            "pdb_info": {
                "filename": original_filename,
                "num_atoms": structure.num_atoms,
                "chains": sorted(chains),
                "box_nm": [round(v, 3) for v in structure.dimensions().tolist()],
                "small_molecules": small_molecules,
            },
            "current_step": "input",
        })

        return {
            "task_id": task_id,
            "filename": original_filename,
            "structure_format": "mmCIF" if is_cif else "PDB",
            "num_atoms": structure.num_atoms,
            "residues": dict(res_counts.most_common(20)),
            "protein_residues": sorted(protein_res),
            "chains": sorted(chains),
            "box_nm": [round(v, 3) for v in structure.dimensions().tolist()],
            "pdb_content": display_text if display_text else pdb_text,
            "sequences": sequences,
            "validation_warnings": validation["warnings"],
            "small_molecules": small_molecules,
        }
    except (ParseError, UnicodeError, ValueError) as exc:
        logger.warning("Structure upload rejected for task %s: %s", task_id, exc)
        message = _redact_server_paths(str(exc)).replace("\n", " ")[:400]
        return JSONResponse(
            {"error": f"Failed to parse structure file: {message}"},
            status_code=400,
        )
    except Exception:
        logger.exception("Upload PDB failed")
        return JSONResponse({"error": "Failed to process structure file"}, status_code=400)
    # NOTE: tmp_path is intentionally kept alive — cleaned up by task TTL expiry


# ---------------------------------------------------------------------------
# API: build

# Build queue
# ---------------------------------------------------------------------------
_build_queue: list[tuple[str, dict]] = []  # [(task_id, data), ...]
_queue_lock = threading.Lock()
_build_admission_lock = threading.Lock()
_queue_event: asyncio.Event | None = None  # created in the active app event loop
_MAX_QUEUED_BUILDS = _positive_environment_integer(
    "GMXBUILDER_MAX_QUEUED_BUILDS", 32
)
_queue_enqueued_at: dict[str, float] = {}
_build_started_at: dict[str, float] = {}
_build_duration_history: deque[float] = deque(maxlen=50)


def _baseline_build_seconds() -> float:
    try:
        value = float(os.environ.get("GMXBUILDER_EXPECTED_BUILD_SECONDS", "45"))
    except ValueError:
        value = 45.0
    return value if math.isfinite(value) and value > 0 else 45.0


def _typical_build_seconds() -> float:
    """Return a robust recent finalization duration for queue estimates."""
    if not _build_duration_history:
        return _baseline_build_seconds()
    ordered = sorted(list(_build_duration_history))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _queue_estimate(position: int) -> dict[str, object]:
    """Estimate when a one-based queued position can acquire a task slot."""
    typical = _typical_build_seconds()
    now = time.time()
    with _tasks_lock:
        active_ids = tuple(_building_tasks)
    remaining = [
        max(1.0, typical - max(0.0, now - _build_started_at.get(task_id, now)))
        for task_id in active_ids
    ]
    # Simulate FIFO assignment to the first available slot. Free slots start
    # at t=0; occupied slots start at their estimated remaining duration.
    availability = remaining + [
        0.0
        for _ in range(max(0, _MAX_CONCURRENT_BUILDS - len(remaining)))
    ]
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
            # Double-check: api_build may have already started this task
            # (race between queue-pop and building_tasks-add).
            with _tasks_lock:
                if task_id in _building_tasks:
                    # Already started by api_build — release slot and skip
                    _build_semaphore.release()
                    continue
                _building_tasks.add(task_id)
                _build_started_at[task_id] = time.time()
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
            task = loop.run_in_executor(
                _get_executor(), _run_queued_build, data, task_id
            )
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
    data: dict[str, Any] = await request.json()
    task_id = data.get("task_id", "")

    # Validate task_id format to prevent path traversal
    task_id = _validate_task_id(task_id)

    # Bind the request to the workflow persisted when this task was created.
    state = task_manager.get_state(task_id)
    if state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    persisted_task_type = (
        (state.get("task_type") or {}).get("id")
        or state.get("task_type_id")
        or "membrane-bilayer"
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
        "membrane-bilayer", "pure-membrane", "solvator", "coarse-grained"
    }:
        return JSONResponse(
            {"error": f"Workflow {persisted_task_type!r} is not available for finalization"},
            status_code=400,
        )

    # Reject malformed expert MDP input before the task consumes a build slot.
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        return JSONResponse(
            {"error": "Build modules must be an object"}, status_code=400
        )
    simparams = modules.get("simparams", {})
    runner = _get_step_runner(task_id, persisted_task_type)
    try:
        if persisted_task_type == "coarse-grained":
            from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol

            checked = runner.load_system("cg_system")
            if checked is None:
                raise ValueError("Final CG System Check is missing")
            if not checked.metadata.get("system_confirmed"):
                raise ValueError(
                    "Inspect and confirm the exact Final CG System before building"
                )
            has_membrane = checked.metadata.get("cg_environment") == "bilayer"
            simparams = normalize_protocol(simparams, has_membrane=has_membrane)
            include_solvent = bool(
                (checked.metadata.get("cg_solvation_config") or {}).get(
                    "include_solvent", True
                )
            )
            export_config = modules.setdefault("export", {})
            if not isinstance(export_config, dict):
                raise ValueError("export settings must be an object")
            export_config["write_mdp"] = include_solvent
        else:
            forcefield_config = modules.get("forcefield", {})
            if not isinstance(forcefield_config, dict):
                forcefield_config = {}
            persisted_forcefield = state.get("step_forcefield_config", {})
            if not isinstance(persisted_forcefield, dict):
                persisted_forcefield = {}
            ff_name = str(
                forcefield_config.get("name")
                or persisted_forcefield.get("name")
                or "amber14sb"
            )
            simulation_context = {
                "force_field": ff_name,
                "force_field_family": (
                    "charmm" if ff_name.lower().startswith("charmm")
                    else "opls" if ff_name.lower().startswith("opls")
                    else "amber"
                ),
                "has_membrane": persisted_task_type in {
                    "membrane-bilayer", "pure-membrane"
                },
            }
            simparams = MDPWriter.normalize_simulation_config(
                simparams, simulation_context
            )
            from gmxbuilder.runtime.hardware import normalize_simulation_hardware

            normalize_simulation_hardware(simparams.get("hardware"))
    except (ModuleConfigError, TypeError, ValueError) as exc:
        return JSONResponse(
            {"error": f"Invalid simulation parameters: {exc}"}, status_code=400
        )
    modules["simparams"] = simparams

    source_step = "cg_system" if persisted_task_type == "coarse-grained" else "ions"
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
    data["task_type"] = persisted_task_type
    data["source_step"] = source_step
    task_manager.update_state(task_id, {
        "simparams": simparams,
        "step_simparams_config": simparams,
        "current_step": "simparams",
    })

    queue_pos = None
    already_queued = False
    queue_full = False
    active_count = 0
    with _build_admission_lock:
        with _tasks_lock:
            already_building = task_id in _building_tasks
        if already_building:
            return JSONResponse({
                "error": "This task is already being built."
            }, status_code=409)

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
        return JSONResponse({
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
        })

    _persist_build_status(
        task_id,
        "running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    # Dispatch build in background — return immediately so HTTP doesn't block
    with _build_logs_lock:
        _build_logs[task_id] = [f"Build starting ({active_count}/{_MAX_CONCURRENT_BUILDS} slots used)..."]
    logger.info("Build %s started immediately (%d/%d slots)", task_id, active_count, _MAX_CONCURRENT_BUILDS)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_get_executor(), _run_background_build, data, task_id)

    return JSONResponse({
        "status": "started",
        "task_id": task_id,
        "message": "Build started. Poll /api/build/{task_id}/log for progress."
    })


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
        logger.exception("Build %s failed", task_id)
        public_error = _redact_server_paths(exc)
        with _tasks_lock:
            _tasks[task_id] = {"status": "failed", "progress": 0, "result": None,
                               "error": public_error}
        with _build_logs_lock:
            _build_logs.setdefault(task_id, []).append(
                f"✗ Finalization failed: {public_error}"
            )
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
        if started_at is not None:
            duration = max(0.001, time.time() - started_at)
            _build_duration_history.append(duration)
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
        export_config["system_name"] = (
            simparams.get("system_name", "martini3_system")
            if task_type_id == "coarse-grained"
            else data.get("system_name", "system")
        )

        with _tasks_lock:
            _tasks[task_id]["status"] = "running"
            _tasks[task_id]["progress"] = 20
        with _build_logs_lock:
            _build_logs[task_id] = [
                f"Finalizing exact {source_step} Check checkpoint...",
                "Coordinate-building modules will not be re-run.",
            ]

        if task_type_id != "coarse-grained":
            _require_task_custom_lipids_ready(task_id)
        lipid_scope = (
            nullcontext()
            if task_type_id == "coarse-grained"
            else task_custom_lipid_scope(task_manager.get_task_dir(task_id))
        )
        with task_manager.active_task(task_id), lipid_scope:
            result = runner.finalize_from_checkpoint(
                source_step,
                topology_config=dict(modules_config.get("topology") or {}),
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
            build_log = [
                _redact_server_paths(line)
                for line in _build_logs.get(task_id, [])
            ]
        summary = {
            "task_id": task_id,
            "num_atoms": system.num_atoms,
            "components": [{"name": c.name, "atoms": len(c.atom_indices), "kind": c.kind.name,
                           "n_molecules": c.metadata.get("n_molecules")}
                          for c in system.components],
            "log": build_log,
            "source_checkpoint": source_step,
            "coordinate_rebuild": False,
            "package_contents": result["package_contents"],
            "download_url": f"/api/task/{task_id}/download",
        }
        return summary

    except Exception:
        logger.exception("Build %s failed", task_id)
        raise


# ---------------------------------------------------------------------------
# API: status / download

@app.get("/api/status/{task_id}")
async def api_status(task_id: str):
    task_id = _validate_task_id(task_id)
    t = _tasks.get(task_id)
    if t is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({
        "task_id": task_id,
        "status": t["status"],
        "progress": t["progress"],
        "error": (
            _redact_server_paths(t.get("error"))
            if t.get("error") is not None else None
        ),
    })


@app.get("/api/download/{task_id}")
async def api_download(task_id: str):
    task_id = _validate_task_id(task_id)
    t = _tasks.get(task_id)
    ready = t is not None and t.get("status") == "completed"
    if not ready:
        state = task_manager.get_state(task_id) or {}
        persisted = state.get("build_status") or {}
        ready = isinstance(persisted, dict) and persisted.get("status") == "completed"
    if not ready:
        return JSONResponse({"error": "Task not ready or not found"}, status_code=404)
    zip_path = _authoritative_task_zip(task_id)
    if zip_path is None or not zip_path.exists():
        return JSONResponse({"error": "ZIP file not found on disk"}, status_code=404)
    return FileResponse(str(zip_path), media_type="application/zip",
                        filename=f"gmxbuilder_{task_id}.zip")


# ---------------------------------------------------------------------------
# API: Step-based incremental checkpoint build
# ---------------------------------------------------------------------------

_step_runners: dict[str, StepRunner] = {}
_step_runners_lock = threading.Lock()


def _get_step_runner(task_id: str, pipeline_type: str = "membrane-bilayer") -> StepRunner:
    """Get or create a StepRunner for the given task."""
    from gmxbuilder.pipeline.step_executor import StepRunner
    with _step_runners_lock:
        if task_id not in _step_runners:
            task_dir = task_manager.get_task_dir(task_id)
            _step_runners[task_id] = StepRunner(task_dir, pipeline_type)
        return _step_runners[task_id]


def _require_task_custom_lipids_ready(task_id: str) -> list[dict]:
    """Reject progression while any submitted molecule is not validated."""
    records = CustomLipidStore(
        task_manager.get_task_dir(task_id)
    ).list_public()
    unavailable = [
        record for record in records if record.get("state") != "ready"
    ]
    if unavailable:
        details = ", ".join(
            f"{record.get('name')} ({record.get('state')}: "
            f"{record.get('phase')})"
            for record in unavailable
        )
        raise ValueError(
            "Custom lipid calculation must finish successfully before this "
            f"task can proceed: {details}"
        )
    return records


def _trusted_membrane_config(task_id: str, config: dict) -> dict:
    """Bind every non-built-in membrane entry to this task's READY record."""
    trusted = copy.deepcopy(config)
    definitions = CustomLipidStore(
        task_manager.get_task_dir(task_id)
    ).load_definitions()
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
                f"Lipid {name!r} is not in the standard library and does not "
                "belong to this task"
            )
        status = CustomLipidStore(
            task_manager.get_task_dir(task_id)
        ).load_status(name)
        if status.get("state") != "ready":
            raise ValueError(
                f"Task-scoped lipid {name} is not ready ({status.get('state')})"
            )
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
        has_checkpoint = runner.has_checkpoint(s)
        preview_available = has_checkpoint
        confirmed = None
        if pipeline_type == "coarse-grained" and s == "cg_system" and has_checkpoint:
            checked_system = runner.load_system(s)
            confirmed = bool(
                checked_system is not None
                and checked_system.metadata.get("system_confirmed")
            )
            # Frontend completion means scientific Check plus explicit WYSIWYG
            # confirmation; retain preview availability as a separate field.
            has_checkpoint = confirmed
        step_status.append({
            "name": s,
            "has_checkpoint": has_checkpoint,
            "preview_available": preview_available,
            "confirmed": confirmed,
        })

    return {
        "task_id": task_id,
        "pipeline_type": pipeline_type,
        "steps": step_status,
    }


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
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"error": "Request body must be valid JSON"}, status_code=400
        )
    if not isinstance(data, dict):
        return JSONResponse(
            {"error": "Request body must be a JSON object"}, status_code=400
        )
    if "config" in data:
        config = data["config"]
    else:
        modules = data.get("modules", {})
        if not isinstance(modules, dict):
            return JSONResponse(
                {"error": "modules must be a JSON object"}, status_code=400
            )
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
                    "mol2_path": str(mol2_path), "str_path": str(str_path),
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
        pipeline_type == "coarse-grained"
        and config.get("include_protein") is False
    )
    if step_name == "input" and not protein_free_cg:
        try:
            pdb_path = _resolve_input_pdb(task_id)
        except ValueError:
            return JSONResponse({"error": "No PDB file found for input step"}, status_code=400)

    # Build initial system for first step
    from gmxbuilder.core.system import System
    from gmxbuilder.core.structure import Structure
    import numpy as np

    seed = data.get("seed", task_state.get("seed", 42))
    initial = System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
        metadata={"seed": seed},
    )

    # Run step in thread pool (all steps are synchronous)
    loop = asyncio.get_event_loop()
    def _run_scoped_step():
        with task_manager.active_task(task_id), task_custom_lipid_scope(
            task_manager.get_task_dir(task_id)
        ):
            return runner.run_step(
                step_name, config, initial_system=initial, pdb_path=pdb_path
            )

    result = await loop.run_in_executor(_get_executor(), _run_scoped_step)

    if result["status"] == "ok":
        # Update task state
        state_update = {
            "current_step": step_name,
            f"step_{step_name}_config": config,
        }
        if step_name == "input":
            state_update["input_modifications"] = dict(
                (result.get("metrics") or {}).get("input_modifications") or {}
            )
            state_update["input_sequences"] = list(
                (result.get("metrics") or {}).get("input_sequences") or []
            )
        if step_name == "structure":
            state_update["modification_geometry"] = list(
                (result.get("metrics") or {}).get("modification_geometry") or []
            )
        if step_name == "orient":
            saved_orientation = dict(
                (result.get("metrics") or {}).get("orientation") or {}
            )
            saved_orientation["method"] = config.get("method", "ppm")
            state_update["orient"] = saved_orientation
        if step_name == "cg_system":
            state_update["cg_system_confirmed"] = False
        task_manager.update_state(task_id, state_update)

    return JSONResponse(_public_step_result(task_id, result))


@app.post("/api/step/{task_id}/cg_system/confirm")
async def api_confirm_cg_system(task_id: str):
    """Confirm an already-built exact CG checkpoint without rebuilding it."""
    task_id = _validate_task_id(task_id)
    task_state = task_manager.get_state(task_id)
    task_type = (
        ((task_state or {}).get("task_type") or {}).get("id")
        or (task_state or {}).get("task_type_id")
    )
    if task_state is None:
        return JSONResponse({"error": "Task not found or expired"}, status_code=404)
    if task_type != "coarse-grained":
        return JSONResponse({"error": "This task is not a Martini 3 workflow"}, status_code=409)
    runner = _get_step_runner(task_id, task_type)
    system = runner.load_system("cg_system")
    if system is None:
        return JSONResponse(
            {"error": "Build and inspect the Final CG System first"}, status_code=409
        )
    scientific_check = system.metadata.get("cg_scientific_check") or {}
    if scientific_check.get("passed") is not True:
        return JSONResponse(
            {"error": "The final CG scientific quality gate has not passed"},
            status_code=409,
        )
    system.metadata["system_confirmed"] = True
    system.save_checkpoint(runner.step_dir("cg_system"))
    task_manager.update_state(task_id, {
        "cg_system_confirmed": True,
        "current_step": "cg_system",
    })
    return {
        "status": "ok",
        "task_id": task_id,
        "confirmed": True,
        "coordinate_checkpoint": "cg_system",
    }


@app.get("/api/step/{task_id}/{step_name}/viewer.pdb")
async def api_step_viewer_pdb(task_id: str, step_name: str):
    """Return the viewer PDB for a completed step."""
    task_id = _validate_task_id(task_id)
    pipeline_type = "membrane-bilayer"  # default
    task_state = task_manager.get_state(task_id)
    if task_state:
        pipeline_type = (task_state.get("task_type") or {}).get("id") or "membrane-bilayer"

    runner = _get_step_runner(task_id, pipeline_type)
    pdb_path = runner.step_dir(step_name) / "viewer.pdb"
    # Martinize2 writes authoritative CONECT records for mapped beads.  Keep
    # those bonds in the mapping preview instead of the generic checkpoint PDB,
    # which intentionally stores coordinates only.
    if pipeline_type == "coarse-grained" and step_name == "cg_mapping":
        mapped_path = runner.step_dir(step_name) / "martinize" / "cg_protein.pdb"
        step_root = runner.step_dir(step_name).resolve()
        resolved = mapped_path.resolve()
        if step_root in resolved.parents and resolved.is_file() and not resolved.is_symlink():
            pdb_path = resolved
    if not pdb_path.exists():
        return JSONResponse({"error": f"No viewer PDB for step '{step_name}'"}, status_code=404)

    return FileResponse(str(pdb_path), media_type="chemical/x-pdb",
                       filename=f"{step_name}_viewer.pdb")


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

    zip_path = max(zip_files, key=lambda p: p.stat().st_mtime_ns)
    return FileResponse(str(zip_path), media_type="application/zip",
                       filename=f"gmxbuilder_{task_id}.zip")


# Import for step runner
from gmxbuilder.pipeline.step_executor import StepRunner, get_pipeline_steps  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers

def _extract_sequences(structure) -> list[dict]:
    """Extract per-chain residue sequences with 1-based residue numbering.

    Protein and nucleic-acid polymer residues are included.  Independent small
    molecules, water, and ions are excluded and rendered separately.

    Returns a list of dicts: {chain_id, sequence: [{resname, resid, is_protein}]}
    """
    from collections import OrderedDict
    from gmxbuilder.io.pdb import _PROTEIN_RESNAMES, _SOLVENT_IONS, _LIPID_DETERGENT
    from gmxbuilder.modules.nucleic_acid.support import (
        classify_nucleic_residue,
        nucleic_polymer_residues,
    )

    # Build a set of residue names that should NOT appear in chain sequences
    _NON_PROTEIN = _SOLVENT_IONS | _LIPID_DETERGENT

    chain_data: dict[str, list[dict]] = OrderedDict()
    seen: dict[str, set] = {}  # chain -> set of seen (resname, resid) pairs

    n = structure.num_atoms
    nucleic_residues = nucleic_polymer_residues(structure)
    for i in range(n):
        chain = structure.chain_ids[i] if i < len(structure.chain_ids) else "?"
        resname = structure.resnames[i] if i < len(structure.resnames) else "UNK"
        resid = structure.resids[i] if i < len(structure.resids) else i + 1

        # Skip solvent-like components; decide polymer identity below.
        if resname in _NON_PROTEIN:
            continue
        is_protein = resname in _PROTEIN_RESNAMES
        is_nucleic = (str(chain), int(resid)) in nucleic_residues
        nucleic_type = (
            nucleic_residues[(str(chain), int(resid))]
            if is_nucleic else classify_nucleic_residue(resname)
        )
        if not is_protein and not is_nucleic:
            continue

        if chain not in chain_data:
            chain_data[chain] = []
            seen[chain] = set()

        key = (resname, resid)
        if key not in seen[chain]:
            seen[chain].add(key)
            record = {
                "resname": resname,
                "resid": resid,
                "is_protein": is_protein,
            }
            if is_nucleic:
                record["is_nucleic"] = True
                record["polymer_type"] = nucleic_type or "modified"
            chain_data[chain].append(record)

    # Convert to list of dicts
    result = []
    for chain_id, residues in chain_data.items():
        if not residues:
            continue
        # Preserve coordinate encounter order.  The StructureProcessor uses
        # this same order for residue indices; sorting by residue number here
        # made frontend modifications target a different residue in PDB files
        # with insertion codes or non-monotonic author numbering.
        result.append({
            "chain_id": chain_id,
            "length": len(residues),
            "residues": residues,
        })
    return result


# ---------------------------------------------------------------------------
# Cleanup import (avoid circular)
from gmxbuilder.modules.input.pdb_input import PDBInputModule  # noqa: E402

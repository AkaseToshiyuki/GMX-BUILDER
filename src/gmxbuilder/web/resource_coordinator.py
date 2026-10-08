"""Admission, persistent Web operations and pressure-driven task reclamation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse

from gmxbuilder.web.resource_policy import ResourcePolicy
from gmxbuilder.web.resource_queue import ResourceQueue
from gmxbuilder.web.resource_workers import SystemdWorkers, available_memory
from gmxbuilder.web.server_parts.http_limits import (
    BodyLimitError,
    body_contract,
    positive_body_limit,
    positive_seconds,
)

TASK_ID = re.compile(r"[0-9a-f]{12}(?:[0-9a-f]{20})?\Z")


async def _managed_upload_form(request):
    """Bound form errors and close partial spools, including parser failures."""
    try:
        from python_multipart.exceptions import MultipartParseError
    except ModuleNotFoundError:  # Older supported python-multipart distributions.
        from multipart.exceptions import MultipartParseError
    from starlette.formparsers import MultiPartException, MultiPartParser

    class CompleteFormParser(MultiPartParser):
        complete = False

        def on_end(self):
            self.complete = True

    parser = CompleteFormParser(request.headers, request.stream())
    succeeded = False
    try:
        form = await parser.parse()
        if not parser.complete:
            raise MultiPartException("Incomplete multipart body")
        succeeded = True
        return form
    except (MultipartParseError, MultiPartException) as exc:
        raise BodyLimitError(400, "Malformed multipart upload; send a complete form body") from exc
    finally:
        if not succeeded:
            # Starlette closes these for MultiPartException, but not low-level
            # multipart errors or cancellation after a file has been opened.
            for spool in parser._files_to_close_on_error:
                spool.close()


HEAVY_POSTS = {
    "upload-pdb",
    "orient-ppm",
    "orient-preview",
    "cg-orient-preview",
    "preview-pdb",
    "filter-pdb",
    "protonate",
    "apply-modifications",
    "forcefield-compatibility",
    "ligand-charge-suggestions",
    "ligand-chemistry",
    "ligand-chemistry-upload",
    "cgenff-upload",
    "step",
    "build",
}


def expensive_request(method, path):
    parts = path.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "api":
        return False
    if method == "POST":
        return parts[1] in HEAVY_POSTS
    return method == "GET" and (
        (parts[1] == "step" and path.endswith(("/viewer.pdb", "/viewer.json")))
        or (parts[1] == "task" and path.endswith("/resume"))
    )


def task_from_path(path):
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and TASK_ID.fullmatch(parts[2]):
        return parts[2]
    return None


class ResourceCoordinator:
    def __init__(self, root: Path, manager, *, workers=None, require_mount=True):
        self.root = root.resolve()
        self.manager = manager
        self.policy = ResourcePolicy.from_environment()
        if manager.root.resolve() != self.root / "tasks":
            raise RuntimeError("Managed task directory must be the quota volume's tasks directory")
        if require_mount:
            mounted = False
            for line in Path("/proc/self/mountinfo").read_text().splitlines():
                left, _, right = line.partition(" - ")
                fields = left.split()
                if fields[4].replace("\\040", " ") == str(self.root):
                    mounted = right.startswith("fuse") and "gmxbuilder-quota" in right
            if not mounted:
                raise RuntimeError(
                    "Managed resource volume is not mounted; refusing unbounded work"
                )
            actual = json.loads(os.getxattr(self.root, "user.gmxbuilder.quota"))
            if actual != {
                "total_bytes": self.policy.total_storage_bytes,
                "task_bytes": self.policy.task_storage_bytes,
            }:
                raise RuntimeError("Configured storage limits differ from the active quota volume")
        self.queue = ResourceQueue(root / ".control" / "queue.sqlite3")
        self.workers = workers or SystemdWorkers(root, self.policy)
        self.pause_reason = None
        self._lock = asyncio.Lock()
        # Backpressure during HTTP upload is separate from queue length. At
        # most a bounded share of the Web memory allowance is used for multipart
        # framing; waiting sockets do not materialize their request bodies.
        upload_limit = positive_body_limit("GMXBUILDER_UPLOAD_BODY_LIMIT", 128 * 1024**2)
        self._ingress = asyncio.Semaphore(
            min(self.policy.slots, max(1, self.policy.task_memory_bytes // (4 * upload_limit)))
        )
        self._control_ingress = asyncio.Semaphore(min(8, self.policy.slots))
        self._upload_clients = {}
        self._loop_task = None
        self._leases = {}
        self._last_maintenance = 0.0
        self._health_probe = None
        self._health_checked = 0.0
        self._health_result = {"ready": False, "reason": "storage-check-pending"}

    def job_directory(self, operation):
        return self.manager.root / operation["task_id"] / ".operations" / operation["id"]

    async def start(self):
        allowed = {
            key: value
            for key, value in os.environ.items()
            if (key.startswith("GMX") or key in {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"})
            and not any(word in key for word in ("TOKEN", "PASSWORD", "AUTH_"))
        }
        environment_path = self.root / ".control" / "worker-environment.json"
        environment_path.write_text(json.dumps(allowed))
        environment_path.chmod(0o600)
        for operation in self.queue.active():
            # A scientific Check interrupted by a restart is not silently
            # replayed against possibly partial output. Its checkpoints remain.
            if operation["unit"]:
                await self.workers.stop(operation["unit"])
            self.queue.update(
                operation["id"],
                status="failed",
                finished=time.time(),
                error="Server restarted during this operation. Review and retry it.",
            )
        self._loop_task = asyncio.create_task(self.run())

    def confine_web_writes(self):
        import logging
        import sys
        import tempfile

        from gmxbuilder.web.installation_guard import prepare_installation_lock
        from gmxbuilder.web.server_parts.task_security import (
            CapabilityRedactionFilter,
            CapabilityRedactionFormatter,
        )
        from gmxbuilder.web.write_sandbox import restrict_writes

        for name in ("logs", "tmp", "cache"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
        # stderr is inherited from systemd and survives a lost quota mount.
        handler = logging.StreamHandler(sys.stderr)
        handler.addFilter(CapabilityRedactionFilter())
        handler.setFormatter(
            CapabilityRedactionFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = logging.getLogger(name)
            logger.handlers = [handler]
            logger.propagate = False
        os.environ["TMPDIR"] = tempfile.tempdir = str(self.root / "tmp")
        os.environ["XDG_CACHE_HOME"] = str(self.root / "cache")
        sys.dont_write_bytecode = True
        prepare_installation_lock()
        restrict_writes(self.root)

    async def readiness(self):
        from gmxbuilder.web.storage_health import probe_storage

        now = time.monotonic()
        if self._health_probe is None or (
            self._health_probe.done() and now - self._health_checked > 2
        ):
            self._health_checked = now
            self._health_probe = asyncio.create_task(
                asyncio.to_thread(
                    probe_storage,
                    self.root,
                    {
                        "total_bytes": self.policy.total_storage_bytes,
                        "task_bytes": self.policy.task_storage_bytes,
                    },
                )
            )
        # Keep at most one outstanding filesystem probe, even if FUSE stalls.
        done, _pending = await asyncio.wait({self._health_probe}, timeout=1)
        if not done:
            return {"ready": False, "reason": "storage-check-timeout"}
        self._health_result = self._health_probe.result()
        if not self._health_result["ready"]:
            return self._health_result
        if self._loop_task is None or self._loop_task.done():
            return {"ready": False, "reason": "coordinator-unavailable"}
        if self.pause_reason == "Resource service needs attention; new starts are paused.":
            return {"ready": False, "reason": "coordinator-unavailable"}
        return self._health_result

    async def close(self):
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        for operation in self.queue.active():
            await self.workers.stop(operation["unit"])
            self.release(operation["id"])
            self.queue.update(
                operation["id"],
                status="failed",
                finished=time.time(),
                error="Server stopped during this operation; checkpoints are retained.",
            )

    def release(self, operation_id):
        lease = self._leases.pop(operation_id, None)
        if lease:
            lease.__exit__(None, None, None)

    def free_bytes(self):
        stats = os.statvfs(self.root)
        return stats.f_bavail * stats.f_frsize

    def reserved_bytes(self):
        # Each admitted task may still grow to its entire task quota. Files
        # already written release the corresponding unused reservation.
        reserved = 0
        for operation in self.queue.active():
            path = self.manager.root / operation["task_id"]
            # Match quota accounting: one 4 KiB entry charge plus rounded file blocks.
            used = 4096
            for p in path.rglob("*"):
                st = p.lstat()
                used += 4096
                if stat.S_ISREG(st.st_mode):
                    used += ((st.st_size + 4095) // 4096) * 4096
            reserved += max(0, self.policy.task_storage_bytes - used)
        return reserved

    def make_room(self, required, *, protected=()):
        # Leave 1 MiB beyond payload reservations for filesystem bookkeeping.
        needed = required + self.reserved_bytes() + 1024 * 1024
        if self.free_bytes() >= needed:
            return True
        candidates = []
        for directory in self.manager.root.iterdir():
            if (
                not directory.is_dir()
                or directory.is_symlink()
                or not TASK_ID.fullmatch(directory.name)
                or directory.name in protected
                or self.queue.task_active(directory.name)
            ):
                continue
            try:
                state = json.loads((directory / "state.json").read_text())
                created = datetime.fromisoformat(state["created_at"]).timestamp()
            except (OSError, ValueError, KeyError, TypeError):
                continue
            candidates.append((created, directory.name))
        for _, task_id in sorted(candidates):
            if self.manager.delete_task(task_id):
                self.queue.cancel_task(task_id, "Task removed to release storage (oldest first)")
            if self.free_bytes() >= needed:
                return True
        return False

    async def tick(self):
        async with self._lock:
            now = time.time()
            initial_active = self.queue.active()
            states = await asyncio.gather(
                *(self.workers.state(op["unit"]) for op in initial_active)
            )
            for operation, state in zip(initial_active, states, strict=True):
                expired = operation["expires"] <= now
                if expired:
                    await self.workers.stop(operation["unit"])
                if expired or state not in {"active", "activating"}:
                    self.release(operation["id"])
                    latest = self.queue.get(operation["id"])
                    if latest and latest["status"] == "running":
                        directory = self.job_directory(operation)
                        success = not expired and (directory / "worker.done").is_file()
                        self.queue.update(
                            operation["id"],
                            status="completed" if success else "failed",
                            finished=now,
                            error=None
                            if success
                            else (
                                "Task expired 24 hours after creation."
                                if expired
                                else "Worker stopped before completion "
                                "(memory, storage or process failure). Existing checkpoints are "
                                "retained; reduce the system or contact the administrator."
                            ),
                        )
                    if expired:
                        self.manager.delete_task(operation["task_id"])
            # A worker can finish its queue update just before exiting. Keep
            # its file lease until systemd confirms all children have exited.
            for operation_id in list(self._leases):
                operation = self.queue.get(operation_id)
                if operation and operation["status"] != "running":
                    if await self.workers.state(operation["unit"]) not in {"active", "activating"}:
                        self.release(operation_id)
            if now - self._last_maintenance >= 5:
                self.manager.cleanup_expired()
                self.queue.prune(now)
                self._last_maintenance = now
            active = self.queue.active()
            self.pause_reason = None
            admission_bytes = 0
            if len(active) < self.policy.slots and self.queue.next() is not None:
                memory_usage = await asyncio.gather(
                    *(self.workers.memory_current(op["unit"]) for op in active)
                )
                unused_memory = sum(
                    max(0, self.policy.task_memory_bytes - used) for used in memory_usage
                )
                admission_bytes = available_memory() - unused_memory
            while len(active) < self.policy.slots:
                operation = self.queue.next()
                if not operation:
                    break
                if (
                    operation["expires"] <= now
                    or self.manager.get_state(operation["task_id"]) is None
                ):
                    self.queue.update(
                        operation["id"],
                        status="cancelled",
                        finished=now,
                        error="Task expired or was removed before this operation started.",
                    )
                    continue
                if self.queue.task_active(operation["task_id"]):
                    self.pause_reason = "Waiting for this task's current operation to finish."
                    break
                # New starts reserve their full limit until the next tick. Do not
                # spend an apparent rise in free memory on a stale usage snapshot.
                if min(admission_bytes, available_memory()) < self.policy.task_memory_bytes:
                    self.pause_reason = "Waiting for sufficient memory."
                    break
                if not self.make_room(
                    self.policy.task_storage_bytes, protected=(operation["task_id"],)
                ):
                    self.pause_reason = "Storage is full. New writes and starts are paused."
                    break
                admission_bytes -= self.policy.task_memory_bytes
                threads = max(1, self.policy.cpu_cores // (len(active) + 1))
                await self.workers.rebalance(active + [{"unit": None}])
                lease = self.manager.active_task(operation["task_id"])
                lease.__enter__()
                self._leases[operation["id"]] = lease
                unit = f"gmxbuilder-op-{operation['id']}.service"
                self.queue.update(
                    operation["id"],
                    status="running",
                    started=time.time(),
                    threads=threads,
                    unit=unit,
                )
                try:
                    operation = self.queue.get(operation["id"])
                    await self.workers.launch(operation, threads)
                except Exception:
                    self.release(operation["id"])
                    self.queue.update(
                        operation["id"],
                        status="failed",
                        finished=time.time(),
                        error="Resource isolation is unavailable; operation was not run.",
                    )
                active = self.queue.active()
            await self.workers.rebalance(active)

    async def run(self):
        import logging

        while True:
            try:
                await self.tick()
            except Exception:
                self.pause_reason = "Resource service needs attention; new starts are paused."
                logging.getLogger("gmxbuilder.web").exception("Resource coordinator tick failed")
            await asyncio.sleep(2)

    async def enqueue_request(self, request):
        contract = body_contract(request.scope)
        semaphore = self._ingress if contract.multipart else self._control_ingress
        identity = getattr(request.state, "gmxbuilder_client_identity", None)
        identity = identity or (request.client.host if request.client else "unknown")
        counted = False
        acquired = False

        async def acquire():
            nonlocal acquired
            await semaphore.acquire()
            acquired = True

        try:
            if contract.multipart:
                maximum = positive_body_limit("GMXBUILDER_CLIENT_UPLOADS", 2)
                if self._upload_clients.get(identity, 0) >= maximum:
                    return JSONResponse(
                        {"error": "Too many concurrent uploads from this client"},
                        status_code=429,
                        headers={"Retry-After": "1"},
                    )
                self._upload_clients[identity] = self._upload_clients.get(identity, 0) + 1
                counted = True
            try:
                await asyncio.wait_for(
                    acquire(), positive_seconds("GMXBUILDER_INGRESS_WAIT_TIMEOUT", 10)
                )
            except TimeoutError:
                return JSONResponse(
                    {"error": "Request admission is busy; retry shortly"},
                    status_code=503,
                    headers={"Retry-After": "1"},
                )
            return await self._enqueue_request(request)
        except BodyLimitError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        finally:
            if acquired:
                semaphore.release()
            if counted:
                remaining = self._upload_clients[identity] - 1
                if remaining:
                    self._upload_clients[identity] = remaining
                else:
                    self._upload_clients.pop(identity)

    async def _enqueue_request(self, request):
        contract = body_contract(request.scope)
        multipart, limit = contract.multipart, contract.limit
        if contract.declared is not None and contract.declared > limit:
            raise BodyLimitError(413, "Request body exceeds the endpoint byte limit")
        # Body framing is cheap and bounded. Chemical parsing happens only in
        # the admitted worker. Starlette's cached body is reused for form fields.
        chunks, size = [], 0
        stream = request.stream().__aiter__()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + positive_seconds("GMXBUILDER_BODY_TOTAL_TIMEOUT", 300)
        while True:
            timeout = min(
                positive_seconds("GMXBUILDER_BODY_IDLE_TIMEOUT", 30), deadline - loop.time()
            )
            if timeout <= 0:
                raise BodyLimitError(408, "Request body deadline exceeded")
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout)
            except StopAsyncIteration:
                break
            except TimeoutError as exc:
                raise BodyLimitError(408, "Request body deadline exceeded") from exc
            if chunk and not contract.media_allowed:
                raise BodyLimitError(415, "Unsupported media type for this endpoint")
            size += len(chunk)
            if size > limit:
                return JSONResponse(
                    {"error": "Request body exceeds the upload limit"}, status_code=413
                )
            chunks.append(chunk)
        body = b"".join(chunks)
        request._body = body
        data = {}
        semantic_body = body
        if multipart:
            from starlette.datastructures import UploadFile

            form = await _managed_upload_form(request)
            try:
                data["task_id"] = form.get("task_id", "")
                parts = []
                for name, value in form.multi_items():
                    if isinstance(value, UploadFile):
                        digest = hashlib.sha256()
                        while chunk := await value.read(65536):
                            digest.update(chunk)
                        parts.append([name, value.filename, digest.hexdigest()])
                    else:
                        parts.append([name, value])
                semantic_body = json.dumps(parts, sort_keys=True).encode()
            finally:
                await form.close()
        elif body:
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError
                semantic_body = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
            except (ValueError, UnicodeError):
                return JSONResponse(
                    {"error": "Request body must be a JSON object"}, status_code=400
                )
        task_id = task_from_path(request.url.path) or data.get("task_id")
        if task_id and (not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id)):
            return JSONResponse({"error": "Invalid task ID"}, status_code=400)
        if task_id and data.get("task_id") and data["task_id"] != task_id:
            return JSONResponse(
                {"error": "Task IDs in the path and body must match"}, status_code=400
            )
        async with self._lock:
            if not self.make_room(size + 65536, protected=(task_id,) if task_id else ()):
                return JSONResponse(
                    {"error": "Storage is full; new writes are paused."}, status_code=507
                )
            state = self.manager.get_state(task_id) if task_id else None
            if task_id and not state:
                return JSONResponse({"error": "Task not found or expired"}, status_code=410)
            if not task_id:
                state = self.manager.create_task("pending upload")
                task_id = state["task_id"]
                self.manager.update_state(task_id, {"resource_pending": True})
            operation_id = uuid.uuid4().hex
            directory = self.manager.root / task_id / ".operations" / operation_id
            with self.manager.active_task(task_id):
                try:
                    directory.mkdir(parents=True, mode=0o700)
                    descriptor = {
                        "method": request.method,
                        "path": request.url.path,
                        "query": request.url.query,
                        "headers": [
                            [key, value]
                            for key, value in request.headers.items()
                            if key.lower() in {"content-type", "content-length", "accept"}
                        ],
                    }
                    (directory / "request.json").write_text(json.dumps(descriptor))
                    (directory / "request.body").write_bytes(body)
                    signature = hashlib.sha256(
                        json.dumps(
                            {key: descriptor[key] for key in ("method", "path", "query")},
                            sort_keys=True,
                        ).encode()
                        + semantic_body
                    )
                    operation = self.queue.enqueue(
                        task_id,
                        signature.hexdigest(),
                        request.url.path.split("/")[2],
                        datetime.fromisoformat(state["expires_at"]).timestamp(),
                        operation_id=operation_id,
                    )
                    if operation["id"] != operation_id:
                        shutil.rmtree(directory)
                except OSError:
                    shutil.rmtree(directory, ignore_errors=True)
                    return JSONResponse(
                        {"error": "Task or server storage quota exceeded."}, status_code=507
                    )
            return JSONResponse(
                self.queue.snapshot(operation["id"], pause_reason=self.pause_reason),
                status_code=202,
                headers={"X-GMXBUILDER-Operation": operation["id"]},
            )

    def operation_response(self, operation_id, *, result=False):
        if not re.fullmatch(r"[0-9a-f]{32}", operation_id):
            return JSONResponse({"error": "Invalid operation ID"}, status_code=400)
        operation = self.queue.get(operation_id)
        if not operation or operation["expires"] <= time.time():
            return JSONResponse({"error": "Operation expired or was removed"}, status_code=410)
        if result:
            path = self.job_directory(operation) / "response.json"
            if path.is_file():
                response = json.loads(path.read_text())
                return FileResponse(
                    path.with_suffix(".body"),
                    status_code=response["status"],
                    headers=dict(response["headers"]),
                )
            if operation["status"] in {"failed", "cancelled"}:
                return JSONResponse({"error": operation["error"]}, status_code=422)
            return JSONResponse({"status": operation["status"]}, status_code=202)
        snapshot = self.queue.snapshot(operation_id, pause_reason=self.pause_reason)
        snapshot["response_ready"] = (self.job_directory(operation) / "response.json").is_file()
        return JSONResponse(snapshot)

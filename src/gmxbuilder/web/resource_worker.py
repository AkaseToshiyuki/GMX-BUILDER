"""Replay one validated Web operation in a bounded process, without a listener.

The coordinator alone writes the request descriptor. This executable has no
public command or function dispatch and runs the same route validation and
scientific implementation used by the interactive application.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gmxbuilder.web.resource_queue import ResourceQueue
from gmxbuilder.web.write_sandbox import restrict_writes


class ScopedExecutor(ThreadPoolExecutor):
    def submit(self, function, /, *args, **kwargs):
        context = contextvars.copy_context()
        return super().submit(context.run, function, *args, **kwargs)


async def replay(application, descriptor, body_path, response_path):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": descriptor["method"],
        "scheme": "http",
        "root_path": "",
        "path": descriptor["path"],
        "raw_path": descriptor["path"].encode(),
        "query_string": descriptor["query"].encode(),
        "headers": [(name.encode(), value.encode()) for name, value in descriptor["headers"]],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 7788),
    }
    complete = asyncio.Event()
    response = {"status": 500, "headers": []}
    output = response_path.with_suffix(".body")
    with body_path.open("rb") as body, output.open("wb") as result:
        ended = False

        async def receive():
            nonlocal ended
            if ended:
                await complete.wait()
                return {"type": "http.disconnect"}
            chunk = body.read(65536)
            ended = len(chunk) < 65536
            return {"type": "http.request", "body": chunk, "more_body": not ended}

        async def send(message):
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                response["headers"] = [
                    (k.decode(), v.decode())
                    for k, v in message["headers"]
                    # Replay stores raw body bytes: retain their encoding and
                    # checkpoint cache identity when the coordinator serves them.
                    if k.lower()
                    in {
                        b"content-type",
                        b"content-disposition",
                        b"content-encoding",
                        b"cache-control",
                        b"etag",
                        b"vary",
                    }
                ]
            elif message["type"] == "http.response.body":
                result.write(message.get("body", b""))
                if not message.get("more_body", False):
                    result.flush()
                    complete.set()

        await application(scope, receive, send)
    temp = response_path.with_suffix(".tmp")
    temp.write_text(json.dumps(response))
    temp.replace(response_path)


async def run(root: Path, operation_id: str, threads: int):
    if not re.fullmatch(r"[0-9a-f]{32}", operation_id):
        raise ValueError("Invalid operation identifier")
    queue = ResourceQueue(root / ".control" / "queue.sqlite3")
    operation = queue.get(operation_id)
    if not operation or operation["status"] != "running" or operation["expires"] <= time.time():
        raise RuntimeError("Operation is missing, not admitted, or expired")
    task_dir = root / "tasks" / operation["task_id"]
    job_dir = task_dir / ".operations" / operation_id
    descriptor = json.loads((job_dir / "request.json").read_text())
    environment = json.loads((root / ".control" / "worker-environment.json").read_text())
    os.environ.update(environment)
    os.environ.update(
        {
            "GMXBUILDER_MANAGED_ROOT": str(root),
            "GMXBUILDER_RESOURCE_WORKER": "1",
            "GMXBUILDER_WORKER_TASK_ID": operation["task_id"],
            "GMXBUILDER_OPERATION_ID": operation_id,
            "GMXBUILDER_TASK_DIR": str(root / "tasks"),
            "GMXBUILDER_GPU_COUNT": "0",
            "GMXBUILDER_PREBUILT_AUTO_INSTALL": "0",
            # Keep the installation's full affinity available for later CPU
            # borrowing. The initial thread scope and cgroup quota set the
            # current share; narrowing affinity here would prevent expansion.
            "GMXBUILDER_CPU_CORES": environment.get("GMXBUILDER_CPU_CORES", str(threads)),
            "GMXBUILDER_TASK_THREADS": str(threads),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    scratch = task_dir / ".scratch"
    scratch.mkdir(exist_ok=True, mode=0o700)
    os.environ["TMPDIR"] = tempfile.tempdir = str(scratch)
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    os.environ["GMXBUILDER_GAFF_CACHE"] = str(root / "cache" / "gaff2")
    from gmxbuilder.runtime.hardware import configure_native_threads

    configure_native_threads(threads)
    os.chdir(task_dir)
    restrict_writes(root)

    import logging

    logging.basicConfig(filename=job_dir / "worker.log", level=logging.INFO)

    from gmxbuilder.runtime.hardware import task_thread_scope
    from gmxbuilder.web import server

    server.ThreadPoolExecutor = ScopedExecutor
    with server.task_manager.active_task(operation["task_id"]), task_thread_scope(threads):

        async def publish_prewarm():
            while True:
                state = server.ligand_prep.status(operation["task_id"])
                temp = task_dir / ".ligand-prep.tmp"
                temp.write_text(json.dumps(state))
                temp.replace(task_dir / ".ligand-prep.json")
                await asyncio.sleep(1)

        monitor = asyncio.create_task(publish_prewarm())
        try:
            await replay(
                server.app, descriptor, job_dir / "request.body", job_dir / "response.json"
            )
            # Finalization is an existing background HTTP operation. Its task
            # scope stays alive until it and speculative prewarm futures exit.
            await asyncio.to_thread(lambda: asyncio.run(server.shutdown_event()))
            final_prewarm = task_dir / ".ligand-prep.tmp"
            final_prewarm.write_text(json.dumps(server.ligand_prep.status(operation["task_id"])))
            final_prewarm.replace(task_dir / ".ligand-prep.json")
            (job_dir / "worker.done").write_text("completed")
        except BaseException as exc:
            # Do not publish arbitrary subprocess paths or credentials.
            message = server._redact_server_paths(str(exc))[:2000]
            try:
                (job_dir / "worker.error").write_text(message)
            except OSError:
                pass
            raise
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


def main():
    root, operation_id, threads = sys.argv[1:]
    try:
        asyncio.run(run(Path(root).resolve(), operation_id, max(1, int(threads))))
    except BaseException:
        import traceback

        queue = ResourceQueue(Path(root) / ".control" / "queue.sqlite3")
        operation = queue.get(operation_id)
        if operation:
            log = (
                Path(root)
                / "tasks"
                / operation["task_id"]
                / ".operations"
                / operation_id
                / "worker.log"
            )
            log.write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

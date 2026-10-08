"""Disk-backed FIFO tickets. Queued operations own no worker or scientific System."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ResourceQueue:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA auto_vacuum=INCREMENTAL;
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, digest TEXT NOT NULL,
                    kind TEXT NOT NULL, enqueued REAL NOT NULL, expires REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', started REAL, finished REAL,
                    unit TEXT, error TEXT, threads INTEGER, progress TEXT
                );
                CREATE INDEX IF NOT EXISTS fifo ON operations(status, enqueued);
                CREATE INDEX IF NOT EXISTS task_status ON operations(task_id, status);
                CREATE INDEX IF NOT EXISTS expiry ON operations(expires, status);
                CREATE UNIQUE INDEX IF NOT EXISTS pending_input
                    ON operations(task_id, digest) WHERE status IN ('queued','running');
            """)
        path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, task_id, digest, kind, expires, *, operation_id=None, now=None):
        operation_id = operation_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO operations "
                "(id,task_id,digest,kind,enqueued,expires) VALUES(?,?,?,?,?,?)",
                (operation_id, task_id, digest, kind, now or time.time(), expires),
            )
            return dict(
                db.execute(
                    "SELECT * FROM operations WHERE task_id=? AND digest=? "
                    "AND status IN ('queued','running')",
                    (task_id, digest),
                ).fetchone()
            )

    def get(self, operation_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
            return dict(row) if row else None

    def update(self, operation_id, **values):
        allowed = {"status", "started", "finished", "unit", "error", "threads", "progress"}
        if not values or set(values) - allowed:
            raise ValueError("Invalid queue update")
        if isinstance(values.get("progress"), dict):
            values["progress"] = json.dumps(values["progress"])
        with self.connect() as db:
            fields = ",".join(f"{key}=?" for key in values)
            db.execute(
                f"UPDATE operations SET {fields} WHERE id=?", (*values.values(), operation_id)
            )

    def active(self):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM operations WHERE status='running' ORDER BY enqueued,id"
                )
            ]

    def counts(self):
        with self.connect() as db:
            return {
                row[0]: row[1]
                for row in db.execute(
                    "SELECT status,COUNT(*) FROM operations WHERE status IN ('running','queued') "
                    "GROUP BY status"
                )
            }

    def next(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM operations q WHERE status='queued' "
                "AND NOT EXISTS (SELECT 1 FROM operations r WHERE "
                "r.task_id=q.task_id AND r.status='running') "
                "ORDER BY enqueued,id LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def task_active(self, task_id):
        with self.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM operations WHERE task_id=? AND status='running' LIMIT 1",
                    (task_id,),
                ).fetchone()
                is not None
            )

    def cancel_task(self, task_id, reason):
        with self.connect() as db:
            db.execute(
                "UPDATE operations SET status='cancelled',error=?,finished=? "
                "WHERE task_id=? AND status='queued'",
                (reason, time.time(), task_id),
            )

    def snapshot(self, operation_id, *, now=None, pause_reason=None):
        now = time.time() if now is None else now
        with self.connect() as db:
            row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
            if row is None:
                return None
            waiting = db.execute("SELECT COUNT(*) FROM operations WHERE status='queued'")
            total = waiting.fetchone()[0]
            ahead = 0
            if row["status"] == "queued":
                ahead = db.execute(
                    "SELECT COUNT(*) FROM operations WHERE status='queued' "
                    "AND (enqueued < ? OR (enqueued=? AND id<?))",
                    (row["enqueued"], row["enqueued"], row["id"]),
                ).fetchone()[0]
            # Estimates only from completed operations of the same kind. An
            # unknown workload must not be represented by the old 45 s guess.
            samples = [
                r[0]
                for r in db.execute(
                    "SELECT finished-started FROM operations WHERE kind=? AND status='completed' "
                    "AND finished>started ORDER BY finished DESC LIMIT 30",
                    (row["kind"],),
                )
            ]
            running = db.execute("SELECT COUNT(*) FROM operations WHERE status='running'")
            running_count = running.fetchone()[0]
            estimate = None
            if row["status"] == "queued" and len(samples) >= 3 and not pause_reason:
                samples.sort()
                estimate = round(samples[len(samples) // 2] * (ahead + 1) / max(1, running_count))
            return {
                "operation_id": row["id"],
                "task_id": row["task_id"],
                "status": row["status"],
                "queue_position": ahead + 1 if row["status"] == "queued" else 0,
                "queue_length": total,
                "ahead": ahead,
                "running": running_count,
                "waited_seconds": max(0, round((row["started"] or now) - row["enqueued"])),
                "estimated_wait_seconds": estimate,
                "expires_at": row["expires"],
                "pause_reason": pause_reason,
                "error": row["error"],
                "progress": json.loads(row["progress"]) if row["progress"] else None,
            }

    def forget_task(self, task_id):
        with self.connect() as db:
            db.execute("DELETE FROM operations WHERE task_id=?", (task_id,))

    def prune(self, now=None):
        with self.connect() as db:
            db.execute(
                "DELETE FROM operations WHERE expires<? AND status NOT IN ('running','queued')",
                (now or time.time(),),
            )
            db.execute("PRAGMA incremental_vacuum(64)")

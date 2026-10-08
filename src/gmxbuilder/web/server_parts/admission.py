"""Bounded admission controls for Web-submitted scientific work."""

from __future__ import annotations

import threading


class WorkQueueFull(RuntimeError):
    """Raised when an executor submission was rejected before queuing."""


class BoundedAdmission:
    """A non-blocking capacity gate placed before an executor submission."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("admission capacity must be positive")
        self.capacity = int(capacity)
        self._semaphore = threading.BoundedSemaphore(self.capacity)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()


class TaskAdmission:
    """Bound both global submissions and concurrent work for one task."""

    def __init__(self, capacity: int):
        self._capacity = BoundedAdmission(capacity)
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity.capacity

    def try_acquire(self, task_id: str) -> str:
        """Return ``accepted``, ``duplicate``, or ``full`` without waiting."""
        with self._lock:
            if task_id in self._active:
                return "duplicate"
            if not self._capacity.try_acquire():
                return "full"
            self._active.add(task_id)
            return "accepted"

    def release(self, task_id: str) -> None:
        with self._lock:
            if task_id not in self._active:
                return
            self._active.remove(task_id)
            self._capacity.release()

    def active(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._active

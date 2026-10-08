"""Task-capability validation and log-safe correlation identifiers."""

from __future__ import annotations

import hashlib
import logging
import re
import secrets

_TASK_ID_PATTERN = re.compile(r"^(?:[a-f0-9]{12}|[a-f0-9]{32})$")
_TASK_TOKEN_PATTERN = re.compile(r"(?<![a-f0-9])(?:[a-f0-9]{32}|[a-f0-9]{12})(?![a-f0-9])")
_LOG_CORRELATION_KEY = secrets.token_bytes(32)


class InvalidTaskId(ValueError):
    """Expected client error for a malformed task capability."""


def validate_task_id(value: object) -> str:
    if not isinstance(value, str) or _TASK_ID_PATTERN.fullmatch(value) is None:
        raise InvalidTaskId("Invalid task ID format")
    return value


def task_log_reference(task_id: str) -> str:
    """Return a process-local, irreversible correlation value for logs."""
    digest = hashlib.blake2s(
        task_id.encode("ascii", errors="ignore"),
        key=_LOG_CORRELATION_KEY,
        digest_size=8,
    ).hexdigest()
    return f"task-{digest}"


def redact_task_capabilities(value: object) -> str:
    return _TASK_TOKEN_PATTERN.sub("<task-capability>", str(value))


class CapabilityRedactionFormatter(logging.Formatter):
    """Redact the complete formatted record, including exception tracebacks."""

    def format(self, record):
        return redact_task_capabilities(super().format(record))


class CapabilityRedactionFilter(logging.Filter):
    """Last-resort protection for application/framework log handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = redact_task_capabilities(message)
        record.args = ()
        return True


def install_capability_log_filter() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(item, CapabilityRedactionFilter) for item in handler.filters):
            handler.addFilter(CapabilityRedactionFilter())

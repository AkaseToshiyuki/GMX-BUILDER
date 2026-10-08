"""One request owns one immutable input snapshot and parsed CIF document."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_source = ContextVar("input_source_snapshot", default=None)


@contextmanager
def source_scope(path):
    path = Path(path).resolve()
    current = _source.get()
    if current is not None and current[0] == path:
        yield
        return
    token = _source.set((path, path.read_bytes(), {}))
    try:
        yield
    finally:
        _source.reset(token)


def source_bytes(path):
    current = _source.get()
    path = Path(path).resolve()
    return current[1] if current is not None and current[0] == path else path.read_bytes()


def source_text(path, *, errors="replace"):
    return (
        source_bytes(path)
        .decode("utf-8-sig", errors=errors)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def parsed_documents():
    current = _source.get()
    return current[2] if current is not None else None

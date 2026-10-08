"""Build-local immutable admission and sampling indices; never a UI cache."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_session = ContextVar("conformer_read_session", default=None)


def stamp(path):
    try:
        info = path.stat()
    except OSError as exc:
        # A lost admitted source is corruption, not a missing-library fallback.
        raise ValueError("Conformer source unavailable; retry construction") from exc
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def entry_stamp(path):
    return stamp(path), stamp(path / "metadata.json")


@dataclass
class _Pinned:
    entry: object
    files: tuple
    sources: tuple
    by_name: dict
    stamps: dict
    identity: str
    protocol: str | None
    lipid: str
    directory_stamp: tuple

    @classmethod
    def prepare(cls, library, lipid, force_field, lipid_ff):
        from gmxbuilder.modules.membrane.lipids import LipidRegistry
        from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

        entry = library.inspect(lipid, force_field, lipid_ff)
        if entry is None:
            return None
        before = entry_stamp(entry.path)
        if entry.admission_stamp is not None and before != entry.admission_stamp:
            raise ValueError("Conformer library changed during admission; retry construction")
        files = tuple(entry.conformer_files)
        stamps = {path: stamp(path) for path in files}
        sources = {}
        if entry.metadata.get("conformer_sampling") == "equal-replica-then-uniform-conformer":
            for item in entry.metadata["conformer_provenance"]:
                sources.setdefault(item["replica"], []).append(item["file"])
        protocol = None
        if entry.metadata.get("v4_protocol"):
            protocol = resolve_protocol(lipid, entry.metadata["parameter_family"])["sha256"]
        result = cls(
            entry,
            files,
            tuple(tuple(sources[k]) for k in sorted(sources)),
            {path.name: path for path in files},
            stamps,
            LipidRegistry.get(lipid).smiles,
            protocol,
            lipid,
            before,
        )
        result.check()
        return result

    def check(self, *, full=False):
        if entry_stamp(self.entry.path) != self.directory_stamp:
            raise ValueError("Conformer library changed during construction; retry")
        if not full:
            return
        from gmxbuilder.modules.membrane.lipids import LipidRegistry
        from gmxbuilder.modules.membrane.parameter_provenance import parameters_current
        from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

        if LipidRegistry.get(self.lipid).smiles != self.identity:
            raise ValueError("Lipid identity changed during construction")
        if not parameters_current(self.entry.metadata):
            raise ValueError("Lipid parameters changed during construction")
        if (
            self.protocol is not None
            and self.protocol
            != resolve_protocol(self.lipid, self.entry.metadata["parameter_family"])["sha256"]
        ):
            raise ValueError("Lipid protocol changed during construction")
        if any(stamp(path) != signature for path, signature in self.stamps.items()):
            raise ValueError("Conformer coordinates changed during construction")


@contextmanager
def conformer_read_scope():
    """Pin each admitted species until the build finishes, then verify sources."""
    if _session.get() is not None:
        yield
        return
    entries = {}
    token = _session.set(entries)
    try:
        yield
        for prepared in entries.values():
            prepared.check(full=True)
    finally:
        _session.reset(token)


def prepared_entry(library, lipid, force_field, lipid_ff):
    entries = _session.get()
    if entries is None:
        return None
    key = library, lipid, force_field, lipid_ff
    if key not in entries:
        prepared = _Pinned.prepare(library, lipid, force_field, lipid_ff)
        if prepared is None:
            return None
        entries[key] = prepared
    prepared = entries[key]
    prepared.check()
    return prepared

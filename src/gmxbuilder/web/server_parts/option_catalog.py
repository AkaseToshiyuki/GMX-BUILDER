"""Bounded, read-only UI snapshots. Production admission never reads this cache.

An old pinned publisher need not emit events: the worker checks file identities
between revisions. Changed entries are hidden before revalidation. No metadata
or conformer arrays are retained, only compact display records and stat digests.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from gmxbuilder.modules.membrane.equilibrated_library import (
    EquilibratedLipidLibrary,
    LibraryEntry,
    configured_library_root,
)
from gmxbuilder.modules.membrane.v4_availability import refresh_availability_list
from gmxbuilder.web.server_parts.web_options import build_ui_options

logger = logging.getLogger(__name__)
POLL_SECONDS = 15.0


def tree_stamp(root: Path, *, suffixes=None) -> str:
    """Include replacements, in-place writes and removals without reading bodies."""
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode())
    if not root.exists():
        return digest.hexdigest()
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {"__pycache__", ".git"})
        for name in sorted(files):
            if suffixes is not None and Path(name).suffix not in suffixes:
                continue
            path = Path(directory) / name
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(
                repr(
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                ).encode()
            )
    return digest.hexdigest()


def dependency_stamp() -> tuple:
    """Installed definitions, implementation/policies and fitted GAFF templates."""
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    package = Path(__file__).resolve().parents[2]
    data = package / "data"
    roots = [package / "modules", package / "io", package / "geometry"]
    roots += [data / name for name in ("forcefields", "forcefield_overlays", "lipid21")]
    roots += [
        Path(os.environ.get("GMXBUILDER_GAFF_CACHE", Path.home() / ".cache/gmxbuilder/gaff2"))
    ]
    return (
        tuple(
            tree_stamp(
                p,
                suffixes={
                    ".py",
                    ".itp",
                    ".rtp",
                    ".tdb",
                    ".r2b",
                    ".hdb",
                    ".atp",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".csv",
                },
            )
            for p in roots
        ),
        tree_stamp(data, suffixes={".json", ".yaml", ".yml", ".csv"}),
        tuple((name, repr(LipidRegistry.get(name))) for name in LipidRegistry.list()),
        os.environ.get("PATH", ""),
    )


class EntryIndex:
    """Per-entry reuse of strict results, bounded by the published inventory."""

    def __init__(self, library):
        self.library = library
        self.roots = library.roots
        self.records = {}
        self.stamps = {}
        self.epoch = None
        self.cancelled = lambda: False

    def inventory(self):
        return {
            (str(root), family.name, entry.name): tree_stamp(entry)
            for root in self.roots
            if root.is_dir()
            for family in root.iterdir()
            if family.is_dir() and not family.is_symlink()
            for entry in family.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        }

    def prepare(self, inventory, epoch):
        if epoch != self.epoch:
            self.records.clear()
        else:
            self.records = {
                key: value
                for key, value in self.records.items()
                if all(inventory.get(k) == self.stamps.get(k) for k in value[0])
            }
        self.stamps, self.epoch = inventory, epoch

    def inspect(self, name, force_field, lipid_ff):
        if self.cancelled():
            return None
        key = (name, force_field, lipid_ff)
        if key in self.records:
            return self.records[key][1]
        directories = self.library._candidate_dirs(name, force_field, lipid_ff)
        watched = tuple((str(p.parent.parent), p.parent.name, p.name) for p in directories)
        entry = self.library.inspect(name, force_field, lipid_ff)
        from gmxbuilder.modules.membrane.v4_evidence import validation_scope

        # Never retain the hundreds of MiB of trajectory evidence in the UI cache.
        compact = (
            None
            if entry is None
            else LibraryEntry(
                entry.path,
                {
                    **{
                        k: entry.metadata.get(k)
                        for k in (
                            "v4_protocol",
                            "n_conformations",
                            "lipid_ff",
                            "equilibration_host",
                        )
                    },
                },
                validation_summary=validation_scope(entry.metadata),
            )
        )
        self.records[key] = (watched, compact)
        return compact


class OptionCatalog:
    def __init__(self, library=None, *, dependencies=dependency_stamp, interval=POLL_SECONDS):
        if library is None:
            library = EquilibratedLipidLibrary([configured_library_root()])
            library.require_v4 = True
        self.index = EntryIndex(library)
        self.dependencies = dependencies
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.index.cancelled = self.stop.is_set
        self.thread = None
        self.revision = 0
        self.options = None
        self.signature = None
        self.availability = self._pending("checking")
        self.checked_at = 0.0

    def _pending(self, status):
        return {
            "schema_version": 1,
            "library_version": 4,
            "entries": [],
            "status": status,
            "revision": str(self.revision),
        }

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._run, daemon=True, name="gmxbuilder-catalog"
                )
                self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def read(self, *, options=False):
        self.start()
        with self.lock:
            availability = self.availability
            # A failed/stalled watcher cannot keep advertising old acceptance.
            if self.checked_at and time.monotonic() - self.checked_at > max(10, self.interval * 3):
                availability = self._pending("checking")
            if not options:
                return copy.deepcopy(availability)
            if self.options is None:
                return {
                    "status": "checking",
                    "lipids": [],
                    "water_models": [],
                    "force_fields": [],
                    "availability": copy.deepcopy(availability),
                }
            result = copy.deepcopy(self.options)
            result["availability"] = copy.deepcopy(availability)
            result["status"] = "ready"
            if availability["status"] != "ready":
                for lipid in result["lipids"]:
                    lipid["parameterizations"] = []
                    lipid["parameterization"] = "Checking availability"
            return result

    def refresh(self):
        epoch = self.dependencies()
        inventory = self.index.inventory()
        signature = (epoch, inventory)
        if signature == self.signature:
            with self.lock:
                self.checked_at = time.monotonic()
            return
        with self.lock:
            self.revision += 1
            self.availability = self._pending("checking")
        self.index.prepare(inventory, epoch)
        if self.options is None:
            base = build_ui_options(availability=self._pending("checking"))
            with self.lock:
                self.options = base
        payload = refresh_availability_list(library=self.index, write=False)
        if self.stop.is_set():
            return
        # Do not publish a result assembled across a concurrent install/publication.
        if self.dependencies() != epoch or self.index.inventory() != inventory:
            self.signature = None
            return
        options = build_ui_options(availability=payload)
        payload.update(status="ready", revision=str(self.revision))
        with self.lock:
            self.signature = signature
            self.options = options
            self.availability = payload
            self.checked_at = time.monotonic()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.refresh()
            except Exception:
                logger.exception("UI catalog refresh failed; production admission remains strict")
                with self.lock:
                    self.signature = None
                    self.availability = self._pending("error")
            self.stop.wait(self.interval)


_catalog = None
_catalog_lock = threading.Lock()


def get_catalog():
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = OptionCatalog()
        return _catalog


def close_catalog():
    global _catalog
    with _catalog_lock:
        catalog, _catalog = _catalog, None
    if catalog is not None:
        catalog.close()

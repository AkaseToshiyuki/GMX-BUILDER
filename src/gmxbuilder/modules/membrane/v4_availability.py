"""Live V4 publication manifest shared by the UI and production builders.

The list is a derived index, never an authority for scientific acceptance.
Strict consumers reconcile on admission. The Web display uses a background
verified snapshot, including changes from older pinned publishers.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.membrane.equilibrated_library import (
    EquilibratedLipidLibrary,
    get_equilibrated_lipid_library,
    lipid_parameter_family,
)
from gmxbuilder.modules.membrane.v4_evidence import validation_scope

_LOCK = threading.Lock()
SOURCES = ("lipid21", "gaff2", "charmm36m", "charmm36")


def accepted_entry(library, name, force_field, lipid_ff):
    """Require assembled V4 evidence, not a schema-4 legacy single run."""
    try:
        entry = library.inspect(name, force_field, lipid_ff)
        if entry is not None and entry.metadata.get("v4_protocol"):
            return entry
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass
    return None


def require_v4_lipids(names, force_field, lipid_ff, *, library=None):
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    builtin = sorted(set(names) & set(LipidRegistry.list_builtin()))
    if not builtin:
        return
    library = library or get_equilibrated_lipid_library()
    missing = [
        name for name in builtin if accepted_entry(library, name, force_field, lipid_ff) is None
    ]
    if missing:
        raise ModuleConfigError(
            f"V4 acceptance pending or invalid for {force_field}/{lipid_ff}: "
            f"{', '.join(missing)}. These lipids unlock automatically after V4 acceptance."
        )


def refresh_availability_list(*, library=None, write: bool = True):
    """Atomically publish list.json from strict current entries; never trust old flags."""
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    library = library or EquilibratedLipidLibrary()
    destination = library.roots[0]
    path = destination / "list.json"
    with _LOCK:
        entries = []
        for name in sorted(LipidRegistry.list_builtin()):
            for source in SOURCES:
                force_field = "amber14sb" if source in {"lipid21", "gaff2"} else source
                entry = accepted_entry(library, name, force_field, source)
                entries.append(
                    {
                        "lipid_name": name,
                        "force_field": force_field,
                        "lipid_ff": source,
                        "parameter_family": lipid_parameter_family(force_field, source),
                        "ready": entry is not None,
                        "state": "ready" if entry else "pending",
                        "validation_scope": (
                            getattr(entry, "validation_summary", None)
                            or validation_scope(entry.metadata)
                        )
                        if entry
                        else None,
                        "reason": "" if entry else "Awaiting valid V4 acceptance",
                        "n_conformations": entry.metadata["n_conformations"] if entry else 0,
                        "amber_mixed_ready": bool(entry)
                        and (
                            source == "lipid21"
                            or (
                                source == "gaff2"
                                and not lipid21_capability(name)[0]
                                and entry.metadata.get("lipid_ff") == "amber-mixed"
                                and entry.metadata.get("equilibration_host")
                                == {
                                    "lipid_name": "POPC",
                                    "ratio_percent": 90,
                                    "target_ratio_percent": 10,
                                }
                            )
                        ),
                    }
                )
        payload = {"schema_version": 1, "library_version": 4, "entries": entries}
        if not write:
            return payload
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        if all(previous.get(key) == value for key, value in payload.items()):
            return previous
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            destination.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".list-", suffix=".json", dir=destination)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except PermissionError:
            # Managed Web workers may read an administrator-owned library but
            # cannot write outside their storage boundary. The derived index
            # is optional; return freshly validated entries, never stale flags.
            pass
        return payload

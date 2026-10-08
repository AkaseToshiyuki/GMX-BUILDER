"""What was built, from what, and how -- recorded as the build happens.

A finished package says what the system *is*. It did not say how it came to
be: which pH the protonation used, which orientation method placed the
protein, how many lipids were asked for against how many were delivered, or
how long any of it took. Reproducing a system, or checking someone else's,
meant asking the person who built it.

Each step records the configuration it was given and the metrics it produced
into ``system.metadata['provenance']``, which rides along in the checkpoint
and reaches the exporter with the system it describes. The exporter turns it
into ``manifest.json``.

**The manifest is not a report about the build; it is the build.** Its
``modules`` block is the configuration the modules actually received, in the
schema ``PipelineConfig`` already reads, so replaying is::

    gmxbuilder build -c manifest.json

``PipelineConfig`` ignores keys it does not define, which is what lets the
provenance record travel in the same file as the configuration that produced
it -- one file to archive, one file to hand a reviewer, and no second copy to
drift out of step with the first.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Where the record lives on the system it describes.
PROVENANCE_KEY = "provenance"

#: Bumped when the manifest's shape changes in a way a reader must notice.
MANIFEST_VERSION = 1

#: Injected by the step runner so a module can find its own working
#: directories. They name locations on the machine that ran the build and mean
#: nothing to anyone replaying it, so they never enter the record.
_INTERNAL_CONFIG_KEYS = frozenset(
    {
        "_task_dir",
        "_step_dir",
        "output_dir",
        "pdb",
        "task_id",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def replayable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return one step's configuration with host-specific keys removed.

    Anything private (a leading underscore) or naming a location on the build
    machine is dropped. What remains is the scientific request: the pH, the
    lipid composition, the ion concentration -- the values that decide what
    gets built rather than where it lands.
    """
    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_") and str(key) not in _INTERNAL_CONFIG_KEYS
    }


def _record(system) -> dict[str, Any]:
    record = system.metadata.get(PROVENANCE_KEY)
    if not isinstance(record, dict):
        record = {"steps": []}
        system.metadata[PROVENANCE_KEY] = record
    if not isinstance(record.get("steps"), list):
        record["steps"] = []
    return record


def adopt_task_context(system, task_dir: Path | str | None) -> None:
    """Record when the task was created and what was uploaded into it.

    Read from the task's own ``state.json`` rather than passed down through
    every module signature. A command-line build has no such file and simply
    contributes nothing here, which is why this never raises: an absent task
    context makes the record smaller, not wrong.
    """
    if task_dir is None:
        return
    record = _record(system)
    if "task" in record:
        return
    state_file = Path(task_dir) / "state.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(state, dict):
        return
    context = {
        key: state[key]
        for key in ("created_at", "original_filename", "uploaded_structure_name", "task_type_id")
        if isinstance(state.get(key), str)
    }
    task_type = state.get("task_type")
    if isinstance(task_type, dict) and isinstance(task_type.get("id"), str):
        context.setdefault("task_type_id", task_type["id"])
    if context:
        record["task"] = context


def record_step(
    system,
    step_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    elapsed_s: float,
) -> None:
    """Record one completed step, replacing any earlier record of it.

    A user can re-run a step with different settings; the record must show the
    configuration that produced the checkpoint on disk, not the first one
    tried. Order follows first execution so the list still reads as the
    pipeline.
    """
    record = _record(system)
    # The only place the uploaded file's identity is knowable. `config["pdb"]`
    # is a path on the build machine and is stripped from the record a line
    # later, so if the digest is not taken here it cannot be taken at all.
    source = config.get("pdb")
    if source and "input_structure" not in record:
        path = Path(str(source))
        if path.is_file():
            identity: dict[str, Any] = {"filename": path.name, "bytes": path.stat().st_size}
            digest = sha256_of(path)
            if digest:
                identity["sha256"] = digest
            record["input_structure"] = identity

    entry = {
        "step": step_name,
        "completed_at": _now(),
        "elapsed_s": round(float(elapsed_s), 2),
        "config": replayable_config(config),
        "metrics": metrics,
    }
    for index, existing in enumerate(record["steps"]):
        if isinstance(existing, dict) and existing.get("step") == step_name:
            record["steps"][index] = entry
            return
    record["steps"].append(entry)


def file_inventory(root: Path) -> list[dict[str, Any]]:
    """List every file in the finished package with its size.

    Sizes are what a user checks a download against, and the archive itself is
    excluded: it contains the very listing being written, so its size cannot
    be known before it exists.
    """
    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix == ".zip":
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
            }
        )
    return entries


def sha256_of(path: Path) -> str | None:
    """Return a file's SHA-256, or None when it cannot be read.

    The input structure is not shipped inside the package -- it is the user's
    own file and may carry terms this project cannot redistribute -- so the
    digest is what lets a replay prove it started from the same coordinates.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def build_manifest(
    system,
    *,
    system_name: str,
    seed: int,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the replayable manifest for a finished system."""
    from gmxbuilder.__version__ import VERSION

    record = _record(system)
    steps = [entry for entry in record["steps"] if isinstance(entry, dict)]
    modules = {
        str(entry["step"]): entry.get("config", {})
        for entry in steps
        if isinstance(entry.get("step"), str)
    }
    build_seconds = round(sum(float(entry.get("elapsed_s", 0.0)) for entry in steps), 2)

    provenance: dict[str, Any] = {
        "gmxbuilder_version": VERSION,
        "exported_at": _now(),
        "build_duration_s": build_seconds,
        "steps": [
            {
                "step": entry.get("step"),
                "completed_at": entry.get("completed_at"),
                "elapsed_s": entry.get("elapsed_s"),
                "inputs": entry.get("config", {}),
                "outputs": entry.get("metrics", {}),
            }
            for entry in steps
        ],
        "files": files,
        "total_bytes": sum(int(entry.get("bytes", 0)) for entry in files),
    }
    task = record.get("task")
    if isinstance(task, dict) and task:
        provenance["task"] = task
    input_structure = record.get("input_structure")
    if isinstance(input_structure, dict) and input_structure:
        provenance["input_structure"] = input_structure
        # The uploaded coordinates are not in the package, so `modules.input`
        # cannot name a file that exists. It names the file the user must
        # supply -- under its original name, beside the manifest -- and the
        # digest in `provenance.input_structure` is what proves they supplied
        # the right one. Without this the replay config would have no input at
        # all, which is the one thing it cannot do without.
        filename = input_structure.get("filename")
        if filename:
            modules.setdefault("input", {})["pdb"] = f"./{filename}"

    return {
        "gmxbuilder_manifest_version": MANIFEST_VERSION,
        "system_name": system_name,
        "seed": int(seed),
        "output_dir": "./output",
        "modules": modules,
        "provenance": provenance,
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


def readable_summary(system) -> str:
    """Render the record as the prose block that goes into README.txt.

    The same facts as ``manifest.json``, for a person rather than a program:
    what was asked for at each step, what came out, and how long it took.
    Per-file sizes are deliberately left to the manifest -- this file is
    written before the inventory can be taken, and a size list is something
    you check with a program anyway.
    """
    record = _record(system)
    steps = [entry for entry in record["steps"] if isinstance(entry, dict)]
    if not steps:
        return "  (this system was not built through the step pipeline)"

    lines: list[str] = []
    task = record.get("task")
    if isinstance(task, dict):
        if task.get("created_at"):
            lines.append(f"  Uploaded:    {task['created_at']}")
        if task.get("original_filename"):
            lines.append(f"  Input file:  {task['original_filename']}")
        if task.get("task_type_id"):
            lines.append(f"  Workflow:    {task['task_type_id']}")
    total = sum(float(entry.get("elapsed_s", 0.0)) for entry in steps)
    lines.append(f"  Build time:  {total:.1f} s across {len(steps)} steps")
    lines.append("")

    for entry in steps:
        lines.append(f"  {entry.get('step', '?')}  ({entry.get('elapsed_s', 0)} s)")
        config = entry.get("config") or {}
        if config:
            lines.append("      requested:")
            for key in sorted(config):
                lines.append(f"        {key} = {_format_value(config[key])}")
        metrics = entry.get("metrics") or {}
        produced = [
            (key, metrics[key]) for key in ("num_atoms", "box_dimensions_nm") if key in metrics
        ]
        components = metrics.get("components")
        if produced or components:
            lines.append("      produced:")
        for key, value in produced:
            lines.append(f"        {key} = {_format_value(value)}")
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict):
                    continue
                count = component.get("n_molecules")
                suffix = f", {count} molecules" if count is not None else ""
                lines.append(
                    f"        {component.get('name')} "
                    f"({component.get('kind')}): {component.get('n_atoms')} atoms{suffix}"
                )
        lines.append("")
    return "\n".join(lines).rstrip()

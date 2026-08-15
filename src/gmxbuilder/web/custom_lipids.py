"""Task-scoped custom lipid persistence and execution context.

Custom molecules are bearer-capability data: definitions, GAFF2 artifacts and
validated NPT conformers live below one task directory and are never added to
the process-wide lipid registry.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Iterator

from gmxbuilder.modules.forcefield.gaff_backend import task_gaff_cache
from gmxbuilder.modules.membrane.equilibrated_library import (
    EquilibratedLipidLibrary,
    task_equilibrated_library,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry, LipidTemplate


_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,4}$")
_TERMINAL_STATES = {"ready", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


class TaskLipidLibrary(EquilibratedLipidLibrary):
    """Library that forbids global fallback for task-owned molecule names."""

    def __init__(self, task_root: Path, isolated_names: set[str]):
        defaults = EquilibratedLipidLibrary()
        super().__init__([task_root, *defaults.roots])
        self.isolated_names = {name.upper() for name in isolated_names}

    def _candidate_dirs(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
    ) -> list[Path]:
        candidates = super()._candidate_dirs(lipid_name, force_field, lipid_ff)
        if str(lipid_name).strip().upper() in self.isolated_names:
            return candidates[:1]
        return candidates


class CustomLipidStore:
    """Own custom lipid records inside one task directory."""

    def __init__(self, task_dir: str | Path):
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.root = self.task_dir / "custom_lipids"
        self.definitions = self.root / "definitions"
        self.statuses = self.root / "status"
        self.gaff_cache = self.root / "gaff2"
        self.library_root = self.root / "library"

    @staticmethod
    def normalize_name(name: str) -> str:
        normalized = str(name).strip().upper().replace(" ", "_")
        if not _NAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Custom lipid residue ID must contain 1-5 uppercase letters, "
                "digits, or underscores and must start with a letter"
            )
        return normalized

    def definition_path(self, name: str) -> Path:
        return self.definitions / f"{self.normalize_name(name)}.json"

    def status_path(self, name: str) -> Path:
        return self.statuses / f"{self.normalize_name(name)}.json"

    def save_submission(self, properties: dict, force_field: str) -> dict:
        name = self.normalize_name(properties.get("name", ""))
        if name in LipidRegistry.list_builtin():
            raise ValueError(f"Custom lipid name {name} conflicts with the built-in library")
        canonical = str(properties.get("canonical_smiles", "")).strip()
        if not canonical:
            raise ValueError("Canonical SMILES identity is missing")
        for existing in self.load_definitions().values():
            if existing.get("canonical_smiles") == canonical:
                raise ValueError(
                    f"This molecule is already submitted to this task as {existing['name']}"
                )
        if self.definition_path(name).exists():
            raise ValueError(f"Custom lipid name {name} is already used in this task")

        definition = dict(properties)
        definition.update(
            {
                "name": name,
                "force_field": str(force_field).lower(),
                "lipid_ff": "gaff2",
                "parameterizations": ["gaff2"],
                "task_scoped": True,
                "submitted_at": _now(),
            }
        )
        _atomic_json(self.definition_path(name), definition)
        status = {
            "name": name,
            "state": "queued",
            "phase": "queued",
            "progress": 0,
            "message": "Waiting for task-scoped force-field calculation",
            "submitted_at": definition["submitted_at"],
            "updated_at": _now(),
        }
        _atomic_json(self.status_path(name), status)
        return self.public_record(name)

    def load_definitions(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if not self.definitions.is_dir():
            return result
        for path in sorted(self.definitions.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                name = self.normalize_name(data.get("name", path.stem))
                result[name] = data
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return result

    def load_status(self, name: str) -> dict:
        path = self.status_path(name)
        if not path.is_file():
            raise KeyError(f"Unknown custom lipid: {self.normalize_name(name)}")
        return json.loads(path.read_text())

    def update_status(self, name: str, **updates) -> dict:
        status = self.load_status(name)
        status.update(updates)
        status["updated_at"] = _now()
        _atomic_json(self.status_path(name), status)
        return status

    def templates(self, *, ready_only: bool = True) -> dict[str, LipidTemplate]:
        templates: dict[str, LipidTemplate] = {}
        for name, definition in self.load_definitions().items():
            try:
                if ready_only and self.load_status(name).get("state") != "ready":
                    continue
                templates[name] = LipidRegistry.custom_template(name, definition)
            except (KeyError, ValueError):
                continue
        return templates

    def library(self, names: set[str] | None = None) -> TaskLipidLibrary:
        isolated = names or set(self.load_definitions())
        return TaskLipidLibrary(self.library_root, isolated)

    def public_record(self, name: str) -> dict:
        normalized = self.normalize_name(name)
        definition = self.load_definitions().get(normalized)
        if definition is None:
            raise KeyError(f"Unknown custom lipid: {normalized}")
        status = self.load_status(normalized)
        safe_definition = {
            key: value for key, value in definition.items() if key not in {"submitted_at"}
        }
        return {**safe_definition, **status, "task_scoped": True}

    def list_public(self) -> list[dict]:
        records = []
        for name in self.load_definitions():
            try:
                records.append(self.public_record(name))
            except (KeyError, OSError, json.JSONDecodeError):
                continue
        return records

    def pending_names(self) -> list[str]:
        names = []
        for name in self.load_definitions():
            try:
                if self.load_status(name).get("state") not in _TERMINAL_STATES:
                    names.append(name)
            except KeyError:
                continue
        return names


@contextmanager
def task_custom_lipid_scope(
    task_dir: str | Path,
    *,
    include_unready: set[str] | None = None,
) -> Iterator[CustomLipidStore]:
    """Activate only this task's custom definitions and artifact roots."""
    store = CustomLipidStore(task_dir)
    templates = store.templates(ready_only=True)
    definitions = store.load_definitions()
    for name in include_unready or set():
        if name in definitions:
            templates[name] = LipidRegistry.custom_template(name, definitions[name])
    library = store.library(set(definitions))
    with ExitStack() as stack:
        stack.enter_context(LipidRegistry.task_scope(templates))
        stack.enter_context(task_gaff_cache(store.gaff_cache, set(definitions)))
        stack.enter_context(task_equilibrated_library(library))
        yield store


def run_custom_lipid_build(task_dir: str | Path, name: str) -> None:
    """Parameterize and build the task-private explicit-solvent NPT library."""
    store = CustomLipidStore(task_dir)
    normalized = store.normalize_name(name)
    definition = store.load_definitions().get(normalized)
    if definition is None:
        raise KeyError(f"Unknown custom lipid: {normalized}")
    force_field = str(definition.get("force_field", "")).lower()
    if not force_field.startswith("amber") or definition.get("lipid_ff") != "gaff2":
        raise ValueError("Custom lipids currently require an Amber + GAFF2 selection")

    with task_custom_lipid_scope(task_dir, include_unready={normalized}):
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
        from gmxbuilder.modules.membrane.lipid_equilibration import (
            LipidEquilibrationBuilder,
        )

        store.update_status(
            normalized,
            state="running",
            phase="parameterization",
            progress=10,
            message="Calculating task-scoped GAFF2/AM1-BCC parameters",
        )
        prepare_gaff_lipid(
            normalized,
            definition["canonical_smiles"],
            int(definition.get("charge", 0)),
        )
        store.update_status(
            normalized,
            state="running",
            phase="pre_equilibration",
            progress=35,
            message="Running explicit-solvent semi-isotropic NPT pre-equilibration",
        )
        npt_ps = float(os.environ.get("GMXBUILDER_CUSTOM_LIPID_NPT_PS", "1000"))
        builder = LipidEquilibrationBuilder(library=store.library({normalized}))
        builder.build(
            normalized,
            force_field,
            "gaff2",
            npt_ps=npt_ps,
            force=True,
        )
        entry = store.library({normalized}).inspect(normalized, force_field, "gaff2")
        if entry is None:
            raise RuntimeError("Pre-equilibration completed without a validated conformer library")
        store.update_status(
            normalized,
            state="ready",
            phase="complete",
            progress=100,
            message="Validated task-scoped lipid library is ready",
            completed_at=_now(),
            n_conformations=len(entry.conformer_files),
        )

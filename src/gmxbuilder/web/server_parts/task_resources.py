"""Task-scoped resource resolution and public-response sanitization."""

from __future__ import annotations

import copy
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

SERVER_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|root|tmp|var|opt|srv|mnt|media)"
    r"(?:/[^\s,;:)\]}]+)+"
)


class TaskResourceHelpers:
    """Resolve task files through the server's current task-manager instance."""

    def __init__(
        self,
        manager_provider: Callable[[], Any],
        task_id_validator: Callable[[str], str],
    ) -> None:
        self._manager_provider = manager_provider
        self._task_id_validator = task_id_validator

    @property
    def manager(self):
        return self._manager_provider()

    def validate_task_resource(self, task_id: str, resource: str | Path) -> Path:
        """Resolve a server-owned file and confine it to exactly one task."""
        task_id = self._task_id_validator(task_id)
        task_dir = self.manager.get_task_dir(task_id).resolve()
        candidate = Path(resource)
        if not candidate.is_absolute():
            candidate = task_dir / candidate
        resolved = candidate.resolve()
        if resolved == task_dir or task_dir not in resolved.parents:
            raise ValueError("Resource is outside task scope")
        return resolved

    def resolve_input_pdb(self, task_id: str) -> str:
        """Resolve the filtered selection or original structure upload."""
        task_id = self._task_id_validator(task_id)
        manager = self.manager
        task_dir = manager.get_task_dir(task_id)

        filtered = task_dir / "filtered.pdb"
        if filtered.exists():
            from gmxbuilder.io.input_document import canonical_path

            canonical = canonical_path(filtered)
            if not canonical.exists():
                raise ValueError(
                    "Reapply the input selection before Check: this legacy task "
                    "has no lossless canonical selection"
                )
            return str(canonical)

        state = manager.get_state(task_id) or {}
        uploaded_name = state.get("uploaded_structure_name")
        if isinstance(uploaded_name, str) and uploaded_name:
            uploaded = self.validate_task_resource(task_id, uploaded_name)
            if uploaded.is_file() and not uploaded.is_symlink():
                return str(uploaded)

        pdb_path = manager.get_pdb_path(task_id)
        if pdb_path and pdb_path.exists():
            return str(pdb_path)
        raise ValueError(f"No structure file found for task {task_id}")

    def resolve_pdb_path(self, task_id: str) -> str:
        """Resolve the best preview PDB, preferring the structure checkpoint."""
        task_id = self._task_id_validator(task_id)
        manager = self.manager
        task_dir = manager.get_task_dir(task_id)

        structure_pdb = task_dir / "steps" / "structure" / "viewer.pdb"
        if structure_pdb.exists():
            return str(structure_pdb)
        filtered = task_dir / "filtered.pdb"
        if filtered.exists():
            return str(filtered)
        pdb_path = manager.get_pdb_path(task_id)
        if pdb_path and pdb_path.exists():
            return str(pdb_path)
        raise ValueError(f"No PDB file found for task {task_id}")

    def propka_source_revision(self, task_id: str):
        """Cheap invalidation key; scientific digests remain checked on use."""
        from gmxbuilder import __version__
        from gmxbuilder.core.checkpoint_status import checkpoint_stamp, file_stamp
        from gmxbuilder.modules.input.validation import INPUT_VALIDATION_VERSION

        task_dir = self.manager.get_task_dir(task_id)
        checkpoint = task_dir / "steps" / "input"
        if all((checkpoint / name).exists() for name in ("system.json", "system.npz")):
            source = ["checkpoint", checkpoint_stamp(checkpoint)]
        else:
            path = Path(self.resolve_input_pdb(task_id))
            source = [path.name, file_stamp(path)]
        return [__version__, INPUT_VALIDATION_VERSION, source]

    def propka_manifest(self, task_id: str):
        from gmxbuilder.core.checkpoint_status import file_stamp, read_json

        directory = self.manager.get_task_dir(task_id)
        try:
            manifest = read_json(directory / ".propka-input-status.json")
            if (
                manifest.get("source") == self.propka_source_revision(task_id)
                and manifest.get("adapter") == file_stamp(directory / "propka-input.pdb")
                and manifest.get("identities")
                == file_stamp(directory / "propka-input.identities.json")
            ):
                return manifest
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def resolve_propka_pdb_path(self, task_id: str) -> str:
        """Resolve the stable pre-protonation structure used by PROPKA."""
        task_id = self._task_id_validator(task_id)
        manager = self.manager
        task_dir = manager.get_task_dir(task_id)

        from gmxbuilder.core.system import System
        from gmxbuilder.io.input_document import read_input

        revision = self.propka_source_revision(task_id)
        destination = task_dir / "propka-input.pdb"
        if self.propka_manifest(task_id) is not None:
            return str(destination)
        checkpoint = task_dir / "steps" / "input"
        if (checkpoint / "system.json").exists() and (checkpoint / "system.npz").exists():
            system = System.load_checkpoint(checkpoint)
            from gmxbuilder.modules.input.validation import INPUT_VALIDATION_VERSION

            report = system.metadata.get("input_validation", {})
            if report.get("policy_version") != INPUT_VALIDATION_VERSION or not report.get(
                "can_proceed"
            ):
                raise ValueError(
                    "Repeat Input Check before analyzing this legacy structure checkpoint"
                )
            structure = system.structure
        else:
            structure = read_input(self.resolve_input_pdb(task_id))
        from gmxbuilder.web.server_parts.protonation_identity import prepare_analysis

        destination = task_dir / "propka-input.pdb"
        prepare_analysis(structure, destination)
        from gmxbuilder.core.checkpoint_status import atomic_json, file_stamp

        if revision != self.propka_source_revision(task_id):
            raise ValueError("Input changed during PROPKA preparation; repeat analysis")
        atomic_json(
            task_dir / ".propka-input-status.json",
            {
                "source": revision,
                "adapter": file_stamp(destination),
                "identities": file_stamp(destination.with_suffix(".identities.json")),
            },
        )
        return str(destination)

    def redact_server_paths(self, value: object) -> str:
        """Remove host filesystem locations from browser-visible messages."""
        redacted = str(value)
        roots = {
            self.manager.root.expanduser().resolve(strict=False),
            Path(tempfile.gettempdir()).expanduser().resolve(strict=False),
        }
        configured_cache = os.environ.get("GMXBUILDER_CACHE_DIR")
        if configured_cache:
            roots.add(Path(configured_cache).expanduser().resolve(strict=False))
        for root in sorted((str(path) for path in roots), key=len, reverse=True):
            redacted = re.sub(
                re.escape(root) + r"(?:/[^\s,;:)\]}]+)*",
                "<server-path>",
                redacted,
            )
        return SERVER_PATH_PATTERN.sub("<server-path>", redacted)

    def sanitize_public_value(self, value: object) -> object:
        """Recursively remove internal path fields and redact path-like strings."""
        if isinstance(value, dict):
            return {
                str(key): self.sanitize_public_value(item)
                for key, item in value.items()
                if not str(key).lower().endswith("_path")
            }
        if isinstance(value, list):
            return [self.sanitize_public_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.sanitize_public_value(item) for item in value]
        if isinstance(value, str):
            return self.redact_server_paths(value)
        return value

    def public_task_state(self, state: dict) -> dict:
        """Return resumable task state without host filesystem paths."""
        public = copy.deepcopy(state)
        chemistry = public.get("ligand_chemistry_uploads")
        if isinstance(chemistry, dict):
            public["ligand_chemistry_uploads"] = {
                name: {
                    "ready": True,
                    "smiles": record.get("smiles"),
                    "sha256": record.get("sha256"),
                }
                for name, record in chemistry.items()
                if isinstance(record, dict)
            }
        uploads = public.get("cgenff_uploads")
        if isinstance(uploads, dict):
            public["cgenff_uploads"] = {
                str(name): {
                    "force_field": package.get("force_field"),
                    "cgenff_version": package.get("cgenff_version"),
                    "maximum_penalty": package.get("maximum_penalty"),
                    "ready": True,
                }
                for name, package in uploads.items()
                if isinstance(package, dict)
            }
        sanitized = self.sanitize_public_value(public)
        assert isinstance(sanitized, dict)
        return sanitized

    def public_step_result(self, task_id: str, result: dict) -> dict:
        """Replace internal StepRunner paths with task-scoped resource URLs."""
        public = dict(result)
        viewer_available = bool(public.pop("viewer_pdb_path", None))
        index_available = bool(public.pop("index_path", None))
        public.pop("zip_path", None)
        step_name = str(public.get("step", ""))
        if viewer_available and step_name:
            public["viewer_pdb_url"] = f"/api/step/{task_id}/{step_name}/viewer.pdb"
        if index_available:
            public["index_available"] = True
        sanitized = self.sanitize_public_value(public)
        assert isinstance(sanitized, dict)
        return sanitized

    def authoritative_task_zip(self, task_id: str) -> Path | None:
        """Return the current export ZIP, preferring it over legacy bundles.

        The exporter records which archive is authoritative, so this does not
        have to infer it. Modification time remains the fallback for tasks
        written before the marker existed.
        """
        from gmxbuilder.modules.export.naming import read_authoritative_archive

        manager = self.manager
        task_dir = manager.get_task_dir(task_id)
        for directory in (
            task_dir / "steps" / "export",
            manager.get_output_dir(task_id),
        ):
            if not directory.is_dir():
                continue
            recorded = read_authoritative_archive(directory)
            if recorded is not None:
                return recorded
            archives = list(directory.glob("*.zip"))
            if archives:
                return max(archives, key=lambda path: path.stat().st_mtime_ns)
        return None

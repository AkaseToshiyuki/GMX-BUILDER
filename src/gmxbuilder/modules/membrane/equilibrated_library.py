"""Force-field-specific, validated lipid conformer library.

Only conformers extracted from an explicit-solvent, semi-isotropic NPT
bilayer are accepted.  The older ``lipid_conformations`` directory contains
geometric bootstrap structures and is deliberately not searched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from gmxbuilder.modules.membrane.lipid_orientation import (
    LipidOrientationError,
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    infer_lipid_orientation,
    outward_orientation,
)


SCHEMA_VERSION = 2
MIN_CONFORMERS = 20
ACCEPTED_METHOD = "explicit_solvent_semiisotropic_npt"


def lipid_parameter_family(force_field: str, lipid_ff: str | None = None) -> str:
    """Return the lipid parameter family used as the on-disk namespace."""
    protein = str(force_field).strip().lower()
    selected = str(lipid_ff or protein).strip().lower()
    if selected == "lipid21":
        if not protein.startswith("amber"):
            raise ValueError("Lipid21 requires an Amber protein force-field family")
        return "amber-lipid21"
    if selected == "gaff2" or (protein.startswith("amber") and selected == protein):
        return "amber-gaff2"
    # CHARMM36 and CHARMM36m are separate parameter releases.  They share
    # combination rules, but not an identity contract for lipid atom types and
    # non-bonded parameters, so their NPT conformers must never alias on disk.
    if selected == "charmm36m" or (selected == protein and protein == "charmm36m"):
        return "charmm36m-lipid"
    if selected == "charmm36" or (selected == protein and protein == "charmm36"):
        return "charmm36-lipid"
    if selected == "oplsaa" or protein == "oplsaa":
        return "oplsaa-lipid"
    raise ValueError(f"Unsupported lipid force-field family: {force_field!r}/{lipid_ff!r}")


def topology_signature(atom_names: list[str], force_field: str, lipid_ff: str) -> str:
    payload = json.dumps(
        {
            "atom_names": [str(name).strip() for name in atom_names],
            "force_field": str(force_field).lower(),
            "lipid_ff": str(lipid_ff).lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LibraryEntry:
    path: Path
    metadata: dict

    @property
    def conformer_files(self) -> list[Path]:
        return sorted(self.path.glob("conf_*.npz"))


class EquilibratedLipidLibrary:
    """Read strict pre-equilibrated lipid conformers from package/cache roots."""

    _SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

    @classmethod
    def _safe_component(cls, value: str, label: str) -> str:
        component = str(value).strip()
        if (
            not cls._SAFE_COMPONENT.fullmatch(component)
            or component in {".", ".."}
            or "/" in component
            or "\\" in component
        ):
            raise ValueError(f"Unsafe {label}: {value!r}")
        return component

    @classmethod
    def _contained_entry(cls, root: Path, family: str, lipid_name: str) -> Path:
        safe_family = cls._safe_component(family, "parameter family")
        safe_name = cls._safe_component(
            str(lipid_name).strip().upper(), "lipid name"
        )
        resolved_root = root.expanduser().resolve()
        candidate = (resolved_root / safe_family / safe_name).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError("Lipid library path escapes the configured root")
        return candidate

    def __init__(self, roots: list[str | Path] | None = None):
        if roots is None:
            from gmxbuilder.runtime.prebuilt_assets import ensure_prebuilt_assets

            ensure_prebuilt_assets()
            bundled = (
                Path(__file__).resolve().parent.parent.parent
                / "data" / "lipid_equilibrated"
            )
            writable = Path(
                os.environ.get(
                    "GMXBUILDER_LIPID_LIBRARY",
                    str(Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated"),
                )
            ).expanduser()
            roots = [writable, bundled]
        self.roots = [Path(root).expanduser() for root in roots]

    def entry_dir(
        self, lipid_name: str, force_field: str, lipid_ff: str | None = None,
        *, writable: bool = False,
    ) -> Path:
        family = lipid_parameter_family(force_field, lipid_ff)
        root = self.roots[0] if writable else self.roots[-1]
        return self._contained_entry(root, family, lipid_name)

    def _candidate_dirs(
        self, lipid_name: str, force_field: str, lipid_ff: str | None = None,
    ) -> list[Path]:
        family = lipid_parameter_family(force_field, lipid_ff)
        return [
            self._contained_entry(root, family, lipid_name) for root in self.roots
        ]

    def inspect(
        self, lipid_name: str, force_field: str, lipid_ff: str | None = None,
    ) -> LibraryEntry | None:
        expected_family = lipid_parameter_family(force_field, lipid_ff)
        for directory in self._candidate_dirs(lipid_name, force_field, lipid_ff):
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, ValueError):
                continue
            files = sorted(directory.glob("conf_*.npz"))
            quality = metadata.get("quality", {})
            orientation = quality.get("orientation", {})
            valid = (
                metadata.get("schema_version") == SCHEMA_VERSION
                and metadata.get("status") == "ready"
                and metadata.get("method") == ACCEPTED_METHOD
                and metadata.get("parameter_family") == expected_family
                and int(metadata.get("n_conformations", 0)) == len(files)
                and len(files) >= MIN_CONFORMERS
                and bool(metadata.get("topology_sha256"))
                and bool(quality.get("passed"))
                and bool(orientation.get("passed"))
                and int(orientation.get("n_lipids_checked", 0)) >= len(files)
            )
            stored_names = [
                str(name).strip() for name in metadata.get("atom_names", [])
            ]
            stored_force_field = str(metadata.get("force_field", "")).lower()
            stored_lipid_ff = str(metadata.get("lipid_ff", "")).lower()
            if stored_names and stored_force_field and stored_lipid_ff:
                valid = valid and metadata.get("topology_sha256") == topology_signature(
                    stored_names, stored_force_field, stored_lipid_ff,
                )
            else:
                valid = False

            # A topology hash over atom names cannot detect a registry identity
            # correction (the former truncated GM1 entry is the motivating
            # case).  Re-canonicalize both structures so an outdated formula,
            # charge, stereoisomer or connectivity is never served as READY.
            try:
                from gmxbuilder.modules.membrane.lipids import (
                    LipidRegistry,
                    canonical_lipid_identity,
                )

                registered = LipidRegistry.get(str(lipid_name).strip().upper())
                stored_identity = canonical_lipid_identity(
                    str(metadata.get("canonical_smiles", ""))
                )
                current_identity = canonical_lipid_identity(registered.smiles)
                valid = valid and (
                    stored_identity["canonical_smiles"]
                    == current_identity["canonical_smiles"]
                )
            except KeyError:
                # User-defined lipids are not in the built-in registry; their
                # immutable topology hash and on-disk metadata remain the
                # identity contract.
                pass
            except (TypeError, ValueError):
                valid = False
            if expected_family in {"charmm36-lipid", "charmm36m-lipid"}:
                valid = valid and str(metadata.get("force_field", "")).lower() == str(force_field).lower()
                valid = valid and str(metadata.get("lipid_ff", "")).lower() == str(lipid_ff or force_field).lower()
            if valid:
                return LibraryEntry(directory, metadata)
        return None

    def has(self, lipid_name: str, force_field: str, lipid_ff: str | None = None) -> bool:
        return self.inspect(lipid_name, force_field, lipid_ff) is not None

    def load_one(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        entry = self.inspect(lipid_name, force_field, lipid_ff)
        if entry is None:
            raise FileNotFoundError(
                f"No validated NPT conformer library for {lipid_name.upper()} "
                f"under {lipid_parameter_family(force_field, lipid_ff)}"
            )
        generator = rng or np.random.default_rng()
        path = entry.conformer_files[int(generator.integers(len(entry.conformer_files)))]
        with np.load(path, allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=float)
            atom_names = [str(value) for value in data["atom_names"].tolist()]
        if coords.shape != (len(atom_names), 3) or not np.isfinite(coords).all():
            raise ValueError(f"Corrupt lipid conformer: {path}")
        expected = entry.metadata.get("atom_names", [])
        if expected and atom_names != expected:
            raise ValueError(f"Atom order does not match metadata: {path}")
        try:
            profile = infer_lipid_orientation(coords, atom_names)
            projection, cosine = outward_orientation(profile, upper=True)
        except LipidOrientationError as exc:
            raise ValueError(f"Invalid lipid orientation in {path}: {exc}") from exc
        if projection < MIN_INWARD_PROJECTION_NM or cosine < MIN_INWARD_COSINE:
            raise ValueError(
                f"Lipid conformer does not have an outward polar head and inward "
                f"hydrophobic region: {path}"
            )
        return coords, atom_names

    def coverage(self, force_fields: list[str] | None = None) -> list[dict]:
        """Return all scientifically compatible built-in library jobs."""
        from gmxbuilder.modules.forcefield.lipid_policy import (
            amber_lipid_backend,
            charmm_lipid_capability,
        )
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        fields = force_fields or [
            "amber14sb", "charmm36m", "charmm36", "amber99sb-ildn", "amber99sb",
        ]
        jobs = []
        for force_field in fields:
            for lipid_name in LipidRegistry.list():
                if force_field.startswith("amber"):
                    lipid_ff, _reason = amber_lipid_backend([lipid_name])
                    compatible = lipid_ff not in {None, "none"}
                else:
                    lipid_ff = force_field
                    compatible = charmm_lipid_capability(
                        lipid_name, force_field
                    )[0]
                if not compatible:
                    continue
                jobs.append({
                    "lipid_name": lipid_name,
                    "force_field": force_field,
                    "lipid_ff": lipid_ff,
                    "parameter_family": lipid_parameter_family(force_field, lipid_ff),
                    "ready": self.has(lipid_name, force_field, lipid_ff),
                })
        return jobs


_library: EquilibratedLipidLibrary | None = None
_task_library: ContextVar[EquilibratedLipidLibrary | None] = ContextVar(
    "gmxbuilder_task_lipid_library", default=None
)


def get_equilibrated_lipid_library() -> EquilibratedLipidLibrary:
    scoped = _task_library.get()
    if scoped is not None:
        return scoped
    global _library
    if _library is None:
        _library = EquilibratedLipidLibrary()
    return _library


@contextmanager
def task_equilibrated_library(
    library: EquilibratedLipidLibrary,
) -> Iterator[None]:
    """Use a task-owned conformer library during one task execution."""
    token = _task_library.set(library)
    try:
        yield
    finally:
        _task_library.reset(token)

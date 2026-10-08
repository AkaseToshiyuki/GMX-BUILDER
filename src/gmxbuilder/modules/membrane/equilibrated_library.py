"""Force-field-specific, validated lipid conformer library.

Only conformers extracted from an explicit-solvent, semi-isotropic NPT
bilayer are accepted. Geometry-only bootstrap structures are deliberately
excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from gmxbuilder.modules.membrane.lipid_orientation import (
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    LipidOrientationError,
    infer_lipid_orientation,
    outward_orientation,
)

SCHEMA_VERSION = 4
MIN_CONFORMERS = 20
ACCEPTED_METHOD = "explicit_solvent_semiisotropic_npt"


@lru_cache(maxsize=128)
def _assembled_admission(
    payload: str, current_protocol_digest: str, autocorrelation_method: str
) -> bool:
    """Cache immutable numeric evidence; installed parameters are checked outside.

    Complete metadata, protocol and estimator identity form the key, so edits
    and analysis upgrades cannot inherit an earlier acceptance decision. Local
    raw observations are recomputed; obsolete bulk diagnostics cannot certify
    equilibrium, but do not discard valid construction-only evidence.
    """
    from gmxbuilder.modules.membrane.v4_evidence import assembled_evidence_valid

    return assembled_evidence_valid(json.loads(payload))


def configured_library_root() -> Path:
    """Resolve the V4 publication root, migrating the historical default override."""
    old = Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated"
    current = old.with_name("lipid_equilibrated_v4") / "library"
    configured = Path(os.environ.get("GMXBUILDER_LIPID_LIBRARY", current)).expanduser()
    return current if configured.resolve() == old.resolve() else configured


def conformer_files(directory: Path) -> list[Path]:
    """Keep source indices ordered beyond the four-digit filename minimum."""
    return sorted(directory.glob("conf_*.npz"), key=lambda p: int(p.stem.removeprefix("conf_")))


def lipid_parameter_family(force_field: str, lipid_ff: str | None = None) -> str:
    """Return the lipid parameter family used as the on-disk namespace."""
    protein = str(force_field).strip().lower()
    selected = str(lipid_ff or protein).strip().lower()
    if selected == "lipid21":
        if not protein.startswith("amber"):
            raise ValueError("Lipid21 requires an Amber protein force-field family")
        return "amber-lipid21"
    if selected in {"gaff2", "amber-mixed"} or (
        protein.startswith("amber") and selected == protein
    ):
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
    admission_stamp: tuple | None = None
    admitted_files: tuple[Path, ...] | None = None
    validation_summary: dict | None = None

    @property
    def conformer_files(self) -> list[Path]:
        return (
            list(self.admitted_files)
            if self.admitted_files is not None
            else conformer_files(self.path)
        )


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
        safe_name = cls._safe_component(str(lipid_name).strip().upper(), "lipid name")
        resolved_root = root.expanduser().resolve()
        candidate = (resolved_root / safe_family / safe_name).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError("Lipid library path escapes the configured root")
        return candidate

    def __init__(self, roots: list[str | Path] | None = None):
        self.require_v4 = roots is None
        if roots is None:
            from gmxbuilder.runtime.prebuilt_assets import ensure_prebuilt_assets

            ensure_prebuilt_assets()
            roots = [configured_library_root()]
        self.roots = [Path(root).expanduser() for root in roots]

    def entry_dir(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        writable: bool = False,
    ) -> Path:
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        lipid_ff = lipid_backend_for(lipid_name, lipid_ff) or None
        family = lipid_parameter_family(force_field, lipid_ff)
        root = self.roots[0] if writable else self.roots[-1]
        return self._contained_entry(root, family, lipid_name)

    def _candidate_dirs(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
    ) -> list[Path]:
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        lipid_ff = lipid_backend_for(lipid_name, lipid_ff) or None
        family = lipid_parameter_family(force_field, lipid_ff)
        return [self._contained_entry(root, family, lipid_name) for root in self.roots]

    def inspect(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        candidate_directory: Path | None = None,
    ) -> LibraryEntry | None:
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        lipid_ff = lipid_backend_for(lipid_name, lipid_ff) or None
        expected_family = lipid_parameter_family(force_field, lipid_ff)
        directories = (
            [Path(candidate_directory)]
            if candidate_directory is not None
            else self._candidate_dirs(lipid_name, force_field, lipid_ff)
        )
        for directory in directories:
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                from gmxbuilder.modules.membrane.conformer_reader import entry_stamp

                admitted_stamp = entry_stamp(directory)
                payload = metadata_path.read_text()
                metadata = json.loads(payload)
            except (OSError, ValueError):
                continue
            from gmxbuilder.modules.membrane.parameter_provenance import parameters_current

            if self.require_v4 and not metadata.get("v4_protocol"):
                continue
            if not parameters_current(metadata):
                continue
            try:
                files = conformer_files(directory)
            except ValueError:
                continue
            quality = metadata.get("quality", {})
            from gmxbuilder.modules.membrane.initial_water import initial_water_evidence_valid

            orientation = quality.get("orientation", {})
            valid = (
                metadata.get("schema_version") == SCHEMA_VERSION
                and metadata.get("coordinate_handedness") == "preserved"
                and metadata.get("leaflet_transform") == "proper_rotation"
                and metadata.get("status") == "ready"
                and metadata.get("method") == ACCEPTED_METHOD
                and metadata.get("parameter_family") == expected_family
                and int(metadata.get("n_conformations", 0)) == len(files)
                and len(files) >= MIN_CONFORMERS
                and bool(metadata.get("topology_sha256"))
                and bool(quality.get("passed"))
                and initial_water_evidence_valid(quality.get("initial_water_exclusion"))
                and bool(orientation.get("passed"))
                and int(orientation.get("n_lipids_checked", 0)) >= len(files)
            )
            stored_names = [str(name).strip() for name in metadata.get("atom_names", [])]
            stored_force_field = str(metadata.get("force_field", "")).lower()
            stored_lipid_ff = str(metadata.get("lipid_ff", "")).lower()
            if stored_names and stored_force_field and stored_lipid_ff:
                valid = valid and metadata.get("topology_sha256") == topology_signature(
                    stored_names,
                    stored_force_field,
                    stored_lipid_ff,
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
                if stored_identity["canonical_smiles"] != current_identity["canonical_smiles"]:
                    # The recorded SMILES is an annotation, not necessarily what
                    # the entry was built from, so a mismatch alone does not mean
                    # the entry holds the wrong molecule. It depends on which
                    # backend defined the chemistry: GAFF2 builds the molecule
                    # from this SMILES, while CHARMM builds from its RTP and
                    # Lipid21 from its ITP and only copy the string in as a note.
                    #
                    # Correcting 46 lipid structures made 162 entries disagree
                    # here. Checking the molecule each entry actually contains
                    # separates them exactly along that line: all 48 GAFF2
                    # entries hold the superseded molecule and must be refused,
                    # and all 114 CHARMM and Lipid21 entries hold the correct
                    # one and were only carrying a stale note.
                    valid = valid and len(stored_names) == _explicit_atom_count(registered.smiles)
            except KeyError:
                # User-defined lipids are not in the built-in registry; their
                # immutable topology hash and on-disk metadata remain the
                # identity contract.
                pass
            except (TypeError, ValueError):
                valid = False
            # A GAFF2 entry is addressed by atom name, and the names come from
            # ACPYPE, which derives them from the SMILES. Where the registry
            # SMILES has been rewritten, ACPYPE can return the same molecule in a
            # different order -- CER16 does -- and then the conformers here and
            # the topology the writer builds disagree about which atom is which.
            # The build fails at that comparison with an error about coordinate
            # order; refusing the entry instead falls back to a generated
            # conformer, which is at least self-consistent. The template is only
            # consulted when it is already fitted: fitting one costs half an hour
            # of AM1-BCC and this is a lookup.
            if valid and expected_family == "amber-gaff2":
                try:
                    from gmxbuilder.modules.forcefield.gaff_backend import (
                        cached_gaff_template,
                    )
                    from gmxbuilder.modules.membrane.lipids import LipidRegistry

                    registered = LipidRegistry.get(str(lipid_name).strip().upper())
                    template = cached_gaff_template(
                        str(lipid_name).strip().upper(),
                        registered.smiles,
                        registered.charge,
                    )
                except (KeyError, TypeError, ValueError):
                    template = None
                if template is not None and tuple(template.atom_names) != tuple(stored_names):
                    valid = False
            if expected_family in {"charmm36-lipid", "charmm36m-lipid"}:
                valid = (
                    valid
                    and str(metadata.get("force_field", "")).lower() == str(force_field).lower()
                )
                valid = (
                    valid
                    and str(metadata.get("lipid_ff", "")).lower()
                    == str(lipid_ff or force_field).lower()
                )
            if valid and metadata.get("v4_protocol"):
                from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

                try:
                    from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD

                    current = resolve_protocol(lipid_name, expected_family)["sha256"]
                    valid = _assembled_admission(payload, current, AUTOCORRELATION_METHOD)
                except (ValueError, KeyError, TypeError):
                    valid = False
                if valid and metadata.get("reused_from"):
                    from gmxbuilder.modules.membrane.v4_reuse import reused_coordinates_valid

                    valid = reused_coordinates_valid(directory, metadata)
            if valid:
                if entry_stamp(directory) != admitted_stamp:
                    raise ValueError("Conformer library changed during admission")
                return LibraryEntry(directory, metadata, admitted_stamp, tuple(files))
        return None

    def has(self, lipid_name: str, force_field: str, lipid_ff: str | None = None) -> bool:
        return self.inspect(lipid_name, force_field, lipid_ff) is not None

    def inspect_failure(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        min_npt_ps: float = 500.0,
    ) -> LibraryEntry | None:
        """Return a current, production-length scientific rejection.

        A failed geometry gate is a classified unavailable result for the
        current library schema, not a transient queue error.  Retaining and
        recognizing it prevents an hourly service from repeating the exact
        same deterministic NPT calculation forever.  A schema bump or an
        explicit manual ``--force`` build remains the retry mechanism after a
        scientific protocol change.
        """
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        lipid_ff = lipid_backend_for(lipid_name, lipid_ff or force_field)
        expected_family = lipid_parameter_family(force_field, lipid_ff)
        expected_force_field = str(force_field).strip().lower()
        expected_lipid_ff = str(lipid_ff or force_field).strip().lower()
        for directory in self._candidate_dirs(lipid_name, force_field, lipid_ff):
            failed_directory = directory.with_name(directory.name + ".failed")
            metadata_path = failed_directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
                atom_names = [str(name).strip() for name in metadata.get("atom_names", [])]
                valid = (
                    metadata.get("schema_version") == SCHEMA_VERSION
                    and metadata.get("status") == "failed"
                    and metadata.get("method") == ACCEPTED_METHOD
                    and metadata.get("parameter_family") == expected_family
                    and str(metadata.get("force_field", "")).lower() == expected_force_field
                    and str(metadata.get("lipid_ff", "")).lower() == expected_lipid_ff
                    and not bool(metadata.get("test_mode"))
                    and float(metadata.get("npt_ps", 0.0)) >= float(min_npt_ps)
                    and bool(atom_names)
                    and metadata.get("topology_sha256")
                    == topology_signature(atom_names, force_field, expected_lipid_ff)
                    and not bool(metadata.get("quality", {}).get("passed"))
                )
                from gmxbuilder.modules.membrane.parameter_provenance import parameters_current

                valid = valid and parameters_current(metadata)
            except (OSError, TypeError, ValueError):
                continue
            if not valid:
                continue
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
                valid = stored_identity["canonical_smiles"] == current_identity["canonical_smiles"]
            except (KeyError, TypeError, ValueError):
                valid = False
            if valid:
                return LibraryEntry(failed_directory, metadata)
        return None

    def load_one(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        from gmxbuilder.modules.membrane.conformer_reader import prepared_entry, stamp

        prepared = prepared_entry(self, lipid_name, force_field, lipid_ff)
        entry = (
            prepared.entry
            if prepared is not None
            else self.inspect(lipid_name, force_field, lipid_ff)
        )
        if entry is None:
            raise FileNotFoundError(
                f"No validated NPT conformer library for {lipid_name.upper()} "
                f"under {lipid_parameter_family(force_field, lipid_ff)}"
            )
        generator = rng or np.random.default_rng()
        files = prepared.files if prepared is not None else entry.conformer_files
        if entry.metadata.get("conformer_sampling") == "equal-replica-then-uniform-conformer":
            if prepared is not None:
                sources = prepared.sources
                by_name = prepared.by_name
            else:
                groups = {}
                for item in entry.metadata["conformer_provenance"]:
                    groups.setdefault(item["replica"], []).append(item["file"])
                sources = [groups[key] for key in sorted(groups)]
                by_name = {path.name: path for path in files}
            selected = sources[int(generator.integers(len(sources)))]
            name = selected[int(generator.integers(len(selected)))]
            if name not in by_name:
                raise ValueError("Conformer provenance references an absent coordinate file")
            path = by_name[name]
        else:
            path = files[int(generator.integers(len(files)))]
        if prepared is not None and stamp(path) != prepared.stamps[path]:
            raise ValueError("Conformer coordinates changed during construction")
        with np.load(path, allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=float)
            atom_names = [str(value) for value in data["atom_names"].tolist()]
        if prepared is not None and stamp(path) != prepared.stamps[path]:
            raise ValueError("Conformer coordinates changed while reading")
        if coords.shape != (len(atom_names), 3) or not np.isfinite(coords).all():
            raise ValueError(f"Corrupt lipid conformer: {path}")
        expected = entry.metadata.get("atom_names", [])
        if expected and atom_names != expected:
            raise ValueError(f"Atom order does not match metadata: {path}")
        if entry.metadata.get("conformer_validation", {}).get("intrinsic_geometry_passed"):
            from gmxbuilder.geometry.molecular_identity import validate_stereochemistry
            from gmxbuilder.modules.membrane.local_conformations import validate_intrinsic_geometry

            definition = entry.metadata["replica_analyses"][0]["local_conformations"]["definition"]
            validate_intrinsic_geometry(
                coords, definition["elements"], tuple(map(tuple, definition["bonds"]))
            )
            validate_stereochemistry(
                definition["smiles"],
                tuple(definition["elements"]),
                tuple(map(tuple, definition["bonds"])),
                coords,
            )
            return coords, atom_names
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
            amber_lipid_backend_candidates,
            charmm_lipid_capability,
        )
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        fields = force_fields or [
            "amber14sb",
            "charmm36m",
            "charmm36",
            "amber99sb-ildn",
            "amber99sb",
        ]
        jobs = []
        for force_field in fields:
            for lipid_name in LipidRegistry.list():
                if force_field.startswith("amber"):
                    lipid_backends = list(amber_lipid_backend_candidates([lipid_name]))
                    compatible = bool(lipid_backends)
                else:
                    lipid_ff = force_field
                    compatible = charmm_lipid_capability(lipid_name, force_field)[0]
                if not compatible:
                    continue
                backends = lipid_backends if force_field.startswith("amber") else [lipid_ff]
                for lipid_ff in backends:
                    ready = self.has(lipid_name, force_field, lipid_ff)
                    jobs.append(
                        {
                            "lipid_name": lipid_name,
                            "force_field": force_field,
                            "lipid_ff": lipid_ff,
                            "parameter_family": lipid_parameter_family(force_field, lipid_ff),
                            "ready": ready,
                            "unavailable": (
                                not ready
                                and self.inspect_failure(
                                    lipid_name,
                                    force_field,
                                    lipid_ff,
                                )
                                is not None
                            ),
                        }
                    )
        return jobs


_library: EquilibratedLipidLibrary | None = None
_task_library: ContextVar[EquilibratedLipidLibrary | None] = ContextVar(
    "gmxbuilder_task_lipid_library", default=None
)


#: Atom count of a lipid with explicit hydrogens, cached: the identity check
#: below asks for it only when a recorded SMILES has gone stale, but it asks
#: repeatedly for the same few molecules while that lasts.
_EXPLICIT_ATOM_COUNTS: dict[str, int] = {}


def _explicit_atom_count(smiles: str) -> int:
    cached = _EXPLICIT_ATOM_COUNTS.get(smiles)
    if cached is not None:
        return cached
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    count = Chem.AddHs(molecule).GetNumAtoms() if molecule is not None else -1
    _EXPLICIT_ATOM_COUNTS[smiles] = count
    return count


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

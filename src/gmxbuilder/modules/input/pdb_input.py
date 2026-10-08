"""Module 0: Read a PDB file and create the initial System.

Performs structure cleaning, solute-only box computation, component
detection, and structure validation — all within the module so that
CLI and Web UI paths produce identical results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES, is_hydrogen
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules import register_module
from gmxbuilder.modules.input.modification_detection import (
    normalize_detected_modifications,
)
from gmxbuilder.modules.input.protein_repair import (
    repair_report,
    repair_standard_protein_heavy_atoms,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult

# ---------------------------------------------------------------------------
# Per-atom helper — lives at module level so it can be reused by server.py
# for the lightweight display filter (no file I/O, no Structure dependency).
# ---------------------------------------------------------------------------

_is_hydrogen = is_hydrogen


# ---------------------------------------------------------------------------
# Residue-name classification sets
# ---------------------------------------------------------------------------

_PROTEIN_RESNAMES = PROTEIN_RESNAMES

_WATER_RESNAMES = {"HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD"}

_ION_RESNAMES = {
    "NA",
    "CL",
    "K",
    "CA",
    "ZN",
    "MG",
    "CD",
    "BR",
    "I",
    "CS",
    "LI",
    "RB",
    "BA",
    "SR",
    "CU",
    "FE",
    "MN",
    "CO",
    "NI",
    "AU",
    "HG",
}

# Crystallisation precipitants and common buffer / artifact molecules.
# These are not protein, not biologically relevant, and would
# artificially inflate the solute box.
_ARTIFACT_RESNAMES = {
    "EDO",
    "GOL",
    "ACT",
    "BOG",
    "PEG",
    "PG4",
    "PGE",
    "1PE",
    "2PE",
    "MPD",
    "IPA",
    "TBU",
    "DMS",
    "DIO",
    "SO4",
    "PO4",
    "CIT",
    "TRS",
    "MES",
    "HEP",
    "EPE",
    "BME",
    "DTE",
    "DTV",
    "BCT",
    "BNG",
    "LDA",
    "LMT",
    "DMU",
    "FMT",
    "ACY",
    "BTB",
    "NO3",
}

# Residue name standardisation map — normalises common non-standard
# names to their canonical forms *before* component detection.
# Only used when the raw PDB has not been pre-cleaned by the server.
_RESIDUE_RENAME_MAP = {
    # HIS protonation → HIS (detection then handles all variants)
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    # CYS variants → CYS
    "CYM": "CYS",
    "CYX": "CYS",
    # ASP / GLU protonation
    "ASH": "ASP",
    "GLH": "GLU",
    # LYS neutral
    "LYN": "LYS",
    # Selenomethionine → methionine (same geometry)
    "MSE": "MET",
}


@register_module
class PDBInputModule(BaseModule):
    """Read a PDB file, clean the structure, and detect molecular components.

    Cleaning (water / hydrogen / alt-conf removal) and solute-only box
    computation are performed *inside* the module so the CLI path and
    the Web-UI path produce identical System checkpoints.
    """

    name = "input"
    description = "Read PDB file and detect components (protein, ligands, etc.)"

    # Re-export so server.py can import from the module directly.
    _PROTEIN_RESNAMES = _PROTEIN_RESNAMES

    # ── format detection ─────────────────────────────────────────────────

    @staticmethod
    def _is_cif_format(path: str | Path) -> bool:
        """Detect CIF/mmCIF format by file extension or content marker."""
        p = Path(path)
        if p.suffix.lower() in (".cif", ".mmcif"):
            return True
        try:
            with open(p) as fh:
                first_line = fh.readline(200).lstrip()
            return first_line.startswith("data_")
        except (OSError, UnicodeDecodeError):
            return False

    # ── config validation ────────────────────────────────────────────────

    def validate_config(self, config: dict) -> bool:
        from gmxbuilder.modules.input.reconstruction import OPTIONS, validate_options

        self.validate_config_keys(config, {"pdb", "task_id", "seed"} | OPTIONS)
        try:
            validate_options(config)
        except ValueError as exc:
            raise ModuleConfigError(str(exc)) from exc
        pdb_path = config.get("pdb")
        if not pdb_path:
            raise ModuleConfigError("'pdb' path is required for input module")
        if not Path(pdb_path).exists():
            raise ModuleConfigError(f"PDB file not found: {pdb_path}")
        return True

    # ── main entry point ─────────────────────────────────────────────────

    def run(self, system: System, config: dict) -> ModuleResult:
        from gmxbuilder.pipeline.progress import report_progress

        pdb_path = config["pdb"]
        log: list[str] = []

        # ---- 1. Parse (PDB or CIF) ----
        report_progress(0.05, "Parsing the uploaded structure")
        from gmxbuilder.io.input_document import read_input

        structure = read_input(pdb_path)
        from gmxbuilder.modules.input.reconstruction import reconstruct_input

        reconstruction = reconstruct_input(structure, config)
        is_cif = structure.source_info.get("format") == "mmcif"
        n_raw = structure.num_atoms
        log.append(f"Read {n_raw} atoms from {pdb_path}{' (CIF→PDB)' if is_cif else ''}")

        # ---- 2. Detect and reversibly normalize modified residues ----
        report_progress(0.2, "Detecting modified residues")
        # This must happen before generic residue-name standardisation and
        # heavy-atom repair, while the deposited PTM residue names still carry
        # their chemical meaning.
        structure, modification_report = normalize_detected_modifications(
            structure, _PROTEIN_RESNAMES
        )
        if modification_report["detected"]:
            log.append(
                "Input modifications: "
                f"{modification_report['detected']} non-standard protein residue(s) detected; "
                f"{modification_report['recognized']} mapped to registered patches"
            )
            for record in modification_report["records"]:
                if record["status"] == "recognized":
                    log.append(
                        f"Normalized {record['chain']}:{record['resid']} "
                        f"{record['original_resname']}→{record['standard_resname']}; "
                        f"recorded {record['patch_id']} for Structure Processing"
                    )
                elif record.get("warning"):
                    log.append("Modification warning: " + record["warning"])

        # ---- 2b. Residue name standardisation ----
        n_renamed = 0
        for i in range(structure.num_atoms):
            old = structure.resnames[i]
            new = _RESIDUE_RENAME_MAP.get(old)
            if new is not None:
                structure.resnames[i] = new
                n_renamed += 1
        if n_renamed:
            n_res_approx = len(set(structure.resnames))
            log.append(
                f"Residue names standardised: {n_renamed} atoms "
                f"across ~{n_res_approx} unique residue types"
            )

        # ---- 3. Structure cleaning ----
        report_progress(0.35, "Cleaning the structure")
        structure, clean_stats = self._clean_structure(structure)
        if clean_stats["n_water"]:
            log.append(f"Removed {clean_stats['n_water']} water atoms")
        if clean_stats["n_hydrogen"]:
            log.append(f"Removed {clean_stats['n_hydrogen']} hydrogen atoms")
        if clean_stats["n_alt_conf"]:
            log.append(f"Removed {clean_stats['n_alt_conf']} alternate-conformation atoms")
        n_clean = structure.num_atoms
        if n_clean < n_raw:
            log.append(f"Cleaned: {n_raw} → {n_clean} atoms ({n_raw - n_clean} removed)")

        # A water-only or hydrogen-only upload cannot seed any downstream
        # pipeline stage.  Stop here instead of allowing empty-coordinate
        # operations to produce warnings and a misleading empty checkpoint.
        if n_clean == 0:
            return ModuleResult(
                success=False,
                system=system,
                log=[
                    "Input contains no solute atoms after cleaning; "
                    "provide a PDB/mmCIF structure with at least one non-water, non-hydrogen atom."
                ],
            )

        # Reject non-finite values before centring or deriving a box.  Both
        # operations would otherwise propagate NaN/Inf into the checkpoint
        # and make later pipeline failures difficult to diagnose.
        if (
            not np.isfinite(structure.coordinates).all()
            or not np.isfinite(structure.box_vectors).all()
        ):
            return ModuleResult(
                success=False,
                system=system,
                log=[
                    "Input contains non-finite coordinates or box vectors; "
                    "replace NaN/Inf values before building the system."
                ],
            )

        # ---- 3b. Conservative protein heavy-atom repair ----
        report_progress(0.45, "Repairing incomplete protein side chains")
        from gmxbuilder.modules.input.protein_repair import normalize_repair_atom_names

        aliases = normalize_repair_atom_names(structure)
        if aliases:
            log.append(f"Normalized {aliases} ILE CD atom names to PDB CD1 before repair")
        from gmxbuilder.modules.input.validation import (
            assess_input_structure,
            read_polymer_metadata,
            validation_message,
        )

        source_metadata = system.metadata.get("input_source_metadata")
        if source_metadata is None:
            source_metadata = structure.source_info.get("polymer_metadata")
        if source_metadata is None:
            source_metadata = read_polymer_metadata(pdb_path)
        input_validation = assess_input_structure(structure, source_metadata)
        if not input_validation["can_proceed"]:
            rejected = System(
                structure=structure,
                metadata={
                    **system.metadata,
                    "input_source_metadata": source_metadata,
                    "input_validation": input_validation,
                },
            )
            return ModuleResult(
                success=False, system=rejected, log=[validation_message(input_validation)]
            )
        # Only complete standard-residue backbones with an unbroken partial
        # side chain are eligible.  Missing loops/backbone atoms and
        # disconnected fragments remain explicit errors.
        structure, repair_records = repair_standard_protein_heavy_atoms(structure)
        input_validation = assess_input_structure(structure, source_metadata)
        if input_validation["repairable_residues"]:
            raise ModuleConfigError("Protein repair left missing atoms; upload a complete model.")
        if not input_validation["can_proceed"]:
            raise ModuleConfigError(validation_message(input_validation))
        repair_metadata = repair_report(repair_records)
        repair_metadata["unrepairable_warnings"] = []
        if repair_records:
            log.append(
                "Automatic protein heavy-atom repair: "
                f"{repair_metadata['residues_repaired']} residue(s), "
                f"{repair_metadata['atoms_added']} atom(s) added with "
                f"{repair_metadata['backend']}"
            )
            for record in repair_records:
                log.append(
                    f"Repaired {record.chain or '?'}:{record.resid} {record.resname}: "
                    f"added {','.join(record.added_atoms)}"
                )
            log.append("Repair validation passed: " + repair_metadata["validation"])

        # ---- 3c. Center solute at origin ----
        # Downstream steps (orient, membrane, solvation) all assume the
        # solute is approximately centred at the coordinate origin.
        # Without centring, large PDB coordinate offsets (e.g. 80–100 Å)
        # cause box misalignment and viewer rendering issues.
        shift = self._center_solute(structure)
        if np.any(np.abs(shift) > 0.005):
            log.append(
                f"Solute centred at origin "
                f"(shift: {shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f} nm)"
            )

        # ---- 4. Box validation ----
        report_progress(0.75, "Validating the periodic box")
        # Prefer CRYST1 when it is physically reasonable; fall back to
        # solute-only extent estimation otherwise.
        dims = structure.dimensions()
        from gmxbuilder.io.cell import classify_cell

        if classify_cell(structure.box_vectors) == "deposited":
            box_source = structure.source_info.get("box_source", "provided")
        else:
            box = self._compute_solute_box(structure)
            structure.box_vectors = box
            dims = np.diag(box)
            box_source = (
                f"estimated from solute extent ({dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} nm)"
            )
        log.append(f"Box: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} nm ({box_source})")

        # ---- 5. Build System and detect components ----
        report_progress(0.85, "Detecting chains, ligands and other components")
        # The loader replaces the empty seed System with the parsed structure,
        # but build-level metadata (notably seed and simparams) originates
        # before this first stage and must remain available downstream.
        system = System(structure=structure, metadata=dict(system.metadata))
        self._detect_components(system)

        system.metadata["pdb_path"] = str(pdb_path)
        system.metadata["num_atoms"] = structure.num_atoms
        system.metadata["input_repair"] = repair_metadata
        system.metadata["input_modifications"] = modification_report
        system.metadata["input_source_metadata"] = source_metadata
        system.metadata["input_validation"] = input_validation
        system.metadata["input_reconstruction"] = reconstruction
        system.metadata["input_provenance"] = {
            key: value
            for key, value in structure.source_info.items()
            if key not in {"atoms", "connections", "polymer_metadata"}
        }
        system.metadata["input_coordinate_transform"] = {"translation_nm": shift.tolist()}

        # ---- 6. Disclose non-blocking construct decisions after hard checks ----
        report_progress(0.95, "Validating the structure")
        for issue in input_validation["warnings"]:
            log.append(issue["message"])

        # ---- 7. Component summary ----
        log.append(self._component_summary(system))

        return ModuleResult(success=True, system=system, log=log)

    # ── structure cleaning ───────────────────────────────────────────────

    def _clean_structure(self, structure: Structure) -> tuple[Structure, dict]:
        """Remove water, hydrogens and alternate-conformation duplicates.

        Returns (cleaned_structure, stats_dict).
        """
        n = structure.num_atoms
        remove = np.zeros(n, dtype=bool)

        # -- water --
        n_water = 0
        for i in range(n):
            if structure.resnames[i] in _WATER_RESNAMES:
                remove[i] = True
                n_water += 1

        # -- hydrogen --
        n_h = 0
        for i in range(n):
            if remove[i]:
                continue
            if _is_hydrogen(
                structure.atom_names[i], structure.elements[i] if structure.elements else ""
            ):
                remove[i] = True
                n_h += 1

        # -- alternate conformations --
        alt_remove = self._find_alt_conf_atoms(structure, remove)
        for idx in alt_remove:
            if not remove[idx]:
                remove[idx] = True
        n_alt = len(alt_remove)

        # -- build cleaned Structure --
        keep = ~remove
        n_kept = int(keep.sum())
        if n_kept == n:
            return structure, {"n_water": 0, "n_hydrogen": 0, "n_alt_conf": 0}

        cleaned = structure.take(np.flatnonzero(keep))
        return cleaned, {"n_water": n_water, "n_hydrogen": n_h, "n_alt_conf": n_alt}

    @staticmethod
    def _find_alt_conf_atoms(structure: Structure, already_removed: np.ndarray) -> set[int]:
        """Return indices of alternate-conformation duplicate atoms.

        For each (chain_id, resid, atom_name) tuple, keep only the atom
        with the highest occupancy.  If occupancies are equal, keep the
        first occurrence.
        """
        seen: dict[tuple[str, int, str], tuple[int, float]] = {}
        to_remove: set[int] = set()

        for i in range(structure.num_atoms):
            if already_removed[i]:
                continue
            key = (
                (structure.chain_ids[i] or "").strip(),
                structure.resids[i],
                (structure.atom_names[i] or "").strip(),
            )
            occ = (
                structure.occupancies[i]
                if structure.occupancies and i < len(structure.occupancies)
                else 1.0
            )

            if key in seen:
                raise ModuleConfigError(f"Duplicate atom identity after normalization: {key}")
            seen[key] = (i, occ)

        return to_remove

    # ── box computation ──────────────────────────────────────────────────

    def _compute_solute_box(self, structure: Structure) -> np.ndarray:
        """Compute an orthorhombic box spanning the solute atoms only.

        Excludes water, ions and crystallisation artifacts so the box
        reflects the actual macromolecular extent, not spurious crystal
        contacts.
        """
        n = structure.num_atoms
        exclude = _WATER_RESNAMES | _ION_RESNAMES | _ARTIFACT_RESNAMES
        solute_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if structure.resnames[i] in exclude:
                solute_mask[i] = False

        if not solute_mask.any():
            return np.eye(3) * 10.0

        coords = structure.coordinates[solute_mask]
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        extent = cmax - cmin

        box_size = max(extent.max() + 3.0, 4.0)
        box_size = min(box_size, 1000.0)
        return np.eye(3) * box_size

    # ── coordinate normalization ─────────────────────────────────────────

    @staticmethod
    def _center_solute(structure: Structure) -> np.ndarray:
        """Shift all coordinates so the solute centre of geometry is at origin.

        Excludes water, ions and crystallisation artifacts from the centre
        calculation so they don't bias the result.  Returns the shift vector
        that was applied (old_centre → origin).
        """
        n = structure.num_atoms
        exclude = _WATER_RESNAMES | _ION_RESNAMES | _ARTIFACT_RESNAMES
        solute_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if structure.resnames[i] in exclude:
                solute_mask[i] = False
        if not solute_mask.any():
            return np.zeros(3)
        center = structure.coordinates[solute_mask].mean(axis=0)
        structure.translate(-center)
        return center

    # ── component detection ──────────────────────────────────────────────

    def _detect_components(self, system: System) -> None:
        """Identify protein, solvent, ion and unknown components.

        Protein membership requires an explicit supported residue name.
        Same-chain crystallographic additives must remain separate; unknown
        covalent modifications are reported for explicit parameterization
        rather than inferred from chain membership.
        """
        structure = system.structure
        n = structure.num_atoms
        if n == 0:
            return

        assigned = np.zeros(n, dtype=bool)

        # Group residues once so canonical and modified nucleotides are never
        # mistaken for independent general small molecules.
        residue_atoms: dict[tuple[str, int], list[int]] = {}
        for index in range(n):
            key = (str(structure.chain_ids[index]), int(structure.resids[index]))
            residue_atoms.setdefault(key, []).append(index)

        from gmxbuilder.modules.nucleic_acid.support import (
            classify_nucleic_residue,
            nucleic_polymer_residues,
        )

        nucleic_residues = nucleic_polymer_residues(structure)

        # -- Pass 1: exact resname match --
        protein_indices: list[int] = []
        for i in range(n):
            if structure.resnames[i] in _PROTEIN_RESNAMES:
                protein_indices.append(i)
                assigned[i] = True

        if protein_indices:
            protein_indices_arr = np.array(sorted(protein_indices))
            system.add_component(
                Component(
                    name="PROTEIN",
                    kind=ComponentKind.PROTEIN,
                    atom_indices=protein_indices_arr,
                    metadata={
                        "n_residues": len(
                            {(structure.chain_ids[i], structure.resids[i]) for i in protein_indices}
                        ),
                    },
                )
            )

        # -- Nucleic-acid polymer runs --
        # One component per contiguous chain/polymer run gives the native
        # topology backend an unambiguous molecule boundary.  DNA/RNA hybrids
        # and modified residues remain classified here, then are rejected with
        # an actionable capability message at force-field selection.
        nucleic_runs: list[tuple[str, str, list[int], list[str]]] = []
        current_key: tuple[str, str] | None = None
        current_indices: list[int] = []
        current_resnames: list[str] = []
        previous_residue: tuple[str, int] | None = None
        for residue_key, indices in residue_atoms.items():
            classification = nucleic_residues.get(residue_key)
            if classification is None:
                if current_indices:
                    nucleic_runs.append((*current_key, current_indices, current_resnames))
                    current_key, current_indices, current_resnames = None, [], []
                previous_residue = None
                continue
            chain, resid = residue_key
            run_key = (chain, classification)
            contiguous = (
                current_key == run_key
                and previous_residue is not None
                and previous_residue[0] == chain
                and int(resid) >= int(previous_residue[1])
            )
            if current_indices and not contiguous:
                nucleic_runs.append((*current_key, current_indices, current_resnames))
                current_indices, current_resnames = [], []
            current_key = run_key
            current_indices.extend(indices)
            current_resnames.append(str(structure.resnames[indices[0]]).strip().upper())
            previous_residue = residue_key
        if current_indices:
            nucleic_runs.append((*current_key, current_indices, current_resnames))

        for run_number, (chain, polymer_type, indices, resnames) in enumerate(
            nucleic_runs, start=1
        ):
            for index in indices:
                assigned[index] = True
            unsupported = sorted(
                {
                    name
                    for name in resnames
                    if classify_nucleic_residue(name) == "modified"
                    or classify_nucleic_residue(name) is None
                }
            )
            system.add_component(
                Component(
                    name=f"NUCLEIC_{chain or run_number}",
                    kind=ComponentKind.NUCLEIC_ACID,
                    atom_indices=np.asarray(indices, dtype=int),
                    metadata={
                        "polymer_type": polymer_type,
                        "chain_id": chain,
                        "n_residues": len(resnames),
                        "residue_names": resnames,
                        "unsupported_residues": unsupported,
                    },
                )
            )

        # -- Solvent --
        solvent_indices = [
            i for i in range(n) if not assigned[i] and structure.resnames[i] in _WATER_RESNAMES
        ]
        if solvent_indices:
            for idx in solvent_indices:
                assigned[idx] = True
            system.add_component(
                Component(
                    name="SOLVENT",
                    kind=ComponentKind.SOLVENT,
                    atom_indices=np.array(solvent_indices),
                )
            )

        # -- Ions --
        ion_indices = [
            i for i in range(n) if not assigned[i] and structure.resnames[i] in _ION_RESNAMES
        ]
        if ion_indices:
            for idx in ion_indices:
                assigned[idx] = True
            system.add_component(
                Component(
                    name="IONS",
                    kind=ComponentKind.IONS,
                    atom_indices=np.array(ion_indices),
                )
            )

        # -- Remaining → UNKNOWN (ligands, cofactors, additives, etc.) --
        unknown_indices = [i for i in range(n) if not assigned[i]]
        if unknown_indices:
            system.add_component(
                Component(
                    name="UNKNOWN",
                    kind=ComponentKind.UNKNOWN,
                    atom_indices=np.array(unknown_indices),
                )
            )

    # ── logging helpers ──────────────────────────────────────────────────

    @staticmethod
    def _component_summary(system: System) -> str:
        """Build a one-line summary of each component."""
        parts: list[str] = ["Components:"]
        for comp in system.components:
            n_atoms = len(comp.atom_indices)
            extra = ""
            if comp.kind == ComponentKind.PROTEIN:
                # Count unique residues
                structure = system.structure
                n_res = len({structure.resids[i] for i in comp.atom_indices})
                chains = sorted({(structure.chain_ids[i] or "").strip() for i in comp.atom_indices})
                extra = f", {len(chains)} chain(s), {n_res} residues"
            elif comp.kind == ComponentKind.NUCLEIC_ACID:
                polymer = str(comp.metadata.get("polymer_type", "nucleic acid"))
                residues = int(comp.metadata.get("n_residues", 0))
                chain = str(comp.metadata.get("chain_id", "")) or "?"
                extra = f", chain {chain}, {residues} {polymer} residue(s)"
                unsupported = comp.metadata.get("unsupported_residues", [])
                if unsupported:
                    extra += f", unsupported modifications={','.join(unsupported)}"
            elif comp.kind == ComponentKind.UNKNOWN:
                # List resname counts for unknown molecules
                from collections import Counter

                structure = system.structure
                rn_counts = Counter(structure.resnames[i] for i in comp.atom_indices)
                if rn_counts:
                    items = [f"{rn}×{cnt}" for rn, cnt in rn_counts.most_common(6)]
                    if len(rn_counts) > 6:
                        items.append(f"+{len(rn_counts) - 6} more")
                    extra = ": " + ", ".join(items)
            parts.append(f"  {comp.name} ({n_atoms} atoms{extra})")
        return "\n".join(parts)

"""Martini 3 input audit; independent from atomistic input modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.cif import CIFParser
from gmxbuilder.io.pdb import PDBParser, PDBWriter
from gmxbuilder.core.structure import Structure
from gmxbuilder.modules.coarse_grained.common import (
    STANDARD_PROTEIN_RESIDUES,
    strict_bool,
    task_step_dir,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGInputModule(BaseModule):
    name = "cg_input"
    description = "Audit an atomistic structure for Martini 3 mapping"

    @staticmethod
    def _is_cif(source: Path) -> bool:
        """Detect mmCIF independently of the atomistic input workflow."""
        if source.suffix.lower() in {".cif", ".mmcif"}:
            return True
        try:
            with source.open(encoding="utf-8-sig", errors="replace") as handle:
                first_line = handle.readline(200).lstrip()
        except OSError:
            return False
        return first_line.startswith("data_")

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "pdb",
                "include_protein",
                "environment",
                "seed",
                "_task_dir",
                "_step_dir",
            },
        )
        environment = str(config.get("environment", "bilayer")).lower()
        if environment not in {"solution", "bilayer"}:
            raise ModuleConfigError("environment must be solution or bilayer")
        include_protein = strict_bool(config, "include_protein", True)
        if not include_protein and environment != "bilayer":
            raise ModuleConfigError("Protein-free mode is only available for a bilayer")
        if include_protein and not config.get("pdb"):
            raise ModuleConfigError("Upload a protein structure or select protein-free bilayer")
        return True

    def run(self, system, config: dict) -> ModuleResult:
        include_protein = strict_bool(config, "include_protein", True)
        environment = str(config.get("environment", "bilayer")).lower()
        output = system.copy()
        output.metadata.update(
            {
                "cg_environment": environment,
                "cg_include_protein": include_protein,
                "resolution": "coarse-grained",
                "force_field": "martini3",
            }
        )
        if not include_protein:
            return ModuleResult(
                True,
                output,
                [
                    "Protein-free Martini 3 bilayer requested",
                    "No atomistic structure will be mapped",
                ],
            )

        source = Path(str(config["pdb"])).resolve()
        if not source.is_file() or source.is_symlink():
            raise ModuleConfigError("Uploaded protein structure is unavailable")
        source_is_cif = self._is_cif(source)
        parsed = CIFParser().parse(source) if source_is_cif else PDBParser().parse(source)
        water_names = {"HOH", "WAT", "SOL", "TIP", "TIP3", "SPC", "SPCE"}
        keep = np.asarray(
            [str(name).strip().upper() not in water_names for name in parsed.resnames], dtype=bool
        )
        ignored_water_atoms = int(np.count_nonzero(~keep))
        if ignored_water_atoms:
            indices = np.flatnonzero(keep)
            parsed = Structure(
                coordinates=parsed.coordinates[indices],
                box_vectors=parsed.box_vectors.copy(),
                atom_names=[parsed.atom_names[index] for index in indices],
                resnames=[parsed.resnames[index] for index in indices],
                resids=[parsed.resids[index] for index in indices],
                chain_ids=[parsed.chain_ids[index] for index in indices],
                segids=[parsed.segids[index] for index in indices],
                elements=[parsed.elements[index] for index in indices],
                occupancies=[parsed.occupancies[index] for index in indices],
                tempfactors=[parsed.tempfactors[index] for index in indices],
            )
        observed = sorted({str(name).strip().upper() for name in parsed.resnames})
        unsupported = [name for name in observed if name not in STANDARD_PROTEIN_RESIDUES]
        if unsupported:
            raise ModuleConfigError(
                "Martini 3 mapping accepts standard protein residues after ignoring "
                "crystallographic water. Ligands, cofactors, PTMs, glycans, nucleic "
                "acids, and terminal cap residues require a separate Martini mapping; "
                "remove or parameterize: " + ", ".join(unsupported)
            )
        if parsed.num_atoms == 0:
            raise ModuleConfigError("Uploaded structure contains no protein atoms")

        destination = task_step_dir(config) / "cg_input.pdb"
        # Martinize2 consumes PDB in the next CG step.  Always write a canonical
        # PDB from the parsed Structure so an mmCIF upload cannot be copied with
        # a misleading .pdb suffix or parsed differently by the two steps.
        PDBWriter.write(parsed, destination, title="Martini 3 atomistic input")
        output.structure = parsed
        output.components = [
            Component(
                name="Atomistic Protein Input",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(parsed.num_atoms, dtype=np.int64),
                metadata={"mapping_target": "martini3001"},
            )
        ]
        output.metadata.update(
            {
                "cg_input_file": "steps/input/cg_input.pdb",
                "cg_input_residues": observed,
                "cg_input_atom_count": parsed.num_atoms,
            }
        )
        log = [
            (
                f"Accepted {parsed.num_atoms} protein atoms for Martini 3 mapping"
                f"{' (mmCIF converted to canonical PDB)' if source_is_cif else ''}"
            ),
            "No ligands, PTMs, glycans, nucleic acids, or unknown residues were discarded",
        ]
        if ignored_water_atoms:
            log.insert(
                0,
                f"Ignored {ignored_water_atoms} crystallographic water atom(s); "
                "CG solvation is generated in the dedicated solvation step",
            )
        return ModuleResult(True, output, log)

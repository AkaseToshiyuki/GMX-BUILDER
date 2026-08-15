"""Protein mapping with pinned Martinize2, owned by the CG workflow."""

from __future__ import annotations

import re
import shutil

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.modules.coarse_grained.common import (
    martinize_executable,
    molecule_types_from_topology,
    run_checked,
    strict_bool,
    task_root,
    task_step_dir,
    topology_texts_from_dir,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGMappingModule(BaseModule):
    name = "cg_mapping"
    description = "Map a standard protein to Martini 3 beads"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "protein_model",
                "secondary_structure",
                "secondary_structure_string",
                "elastic",
                "elastic_force",
                "elastic_lower",
                "elastic_upper",
                "seed",
                "_task_dir",
                "_step_dir",
            },
        )
        model = str(config.get("protein_model", "folded")).lower()
        if model not in {"folded", "tm_helix", "disordered"}:
            raise ModuleConfigError("protein_model must be folded, tm_helix, or disordered")
        secondary = str(config.get("secondary_structure", "auto")).lower()
        if secondary not in {"auto", "manual"}:
            raise ModuleConfigError("secondary_structure must be auto or manual")
        if secondary == "manual":
            sequence = str(config.get("secondary_structure_string", "")).strip().upper()
            if not sequence or not re.fullmatch(r"[HBEGITSCT]+", sequence):
                raise ModuleConfigError(
                    "Manual secondary structure must use DSSP letters H/B/E/G/I/T/S/C"
                )
        elastic = strict_bool(config, "elastic", model == "folded")
        if model == "disordered" and elastic:
            raise ModuleConfigError(
                "A generic elastic network is not allowed for disordered proteins"
            )
        force = float(config.get("elastic_force", 700.0))
        lower = float(config.get("elastic_lower", 0.5))
        upper = float(config.get("elastic_upper", 0.9))
        if elastic and not (500.0 <= force <= 1500.0 and 0.0 <= lower < upper <= 1.2):
            raise ModuleConfigError(
                "Elastic network requires force 500-1500 kJ/mol/nm² and 0 <= lower < upper <= 1.2 nm"
            )
        return True

    def run(self, system, config: dict) -> ModuleResult:
        if not system.metadata.get("cg_include_protein", True):
            output = system.copy()
            output.metadata.update(
                {
                    "cg_mapping": {"status": "not_applicable", "molecule_types": []},
                    "cg_topology_texts": {},
                }
            )
            return ModuleResult(True, output, ["Protein mapping skipped for protein-free bilayer"])

        root = task_root(config)
        input_path = root / str(system.metadata.get("cg_input_file", ""))
        if not input_path.is_file() or input_path.is_symlink():
            raise ModuleConfigError("Authoritative Martini input structure is missing")
        work = task_step_dir(config) / "martinize"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        local_input = work / "input.pdb"
        shutil.copy2(input_path, local_input)

        model = str(config.get("protein_model", "folded")).lower()
        elastic = strict_bool(config, "elastic", model == "folded")
        args = [
            str(martinize_executable()),
            "-f",
            local_input.name,
            "-x",
            "cg_protein.pdb",
            "-o",
            "protein.top",
            "-ff",
            "martini3001",
            "-from",
            "charmm",
            "-resid",
            "input",
            "-p",
            "backbone",
            "-cys",
            "auto",
            "-ignh",
        ]
        if str(config.get("secondary_structure", "auto")).lower() == "manual":
            args.extend(["-ss", str(config["secondary_structure_string"]).strip().upper()])
        else:
            args.append("-dssp")
        if elastic:
            args.extend(
                [
                    "-elastic",
                    "-ef",
                    str(float(config.get("elastic_force", 700.0))),
                    "-el",
                    str(float(config.get("elastic_lower", 0.5))),
                    "-eu",
                    str(float(config.get("elastic_upper", 0.9))),
                    "-ea",
                    "0",
                    "-ep",
                    "0",
                ]
            )
        result = run_checked(args, cwd=work, timeout=600.0)

        cg_path = work / "cg_protein.pdb"
        master = work / "protein.top"
        if not cg_path.is_file() or not master.is_file():
            raise ModuleConfigError("Martinize2 completed without CG coordinates/topology")
        structure = PDBParser().parse(cg_path)
        if structure.num_atoms == 0 or not np.isfinite(structure.coordinates).all():
            raise ModuleConfigError("Martinize2 produced an empty or non-finite structure")
        molecule_types: list[str] = []
        for path in sorted(work.glob("*.itp")):
            molecule_types.extend(molecule_types_from_topology(path.read_text()))
        if not molecule_types:
            raise ModuleConfigError("Martinize2 output contains no protein molecule type")
        texts = topology_texts_from_dir(work)
        if elastic and not any(
            re.search(r";\s*Rubber band\s*(?:\r?\n)+\s*\d", text, re.IGNORECASE)
            for text in texts.values()
        ):
            raise ModuleConfigError(
                "Elastic-network mapping was requested, but Martinize2 generated "
                "no elastic bonds. Review the lower/upper cutoff or disable the "
                "elastic network for a genuinely disordered or very small protein."
            )

        output = system.copy()
        output.structure = structure
        output.components = [
            Component(
                name="Martini 3 Protein",
                kind=ComponentKind.PROTEIN,
                atom_indices=np.arange(structure.num_atoms, dtype=np.int64),
                metadata={"molecule_types": molecule_types, "elastic_network": elastic},
            )
        ]
        output.metadata.update(
            {
                "cg_topology_texts": texts,
                "cg_protein_pdb": "steps/cg_mapping/martinize/cg_protein.pdb",
                "cg_connectivity_pdb": "steps/cg_mapping/martinize/cg_protein.pdb",
                "cg_mapping": {
                    "status": "mapped",
                    "protein_model": model,
                    "elastic_network": elastic,
                    "elastic_force": float(config.get("elastic_force", 700.0)) if elastic else None,
                    "elastic_lower_nm": float(config.get("elastic_lower", 0.5))
                    if elastic
                    else None,
                    "elastic_upper_nm": float(config.get("elastic_upper", 0.9))
                    if elastic
                    else None,
                    "secondary_structure": str(config.get("secondary_structure", "auto")),
                    "molecule_types": molecule_types,
                    "beads": structure.num_atoms,
                    "protein_extent_nm": [
                        round(float(value), 4) for value in np.ptp(structure.coordinates, axis=0)
                    ],
                },
            }
        )
        tail = [line for line in result.stdout.splitlines() if line.strip()][-8:]
        return ModuleResult(
            True,
            output,
            [
                f"Mapped protein to {structure.num_atoms} Martini 3 beads",
                f"Protein molecule types: {', '.join(molecule_types)}",
                f"Elastic network: {'enabled' if elastic else 'disabled'}",
                *tail,
            ],
        )

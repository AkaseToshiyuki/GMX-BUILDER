"""Independent Martini 3 water-stage builder."""

from __future__ import annotations

from gmxbuilder.modules.coarse_grained.backend import build_with_coby, normalize_solvation
from gmxbuilder.modules.coarse_grained.common import system_from_gro
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGSolvationModule(BaseModule):
    name = "cg_solvation"
    description = "Add regular Martini water without target bulk salt"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, {"include_solvent", "salt_molarity", "seed", "_task_dir", "_step_dir"})
        return True

    def run(self, system, config: dict) -> ModuleResult:
        output = system.copy()
        normalized = normalize_solvation(config, output.metadata)
        output.metadata["cg_solvation_config"] = normalized
        if not normalized["include_solvent"]:
            output.metadata["cg_dry_export"] = True
            return ModuleResult(True, output, ["Dry bilayer selected; water and ions are disabled"])
        gro, top, _log = build_with_coby(output, config, solvate=True, final_salt=False)
        topology_text = top.read_text(encoding="utf-8")
        built = system_from_gro(gro, topology_text, metadata=output.metadata)
        built.metadata["cg_master_topology"] = topology_text
        return ModuleResult(True, built, [
            "Added regular Martini water (W)",
            "This preview includes only counterions required for charge neutrality",
            f"Solvated CG beads: {built.num_atoms}",
        ])

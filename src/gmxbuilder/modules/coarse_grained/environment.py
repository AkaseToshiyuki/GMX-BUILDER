"""Independent Martini 3 environment and membrane construction."""

from __future__ import annotations

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained.backend import (
    build_with_coby,
    normalize_environment,
    validate_protein_box,
)
from gmxbuilder.modules.coarse_grained.common import system_from_gro
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGEnvironmentModule(BaseModule):
    name = "cg_environment"
    description = "Place CG protein and construct an optional flat Martini 3 bilayer"

    _allowed = {"environment", "box_xy", "box_z", "rotate_x", "rotate_y", "rotate_z",
                "z_offset", "upper_leaflet", "lower_leaflet", "asymmetric", "seed",
                "_task_dir", "_step_dir"}

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, self._allowed)
        return True

    def run(self, system, config: dict) -> ModuleResult:
        output = system.copy()
        normalized = normalize_environment(config, output.metadata)
        if normalized["environment"] == "solution" and not normalized["include_protein"]:
            raise ModuleConfigError("A solution-phase CG task requires a protein")
        validate_protein_box(output, normalized)
        output.metadata["cg_environment_config"] = normalized
        gro, top, _log = build_with_coby(output, config, solvate=False, final_salt=False)
        topology_text = top.read_text(encoding="utf-8")
        built = system_from_gro(gro, topology_text, metadata=output.metadata)
        built.metadata["cg_master_topology"] = topology_text
        environment = normalized["environment"]
        logs = [f"Constructed dry Martini 3 {environment} environment",
                f"Box: {normalized['box_xy']:.2f} × {normalized['box_xy']:.2f} × {normalized['box_z']:.2f} nm",
                f"CG beads: {built.num_atoms}"]
        if environment == "bilayer":
            logs.append("Built independent upper and lower leaflets with tails facing the bilayer core")
        return ModuleResult(True, built, logs)

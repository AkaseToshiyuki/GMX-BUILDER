"""Topology integrity gate for the exact CG checkpoint."""

from __future__ import annotations

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained.common import molecules_table
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGTopologyModule(BaseModule):
    name = "cg_topology"
    description = "Validate immutable Martini 3 topology metadata"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, {"seed", "_task_dir", "_step_dir"})
        return True

    def run(self, system, config: dict) -> ModuleResult:
        text = str(system.metadata.get("cg_master_topology", ""))
        if "[ molecules ]" not in text or not molecules_table(text):
            raise ModuleConfigError("Final Martini topology has no [ molecules ] table")
        output = system.copy()
        output.metadata["topology_assigned"] = True
        output.metadata["topology_backend"] = "martini3-coby"
        return ModuleResult(
            True, output, ["Validated exact COBY topology; coordinates were not changed"]
        )

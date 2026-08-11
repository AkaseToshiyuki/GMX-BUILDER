"""Solvation dedicated to solution-phase systems."""

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.solvation.solvate import SolvationBuilder
from gmxbuilder.pipeline.base import ModuleResult


class SolutionSolvationBuilder(SolvationBuilder):
    """Build a six-face solvent box around a non-membrane solute."""

    description = "Solvate a solution-phase solute on all six faces"

    def run(self, system: System, config: dict) -> ModuleResult:
        if system.component_by_kind(ComponentKind.MEMBRANE):
            raise ModuleConfigError(
                "Solution Solvator cannot consume a membrane checkpoint"
            )
        result = super().run(system, config)
        if not result.system.component_by_kind(ComponentKind.SOLVENT):
            raise ModuleConfigError("Solution solvation produced no solvent component")
        return result

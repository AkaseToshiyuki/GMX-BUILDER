"""Optional solvation for protein-free bilayers."""

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.solvation.solvate import SolvationBuilder
from gmxbuilder.pipeline.base import ModuleResult


class PureMembraneSolvationBuilder(SolvationBuilder):
    """Add symmetric solvent slabs above and below a pure bilayer."""

    description = "Solvate a pure bilayer symmetrically"

    def run(self, system: System, config: dict) -> ModuleResult:
        if not system.component_by_kind(ComponentKind.MEMBRANE):
            raise ModuleConfigError("Pure bilayer solvation requires a membrane checkpoint")
        return super().run(system, config)

"""Ion placement dedicated to solution-phase systems."""

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.ions.add_ions import IonBuilder
from gmxbuilder.pipeline.base import ModuleResult


class SolutionIonBuilder(IonBuilder):
    """Replace solution water sites with the requested ions."""

    description = "Add ions to a solution-phase solvent box"

    def run(self, system: System, config: dict) -> ModuleResult:
        if system.component_by_kind(ComponentKind.MEMBRANE):
            raise ModuleConfigError("Solution ion placement cannot consume a membrane")
        if not system.component_by_kind(ComponentKind.SOLVENT):
            raise ModuleConfigError("Solution ion placement requires a solvent checkpoint")
        return super().run(system, config)

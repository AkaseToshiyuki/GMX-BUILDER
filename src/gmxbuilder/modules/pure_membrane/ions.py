"""Optional ion placement for protein-free bilayers."""

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.ions.add_ions import IonBuilder
from gmxbuilder.pipeline.base import ModuleResult


class PureMembraneIonBuilder(IonBuilder):
    """Replace water sites with ions around a pure bilayer."""

    description = "Add ions to a solvated pure bilayer"

    def run(self, system: System, config: dict) -> ModuleResult:
        if not system.component_by_kind(ComponentKind.MEMBRANE):
            raise ModuleConfigError("Pure bilayer ion placement requires a membrane")
        if not system.component_by_kind(ComponentKind.SOLVENT):
            raise ModuleConfigError(
                "Ion placement is unavailable when pure bilayer solvation is disabled"
            )
        return super().run(system, config)

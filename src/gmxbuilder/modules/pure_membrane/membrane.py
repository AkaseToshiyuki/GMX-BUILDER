"""Membrane construction for protein-free bilayers."""

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from gmxbuilder.pipeline.base import ModuleResult


class PureMembraneBuilder(MembraneBuilder):
    """Build only a lipid bilayer and enforce the absence of solute atoms."""

    description = "Build and relax a protein-free lipid bilayer"

    def run(self, system: System, config: dict) -> ModuleResult:
        if system.num_atoms or system.components:
            raise ModuleConfigError("Pure Bilayer System must start from an empty structure")
        result = super().run(system, config)
        membranes = result.system.component_by_kind(ComponentKind.MEMBRANE)
        if len(membranes) != 1 or result.system.num_atoms == 0:
            raise ModuleConfigError("Pure bilayer construction produced no membrane")
        return result

"""Pipeline orchestrator — executes modules sequentially."""

from __future__ import annotations

from collections import OrderedDict

from gmxbuilder.core.system import System
from gmxbuilder.core.exceptions import PipelineError, ModuleConfigError
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.pipeline.config import PipelineConfig


class Pipeline:
    """Orchestrates sequential execution of BaseModule instances.

    Modules are stored in an ordered dictionary and executed in order.
    Each module receives the System from the previous stage.
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._modules: OrderedDict[str, BaseModule] = OrderedDict()

    # ------------------------------------------------------------------
    # Module management
    # ------------------------------------------------------------------

    def add_module(self, module: BaseModule, after: str | None = None) -> Pipeline:
        """Add a module to the pipeline.

        If *after* is given, the module is inserted after that named stage.
        Otherwise it is appended to the end.  Raises KeyError if *after*
        is specified but not found (consistent with insert_before).
        """
        if after is not None:
            if after not in self._modules:
                raise KeyError(f"Module '{after}' not found — cannot insert after")
            # Insert after the named module
            # Insert after the named module
            items = list(self._modules.items())
            new_items = OrderedDict()
            for key, val in items:
                new_items[key] = val
                if key == after:
                    new_items[module.name] = module
            self._modules = new_items
        else:
            self._modules[module.name] = module
        return self

    def remove_module(self, name: str) -> Pipeline:
        """Remove a module by name. No-op if the module is not present."""
        self._modules.pop(name, None)
        return self

    def replace_module(self, name: str, module: BaseModule) -> Pipeline:
        """Replace the module named *name* with *module*, or append if not found."""
        if name in self._modules:
            old_pos = list(self._modules.keys()).index(name)
            items = list(self._modules.items())
            items[old_pos] = (module.name, module)
            self._modules = OrderedDict(items)
        else:
            self._modules[module.name] = module
        return self

    def insert_before(self, target: str, module: BaseModule) -> Pipeline:
        """Insert *module* before the stage named *target*."""
        if target not in self._modules:
            raise KeyError(f"Target module {target!r} not in pipeline")
        items = list(self._modules.items())
        new_items = OrderedDict()
        for key, val in items:
            if key == target:
                new_items[module.name] = module
            new_items[key] = val
        self._modules = new_items
        return self

    def list_modules(self) -> list[tuple[str, str]]:
        """Return [(name, description)] of all registered modules."""
        return [(m.name, m.description) for m in self._modules.values()]

    def get_module(self, name: str) -> BaseModule | None:
        """Return the module with the given name, or None if not present."""
        return self._modules.get(name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_system: System, config: PipelineConfig) -> ModuleResult:
        """Execute all configured modules in order.

        Modules that are not present in *config.modules* are skipped.
        """
        system = initial_system
        stage_errors: dict[str, list[str]] = {}

        for mod_name, module in self._modules.items():
            mod_config = config.modules.get(mod_name)
            if mod_config is None:
                # Module not configured — skip
                continue

            # Force-field selection precedes membrane construction, but its
            # compatibility decision depends on the later membrane config.
            # Derive this input here as well as in the Web adapter so YAML/CLI
            # pipelines cannot accidentally bypass the Amber/GAFF2 policy.
            if mod_name == "forcefield" and "lipid_names" not in mod_config:
                from gmxbuilder.modules.forcefield.lipid_policy import (
                    membrane_lipid_names,
                )

                lipid_names = membrane_lipid_names(config.modules.get("membrane", {}))
                if lipid_names:
                    mod_config = {**mod_config, "lipid_names": list(lipid_names)}

            try:
                module.validate_config(mod_config)
            except ModuleConfigError as exc:
                stage_errors[mod_name] = [str(exc)]
                break

            result = module.execute(system, mod_config)
            if not result.success:
                stage_errors[mod_name] = result.log or ["Module reported failure"]
                break

            system = result.system

            if result.warnings:
                import warnings

                for w in result.warnings:
                    warnings.warn(f"[{mod_name}] {w}")

        if stage_errors:
            raise PipelineError(stage_errors)

        return ModuleResult(success=True, system=system)

    def validate(self, config: PipelineConfig) -> list[str]:
        """Validate all configured modules. Returns list of error messages."""
        errors = []
        for mod_name, module in self._modules.items():
            mod_config = config.modules.get(mod_name)
            if mod_config is None:
                continue
            try:
                module.validate_config(mod_config)
            except ModuleConfigError as exc:
                errors.append(f"[{mod_name}] {exc}")
        return errors

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_default(cls) -> Pipeline:
        """Create the default membrane-bilayer pipeline."""
        return cls._build("default", with_membrane=True)

    @classmethod
    def create_solvator(cls) -> Pipeline:
        """Create the independently maintained solution-phase pipeline."""
        from gmxbuilder.modules.solution import (
            SolutionExportModule,
            SolutionForceFieldSelector,
            SolutionInputModule,
            SolutionIonBuilder,
            SolutionSolvationBuilder,
            SolutionStructureProcessor,
            SolutionTopologyAssigner,
        )

        pipeline = cls(name="solvator")
        pipeline.add_module(SolutionInputModule())
        pipeline.add_module(SolutionForceFieldSelector())
        pipeline.add_module(SolutionStructureProcessor())
        pipeline.add_module(SolutionSolvationBuilder())
        pipeline.add_module(SolutionIonBuilder())
        pipeline.add_module(SolutionTopologyAssigner())
        pipeline.add_module(SolutionExportModule())
        return pipeline

    @classmethod
    def create_pure_membrane(cls) -> Pipeline:
        """Create the protein-free pure bilayer pipeline."""
        from gmxbuilder.modules.pure_membrane import (
            PureMembraneBuilder,
            PureMembraneExportModule,
            PureMembraneForceFieldSelector,
            PureMembraneIonBuilder,
            PureMembraneSolvationBuilder,
            PureMembraneTopologyAssigner,
        )

        pipeline = cls(name="pure_membrane")
        pipeline.add_module(PureMembraneForceFieldSelector())
        pipeline.add_module(PureMembraneBuilder())
        pipeline.add_module(PureMembraneSolvationBuilder())
        pipeline.add_module(PureMembraneIonBuilder())
        pipeline.add_module(PureMembraneTopologyAssigner())
        pipeline.add_module(PureMembraneExportModule())
        return pipeline

    @classmethod
    def create_liquid(cls) -> Pipeline:
        """Create a pure-liquid pipeline (no protein, no membrane)."""
        from gmxbuilder.modules.solvation.solvate import SolvationBuilder
        from gmxbuilder.modules.ions.add_ions import IonBuilder
        from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
        from gmxbuilder.modules.export.exporter import ExportModule

        pipeline = cls(name="liquid")
        pipeline.add_module(SolvationBuilder())
        pipeline.add_module(IonBuilder())
        pipeline.add_module(ForceFieldAssigner())
        pipeline.add_module(ExportModule())
        return pipeline

    @classmethod
    def _build(cls, name: str, *, with_membrane: bool) -> Pipeline:
        """Shared builder — toggles membrane-related modules."""
        from gmxbuilder.modules.input.pdb_input import PDBInputModule
        from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
        from gmxbuilder.modules.modifications.processor import StructureProcessor
        from gmxbuilder.modules.membrane.orient_module import OrientModule
        from gmxbuilder.modules.membrane.builder import MembraneBuilder
        from gmxbuilder.modules.solvation.solvate import SolvationBuilder
        from gmxbuilder.modules.ions.add_ions import IonBuilder
        from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
        from gmxbuilder.modules.export.exporter import ExportModule

        pipeline = cls(name=name)
        pipeline.add_module(PDBInputModule())
        pipeline.add_module(ForceFieldSelector())
        pipeline.add_module(StructureProcessor())
        if with_membrane:
            # Structure chemistry must be finalized before orientation and membrane insertion.
            pipeline.add_module(OrientModule())
            pipeline.add_module(MembraneBuilder())
        pipeline.add_module(SolvationBuilder())
        pipeline.add_module(IonBuilder())
        pipeline.add_module(ForceFieldAssigner())
        pipeline.add_module(ExportModule())
        return pipeline

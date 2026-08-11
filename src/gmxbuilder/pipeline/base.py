"""Base classes for the module/pipeline system."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from gmxbuilder.core.system import System
from gmxbuilder.core.exceptions import ModuleConfigError


@dataclass
class ModuleResult:
    """Result returned by a module's run() method."""

    success: bool
    system: System
    log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timing: float = 0.0


class BaseModule(ABC):
    """Abstract base class for all pipeline modules.

    To add a new module type:
    1. Subclass BaseModule
    2. Set *name* and *description* class variables
    3. Implement validate_config() and run()

    Registration can be done via entry_points or programmatically
    through the module registry.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0"

    def validate_config_keys(self, config: dict, allowed: set[str]) -> None:
        """Reject unconsumed inputs instead of silently ignoring them."""
        unknown = sorted(set(config) - set(allowed))
        if unknown:
            raise ModuleConfigError(
                f"Unsupported {self.name} option(s): {', '.join(unknown)}"
            )

    @abstractmethod
    def validate_config(self, config: dict) -> bool:
        """Validate module-specific configuration.

        Should return True if valid, or raise ModuleConfigError.
        """
        ...

    @abstractmethod
    def run(self, system: System, config: dict) -> ModuleResult:
        """Execute the module.

        Receives a System (possibly from a previous pipeline stage) and
        module-specific configuration. Returns a ModuleResult with the
        (potentially modified) system.
        """
        ...

    def pre_run_hook(self, system: System, config: dict) -> None:
        """Optional hook called before run()."""
        pass

    def post_run_hook(self, result: ModuleResult) -> None:
        """Optional hook called after run()."""
        pass

    def execute(self, system: System, config: dict) -> ModuleResult:
        """Full execution with hooks and timing."""
        self.pre_run_hook(system, config)
        t0 = time.time()
        result = self.run(system, config)
        result.timing = time.time() - t0
        self.post_run_hook(result)
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

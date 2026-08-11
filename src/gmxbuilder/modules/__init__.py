"""Built-in pipeline modules and module registry.

Modules are auto-discovered from sub-packages and can be extended
via the entry_points mechanism.
"""

from __future__ import annotations

import importlib.metadata
from typing import Type

from gmxbuilder.pipeline.base import BaseModule

# Global module registry: name -> Module class
_registry: dict[str, Type[BaseModule]] = {}


def register_module(cls: Type[BaseModule]) -> Type[BaseModule]:
    """Register a module class in the global registry."""
    if not cls.name:
        raise ValueError(f"Module {cls} must define a non-empty 'name' class variable")
    _registry[cls.name] = cls
    return cls


def get_module(name: str) -> Type[BaseModule] | None:
    """Look up a registered module by name."""
    return _registry.get(name)


def list_modules() -> list[tuple[str, str]]:
    """Return [(name, description)] of all registered modules."""
    return [(cls.name, cls.description) for cls in _registry.values()]


def discover_modules() -> dict[str, Type[BaseModule]]:
    """Discover and register all available modules.

    Checks:
    1. Built-in modules (imported via sub-packages)
    2. Entry points in group "gmxbuilder.modules"
    """
    # Import built-in modules to trigger their @register_module calls
    _import_builtin_modules()

    # Discover entry points
    try:
        for ep in importlib.metadata.entry_points(group="gmxbuilder.modules"):
            try:
                cls = ep.load()
                if issubclass(cls, BaseModule):
                    register_module(cls)
            except (ImportError, AttributeError, TypeError):
                pass  # Skip broken/unloadable plugins
            except Exception:
                import logging
                logging.getLogger("gmxbuilder.modules").warning(
                    "Unexpected error loading plugin %s", ep, exc_info=True)
    except Exception:
        pass  # entry_points() can fail in some environments

    return dict(_registry)


def _import_builtin_modules() -> None:
    """Import all built-in module sub-packages to trigger registration."""
    try:
        from gmxbuilder.modules.input import pdb_input  # noqa
    except ImportError:
        pass
    try:
        from gmxbuilder.modules.membrane import orient_module  # noqa
        from gmxbuilder.modules.membrane import builder  # noqa
    except ImportError:
        pass
    try:
        from gmxbuilder.modules.modifications import processor  # noqa
    except ImportError:
        pass
        pass
    try:
        from gmxbuilder.modules.solvation import solvate  # noqa
    except ImportError:
        pass
    try:
        from gmxbuilder.modules.ions import add_ions  # noqa
    except ImportError:
        pass
    try:
        from gmxbuilder.modules.forcefield import assign  # noqa
        from gmxbuilder.modules.forcefield import selector  # noqa
    except ImportError:
        pass
    try:
        from gmxbuilder.modules.export import exporter  # noqa
    except ImportError:
        pass
    # System verification is intentionally not a pipeline module: final
    # export consumes the exact checked coordinate checkpoint, so rebuilding
    # or comparing a second representation would add no scientific evidence.

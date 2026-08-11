"""Plugin discovery and loading.

Discovers modules from:
1. Built-in modules (gmxbuilder.modules)
2. Entry points ("gmxbuilder.modules" group)
3. Filesystem plugin directories
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Type

from gmxbuilder.pipeline.base import BaseModule
from gmxbuilder.modules import register_module


class PluginLoader:
    """Discovers and loads plugin modules from various sources."""

    @classmethod
    def discover(cls) -> dict[str, Type[BaseModule]]:
        """Run full discovery and return registered modules."""
        cls._discover_builtin()
        cls._discover_entry_points()
        cls._discover_filesystem()
        from gmxbuilder.modules import _registry
        return dict(_registry)

    @classmethod
    def _discover_builtin(cls) -> None:
        """Import built-in module sub-packages to trigger @register_module calls."""
        from gmxbuilder.modules import _import_builtin_modules
        _import_builtin_modules()

    @classmethod
    def _discover_entry_points(cls) -> None:
        """Discover modules via entry_points."""
        try:
            for ep in importlib.metadata.entry_points(group="gmxbuilder.modules"):
                try:
                    cls_ = ep.load()
                    if isinstance(cls_, type) and issubclass(cls_, BaseModule):
                        register_module(cls_)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def _discover_filesystem(cls) -> None:
        """Discover plugins from GMXBUILDER_PLUGIN_PATH directories."""
        plugin_paths = []
        env_path = sys.environ.get("GMXBUILDER_PLUGIN_PATH", "")
        if env_path:
            plugin_paths.extend(env_path.split(":"))

        # Also check ~/.gmxbuilder/plugins
        user_plugin_dir = Path.home() / ".gmxbuilder" / "plugins"
        if user_plugin_dir.is_dir():
            plugin_paths.append(str(user_plugin_dir))

        for pp in plugin_paths:
            pp = Path(pp)
            if not pp.is_dir():
                continue
            for py_file in pp.glob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(
                        py_file.stem, str(py_file)
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                except Exception:
                    pass

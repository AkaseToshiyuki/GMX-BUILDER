"""Force field registry — discover and manage force field implementations."""

from __future__ import annotations

from gmxbuilder.modules.forcefield.base_ff import ForceField


class ForceFieldRegistry:
    """Registry of available force field implementations."""

    _forcefields: dict[str, type[ForceField]] = {}

    @classmethod
    def register(cls, ff_class: type[ForceField]) -> type[ForceField]:
        cls._forcefields[ff_class.name] = ff_class
        return ff_class

    @classmethod
    def get(cls, name: str) -> ForceField:
        cls._ensure_loaded()
        name = name.lower()
        if name not in cls._forcefields:
            raise KeyError(f"Unknown force field: {name!r}. Available: {cls.list()}")
        return cls._forcefields[name]()

    @classmethod
    def list(cls) -> list[str]:
        cls._ensure_loaded()
        return sorted(cls._forcefields.keys())

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Import built-in force field implementations to trigger registration."""
        try:
            from gmxbuilder.modules.forcefield import charmm36  # noqa
        except ImportError:
            pass

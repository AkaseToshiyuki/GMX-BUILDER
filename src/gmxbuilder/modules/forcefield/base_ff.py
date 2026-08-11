"""Force field abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gmxbuilder.core.topology import Topology
from gmxbuilder.core.system import System


class ForceField(ABC):
    """Abstract base class for force field implementations.

    Subclasses implement parameter assignment from a specific force field
    (CHARMM36, Amber ff19SB, etc.).
    """

    name: str = ""
    version: str = ""
    water_model: str = ""
    supported_lipids: list[str] = []
    data_dir: Path | None = None

    @abstractmethod
    def build_system_topology(self, system: System) -> Topology:
        """Assign atom types, charges, and bonded parameters to the system.

        Returns a fully populated Topology object.
        """
        ...

    @abstractmethod
    def get_ff_includes(self) -> list[str]:
        """Return #include directives for the master .top file."""
        ...

    def copy_ff_data(self, output_dir: Path) -> None:
        """Copy force field data files to the output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.data_dir and self.data_dir.is_dir():
            import shutil
            for item in self.data_dir.iterdir():
                dest = output_dir / item.name
                if item.is_dir():
                    if not dest.exists():
                        shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

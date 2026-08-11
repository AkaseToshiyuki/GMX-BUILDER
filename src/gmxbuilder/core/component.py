"""Component — a named molecular subsystem within a System."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gmxbuilder.core.enums import ComponentKind


@dataclass
class Component:
    """A named subsystem of a molecular system.

    A Component tracks a range of atom indices within the parent System,
    together with a kind (protein, membrane, solvent, etc.) and
    arbitrary metadata for module-specific state.
    """

    name: str
    kind: ComponentKind
    atom_indices: np.ndarray  # 1D integer array, views into System arrays
    metadata: dict = field(default_factory=dict)

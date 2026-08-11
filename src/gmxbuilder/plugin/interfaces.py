"""Additional plugin interfaces for extending GMXBUILDER.

These interfaces allow third-party code to extend specific subsystems
beyond the BaseModule pipeline mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class LipidPlugin(ABC):
    """Interface for adding custom lipid types.

    Implement this and register via entry_points or the LipidRegistry.
    """

    @abstractmethod
    def get_template(self) -> dict:
        """Return a dict that can be passed to LipidTemplate(**kwargs)."""
        ...

    @abstractmethod
    def get_coordinates(self) -> np.ndarray:
        """Return (N, 3) coordinates of the pre-equilibrated lipid."""
        ...

"""Force field module — topology assignment and parameter management."""

# Import all force field implementations to trigger @ForceFieldRegistry.register
from gmxbuilder.modules.forcefield import (
    amber,  # noqa: F401
    charmm36,  # noqa: F401
    opls,  # noqa: F401
)

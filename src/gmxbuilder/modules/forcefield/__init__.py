"""Force field module — topology assignment and parameter management."""

# Import all force field implementations to trigger @ForceFieldRegistry.register
from gmxbuilder.modules.forcefield import charmm36  # noqa: F401
from gmxbuilder.modules.forcefield import amber  # noqa: F401
from gmxbuilder.modules.forcefield import opls  # noqa: F401

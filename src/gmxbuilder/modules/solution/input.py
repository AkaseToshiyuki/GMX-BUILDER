"""Structure input dedicated to solution-phase systems."""

from gmxbuilder.modules.input.pdb_input import PDBInputModule


class SolutionInputModule(PDBInputModule):
    """Read the solute without importing membrane workflow state."""

    description = "Read and validate a solution-phase solute structure"

"""Topology assignment for protein-free bilayers."""

from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner


class PureMembraneTopologyAssigner(ForceFieldAssigner):
    """Assign lipid, optional water, and optional ion topology."""

    description = "Assign pure bilayer topology and parameters"

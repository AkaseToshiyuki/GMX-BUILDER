"""Topology assignment dedicated to solution-phase systems."""

from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner


class SolutionTopologyAssigner(ForceFieldAssigner):
    """Assign the topology for a solution-only system."""

    description = "Assign solution-phase topology and parameters"

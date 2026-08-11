"""Export dedicated to solution-phase systems."""

from gmxbuilder.modules.export.exporter import ExportModule


class SolutionExportModule(ExportModule):
    """Export a solution-only GROMACS package."""

    description = "Export a solution-phase GROMACS package"

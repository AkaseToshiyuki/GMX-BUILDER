"""Force-field selection dedicated to solution-phase systems."""

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector


class SolutionForceFieldSelector(ForceFieldSelector):
    """Reject membrane parameters in a solution-only workflow."""

    description = "Select solution-phase protein, ligand, and water parameters"
    supports_nucleic_acids = True

    def validate_config(self, config: dict) -> bool:
        super().validate_config(config)
        if config.get("lipid_names"):
            raise ModuleConfigError(
                "Solution Solvator does not accept membrane lipid selections"
            )
        lipid_ff = str(config.get("lipid_ff", "none")).strip().lower()
        if lipid_ff not in {"", "none"}:
            raise ModuleConfigError(
                "Solution Solvator requires lipid_ff='none'"
            )
        return True

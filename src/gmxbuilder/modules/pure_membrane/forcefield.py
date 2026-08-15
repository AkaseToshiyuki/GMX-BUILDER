"""Force-field selection for a protein-free bilayer."""

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector


class PureMembraneForceFieldSelector(ForceFieldSelector):
    """Require an explicit lipid parameter family and no ligand parameters."""

    description = "Select force field and water model for a pure bilayer"

    def validate_config(self, config: dict) -> bool:
        super().validate_config(config)
        if not config.get("lipid_names"):
            raise ModuleConfigError("Pure Bilayer System requires at least one selected lipid")
        ligand_ff = str(config.get("ligand_ff", "none")).strip().lower()
        if ligand_ff not in {"", "none"}:
            raise ModuleConfigError(
                "Pure Bilayer System does not accept ligand force-field parameters"
            )
        return True

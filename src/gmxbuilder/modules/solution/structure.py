"""Structure processing dedicated to solution-phase systems."""

from gmxbuilder.modules.modifications.processor import StructureProcessor


class SolutionStructureProcessor(StructureProcessor):
    """Apply solution-specific protein repairs and user-selected chemistry."""

    description = "Process solution-phase solute structure"

    def run(self, system, config):
        result = super().run(system, config)
        if not result.success:
            return result
        from gmxbuilder.modules.nucleic_acid.native import prepare_nucleic_acids

        prepared, nucleic_log = prepare_nucleic_acids(result.system)
        result.system = prepared
        result.log.extend(nucleic_log)
        return result

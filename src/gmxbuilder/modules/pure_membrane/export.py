"""Export for protein-free bilayers."""

from pathlib import Path

from gmxbuilder.core.system import System
from gmxbuilder.modules.export.exporter import ExportModule
from gmxbuilder.modules.export.naming import confined_archive_path, validated_system_name
from gmxbuilder.pipeline.base import ModuleResult


class PureMembraneExportModule(ExportModule):
    """Export dry or solvated pure bilayer coordinates and topology."""

    description = "Export a pure bilayer GROMACS package"

    def run(self, system: System, config: dict) -> ModuleResult:
        result = super().run(system, config)
        if config.get("write_mdp", True):
            return result

        output_dir = Path(config.get("output_dir", "./output"))
        # Must match ExportModule.run: a different default would leave the
        # parent's solvated archive beside this dry one, with nothing but the
        # timestamp to tell consumers which is current.
        system_name = validated_system_name(config.get("system_name"), default="system")
        run_script = output_dir / "run_md.sh"
        run_script.unlink(missing_ok=True)
        (output_dir / "README.txt").write_text(
            "GMXBUILDER Pure Bilayer — dry structure export\n"
            "================================================\n\n"
            "Solvation was disabled by the user. This package contains the relaxed "
            "protein-free bilayer coordinates, force-field files, topology, and "
            "index groups. It intentionally contains no solvent, ions, MDP files, "
            "or simulation launcher. Add a physically appropriate environment "
            "before using the dry bilayer for molecular dynamics.\n\n"
            "Coordinates are in structure/, the topology and its force-field "
            "database in topology/. manifest.json records how this bilayer was "
            "built and rebuilds it with `gmxbuilder build -c manifest.json`.\n",
            encoding="utf-8",
        )
        zip_path = confined_archive_path(output_dir, system_name)
        self._write_archive(output_dir, zip_path, [], include_run_script=False)
        result.log = [line for line in result.log if "run_md.sh" not in line and "mdp/" not in line]
        result.log.append("Dry bilayer export: omitted solvent, ions, MDP files, and run_md.sh")
        return result

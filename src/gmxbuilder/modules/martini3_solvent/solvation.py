"""Independent Martini 3 water-stage builder."""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained.assets import load_manifest
from gmxbuilder.modules.coarse_grained.backend import build_with_coby, normalize_solvation
from gmxbuilder.modules.coarse_grained.common import system_from_gro
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGSolvationModule(BaseModule):
    name = "cg_solvation"
    description = "Add regular Martini water without target bulk salt"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, {
            "include_solvent", "salt_molarity", "padding_nm", "seed",
            "_task_dir", "_step_dir",
        })
        return True

    def run(self, system, config: dict) -> ModuleResult:
        if system.metadata.get("cg_environment") != "solution":
            raise ModuleConfigError("Martini 3 Solvent Builder accepts only solution tasks")
        if config.get("include_solvent", True) is not True:
            raise ModuleConfigError("Martini 3 Solvent Builder requires Martini water")
        config = dict(config)
        config["include_solvent"] = True
        output = system.copy()
        normalized = normalize_solvation(config, output.metadata)
        output.metadata["cg_solvation_config"] = normalized
        if not normalized["include_solvent"]:
            output.metadata["cg_dry_export"] = True
            return ModuleResult(True, output, ["Dry bilayer selected; water and ions are disabled"])
        environment = dict(output.metadata.get("cg_environment_config") or {})
        padding = float(normalized["padding_nm"])
        protein_extent = environment.get("protein_extent_nm") or [0.0, 0.0, 0.0]
        if environment.get("environment") == "bilayer":
            interface_thickness = self._bilayer_interface_thickness(output)
            environment["box_z"] = max(
                interface_thickness + 2.0 * padding,
                float(protein_extent[2])
                + 2.0 * abs(float(environment.get("z_offset", 0.0)))
                + 1.0,
            )
            environment["headgroup_interface_thickness_nm"] = interface_thickness
        else:
            environment["box_xy"] = max(
                float(max(protein_extent[0], protein_extent[1])) + 2.0 * padding,
                5.0,
            )
            environment["box_z"] = max(
                float(protein_extent[2])
                + 2.0 * abs(float(environment.get("z_offset", 0.0)))
                + 2.0 * padding,
                6.0,
            )
        output.metadata["cg_environment_config"] = environment
        gro, top, _log = build_with_coby(output, config, solvate=True, final_salt=False)
        topology_text = top.read_text(encoding="utf-8")
        built = system_from_gro(gro, topology_text, metadata=output.metadata)
        built.metadata["cg_master_topology"] = topology_text
        return ModuleResult(True, built, [
            "Added regular Martini water (W)",
            (
                f"Applied {padding:.2f} nm padding from each bilayer interface"
                if environment.get("environment") == "bilayer"
                else f"Applied {padding:.2f} nm padding on all protein sides"
            ),
            "This preview includes only counterions required for charge neutrality",
            f"Solvated CG beads: {built.num_atoms}",
        ])

    @staticmethod
    def _bilayer_interface_thickness(system) -> float:
        """Measure DHH-like headgroup-plane separation from the dry checkpoint."""
        manifest = load_manifest()["lipids"]
        z_values: list[float] = []
        for index, (resname, atom_name) in enumerate(
            zip(system.structure.resnames, system.structure.atom_names)
        ):
            lipid = manifest.get(str(resname).upper())
            if lipid and str(atom_name).upper() in set(lipid["head_beads"]):
                z_values.append(float(system.structure.coordinates[index, 2]))
        if len(z_values) < 4:
            raise ModuleConfigError("Cannot determine both dry-bilayer headgroup interfaces")
        values = np.asarray(z_values, dtype=float)
        center = float(np.median(values))
        upper = values[values > center]
        lower = values[values < center]
        if not len(upper) or not len(lower):
            raise ModuleConfigError("Cannot determine both dry-bilayer headgroup interfaces")
        thickness = float(np.mean(upper) - np.mean(lower))
        if not 2.5 <= thickness <= 6.0:
            raise ModuleConfigError(
                f"Dry Martini headgroup separation {thickness:.3f} nm is outside 2.5-6.0 nm"
            )
        return thickness

"""Final Martini 3 system construction and scientific quality gates."""

from __future__ import annotations

from collections import Counter

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.coarse_grained.backend import build_with_coby, normalize_solvation
from gmxbuilder.modules.coarse_grained.assets import load_manifest
from gmxbuilder.modules.coarse_grained.common import (
    molecule_type_charges,
    molecules_table,
    strict_bool,
    system_from_gro,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


class CGSystemCheckModule(BaseModule):
    name = "cg_system"
    description = "Build and validate the exact Martini 3 system to export"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config, {"salt_molarity", "confirm_system", "seed", "_task_dir", "_step_dir"}
        )
        if strict_bool(config, "confirm_system", False):
            raise ModuleConfigError(
                "confirm_system cannot be set during construction; inspect the exact "
                "checkpoint and use the confirmation endpoint"
            )
        return True

    def run(self, system, config: dict) -> ModuleResult:
        if system.metadata.get("cg_environment") != "bilayer":
            raise ModuleConfigError("Martini 3 Bilayer Builder accepts only bilayer tasks")
        output = system.copy()
        previous = dict(output.metadata.get("cg_solvation_config") or {})
        normalized = normalize_solvation(
            {
                "include_solvent": previous.get("include_solvent", True),
                "salt_molarity": config.get("salt_molarity", previous.get("salt_molarity", 0.15)),
                "padding_nm": previous.get("padding_nm", 2.0),
            },
            output.metadata,
        )
        output.metadata["cg_solvation_config"] = normalized
        if normalized["include_solvent"]:
            gro, top, _log = build_with_coby(output, config, solvate=True, final_salt=True)
            topology_text = top.read_text(encoding="utf-8")
            built = system_from_gro(gro, topology_text, metadata=output.metadata)
            built.metadata["cg_master_topology"] = topology_text
        else:
            built = output
            topology_text = str(built.metadata.get("cg_master_topology", ""))

        if built.num_atoms == 0 or not np.isfinite(built.structure.coordinates).all():
            raise ModuleConfigError("Final CG system is empty or contains non-finite coordinates")
        lengths = np.linalg.norm(built.structure.box_vectors, axis=1)
        if np.any(lengths < 4.0):
            raise ModuleConfigError("Final CG periodic box is physically too small")
        residue_counts = Counter(str(name).upper() for name in built.structure.resnames)
        molecule_counts: Counter[str] = Counter()
        for name, count in molecules_table(topology_text):
            molecule_counts[name] += count
        if normalized["include_solvent"] and residue_counts["W"] == 0:
            raise ModuleConfigError("Solvated CG system contains no Martini water")
        charge_by_type = {"W": 0.0, "NA": 1.0, "CL": -1.0}
        charge_by_type.update(
            {name: float(values["charge"]) for name, values in load_manifest()["lipids"].items()}
        )
        charge_by_type.update(
            molecule_type_charges(dict(output.metadata.get("cg_topology_texts") or {}).values())
        )
        unknown_types = sorted(name for name in molecule_counts if name not in charge_by_type)
        if unknown_types:
            raise ModuleConfigError(
                "Cannot verify final net charge for molecule type(s): " + ", ".join(unknown_types)
            )
        net_charge = sum(charge_by_type[name] * count for name, count in molecule_counts.items())
        if abs(net_charge) > 1e-4:
            raise ModuleConfigError(
                f"Final Martini system is not neutral (net charge {net_charge:+.4f} e)"
            )

        actual_salt = None
        if normalized["include_solvent"]:
            salt_pairs = min(molecule_counts.get("NA", 0), molecule_counts.get("CL", 0))
            water_beads = molecule_counts.get("W", 0)
            if water_beads <= 0:
                raise ModuleConfigError("Cannot verify bulk salt without Martini water beads")
            # Regular Martini water maps four waters to one W bead.  COBY's
            # molarity target is therefore a salt-pair/W-bead ratio referenced
            # to 55.5/4 M, not to the full box volume occupied by membrane and
            # protein as well as solvent.
            water_bead_molarity = 55.5 / 4.0
            actual_salt = salt_pairs * water_bead_molarity / water_beads
            target = float(normalized["salt_molarity"])
            # One integer ion pair is the intrinsic concentration resolution.
            tolerance = max(0.01, 0.55 * water_bead_molarity / water_beads)
            if target > 0.0:
                tolerance = min(tolerance, 0.5 * target)
            if abs(actual_salt - target) > tolerance:
                raise ModuleConfigError(
                    "Final Martini salt concentration differs from the target: "
                    f"{actual_salt:.4f} M vs {target:.4f} M"
                )

        orientation = None
        solvent_layers = None
        protein_placement = None
        if output.metadata.get("cg_environment") == "bilayer":
            membrane = next((c for c in built.components if c.kind == ComponentKind.MEMBRANE), None)
            if membrane is None or len(membrane.atom_indices) == 0:
                raise ModuleConfigError("Final CG bilayer contains no validated membrane beads")
            orientation = self._validate_bilayer_orientation(built)
            requested = int(
                (output.metadata.get("cg_environment_config") or {}).get("n_lipids_per_leaflet", 0)
            )
            if requested and (
                orientation["upper_leaflet_lipids"] != requested
                or orientation["lower_leaflet_lipids"] != requested
            ):
                raise ModuleConfigError(
                    "Built Martini leaflet size differs from the explicit request: "
                    f"upper={orientation['upper_leaflet_lipids']}, "
                    f"lower={orientation['lower_leaflet_lipids']}, "
                    f"requested={requested}. The system was rejected rather than "
                    "silently changing lipid count."
                )
            if normalized["include_solvent"]:
                solvent_layers = self._validate_solvent_layers(built, orientation)
            protein_placement = self._report_protein_placement(built, orientation)
        built.metadata.update(
            {
                "cg_scientific_check": {
                    "passed": True,
                    "box_nm": lengths.tolist(),
                    "molecule_counts": dict(molecule_counts),
                    "net_charge_e": net_charge,
                    "target_salt_molarity": normalized["salt_molarity"]
                    if normalized["include_solvent"]
                    else None,
                    "actual_salt_molarity": actual_salt,
                    "bilayer_orientation": orientation,
                    "solvent_layers": solvent_layers,
                    "protein_placement": protein_placement,
                    "coordinate_source": "cg_system checkpoint",
                },
                # The exact checkpoint is now available for inspection.  A separate
                # confirmation request flips this flag without rebuilding coordinates.
                "system_confirmed": False,
            }
        )
        return ModuleResult(
            True,
            built,
            [
                "Final Martini 3 coordinates and topology passed structural checks",
                f"Target NaCl concentration: {normalized['salt_molarity']:.3f} M"
                if normalized["include_solvent"]
                else "Dry bilayer: no water or ions",
                f"Exact export checkpoint contains {built.num_atoms} beads",
                "Inspect this exact checkpoint and confirm it before finalization",
            ],
        )

    @staticmethod
    def _validate_bilayer_orientation(system) -> dict:
        """Require polar heads outside and hydrophobic beads toward the midplane."""
        lipid_manifest = load_manifest()["lipids"]
        lipids = set(lipid_manifest)
        structure = system.structure
        molecules: list[list[int]] = []
        current: list[int] = []
        previous = None
        for index, (name, resid) in enumerate(zip(structure.resnames, structure.resids)):
            key = (str(name).upper(), int(resid))
            if key[0] not in lipids:
                if current:
                    molecules.append(current)
                    current = []
                previous = None
                continue
            if previous is not None and key != previous:
                molecules.append(current)
                current = []
            current.append(index)
            previous = key
        if current:
            molecules.append(current)
        oriented = 0
        upper_heads: list[float] = []
        lower_heads: list[float] = []
        evaluated = 0
        center_markers = [
            structure.coordinates[index, 2]
            for molecule in molecules
            for index in molecule
            if str(structure.atom_names[index]).upper()
            in set(lipid_manifest[str(structure.resnames[index]).upper()]["midplane_beads"])
        ]
        if not center_markers:
            raise ModuleConfigError("Martini lipid manifest has no usable midplane beads")
        center = float(np.median(center_markers))
        for molecule in molecules:
            names = {str(structure.atom_names[index]).upper(): index for index in molecule}
            lipid_name = str(structure.resnames[molecule[0]]).upper()
            definition = lipid_manifest[lipid_name]
            head_candidates = [names[name] for name in definition["head_beads"] if name in names]
            tail_candidates = [names[name] for name in definition["tail_beads"] if name in names]
            if not head_candidates or not tail_candidates:
                raise ModuleConfigError(
                    f"Cannot validate {lipid_name}: expected head/tail beads from "
                    "the Martini lipid manifest are missing"
                )
            head_z = float(np.mean(structure.coordinates[head_candidates, 2]))
            tail_z = float(np.mean(structure.coordinates[tail_candidates, 2]))
            evaluated += 1
            if head_z >= center:
                upper_heads.append(head_z)
                oriented += int(head_z > tail_z)
            else:
                lower_heads.append(head_z)
                oriented += int(head_z < tail_z)
        if evaluated == 0 or not upper_heads or not lower_heads:
            raise ModuleConfigError("Could not identify both Martini bilayer leaflets")
        fraction = oriented / evaluated
        separation = float(np.mean(upper_heads) - np.mean(lower_heads))
        if fraction < 0.98:
            raise ModuleConfigError(
                f"Bilayer orientation failed: only {fraction:.1%} of lipids have heads facing solvent"
            )
        if not 2.5 <= separation <= 6.0:
            raise ModuleConfigError(
                f"Bilayer headgroup separation {separation:.3f} nm is outside 2.5-6.0 nm"
            )
        return {
            "evaluated_lipids": evaluated,
            "upper_leaflet_lipids": len(upper_heads),
            "lower_leaflet_lipids": len(lower_heads),
            "correct_fraction": fraction,
            "headgroup_separation_nm": separation,
            "midplane_z_nm": center,
            "lower_headgroup_z_nm": float(np.mean(lower_heads)),
            "upper_headgroup_z_nm": float(np.mean(upper_heads)),
        }

    @staticmethod
    def _validate_solvent_layers(system, orientation: dict) -> dict:
        """Require a genuine solvent region outside both bilayer interfaces."""
        structure = system.structure
        water_indices = np.array(
            [
                index
                for index, name in enumerate(structure.resnames)
                if str(name).strip().upper() == "W"
            ],
            dtype=np.int64,
        )
        if water_indices.size == 0:
            raise ModuleConfigError("Solvated bilayer contains no Martini water beads")
        water_z = structure.coordinates[water_indices, 2]
        lower = float(orientation["lower_headgroup_z_nm"])
        upper = float(orientation["upper_headgroup_z_nm"])
        # A 0.30 nm margin is larger than coordinate rounding yet smaller than
        # one regular Martini bead diameter.  It rejects one-sided/inside-only
        # solvation without prescribing a user-specific bulk-water thickness.
        below = int(np.count_nonzero(water_z < lower - 0.30))
        above = int(np.count_nonzero(water_z > upper + 0.30))
        if below < 10 or above < 10:
            raise ModuleConfigError(
                "Solvated bilayer must contain bulk Martini water on both sides "
                f"(below={below}, above={above} beads)"
            )
        return {
            "water_beads_below": below,
            "water_beads_above": above,
            "water_z_min_nm": float(np.min(water_z)),
            "water_z_max_nm": float(np.max(water_z)),
        }

    @staticmethod
    def _report_protein_placement(system, orientation: dict) -> dict | None:
        """Report membrane spanning and enforce it for explicit TM-helix mode."""
        structure = system.structure
        protein = next(
            (
                component
                for component in system.components
                if component.kind == ComponentKind.PROTEIN
            ),
            None,
        )
        if protein is None or len(protein.atom_indices) == 0:
            return None
        bb_indices = np.array(
            [
                int(index)
                for index in protein.atom_indices
                if str(structure.atom_names[int(index)]).strip().upper() == "BB"
            ],
            dtype=np.int64,
        )
        if bb_indices.size == 0:
            raise ModuleConfigError("Mapped membrane protein contains no Martini BB beads")
        bb_z = structure.coordinates[bb_indices, 2]
        lower = float(orientation["lower_headgroup_z_nm"])
        upper = float(orientation["upper_headgroup_z_nm"])
        spans = bool(float(np.min(bb_z)) < lower and float(np.max(bb_z)) > upper)
        mapping = dict(system.metadata.get("cg_mapping") or {})
        model = str(mapping.get("protein_model", "folded"))
        if model == "tm_helix" and not spans:
            raise ModuleConfigError(
                "TM-helix mode requires protein backbone beads to span both membrane interfaces; "
                "adjust rotation or Z offset"
            )
        return {
            "protein_model": model,
            "backbone_beads": int(bb_indices.size),
            "backbone_z_min_nm": float(np.min(bb_z)),
            "backbone_z_max_nm": float(np.max(bb_z)),
            "spans_both_headgroup_planes": spans,
            "interpretation": (
                "required and passed"
                if model == "tm_helix"
                else "reported for review; folded proteins require visual orientation confirmation"
            ),
        }

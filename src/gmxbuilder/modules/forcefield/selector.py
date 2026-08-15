"""Pipeline module: force field selection.

Runs early in the pipeline (after input, before structure processing)
so that downstream modules can read the chosen force field from system
metadata and adapt their behaviour accordingly (HDB hydrogen addition,
water model defaults, supported lipid filtering, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.system import System
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.modules import register_module


@register_module
class ForceFieldSelector(BaseModule):
    """Record the user's force field choice into system metadata."""

    name = "forcefield"
    description = "Select force field for the system build"

    _DEFAULT_FF = "amber14sb"
    supports_nucleic_acids = False

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "name",
                "lipid_names",
                "lipid_ff",
                "ligand_ff",
                "ligand_charges",
                "ligand_pH",
                "cgenff_parameters",
                "water_model",
                "system_name",
                "seed",
            },
        )
        name = config.get("name", self._DEFAULT_FF)
        if not isinstance(name, str):
            raise ModuleConfigError("Force field name must be a string")
        name = name.strip()
        if not name:
            raise ModuleConfigError("Force field name must not be empty")
        from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry

        if name.lower() not in ForceFieldRegistry.list():
            available = ", ".join(ForceFieldRegistry.list())
            raise ModuleConfigError(
                f"Unknown force field {name!r}. Available force fields: {available}"
            )
        configured_water = config.get("water_model")
        if configured_water is not None:
            from gmxbuilder.modules.solvation.water_models import (
                WaterRegistry,
                water_model_supported,
            )

            water_name = str(configured_water).strip().lower()
            try:
                WaterRegistry.get(water_name)
            except KeyError as exc:
                raise ModuleConfigError(str(exc)) from exc
            if not water_model_supported(name, water_name):
                raise ModuleConfigError(
                    f"Water model {water_name!r} is not bundled for force field {name!r}"
                )
        lipid_names = config.get("lipid_names", [])
        if not isinstance(lipid_names, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in lipid_names
        ):
            raise ModuleConfigError("forcefield.lipid_names must be a list of names")
        for key in ("lipid_ff", "ligand_ff"):
            if key in config and not isinstance(config[key], str):
                raise ModuleConfigError(f"{key} must be a string")
        ligand_charges = config.get("ligand_charges", {})
        if not isinstance(ligand_charges, dict):
            raise ModuleConfigError("ligand_charges must be an object")
        for ligand, charge in ligand_charges.items():
            if not isinstance(ligand, str) or not ligand.strip():
                raise ModuleConfigError("ligand_charges keys must be molecule names")
            if isinstance(charge, bool) or not isinstance(charge, int):
                raise ModuleConfigError(f"Net charge for {ligand} must be an integer")
        ligand_pH = config.get("ligand_pH", 7.0)
        if isinstance(ligand_pH, bool) or not isinstance(ligand_pH, (int, float)):
            raise ModuleConfigError("ligand_pH must be numeric")
        if not 1.0 <= float(ligand_pH) <= 13.0:
            raise ModuleConfigError("ligand_pH must be between 1.0 and 13.0")
        cgenff_parameters = config.get("cgenff_parameters", {})
        if not isinstance(cgenff_parameters, dict):
            raise ModuleConfigError("cgenff_parameters must be an object")
        for ligand, package in cgenff_parameters.items():
            if not isinstance(ligand, str) or not ligand.strip() or not isinstance(package, dict):
                raise ModuleConfigError("Each CGenFF package must be keyed by a molecule name")
            if set(package) != {"mol2_path", "str_path"} or not all(
                isinstance(value, str) and value.strip() for value in package.values()
            ):
                raise ModuleConfigError(
                    f"CGenFF package for {ligand} requires mol2_path and str_path"
                )
        system_name = config.get("system_name")
        if system_name is not None:
            if not isinstance(system_name, str) or not system_name.strip():
                raise ModuleConfigError("system_name must be a non-empty string")
            if any(
                ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for ch in system_name
            ):
                raise ModuleConfigError(
                    "system_name may contain only letters, numbers, '_' and '-'"
                )
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        requested_ff = config.get("name", self._DEFAULT_FF).strip().lower()
        nucleic_components = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
        if nucleic_components and not self.supports_nucleic_acids:
            raise ModuleConfigError(
                "DNA/RNA polymers are currently supported only by the Solution "
                "Solvator workflow; membrane embedding and Martini nucleic-acid "
                "models remain unavailable"
            )
        if nucleic_components:
            from gmxbuilder.modules.nucleic_acid.support import (
                nucleic_force_field_capability,
                validate_nucleic_backbone,
            )

            capable, capability_reason = nucleic_force_field_capability(requested_ff)
            if not capable:
                raise ModuleConfigError(
                    f"Nucleic-acid force-field selection is unavailable: "
                    f"{capability_reason}. Select CHARMM36m."
                )
            unsupported = sorted(
                {
                    str(residue)
                    for component in nucleic_components
                    for residue in component.metadata.get("unsupported_residues", [])
                }
            )
            if unsupported:
                raise ModuleConfigError(
                    "Modified or noncanonical nucleotide residue(s) require an "
                    "explicit polymer residue topology and are not automatically "
                    "converted to canonical chemistry: " + ", ".join(unsupported)
                )
            polymer_types = {
                str(component.metadata.get("polymer_type", "")) for component in nucleic_components
            }
            if "modified" in polymer_types:
                raise ModuleConfigError(
                    "A nucleic-acid polymer contains an unparameterized modified residue"
                )
            types_by_chain: dict[str, set[str]] = {}
            for component in nucleic_components:
                chain = str(component.metadata.get("chain_id", ""))
                types_by_chain.setdefault(chain, set()).add(
                    str(component.metadata.get("polymer_type", ""))
                )
            hybrid_chains = sorted(
                chain or "?" for chain, types in types_by_chain.items() if len(types) > 1
            )
            if hybrid_chains:
                raise ModuleConfigError(
                    "Covalent DNA/RNA hybrid chain(s) require explicit hybrid "
                    "terminal/linkage validation and are currently unavailable: "
                    + ", ".join(hybrid_chains)
                )
            backbone_issues = [
                issue
                for component in nucleic_components
                for issue in validate_nucleic_backbone(system.structure, component)
            ]
            if backbone_issues:
                raise ModuleConfigError(
                    "Nucleic-acid backbone continuity validation failed: "
                    + "; ".join(backbone_issues)
                )
        lipid_names = config.get("lipid_names") or []
        if not isinstance(lipid_names, (list, tuple)):
            raise ModuleConfigError("forcefield.lipid_names must be a list")
        from gmxbuilder.modules.forcefield.compatibility import (
            compatibility_report,
            enabled_values,
            molecule_groups,
        )

        report = compatibility_report(system, requested_ff, lipid_names)
        lipid_ff = str(config.get("lipid_ff", "none" if not lipid_names else "")).lower()
        ligand_names = sorted(molecule_groups(system))
        ligand_ff = str(config.get("ligand_ff", "none" if not ligand_names else "")).lower()
        if lipid_ff not in enabled_values(report["lipid_options"]):
            reasons = "; ".join(
                option.get("reason", "")
                for option in report["lipid_options"]
                if option.get("reason")
            )
            raise ModuleConfigError(
                f"Lipid force field {lipid_ff!r} is incompatible with protein "
                f"force field {requested_ff!r}. {reasons}".strip()
            )
        if ligand_ff not in enabled_values(report["ligand_options"]):
            reasons = "; ".join(
                option.get("reason", "")
                for option in report["ligand_options"]
                if option.get("reason")
            )
            raise ModuleConfigError(
                f"Small-molecule force field {ligand_ff!r} is incompatible with "
                f"protein force field {requested_ff!r}. {reasons}".strip()
            )

        system = system.copy()
        ligand_parameters: dict[str, dict] = {}
        if ligand_ff == "gaff2":
            charges = {
                str(name).upper(): value for name, value in config.get("ligand_charges", {}).items()
            }
            missing_charges = [name for name in ligand_names if name not in charges]
            if missing_charges:
                raise ModuleConfigError(
                    "Explicit integer net charge required for GAFF2 molecule(s): "
                    + ", ".join(missing_charges)
                )
            ligand_pH = float(config.get("ligand_pH", 7.0))
            system, ligand_parameters = self._parameterize_gaff2_ligands(
                system,
                charges,
                ligand_pH,
            )
        elif ligand_ff == "rtp":
            system, ligand_parameters = self._prepare_rtp_ligands(system, requested_ff)
        elif ligand_ff == "cgenff":
            packages = {
                str(name).strip().upper(): value
                for name, value in config.get("cgenff_parameters", {}).items()
            }
            missing_packages = [name for name in ligand_names if name not in packages]
            if missing_packages:
                raise ModuleConfigError(
                    "Input molecule(s) are incompatible with CGenFF until the matching "
                    "ParamChem MOL2 and STR files are uploaded: " + ", ".join(missing_packages)
                )
            system, ligand_parameters = self._parameterize_cgenff_ligands(
                system,
                packages,
                requested_ff,
            )
            high_penalty = {
                name: float(parameters["maximum_penalty"])
                for name, parameters in ligand_parameters.items()
                if parameters.get("maximum_penalty") is not None
                and float(parameters["maximum_penalty"]) >= 50.0
            }
            if high_penalty:
                details = ", ".join(
                    f"{name}={penalty:.1f}" for name, penalty in high_penalty.items()
                )
                raise ModuleConfigError(
                    "CGenFF parameters with penalty >=50 are not accepted as "
                    "simulation-ready because they require manual quantum-chemical "
                    f"validation/refitting ({details}). This capability is currently "
                    "marked unavailable rather than silently exporting uncertain terms."
                )

        ff_name = requested_ff

        from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry
        from gmxbuilder.modules.solvation.water_models import water_model_supported

        configured_water = config.get("water_model")
        water_model = (
            str(configured_water).strip().lower()
            if configured_water is not None
            else ForceFieldRegistry.get(ff_name).water_model
        )
        if not water_model_supported(ff_name, water_model):
            raise ModuleConfigError(
                f"Water model {water_model!r} is not available for the effective "
                f"force field {ff_name!r}"
            )

        # Store choices in system metadata — downstream modules read from here
        system.metadata["force_field"] = ff_name
        system.metadata["requested_force_field"] = requested_ff
        from gmxbuilder.modules.forcefield.catalog import get_force_field_profile

        profile = get_force_field_profile(ff_name)
        system.metadata["force_field_release"] = profile.release
        system.metadata["force_field_family"] = profile.family
        system.metadata["force_field_defaults_signature"] = list(profile.defaults_signature)
        system.metadata["cgenff_version"] = profile.cgenff_version
        system.metadata["lipid_ff"] = lipid_ff
        system.metadata["ligand_ff"] = ligand_ff
        system.metadata["gaff_lipids"] = (
            sorted({str(name).strip().upper() for name in lipid_names})
            if lipid_ff == "gaff2"
            else []
        )
        system.metadata["lipid21_lipids"] = (
            sorted({str(name).strip().upper() for name in lipid_names})
            if lipid_ff == "lipid21"
            else []
        )
        system.metadata["selected_lipid_names"] = sorted(
            {str(name).strip().upper() for name in lipid_names}
        )
        system.metadata["ligand_parameters"] = ligand_parameters
        if ligand_ff == "gaff2":
            system.metadata["ligand_protonation_pH"] = float(config.get("ligand_pH", 7.0))
        system.metadata["water_model"] = water_model
        system.metadata["ff_water_model"] = water_model
        if config.get("system_name"):
            system.metadata["system_name"] = config["system_name"].strip()

        log = [
            f"Force field: {ff_name} (lipids: {lipid_ff}, ligands: {ligand_ff}, "
            f"water: {water_model})"
        ]
        if nucleic_components:
            dna = sum(
                int(component.metadata.get("n_residues", 0))
                for component in nucleic_components
                if component.metadata.get("polymer_type") == "DNA"
            )
            rna = sum(
                int(component.metadata.get("n_residues", 0))
                for component in nucleic_components
                if component.metadata.get("polymer_type") == "RNA"
            )
            log.append(
                "Nucleic-acid backend: native GROMACS/CHARMM36 "
                f"({dna} DNA residue(s), {rna} RNA residue(s)); canonical polymers only"
            )
        if lipid_ff == "gaff2":
            log.append(
                f"Compatibility policy: protein force field {ff_name}; selected lipids use GAFF2"
            )
        elif lipid_ff == "lipid21":
            log.append(
                f"Compatibility policy: protein force field {ff_name}; "
                "selected lipids use exact Amber Lipid21 v1.0 parameters"
            )
        if ligand_ff == "cgenff":
            for ligand, parameters in ligand_parameters.items():
                penalty = parameters.get("maximum_penalty")
                version = parameters.get("cgenff_version") or "not declared"
                message = f"CGenFF import {ligand}: stream version {version}"
                if penalty is not None:
                    message += f", maximum penalty {penalty:.1f}"
                    if penalty >= 10:
                        message += " (review assigned charges and parameters before production MD)"
                log.append(message)
        return ModuleResult(
            success=True,
            system=system,
            log=log,
        )

    @staticmethod
    def _kabsch_transform(
        source: np.ndarray, target: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        u, _singular, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return (coordinates - source_center) @ rotation + target_center

    @classmethod
    def _parameterize_gaff2_ligands(
        cls,
        system: System,
        charges: dict[str, int],
        target_pH: float,
    ):
        from gmxbuilder.core.component import Component
        from gmxbuilder.core.structure import Structure
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_molecule

        groups = molecule_groups(system)
        templates = {
            name: prepare_gaff_molecule(
                name,
                system.structure,
                instances[0],
                charges[name],
                target_pH=target_pH,
            )
            for name, instances in groups.items()
        }
        instance_by_first = {
            indices[0]: (name, indices)
            for name, instances in groups.items()
            for indices in instances
        }
        instance_atoms = {
            index for _name, indices in instance_by_first.values() for index in indices
        }
        old_to_new: dict[int, int] = {}
        ligand_new_indices: list[int] = []
        coords: list[np.ndarray] = []
        fields = {
            key: []
            for key in (
                "atom_names",
                "resnames",
                "resids",
                "chain_ids",
                "segids",
                "elements",
                "occupancies",
                "tempfactors",
            )
        }

        def append_old(index: int):
            old_to_new[index] = len(coords)
            coords.append(system.structure.coordinates[index].copy())
            for key in fields:
                fields[key].append(getattr(system.structure, key)[index])

        # Match TopologyWriter order: macromolecules first, then each retained
        # small-molecule instance. This is required because GROMACS assigns
        # coordinate blocks in [molecules] order.
        for index in range(system.num_atoms):
            if index not in instance_atoms:
                append_old(index)
        for index in sorted(instance_by_first):
            name, indices = instance_by_first[index]
            template = templates[name]
            if (
                tuple(system.structure.atom_names[i].strip() for i in indices)
                != template.atom_names[: len(indices)]
            ):
                raise ModuleConfigError(f"GAFF2 atom order mismatch for {name}")
            transformed = cls._kabsch_transform(
                template.coordinates[: len(indices)],
                system.structure.coordinates[indices],
                template.coordinates,
            )
            start = len(coords)
            for old_index in indices:
                append_old(old_index)
            for template_index in range(len(indices), len(template.atom_names)):
                if not template.atom_names[template_index].upper().startswith("H"):
                    raise ModuleConfigError(
                        f"GAFF2 introduced unexpected non-hydrogen atom "
                        f"{template.atom_names[template_index]!r} for {name}"
                    )
                coords.append(transformed[template_index])
                fields["atom_names"].append(template.atom_names[template_index])
                fields["resnames"].append(name)
                fields["resids"].append(system.structure.resids[indices[0]])
                fields["chain_ids"].append(system.structure.chain_ids[indices[0]])
                fields["segids"].append(system.structure.segids[indices[0]])
                fields["elements"].append("H")
                fields["occupancies"].append(1.0)
                fields["tempfactors"].append(0.0)
            ligand_new_indices.extend(range(start, len(coords)))

        system.structure = Structure(
            coordinates=np.asarray(coords, dtype=float),
            box_vectors=system.structure.box_vectors.copy(),
            **fields,
        )
        new_components = []
        for component in system.components:
            if component.kind == ComponentKind.UNKNOWN:
                continue
            mapped = [old_to_new[int(index)] for index in component.atom_indices]
            new_components.append(
                Component(
                    name=component.name,
                    kind=component.kind,
                    atom_indices=np.asarray(mapped, dtype=int),
                    metadata=dict(component.metadata),
                )
            )
        new_components.append(
            Component(
                name="LIGANDS",
                kind=ComponentKind.LIGAND,
                atom_indices=np.asarray(sorted(ligand_new_indices), dtype=int),
                metadata={
                    "molecule_charges": dict(charges),
                    "n_molecules": sum(len(instances) for instances in groups.values()),
                },
            )
        )
        system.components = new_components
        parameters = {
            name: {
                "source": "gaff2",
                "net_charge": int(charges[name]),
                "charge_method": templates[name].charge_method,
                "molecule_type": templates[name].name,
                "itp_path": str(templates[name].itp_path),
                "atomtypes_path": str(templates[name].atomtypes_path),
            }
            for name in groups
        }
        return system, parameters

    @staticmethod
    def _prepare_rtp_ligands(system: System, force_field: str):
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups

        groups = molecule_groups(system)
        for component in system.components:
            if component.kind == ComponentKind.UNKNOWN:
                component.kind = ComponentKind.LIGAND
                component.name = "LIGANDS"
        # RTP ligands must already contain the complete atom set. The input
        # loader removes hydrogens, so a later enhancement must add them from
        # the matching HDB before enabling this path.
        from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

        rtp = load_force_field_rtp(force_field)
        for name, instances in groups.items():
            expected = {atom[0].strip() for atom in rtp.get_residue(name)["atoms"]}
            for indices in instances:
                observed = {system.structure.atom_names[index].strip() for index in indices}
                if observed != expected:
                    raise ModuleConfigError(
                        f"{name} matches {force_field} heavy atoms but lacks the complete "
                        "RTP atom set; automatic HDB completion is not yet available"
                    )
        return system, {name: {"source": "rtp", "net_charge": 0} for name in groups}

    @classmethod
    def _parameterize_cgenff_ligands(
        cls,
        system: System,
        packages: dict[str, dict],
        force_field: str,
    ):
        """Import exact ParamChem packages and add their hydrogen coordinates."""
        from gmxbuilder.core.component import Component
        from gmxbuilder.core.structure import Structure
        from gmxbuilder.modules.forcefield.cgenff_import import prepare_cgenff_molecule
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups

        groups = molecule_groups(system)
        templates = {
            name: prepare_cgenff_molecule(
                name,
                packages[name]["mol2_path"],
                packages[name]["str_path"],
                force_field,
                Path(packages[name]["str_path"]).parent / "generated",
            )
            for name in groups
        }
        instance_by_first = {
            indices[0]: (name, indices)
            for name, instances in groups.items()
            for indices in instances
        }
        instance_atoms = {
            index for _name, indices in instance_by_first.values() for index in indices
        }
        old_to_new: dict[int, int] = {}
        ligand_new_indices: list[int] = []
        coordinates: list[np.ndarray] = []
        fields = {
            key: []
            for key in (
                "atom_names",
                "resnames",
                "resids",
                "chain_ids",
                "segids",
                "elements",
                "occupancies",
                "tempfactors",
            )
        }

        def append_old(index: int, *, atom_name: str | None = None):
            old_to_new[index] = len(coordinates)
            coordinates.append(system.structure.coordinates[index].copy())
            for key in fields:
                value = getattr(system.structure, key)[index]
                fields[key].append(atom_name if key == "atom_names" and atom_name else value)

        for index in range(system.num_atoms):
            if index not in instance_atoms:
                append_old(index)

        for first_index in sorted(instance_by_first):
            name, indices = instance_by_first[first_index]
            template = templates[name]
            observed = {system.structure.atom_names[index].strip(): index for index in indices}
            if len(observed) != len(indices):
                raise ModuleConfigError(f"CGenFF molecule {name} has duplicate PDB atom names")
            template_heavy = {
                atom
                for atom, element in zip(template.atom_names, template.elements)
                if element != "H"
            }
            if set(observed) != template_heavy:
                missing = sorted(template_heavy - set(observed))
                extra = sorted(set(observed) - template_heavy)
                raise ModuleConfigError(
                    f"CGenFF heavy-atom names for {name} do not match the retained structure; "
                    f"missing={missing}, extra={extra}"
                )
            heavy_positions = [
                index for index, element in enumerate(template.elements) if element != "H"
            ]
            source = template.coordinates[heavy_positions]
            target = np.asarray(
                [
                    system.coordinates[observed[template.atom_names[index]]]
                    for index in heavy_positions
                ]
            )
            transformed = cls._kabsch_transform(source, target, template.coordinates)
            start = len(coordinates)
            for template_index, (atom, element) in enumerate(
                zip(template.atom_names, template.elements)
            ):
                if atom in observed:
                    old_index = observed[atom]
                    append_old(old_index, atom_name=atom)
                    coordinates[-1] = transformed[template_index]
                    fields["elements"][-1] = element.title()
                else:
                    if element != "H":
                        raise ModuleConfigError(
                            f"CGenFF would introduce unexpected heavy atom {atom} for {name}"
                        )
                    coordinates.append(transformed[template_index])
                    fields["atom_names"].append(atom)
                    fields["resnames"].append(name)
                    fields["resids"].append(system.structure.resids[first_index])
                    fields["chain_ids"].append(system.structure.chain_ids[first_index])
                    fields["segids"].append(system.structure.segids[first_index])
                    fields["elements"].append("H")
                    fields["occupancies"].append(1.0)
                    fields["tempfactors"].append(0.0)
            ligand_new_indices.extend(range(start, len(coordinates)))

        system.structure = Structure(
            coordinates=np.asarray(coordinates, dtype=float),
            box_vectors=system.structure.box_vectors.copy(),
            **fields,
        )
        components = []
        for component in system.components:
            if component.kind == ComponentKind.UNKNOWN:
                continue
            mapped = [old_to_new[int(index)] for index in component.atom_indices]
            components.append(
                Component(
                    name=component.name,
                    kind=component.kind,
                    atom_indices=np.asarray(mapped, dtype=int),
                    metadata=dict(component.metadata),
                )
            )
        charges = {name: template.net_charge for name, template in templates.items()}
        components.append(
            Component(
                name="LIGANDS",
                kind=ComponentKind.LIGAND,
                atom_indices=np.asarray(sorted(ligand_new_indices), dtype=int),
                metadata={
                    "molecule_charges": charges,
                    "n_molecules": sum(len(instances) for instances in groups.values()),
                },
            )
        )
        system.components = components
        parameters = {
            name: {
                "source": "cgenff",
                "net_charge": template.net_charge,
                "molecule_type": name,
                "itp_path": str(template.itp_path),
                "atomtypes_path": str(template.atomtypes_path),
                "cgenff_version": template.cgenff_version,
                "maximum_penalty": template.maximum_penalty,
            }
            for name, template in templates.items()
        }
        return system, parameters

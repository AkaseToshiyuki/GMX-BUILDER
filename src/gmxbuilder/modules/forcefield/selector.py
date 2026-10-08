"""Pipeline module: force field selection.

Runs early in the pipeline (after input, before structure processing)
so that downstream modules can read the chosen force field from system
metadata and adapt their behaviour accordingly (HDB hydrogen addition,
water model defaults, supported lipid filtering, etc.).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import PER_ATOM_FIELDS
from gmxbuilder.core.system import System
from gmxbuilder.modules import register_module
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


@register_module
class ForceFieldSelector(BaseModule):
    """Record the user's force field choice into system metadata."""

    name = "forcefield"
    description = "Select force field for the system build"

    _DEFAULT_FF = "amber14sb"
    supports_nucleic_acids = False

    # Ligand parameterization's share of this module. Everything else it does
    # is metadata bookkeeping; when a molecule needs GAFF2 this is the module.
    _LIGAND_PROGRESS_START = 0.05
    _LIGAND_PROGRESS_END = 0.95

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
                "charmm_compat_smiles",
                "charmm_compat_mol2",
                "_ligand_source_path",
                "charmm_compat_allow_research",
                "_task_dir",
                "water_model",
                "allow_unvalidated_water_model",
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
        allow_unvalidated_water = config.get("allow_unvalidated_water_model", False)
        if not isinstance(allow_unvalidated_water, bool):
            raise ModuleConfigError("allow_unvalidated_water_model must be a boolean")
        if configured_water is not None:
            from gmxbuilder.modules.solvation.water_models import (
                water_model_compatibility,
            )

            water_name = str(configured_water).strip().lower()
            compatibility = water_model_compatibility(name, water_name)
            if compatibility.status == "prohibited":
                raise ModuleConfigError(compatibility.reason)
            if compatibility.status == "expert-unvalidated" and not allow_unvalidated_water:
                raise ModuleConfigError(
                    f"Water model {water_name!r} with force field {name!r} is "
                    f"expert-unvalidated under policy {compatibility.policy_version}: "
                    f"{compatibility.reason}. Set allow_unvalidated_water_model=true "
                    "only after reviewing the scientific basis."
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
        local_smiles = config.get("charmm_compat_smiles", {})
        if not isinstance(local_smiles, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or len(v) > 4096
            for k, v in local_smiles.items()
        ):
            raise ModuleConfigError(
                "charmm_compat_smiles must map molecule names to SMILES strings"
            )
        if not isinstance(config.get("charmm_compat_allow_research", False), bool):
            raise ModuleConfigError("charmm_compat_allow_research must be a boolean")
        local_mol2 = config.get("charmm_compat_mol2", {})
        if not isinstance(local_mol2, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v
            for k, v in local_mol2.items()
        ):
            raise ModuleConfigError("charmm_compat_mol2 must map molecule names to MOL2 paths")
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
        default_ligand = "charmm_compat" if report["family"] == "charmm" else ""
        ligand_ff = str(
            config.get("ligand_ff", "none" if not ligand_names else default_ligand)
        ).lower()
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
        elif ligand_ff == "charmm_compat":
            smiles = {
                str(k).strip().upper(): v for k, v in config.get("charmm_compat_smiles", {}).items()
            }
            output_dir = (
                Path(config["_task_dir"]) / "charmm_compat"
                if config.get("_task_dir")
                else Path.home() / ".cache" / "gmxbuilder" / "charmm_compat"
            )
            system, ligand_parameters = self._parameterize_cgenff_ligands(
                system,
                {},
                requested_ff,
                local_smiles=smiles,
                local_mol2={
                    str(k).strip().upper(): v
                    for k, v in config.get("charmm_compat_mol2", {}).items()
                },
                source_path=config.get("_ligand_source_path") or system.metadata.get("pdb_path"),
                allow_research=config.get("charmm_compat_allow_research", False),
                environment_pH=float(config.get("ligand_pH", 7.0)),
                output_dir=output_dir,
            )
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
            unknown_penalty = [
                name
                for name, parameters in ligand_parameters.items()
                if parameters.get("maximum_penalty") is None
            ]
            if unknown_penalty:
                raise ModuleConfigError(
                    "CGenFF penalty evidence is missing for "
                    + ", ".join(unknown_penalty)
                    + "; missing penalties cannot be treated as validated low penalties"
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
        from gmxbuilder.modules.solvation.water_models import water_model_compatibility

        configured_water = config.get("water_model")
        water_model = (
            str(configured_water).strip().lower()
            if configured_water is not None
            else ForceFieldRegistry.get(ff_name).water_model
        )
        water_compatibility = water_model_compatibility(ff_name, water_model)
        if water_compatibility.status == "prohibited":
            raise ModuleConfigError(water_compatibility.reason)
        if (
            water_compatibility.status == "expert-unvalidated"
            and config.get("allow_unvalidated_water_model", False) is not True
        ):
            raise ModuleConfigError(
                f"Water model {water_model!r} with force field {ff_name!r} is "
                f"expert-unvalidated under policy {water_compatibility.policy_version}: "
                f"{water_compatibility.reason}. Set allow_unvalidated_water_model=true "
                "only after reviewing the scientific basis."
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
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        system.metadata["gaff_lipids"] = sorted(
            name for name in lipid_names if lipid_backend_for(name, lipid_ff) == "gaff2"
        )
        system.metadata["lipid21_lipids"] = sorted(
            name for name in lipid_names if lipid_backend_for(name, lipid_ff) == "lipid21"
        )
        system.metadata["selected_lipid_names"] = sorted(
            {str(name).strip().upper() for name in lipid_names}
        )
        system.metadata["ligand_parameters"] = ligand_parameters
        if ligand_names:
            system.metadata["ligand_environment_pH"] = float(config.get("ligand_pH", 7.0))
            system.metadata["ligand_protonation_policy"] = (
                "pH-dependent GAFF2 preparation"
                if ligand_ff == "gaff2"
                else "automatic solution-pH model or explicit override; see ligand identity report"
                if ligand_ff == "charmm_compat"
                else "explicit molecular state; pH does not rewrite supplied hydrogens or charges"
            )
        system.metadata.pop("ligand_protonation_pH", None)
        if ligand_ff == "gaff2":
            system.metadata["ligand_protonation_pH"] = float(config.get("ligand_pH", 7.0))
        system.metadata["water_model"] = water_model
        system.metadata["ff_water_model"] = water_model
        system.metadata["water_model_compatibility"] = water_compatibility.status
        system.metadata["water_model_policy_version"] = water_compatibility.policy_version
        system.metadata["water_model_compatibility_reason"] = water_compatibility.reason
        if config.get("system_name"):
            system.metadata["system_name"] = config["system_name"].strip()

        log = [
            f"Force field: {ff_name} (lipids: {lipid_ff}, ligands: {ligand_ff}, "
            f"water: {water_model})",
            "Water compatibility: "
            f"{water_compatibility.status} (policy {water_compatibility.policy_version}) — "
            f"{water_compatibility.reason}",
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
        elif lipid_ff == "amber-mixed":
            log.append("Lipid assignment: Lipid21 where covered; GAFF2 only for missing species")
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
        if ligand_ff == "charmm_compat":
            log.append(
                "Local CHARMM: source and atom order checked; numerical validation and "
                "GROMACS preprocessing passed. Assignment reports accompany the topology."
            )
            if config.get("charmm_compat_allow_research", False):
                log.append(
                    "Experimental assignment enabled: physical accuracy for new chemistry "
                    "has not been validated; this is not official CGenFF output."
                )
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

    @staticmethod
    def _ligand_phase(jobs, pending, charge_label, charge_cost) -> str:
        """Say what is about to happen, in the terms the wait deserves.

        Nothing pending is a lookup, not a wait. One molecule can be named.
        Several are parameterized at once, so naming one of them would be
        misleading about what the bar is waiting for.
        """
        if not pending:
            return "Reusing stored GAFF2 parameters for " + ", ".join(job.name for job in jobs)
        if len(pending) == 1:
            return f"Parameterizing {pending[0]} with GAFF2/{charge_label} — {charge_cost}"
        return (
            f"Parameterizing {len(pending)} molecules with GAFF2/{charge_label} "
            f"in parallel — {charge_cost}"
        )

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
        from gmxbuilder.modules.forcefield.gaff_backend import (
            MoleculeJob,
            describe_gaff_charge_method,
            gaff_molecule_is_cached,
            parameterize_molecules,
        )
        from gmxbuilder.pipeline.progress import report_progress

        groups = molecule_groups(system)
        # Reported one molecule at a time, and named. An uncached molecule is
        # a semi-empirical charge calculation: measured at 214 s inside a real
        # Check for a 71-atom ligand, and the whole of that step's duration. A
        # user watching a bar that says only "running" cannot tell that from a
        # hang, so the phase names the molecule, names the method actually in
        # use, and says whether the wait is real work or a cache lookup.
        charge_label, charge_cost = describe_gaff_charge_method()
        jobs = [
            MoleculeJob(name, system.structure, instances[0], charges[name])
            for name, instances in groups.items()
        ]
        pending = [
            job.name
            for job in jobs
            if not gaff_molecule_is_cached(
                job.name,
                job.structure,
                job.atom_indices,
                job.net_charge,
                target_pH=target_pH,
            )
        ]
        report_progress(
            cls._LIGAND_PROGRESS_START, cls._ligand_phase(jobs, pending, charge_label, charge_cost)
        )

        span = cls._LIGAND_PROGRESS_END - cls._LIGAND_PROGRESS_START

        def announce(completed: int, total: int, _name: str) -> None:
            report_progress(
                cls._LIGAND_PROGRESS_START + span * (completed / total),
                f"Parameterized {completed} of {total} molecules",
            )

        templates = parameterize_molecules(
            jobs,
            target_pH=target_pH,
            on_progress=announce if len(jobs) > 1 else None,
        )
        report_progress(cls._LIGAND_PROGRESS_END, "Building ligand topologies")
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
        fields = {key: [] for key in PER_ATOM_FIELDS}

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
                fields["source_ids"].append("")
            ligand_new_indices.extend(range(start, len(coords)))

        system.structure = Structure(
            coordinates=np.asarray(coords, dtype=float),
            box_vectors=system.structure.box_vectors.copy(),
            source_info=copy.deepcopy(system.structure.source_info),
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
        parameters = {}
        for name in groups:
            charge = sum(float(atom[2]) for atom in rtp.get_residue(name)["atoms"])
            rounded = round(charge)
            if abs(charge - rounded) > 1e-3:
                raise ModuleConfigError(f"{name} RTP charge is not integral: {charge}")
            parameters[name] = {"source": "rtp", "net_charge": rounded}
        return system, parameters

    @classmethod
    def _parameterize_cgenff_ligands(
        cls,
        system: System,
        packages: dict[str, dict],
        force_field: str,
        *,
        local_smiles: dict[str, str] | None = None,
        local_mol2: dict[str, str] | None = None,
        source_path: str | None = None,
        allow_research: bool = False,
        environment_pH: float | None = None,
        output_dir: Path | None = None,
    ):
        """Place imported or local CHARMM models in their fixed ITP atom order.

        Local generation runs for each instance so its heavy coordinates and
        conformation are preserved. Imported package placement retains its
        existing rigid-alignment behavior.
        """
        from gmxbuilder.core.component import Component
        from gmxbuilder.core.structure import Structure
        from gmxbuilder.modules.forcefield.cgenff_import import prepare_cgenff_molecule
        from gmxbuilder.modules.forcefield.compatibility import molecule_groups

        groups = molecule_groups(system)
        local_instances = {}
        if local_smiles is not None:
            from gmxbuilder.modules.forcefield.charmm_compat import prepare_local_molecule
            from gmxbuilder.modules.forcefield.ligand_identity import resolve_identity

            for name, instances in groups.items():
                for indices in instances:
                    if name in local_smiles and not local_smiles[name].strip():
                        raise ModuleConfigError(
                            f"Enter a SMILES for {name}, or select automatic identification"
                        )
                    identity = resolve_identity(
                        name,
                        system.structure,
                        indices,
                        smiles=local_smiles.get(name, "").strip(),
                        mol2_path=(local_mol2 or {}).get(name),
                        source_path=source_path,
                        pH=environment_pH if environment_pH is not None else 7.0,
                    )
                    local_instances[indices[0]] = prepare_local_molecule(
                        name,
                        system.structure,
                        indices,
                        identity["mapped_smiles"],
                        force_field,
                        output_dir,
                        allow_research=allow_research,
                        environment_pH=environment_pH,
                        chemical_identity_report=identity,
                    )
            templates = {
                name: local_instances[instances[0][0]] for name, instances in groups.items()
            }
            for name, instances in groups.items():
                if any(
                    local_instances[indices[0]].itp_path.read_text()
                    != templates[name].itp_path.read_text()
                    for indices in instances
                ):
                    raise ModuleConfigError(
                        f"Local CHARMM molecule {name}: instances require different atom mappings"
                    )
        else:
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
        fields = {key: [] for key in PER_ATOM_FIELDS}

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
            template = local_instances.get(first_index, templates[name])
            observed = {system.structure.atom_names[index].strip(): index for index in indices}
            if len(observed) != len(indices):
                raise ModuleConfigError(f"CGenFF molecule {name} has duplicate PDB atom names")
            template_heavy = {
                atom
                for atom, element in zip(template.atom_names, template.elements, strict=True)
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
            transformed = (
                template.coordinates
                if local_smiles is not None
                else cls._kabsch_transform(source, target, template.coordinates)
            )
            start = len(coordinates)
            for template_index, (atom, element) in enumerate(
                zip(template.atom_names, template.elements, strict=True)
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
                    fields["source_ids"].append("")
            ligand_new_indices.extend(range(start, len(coordinates)))

        system.structure = Structure(
            coordinates=np.asarray(coordinates, dtype=float),
            box_vectors=system.structure.box_vectors.copy(),
            source_info=copy.deepcopy(system.structure.source_info),
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
                "source": "charmm_compat" if local_smiles is not None else "cgenff",
                "net_charge": template.net_charge,
                "molecule_type": name,
                "itp_path": str(template.itp_path),
                "atomtypes_path": str(template.atomtypes_path),
                "cgenff_version": template.cgenff_version,
                "maximum_penalty": template.maximum_penalty,
            }
            for name, template in templates.items()
        }
        if local_smiles is not None:
            import json

            for name, template in templates.items():
                assignment = json.loads((template.itp_path.parent / "report.json").read_text())
                parameters[name]["assignment_method"] = assignment["assignment_method"]
                parameters[name]["export_eligibility"] = assignment["export_eligibility"]
        return system, parameters

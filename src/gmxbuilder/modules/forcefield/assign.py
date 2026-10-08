"""Module: Topology assignment — build system topology from force field parameters."""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ForceFieldError, ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Bond
from gmxbuilder.modules import register_module
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


@register_module
class ForceFieldAssigner(BaseModule):
    """Assign force field parameters and build the system topology."""

    name = "topology"
    description = "Assign force field atom types, charges, and bonded parameters"

    _DEFAULT_FF = "amber14sb"

    def validate_config(self, config: dict) -> bool:
        # FF name is read from system.metadata (set by ForceFieldSelector);
        # config may override it for compatibility.
        self.validate_config_keys(config, {"name", "protein", "seed"})
        for key in ("name", "protein"):
            if key in config and (not isinstance(config[key], str) or not config[key].strip()):
                raise ModuleConfigError(f"topology.{key} must be a non-empty string")
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        # Read FF name from metadata (set by the earlier forcefield step),
        # falling back to config for backward compatibility.
        ff_name = system.metadata.get("force_field") or config.get(
            "protein", config.get("name", self._DEFAULT_FF)
        )
        nucleic_components = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
        if nucleic_components:
            native_records = system.metadata.get("native_nucleic_topologies")
            if not isinstance(native_records, list) or len(native_records) != len(
                nucleic_components
            ):
                raise ForceFieldError(
                    "Every DNA/RNA chain requires an exact native GROMACS topology"
                )
            for component in nucleic_components:
                if not component.metadata.get("prepared") or not isinstance(
                    component.metadata.get("native_topology"), dict
                ):
                    raise ForceFieldError(
                        f"Nucleic-acid component {component.name} was not prepared "
                        "by the native polymer backend"
                    )
        from gmxbuilder.modules.forcefield.lipid_policy import (
            charmm_lipid_capability,
            lipid_has_rtp,
        )
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        registered_lipids = set(LipidRegistry.list())
        lipid_names = sorted(set(system.structure.resnames) & registered_lipids)
        lipid_ff = str(system.metadata.get("lipid_ff", ff_name)).lower()
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

        if lipid_ff == "amber-mixed":
            from gmxbuilder.modules.forcefield.lipid_policy import amber_mixed_validation_reason

            validation_reason = amber_mixed_validation_reason(lipid_names)
            if validation_reason:
                raise ForceFieldError(validation_reason)
        if lipid_ff in {"gaff2", "amber-mixed"}:
            from gmxbuilder.modules.forcefield.lipid_policy import gaff_lipid_capability

            blocked_gaff = [
                (name, gaff_lipid_capability(name)[1])
                for name in lipid_names
                if lipid_backend_for(name, lipid_ff) == "gaff2"
                and not gaff_lipid_capability(name)[0]
            ]
            if blocked_gaff:
                raise ForceFieldError("; ".join(reason for _name, reason in blocked_gaff))
        missing_rtp = [name for name in lipid_names if not lipid_has_rtp(name, ff_name)]
        blocked_charmm = [
            (name, charmm_lipid_capability(name, ff_name)[1])
            for name in lipid_names
            if str(ff_name).lower().startswith("charmm")
            and not charmm_lipid_capability(name, ff_name)[0]
        ]
        if blocked_charmm:
            raise ForceFieldError("; ".join(reason for _name, reason in blocked_charmm))
        if lipid_ff in {"gaff2", "amber-mixed"} and not str(ff_name).lower().startswith("amber"):
            raise ForceFieldError(
                "GAFF2 lipids require an Amber protein force field because "
                "their combination and 1-4 scaling rules must match"
            )
        if lipid_ff in {"lipid21", "amber-mixed"}:
            from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability

            unsupported = [
                name
                for name in lipid_names
                if lipid_backend_for(name, lipid_ff) == "lipid21"
                and not lipid21_capability(name)[0]
            ]
            if unsupported:
                raise ForceFieldError(
                    f"Lipids {unsupported} do not have exact bundled Lipid21 parameters"
                )
            if not str(ff_name).lower().startswith("amber"):
                raise ForceFieldError("Lipid21 requires an Amber protein force-field family")
        if missing_rtp and lipid_ff not in {"gaff2", "lipid21", "amber-mixed"}:
            raise ForceFieldError(
                f"Lipids {missing_rtp} have no {ff_name} RTP parameters and "
                f"were not routed through the Amber/GAFF2 policy"
            )
        ff = ForceFieldRegistry.get(ff_name)

        topology = ff.build_system_topology(system)
        # Force-field assignment refreshes atom types at finalization time.
        # Reconstruct dedicated cross-residue bonds from the authoritative,
        # validated Step 3 metadata so that this refresh cannot discard a
        # disulfide before TopologyWriter merges linked chains and exports it.
        crosslinks = system.metadata.get("crosslinks", [])
        if not isinstance(crosslinks, list):
            raise ForceFieldError("Invalid crosslink metadata in structure checkpoint")
        if crosslinks:
            from gmxbuilder.modules.modifications.patches import disulfide_capability

            supported, reason, target_distance = disulfide_capability(str(ff_name))
            if not supported or target_distance is None:
                raise ForceFieldError(f"Saved disulfides are unavailable for {ff_name}: {reason}")
        endpoint_indices = {}
        if crosslinks:
            structure = system.structure
            for index, (chain, resid, name, residue) in enumerate(
                zip(
                    structure.chain_ids,
                    structure.resids,
                    structure.atom_names,
                    structure.resnames,
                    strict=True,
                )
            ):
                if str(name).strip() == "SG" and str(residue).strip().upper() == "CYX":
                    endpoint_indices.setdefault((str(chain), int(resid)), []).append(index)
        for record in crosslinks:
            if not isinstance(record, dict) or record.get("type") != "disulfide":
                raise ForceFieldError("Invalid crosslink metadata in structure checkpoint")
            if record.get("status") != "passed":
                raise ForceFieldError("Unvalidated crosslink cannot be assigned a topology")
            endpoints = []
            for label in ("first", "second"):
                endpoint = record.get(label)
                if not isinstance(endpoint, dict):
                    raise ForceFieldError(f"Disulfide metadata is missing {label} endpoint")
                chain = str(endpoint.get("chain", ""))
                try:
                    resid = int(endpoint["resid"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ForceFieldError(
                        f"Disulfide metadata has invalid {label} residue"
                    ) from error
                matches = endpoint_indices.get((chain, resid), [])
                if len(matches) != 1:
                    raise ForceFieldError(
                        f"Validated disulfide endpoint {chain or '?'}:{resid} no longer "
                        "maps to exactly one CYX SG atom"
                    )
                endpoints.append(matches[0])
            pair = tuple(sorted(endpoints))
            if pair[0] == pair[1]:
                raise ForceFieldError("Disulfide endpoints resolve to the same SG atom")
            observed = float(
                np.linalg.norm(
                    system.structure.coordinates[pair[0]] - system.structure.coordinates[pair[1]]
                )
            )
            if abs(observed - target_distance) > 0.04:
                raise ForceFieldError(
                    f"Saved disulfide SG-SG distance changed to {observed:.3f} nm; "
                    "the validated structure checkpoint is no longer consistent"
                )
            topology.bonds.append(Bond(pair[0], pair[1]))
        system.topology = topology
        system.metadata["force_field"] = ff_name

        lipid_ff = system.metadata.get("lipid_ff", ff_name)
        ligand_ff = system.metadata.get("ligand_ff", ff_name)
        return ModuleResult(
            success=True,
            system=system,
            log=[
                f"Force field: {ff_name} (version {ff.version})",
                f"Protein FF: {ff_name}",
                f"Lipid FF:   {lipid_ff}",
                f"Ligand FF:  {ligand_ff}",
                *(
                    [
                        "Nucleic-acid FF: CHARMM36 native GROMACS topology "
                        f"({len(nucleic_components)} polymer chain(s))"
                    ]
                    if nucleic_components
                    else []
                ),
            ],
        )

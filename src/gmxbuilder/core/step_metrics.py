"""Metrics derived during admitted scientific work, reusable by checkpoint writers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gmxbuilder.core.enums import ComponentKind

if TYPE_CHECKING:
    from gmxbuilder.core.system import System


def compute_step_metrics(system: System, step_name: str) -> dict:
    """Compute frontend-relevant metrics for this step."""
    metrics: dict[str, Any] = {
        "num_atoms": system.num_atoms,
        "box_dimensions_nm": [round(v, 3) for v in system.structure.dimensions().tolist()],
        "components": [],
        "membrane_composition_warnings": system.metadata.get("membrane_composition_warnings", []),
    }

    for comp in system.components:
        info = {
            "name": comp.name,
            "kind": comp.kind.name,
            "n_atoms": len(comp.atom_indices),
        }
        n_mol = comp.metadata.get("n_molecules")
        if n_mol is not None:
            info["n_molecules"] = n_mol
        for key in ("water_model", "volume_nm3"):
            if key in comp.metadata:
                info[key] = comp.metadata[key]
        metrics["components"].append(info)

    if step_name == "solvation":
        metrics["water_model"] = system.metadata.get("water_model")
        metrics["solvation"] = system.metadata.get("solvation", {})

    if step_name == "membrane":
        membranes = system.component_by_kind(ComponentKind.MEMBRANE)
        if membranes:
            metadata = membranes[0].metadata
            metrics["membrane"] = {
                key: metadata[key]
                for key in (
                    "n_lipids_upper",
                    "n_lipids_lower",
                    "lipid_counts_upper",
                    "lipid_counts_lower",
                    "requested_lipids_per_leaflet",
                    "leaflet_count_policy",
                    "solute_footprint_nm2",
                    "target_area_per_lipid_nm2",
                )
                if key in metadata
            }

    if step_name == "ions":
        metrics["ions"] = system.metadata.get("ions", {})

    if step_name == "cg_mapping":
        metrics["cg_mapping"] = system.metadata.get("cg_mapping", {})

    if step_name == "input":
        metrics["input_validation"] = system.metadata.get("input_validation")
        metrics["input_reconstruction"] = system.metadata.get("input_reconstruction", {})
        metrics["input_repair"] = system.metadata.get(
            "input_repair",
            {
                "status": "not_needed",
                "residues_repaired": 0,
                "atoms_added": 0,
                "residues": [],
                "validation": "No missing standard protein heavy atoms detected.",
            },
        )
        metrics["input_modifications"] = system.metadata.get(
            "input_modifications",
            {"detected": 0, "recognized": 0, "records": [], "warnings": []},
        )
        protein_atoms = {
            int(index)
            for component in system.component_by_kind(ComponentKind.PROTEIN)
            for index in component.atom_indices
        }
        chains: dict[str, list[dict]] = {}
        seen_residues: set[tuple[str, int]] = set()
        for index in range(system.structure.num_atoms):
            if index not in protein_atoms:
                continue
            chain = str(system.structure.chain_ids[index]).strip() or "A"
            resid = int(system.structure.resids[index])
            key = (chain, resid)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            chains.setdefault(chain, []).append(
                {
                    "resname": str(system.structure.resnames[index]).strip().upper(),
                    "resid": resid,
                    "is_protein": True,
                }
            )
        metrics["input_sequences"] = [
            {"chain_id": chain, "length": len(residues), "residues": residues}
            for chain, residues in chains.items()
        ]
        metrics["input_nucleic_acids"] = [
            {
                "name": component.name,
                "chain_id": component.metadata.get("chain_id", ""),
                "polymer_type": component.metadata.get("polymer_type", "unknown"),
                "n_residues": component.metadata.get("n_residues", 0),
                "unsupported_residues": component.metadata.get("unsupported_residues", []),
            }
            for component in system.component_by_kind(ComponentKind.NUCLEIC_ACID)
        ]

    if step_name == "forcefield":
        metrics["forcefield_resolution"] = {
            "requested_protein_ff": system.metadata.get("requested_force_field"),
            "effective_protein_ff": system.metadata.get("force_field"),
            "effective_lipid_ff": system.metadata.get("lipid_ff"),
            "effective_ligand_ff": system.metadata.get("ligand_ff"),
            "water_model": system.metadata.get("water_model"),
            "gaff_lipids": system.metadata.get("gaff_lipids", []),
            "ligand_parameters": system.metadata.get("ligand_parameters", {}),
            "nucleic_acid_backend": (
                "gromacs-pdb2gmx-charmm36"
                if system.component_by_kind(ComponentKind.NUCLEIC_ACID)
                else "none"
            ),
        }

    if step_name == "structure":
        metrics["modification_geometry"] = system.metadata.get("modification_geometry", [])
        metrics["crosslinks"] = system.metadata.get("crosslinks", [])
        metrics["nucleic_acids"] = [
            {
                key: record.get(key)
                for key in (
                    "molecule_type",
                    "polymer_type",
                    "chain_id",
                    "net_charge",
                    "atom_count",
                    "residue_count",
                    "backend",
                )
            }
            for record in system.metadata.get("native_nucleic_topologies", [])
        ]

    if step_name == "orient":
        metrics["orientation"] = system.metadata.get("_orient_params", {})
        metrics["orientation_method"] = system.metadata.get("_orientation_method")
        metrics["orientation_quality"] = system.metadata.get("_orientation_quality", {})

    return metrics

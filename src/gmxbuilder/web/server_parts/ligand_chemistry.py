"""Task-owned ligand chemistry inputs shared by preview and construction."""

from __future__ import annotations

from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield.compatibility import molecule_groups
from gmxbuilder.modules.forcefield.ligand_identity import resolve_identity


def trusted_config(task_id, config, manager, validate_resource):
    state = manager.get_state(task_id) or {}
    result = dict(config)
    result.pop("_ligand_source_path", None)
    source = state.get("uploaded_structure_name")
    if source:
        result["_ligand_source_path"] = str(validate_resource(task_id, source))
    else:
        original = manager.get_pdb_path(task_id)
        if original:
            result["_ligand_source_path"] = str(validate_resource(task_id, original))
    requested = config.get("charmm_compat_mol2", {})
    if not isinstance(requested, dict):
        raise ValueError("charmm_compat_mol2 must be an object")
    paths = {}
    root = manager.get_task_dir(task_id) / "ligand_chemistry"
    for name, selected in requested.items():
        if (
            not isinstance(selected, dict)
            or set(selected) != {"uploaded", "sha256"}
            or selected.get("uploaded") is not True
        ):
            raise ValueError("MOL2 inputs must reference a task-owned upload")
        package = (state.get("ligand_chemistry_uploads") or {}).get(name)
        if not isinstance(package, dict) or not package.get("file"):
            raise ValueError(f"Upload a MOL2 for {name} first")
        if not selected.get("sha256") or selected["sha256"] != package.get("sha256"):
            raise ValueError(f"MOL2 selection for {name} changed; validate the selected file again")
        candidate = validate_resource(task_id, root / Path(package["file"]).name)
        if candidate.parent != root.resolve() or not candidate.is_file():
            raise ValueError(f"MOL2 upload for {name} is unavailable")
        paths[name] = str(candidate)
    result["charmm_compat_mol2"] = paths
    return result


def identify_groups(system, config, *, only=None):
    result = {}
    smiles = config.get("charmm_compat_smiles", {})
    if not isinstance(smiles, dict) or any(not isinstance(v, str) for v in smiles.values()):
        raise ValueError("SMILES inputs must be an object of strings")
    for name, instances in molecule_groups(system).items():
        if only is not None and name != only:
            continue
        try:
            if name in smiles and not smiles[name].strip():
                raise ValueError(f"Enter a SMILES for {name}, or select automatic identification")
            records = [
                resolve_identity(
                    name,
                    system.structure,
                    indices,
                    pH=config.get("ligand_pH", 7.0),
                    smiles=smiles.get(name, "").strip(),
                    mol2_path=config.get("charmm_compat_mol2", {}).get(name),
                    source_path=config.get("_ligand_source_path"),
                )
                for indices in instances
            ]
            if len({r["smiles"] for r in records}) != 1:
                raise ValueError(
                    "Instances have different chemical states; separate their residue IDs"
                )
            result[name] = records[0]
            result[name]["instances"] = len(records)
        except (ValueError, RuntimeError, OSError, KeyError, ModuleConfigError) as exc:
            result[name] = {"status": "needs_input", "error": str(exc)}
    return result


def preview(checkpoint, config):
    return identify_groups(System.load_checkpoint(checkpoint), config)

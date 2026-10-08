"""Scientific modification-preview calculation independent of HTTP routing."""

from __future__ import annotations

from pathlib import Path

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.modules.modifications.patches import (
    effective_patch_charge_shift,
    get_patch,
)
from gmxbuilder.modules.modifications.protonation import (
    assign_all_protonations,
    compute_net_charge_from_protonation,
)


def build_modification_preview(
    pdb_path: str | Path,
    pH: object,
    his_tautomer: object,
    modifications: list,
    force_field: str,
    nter_patch: object,
    cter_patch: object,
) -> dict:
    """Return residue changes and the resulting approximate net charge."""
    structure = PDBParser().parse(pdb_path)
    keys = list(
        dict.fromkeys(
            (chain, resid, name)
            for chain, resid, name in zip(
                structure.chain_ids, structure.resids, structure.resnames, strict=True
            )
            if name in PROTEIN_RESNAMES
        )
    )
    residues = [key[2] for key in keys]
    assignments = assign_all_protonations(
        list(residues),
        pH=float(pH),
        his_tautomer=his_tautomer,
        force_field=force_field,
    )

    applied: list[dict] = []
    for modification in modifications:
        index = modification.get("index")
        patch_id = modification.get("patch_id")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(residues):
            raise ModuleConfigError("Modification index must identify one protein residue")
        if get_patch(patch_id) is None:
            raise ModuleConfigError(f"Unknown modification patch: {patch_id}")
        if index is not None:
            applied.append(
                {
                    "index": index,
                    "original_resname": residues[index],
                    "patch_id": patch_id,
                }
            )

    residue_changes = [
        {
            "index": assignment["index"],
            "original": assignment["original"],
            "new_name": assignment["assigned_name"],
            "charge": assignment["charge"],
            "pKa": assignment["pKa"],
            "state": assignment["state_label"],
        }
        for assignment in assignments
        if assignment["is_titratable"]
    ]

    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    # Include fixed-charge residues such as ARG and existing modified residues.
    for assignment in assignments:
        if assignment.get("force_field_lacks_state"):
            raise ModuleConfigError("The predicted protonation state requires review")
        if not assignment["is_titratable"]:
            template = rtp.get_residue(assignment["assigned_name"])
            if template is None:
                raise ModuleConfigError(f"No charge template for {assignment['assigned_name']}")
            assignment["charge"] = sum(atom[2] for atom in template["atoms"])
    net_charge = compute_net_charge_from_protonation(assignments)
    for modification in applied:
        patch = get_patch(modification["patch_id"])
        if patch:
            net_charge += effective_patch_charge_shift(
                modification["patch_id"],
                force_field,
            )

    # The legacy API supplies one cap choice for all protein chains.
    from gmxbuilder.modules.modifications.processor import terminal_capabilities

    caps = terminal_capabilities(force_field)
    chains = list(dict.fromkeys(key[0] for key in keys))
    for chain in chains:
        indices = [i for i, key in enumerate(keys) if key[0] == chain]
        for patch_id, index, term, free_charge in (
            (nter_patch, indices[0], "N", 1),
            (cter_patch, indices[-1], "C", -1),
        ):
            if not patch_id:
                net_charge += free_charge
                continue
            capability = caps.get(patch_id, {})
            if capability.get("end") != term or not capability.get("supported"):
                raise ModuleConfigError(f"Invalid {term}-terminal cap: {patch_id}")
            net_charge += sum(atom[2] for atom in rtp.get_residue(patch_id)["atoms"])
            applied.append(
                {
                    "index": index,
                    "original_resname": residues[index],
                    "patch_id": patch_id,
                    "term": term,
                    "chain": chain,
                }
            )

    return {
        "pH": pH,
        "charge_scope": "protein residues and canonical free/capped termini",
        "protonation_count": len(residue_changes),
        "residue_changes": residue_changes,
        "modifications_applied": len(applied),
        "modifications": applied,
        "nter_patch": nter_patch,
        "cter_patch": cter_patch,
        "net_charge": net_charge,
    }

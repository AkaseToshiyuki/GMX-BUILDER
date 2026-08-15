"""Force-field family compatibility and molecule-template inspection."""

from __future__ import annotations

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield.gaff_backend import gaff_available
from gmxbuilder.modules.forcefield.lipid_policy import (
    amber_lipid_backend,
    charmm_lipid_capability,
    gaff_lipid_capability,
    lipid_has_rtp,
)
from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
from gmxbuilder.modules.forcefield.catalog import force_field_family


def molecule_groups(system: System) -> dict[str, list[list[int]]]:
    """Return retained UNKNOWN/LIGAND molecule instances grouped by resname."""
    eligible = {
        int(index)
        for component in system.components
        if component.kind in (ComponentKind.UNKNOWN, ComponentKind.LIGAND)
        for index in component.atom_indices
    }
    groups: dict[str, dict[tuple[str, int], list[int]]] = {}
    structure = system.structure
    for index in sorted(eligible):
        name = str(structure.resnames[index]).strip().upper()
        key = (str(structure.chain_ids[index]), int(structure.resids[index]))
        groups.setdefault(name, {}).setdefault(key, []).append(index)
    return {name: list(instances.values()) for name, instances in groups.items()}


def _is_hydrogen(system: System, index: int) -> bool:
    element = str(system.structure.elements[index]).strip().upper()
    atom_name = str(system.structure.atom_names[index]).strip().upper()
    return element == "H" or atom_name.startswith("H")


def rtp_heavy_atom_match(system: System, force_field: str, name: str) -> tuple[bool, str]:
    """Require every instance to match one RTP template by heavy atom names."""
    template = load_force_field_rtp(force_field).get_residue(name.upper())
    if template is None:
        return False, f"no {force_field} RTP template"
    expected_all = {atom[0].strip() for atom in template["atoms"]}
    expected = {name for name in expected_all if not name.upper().startswith("H")}
    instances = molecule_groups(system).get(name.upper(), [])
    if not instances:
        return False, "molecule is absent from the input checkpoint"
    for indices in instances:
        observed = {
            str(system.structure.atom_names[index]).strip()
            for index in indices
            if not _is_hydrogen(system, index)
        }
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            return False, f"RTP identity mismatch (missing={missing}, extra={extra})"
        observed_all = {str(system.structure.atom_names[index]).strip() for index in indices}
        if observed_all != expected_all:
            return False, (
                f"heavy atoms match {force_field} RTP, but the complete template "
                "atom set (including hydrogens) is not present"
            )
    return True, f"exact atom match to {force_field} RTP"


def compatibility_report(
    system: System, protein_ff: str, lipid_names: list[str] | tuple[str, ...]
) -> dict:
    protein_ff = protein_ff.strip().lower()
    family = force_field_family(protein_ff)
    lipids = sorted({str(name).strip().upper() for name in lipid_names if str(name).strip()})
    ligands = sorted(molecule_groups(system))
    nucleic_components = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
    from gmxbuilder.modules.nucleic_acid.support import (
        nucleic_force_field_capability,
        validate_nucleic_backbone,
    )

    nucleic_enabled, nucleic_reason = nucleic_force_field_capability(protein_ff)
    if nucleic_components and nucleic_enabled:
        unsupported = sorted(
            {
                str(residue)
                for component in nucleic_components
                for residue in component.metadata.get("unsupported_residues", [])
            }
        )
        polymer_types_by_chain: dict[str, set[str]] = {}
        for component in nucleic_components:
            chain = str(component.metadata.get("chain_id", ""))
            polymer_types_by_chain.setdefault(chain, set()).add(
                str(component.metadata.get("polymer_type", ""))
            )
        hybrids = sorted(
            chain or "?" for chain, kinds in polymer_types_by_chain.items() if len(kinds) > 1
        )
        backbone_issues = [
            issue
            for component in nucleic_components
            for issue in validate_nucleic_backbone(system.structure, component)
        ]
        if unsupported:
            nucleic_enabled = False
            nucleic_reason = (
                "modified/noncanonical nucleotide topology is unavailable: "
                + ", ".join(unsupported)
            )
        elif hybrids:
            nucleic_enabled = False
            nucleic_reason = "covalent DNA/RNA hybrid chains are unavailable: " + ", ".join(hybrids)
        elif backbone_issues:
            nucleic_enabled = False
            nucleic_reason = "backbone continuity failed: " + "; ".join(backbone_issues)

    def missing_lipid_reason(missing: list[str], requested: str) -> str:
        alternatives = []
        for name in missing:
            available = []
            for candidate, label in (
                ("charmm36m", "CHARMM36m"),
                ("charmm36", "CHARMM36"),
            ):
                if candidate != requested and lipid_has_rtp(name, candidate):
                    available.append(label)
            if gaff_available() and gaff_lipid_capability(name)[0]:
                available.append("Amber14SB + GAFF2")
            alternatives.append(
                f"{name} -> {', '.join(available) if available else 'no installed force field'}"
            )
        return (
            f"missing validated {requested} lipid topology: {', '.join(missing)}. "
            "Available alternatives: " + "; ".join(alternatives)
        )

    if not lipids:
        lipid_options = [{"value": "none", "label": "No membrane lipids", "enabled": True}]
    elif family == "charmm":
        topology_missing = [name for name in lipids if not lipid_has_rtp(name, protein_ff)]
        quality_blocked = [
            name
            for name in lipids
            if name not in topology_missing and not charmm_lipid_capability(name, protein_ff)[0]
        ]
        quality_reasons = [
            charmm_lipid_capability(name, protein_ff)[1]
            for name in quality_blocked
            if charmm_lipid_capability(name, protein_ff)[1]
        ]
        missing = topology_missing + quality_blocked
        reason_parts = []
        if topology_missing:
            reason_parts.append(missing_lipid_reason(topology_missing, protein_ff))
        reason_parts.extend(quality_reasons)
        lipid_options = [
            {
                "value": protein_ff,
                "label": f"CHARMM36 lipids bundled with {protein_ff}",
                "enabled": not missing,
                "reason": ("" if not missing else "; ".join(reason_parts)),
            }
        ]
    elif family == "amber":
        backend, backend_reason = amber_lipid_backend(lipids)
        labels = {
            "lipid21": "Amber Lipid21 v1.0 (exact)",
            "gaff2": "GAFF2 fallback (Amber-compatible)",
        }
        lipid_options = [
            {
                "value": backend or "unavailable",
                "label": labels.get(backend, "No validated Amber lipid backend"),
                "enabled": backend is not None,
                "reason": backend_reason,
            }
        ]
        # Lipid21 remains the preferred exact backend, but GAFF2 must stay an
        # explicit coherent whole-membrane option when every selected lipid
        # supports it.  Task-scoped custom lipids can only join a GAFF2
        # membrane, so hiding this valid alternative makes their workflow
        # impossible to complete.
        gaff_supported = gaff_available() and all(gaff_lipid_capability(name)[0] for name in lipids)
        if backend == "lipid21" and gaff_supported:
            lipid_options.append(
                {
                    "value": "gaff2",
                    "label": labels["gaff2"],
                    "enabled": True,
                    "reason": (
                        "all selected lipids support one coherent GAFF2 membrane; "
                        "select this backend before adding a custom lipid"
                    ),
                }
            )
    else:
        missing = [name for name in lipids if not lipid_has_rtp(name, protein_ff)]
        lipid_options = [
            {
                "value": "oplsaa",
                "label": "OPLS-AA compatible lipid parameters",
                "enabled": not missing,
                "reason": "" if not missing else missing_lipid_reason(missing, protein_ff),
            }
        ]

    ligand_details = []
    rtp_all = True
    for name in ligands:
        match, reason = rtp_heavy_atom_match(system, protein_ff, name)
        rtp_all = rtp_all and match
        ligand_details.append({"name": name, "rtp_match": match, "rtp_reason": reason})

    if not ligands:
        ligand_options = [{"value": "none", "label": "No retained small molecule", "enabled": True}]
    elif family == "charmm":
        ligand_options = [
            {
                "value": "rtp",
                "label": f"{protein_ff} RTP template",
                "enabled": rtp_all,
                "reason": ""
                if rtp_all
                else "one or more molecules do not exactly match a CHARMM template",
            },
            {
                "value": "cgenff",
                "label": "CGenFF / ParamChem import",
                "enabled": True,
                "reason": "requires the matching ParamChem MOL2 and STR output for every molecule",
            },
        ]
    elif family == "amber":
        ligand_options = [
            {
                "value": "gaff2",
                "label": "GAFF2 + AM1-BCC",
                "enabled": gaff_available(),
                "reason": "" if gaff_available() else "AmberTools/ACPYPE is unavailable",
            }
        ]
    else:
        ligand_options = [
            {
                "value": "rtp",
                "label": "Bundled OPLS-AA template",
                "enabled": rtp_all,
                "reason": ""
                if rtp_all
                else "no exact bundled OPLS template; no general OPLS generator is installed",
            }
        ]

    return {
        "protein_ff": protein_ff,
        "family": family,
        "lipid_names": lipids,
        "ligand_names": ligands,
        "lipid_options": lipid_options,
        "ligand_options": ligand_options,
        "ligands": ligand_details,
        "nucleic_acid": {
            "present": bool(nucleic_components),
            "enabled": nucleic_enabled if nucleic_components else True,
            "reason": nucleic_reason if nucleic_components else "",
            "chains": len(nucleic_components),
            "residues": sum(
                int(component.metadata.get("n_residues", 0)) for component in nucleic_components
            ),
            "polymer_types": sorted(
                {
                    str(component.metadata.get("polymer_type", "unknown"))
                    for component in nucleic_components
                }
            ),
        },
    }


def enabled_values(options: list[dict]) -> set[str]:
    return {str(option["value"]) for option in options if option.get("enabled")}

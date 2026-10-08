"""Force-field-independent input readiness and deposited sequence evidence."""

from __future__ import annotations

import copy
import re
from pathlib import Path

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES, is_hydrogen
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.polymer import peptide_connectivity_issues
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.residue_identity import clean_identifier
from gmxbuilder.modules.input.protein_repair import (
    assess_repairable_missing_atoms,
    normalize_repair_atom_names,
)

INPUT_VALIDATION_VERSION = 5


def read_polymer_metadata(path: str | Path) -> dict:
    """Keep deposited sequences/missing residues independently of viewer PDBs.

    A numbering jump alone does not establish absent residues. Only explicit
    deposition records are called missing here; coordinates independently
    establish whether the retained peptide can be built.
    """
    path = Path(path)
    from gmxbuilder.io.source_document import source_text

    raw = source_text(path)
    sequences: dict[str, list[dict]] = {}
    missing: list[dict] = []
    from gmxbuilder.io.cif_document import is_cif

    if is_cif(raw):
        from gmxbuilder.io.cif import CIFParser

        def rows(prefix):
            fields, values = CIFParser._extract_loop(raw, prefix)
            if not fields or len(values) % len(fields):
                return []
            names = [name.split(".", 1)[1] for name in fields]
            return [
                dict(zip(names, values[i : i + len(fields)], strict=True))
                for i in range(0, len(values), len(fields))
            ]

        for row in rows("_pdbx_poly_seq_scheme."):
            chain = row.get("pdb_strand_id", "")
            if chain in {".", "?"}:
                chain = row.get("asym_id", "")
            number = row.get("pdb_seq_num", "?")
            sequences.setdefault(chain, []).append(
                {
                    "resname": row.get("mon_id", "UNK"),
                    "resid": int(number) if re.fullmatch(r"-?\d+", number) else None,
                    "label_seq_id": row.get("seq_id"),
                    "insertion_code": clean_identifier(row.get("pdb_ins_code")),
                }
            )
        fields, values = CIFParser._extract_loop(raw, "_atom_site.")
        model_field = "_atom_site.pdbx_PDB_model_num"
        model = values[fields.index(model_field)] if model_field in fields and values else "1"
        for row in rows("_pdbx_unobs_or_zero_occ_residues."):
            if row.get("PDB_model_num", model) != model or row.get("polymer_flag") == "N":
                continue
            number = row.get("auth_seq_id", "?")
            if re.fullmatch(r"-?\d+", number):
                missing.append(
                    {
                        "chain": row.get("auth_asym_id", ""),
                        "resid": int(number),
                        "resname": row.get("auth_comp_id", "UNK"),
                        "source": "mmCIF unobserved/zero-occupancy residues",
                        "label_seq_id": row.get("label_seq_id"),
                        "insertion_code": clean_identifier(row.get("PDB_ins_code")),
                    }
                )
    else:
        model = next(
            (line[10:14].strip() for line in raw.splitlines() if line.startswith("MODEL ")),
            "1",
        )
        for line in raw.splitlines():
            if line.startswith("SEQRES"):
                chain = line[11:12].strip()
                sequences.setdefault(chain, []).extend(
                    {"resname": name, "resid": None} for name in line[19:].split()
                )
            if line.startswith("REMARK 465"):
                match = re.fullmatch(
                    r"\s*(?:(\d+)\s+)?([A-Z]{3})\s+([A-Za-z0-9]?)\s+(-?\d+)([A-Za-z]?)\s*",
                    line[10:],
                )
                if match:
                    record_model, residue, chain, number, insertion = match.groups()
                    if record_model is not None and record_model != model:
                        continue
                    missing.append(
                        {
                            "chain": chain,
                            "resid": int(number),
                            "resname": residue,
                            "source": "PDB REMARK 465",
                            "insertion_code": insertion,
                        }
                    )
    return {"sequences": sequences, "missing_residues": missing}


def assess_input_structure(structure: Structure, source: dict | None = None) -> dict:
    """Report all currently known blockers and safe repair candidates.

    This does not assign a force field, invent loops, or choose new termini.
    Missing terminal sequence is disclosed rather than silently reconstructed.
    """
    # This assessment only changes atom names. Share read-only coordinates and
    # deposition evidence, preserving the general Structure.copy deep-copy contract.
    normalized = copy.copy(structure)
    normalized.atom_names = list(structure.atom_names)
    try:
        normalize_repair_atom_names(normalized)
        candidates, damage = assess_repairable_missing_atoms(normalized)
    except ModuleConfigError as exc:
        candidates, damage = {}, [str(exc)]
    errors = [
        {
            "code": "protein_atom_damage",
            "message": message
            + ". Upload a repaired structure; automatic repair requires a complete "
            "backbone and a connected partial side chain.",
        }
        for message in damage
    ]
    water = {"HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD"}
    if not any(
        name not in water and not is_hydrogen(atom, element)
        for name, atom, element in zip(
            structure.resnames, structure.atom_names, structure.elements, strict=True
        )
    ):
        errors.append(
            {
                "code": "empty_solute",
                "message": "Input contains no solute atoms after cleaning; provide a structure "
                "with at least one non-water, non-hydrogen atom.",
            }
        )
    indices = [i for i, name in enumerate(structure.resnames) if name in PROTEIN_RESNAMES]
    errors.extend(peptide_connectivity_issues(structure, indices))
    # Apply the existing polymer continuity contract whether or not the input
    # redundantly declares canonical O3'-P connections.
    from gmxbuilder.core.component import Component
    from gmxbuilder.core.enums import ComponentKind
    from gmxbuilder.modules.nucleic_acid.support import (
        nucleic_polymer_residues,
        validate_nucleic_backbone,
    )

    polymers = nucleic_polymer_residues(structure)
    chains = {}
    for index, key in enumerate(zip(structure.chain_ids, structure.resids, strict=True)):
        if key in polymers:
            chains.setdefault(key[0], []).append(index)
    for chain, atoms in chains.items():
        component = Component(chain, ComponentKind.NUCLEIC_ACID, atoms)
        errors.extend(
            {"code": "nucleic_backbone", "message": message}
            for message in validate_nucleic_backbone(structure, component)
        )
    from gmxbuilder.modules.input.geometry_validation import (
        coincident_atom_issues,
        protein_geometry_issues,
    )

    errors.extend(protein_geometry_issues(normalized))
    errors.extend(coincident_atom_issues(normalized))
    from gmxbuilder.modules.input.connections import connection_issues

    errors.extend(connection_issues(normalized))
    warnings = []
    if structure.num_atoms and all(value == 0 for value in structure.occupancies):
        warnings.append(
            {
                "code": "zero_occupancy",
                "message": "All input occupancies are zero. Coordinates are retained; "
                "confirm the source model rather than interpreting these as measured populations.",
            }
        )
    reconstruction = structure.source_info.get("input_reconstruction", {})
    if reconstruction.get("allow_incomplete_protein"):
        warnings.append(
            {
                "code": "accepted_coordinate_fragments",
                "message": "You accepted the observed protein construct. Missing residues are not "
                "rebuilt. Coordinate fragments remain parts of the same source protein but "
                "have independent topology termini. This is an approximate missing-loop model, "
                "not a restored continuous protein. Review new internal ends in Structure "
                "Processing or Martini mapping; capping does not restore the missing loop.",
            }
        )
    # Group by the pre-split working chain: author IDs alone can be shared by
    # assembly copies, while fragment IDs would duplicate an internal gap.
    lineage = {
        (row["chain"], row["resid"]): row for row in reconstruction.get("residue_mapping", [])
    }

    def source_chain(index):
        chain = structure.chain_ids[index]
        return lineage.get((chain, structure.resids[index]), {}).get("original_chain", chain)

    observed: dict[str, set[tuple[int, str]]] = {}
    fragments: dict[str, list[str]] = {}
    source_atoms = structure.source_info.get("atoms", {})

    def author_resid(index):
        atom = source_atoms.get(structure.source_ids[index], {})
        residue = lineage.get((structure.chain_ids[index], structure.resids[index]), {})
        value = residue.get(
            "author_resid",
            atom.get("resid", atom.get("_atom_site.auth_seq_id", structure.resids[index])),
        )
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(structure.resids[index])

    def author_identity(index):
        atom = source_atoms.get(structure.source_ids[index], {})
        residue = lineage.get((structure.chain_ids[index], structure.resids[index]), {})
        insertion = residue.get(
            "insertion_code", atom.get("icode", atom.get("_atom_site.pdbx_PDB_ins_code", ""))
        )
        return author_resid(index), clean_identifier(insertion)

    def missing_identity(record):
        return record["resid"], clean_identifier(record.get("insertion_code", ""))

    for index in indices:
        chain = source_chain(index)
        observed.setdefault(chain, set()).add(author_identity(index))
        working = fragments.setdefault(chain, [])
        if structure.chain_ids[index] not in working:
            working.append(structure.chain_ids[index])
    # Source records apply only to retained chains, and never override atoms
    # that are actually present in an explicitly repaired replacement model.
    missing_by_chain: dict[str, list[dict]] = {}
    author_to_logical = {}
    sequence_positions = {}
    for index in indices:
        atom = source_atoms.get(structure.source_ids[index], {})
        logical = source_chain(index)
        residue = lineage.get((structure.chain_ids[index], structure.resids[index]), {})
        author = residue.get(
            "author_chain", atom.get("author_chain", atom.get("_atom_site.auth_asym_id", logical))
        )
        author_to_logical.setdefault(author, set()).add(logical)
        label = atom.get("_atom_site.label_seq_id", "")
        if str(label).isdigit():
            sequence_positions[(logical, author_identity(index))] = int(label)
    deposited = []
    for record in (source or {}).get("missing_residues", []):
        for chain in author_to_logical.get(record["chain"], {record["chain"]}):
            deposited.append({**record, "chain": chain})
    for record in deposited:
        chain, resid = record["chain"], missing_identity(record)
        if chain in observed and resid not in observed[chain]:
            missing_by_chain.setdefault(chain, []).append(record)
    for chain, records in missing_by_chain.items():
        first, last = min(observed[chain]), max(observed[chain])
        positions = [sequence_positions.get((chain, resid)) for resid in observed[chain]]
        if (
            positions
            and all(position is not None for position in positions)
            and all(str(r.get("label_seq_id", "")).isdigit() for r in records)
        ):
            low, high = min(positions), max(positions)
            internal = [r for r in records if low < int(r["label_seq_id"]) < high]
        else:
            ordered_ids = list(
                dict.fromkeys(author_identity(i) for i in indices if source_chain(i) == chain)
            )
            if ordered_ids != sorted(ordered_ids):
                target = warnings if reconstruction.get("allow_incomplete_protein") else errors
                target.append(
                    {
                        "code": "ambiguous_missing_sequence",
                        "chain": chain,
                        "message": f"Chain {chain}: deposited missing residues cannot be located "
                        "unambiguously in non-monotonic author numbering; "
                        "provide the polymer sequence mapping",
                    }
                )
                continue
            internal = [r for r in records if first < missing_identity(r) < last]
        terminal = [r for r in records if r not in internal]
        if internal:
            description = _residue_ranges(internal)
            target = warnings if reconstruction.get("allow_incomplete_protein") else errors
            target.append(
                {
                    "code": "deposited_missing_segment",
                    "chain": chain,
                    "residues": internal,
                    "working_fragments": fragments[chain],
                    "message": f"Source protein {chain or '<unnamed>'} has "
                    f"{len(internal)} deposited "
                    f"internal residues without coordinates: {description}. "
                    + (
                        "Missing sequence was explicitly accepted and will not be rebuilt."
                        if reconstruction.get("allow_incomplete_protein")
                        else "Repair the missing segment or explicitly define separate chains "
                        "with appropriate termini before continuing."
                    ),
                }
            )
        if terminal:
            warnings.append(
                {
                    "code": "deposited_missing_termini",
                    "chain": chain,
                    "residues": terminal,
                    "working_fragments": fragments[chain],
                    "message": f"Chain {chain or '<unnamed>'}: {len(terminal)} deposited "
                    "terminal residues have no coordinates. Confirm the intended construct "
                    "and termini; these residues will not be added automatically.",
                }
            )
    repairs = [
        {"chain": chain, "resid": resid, "resname": name, "atoms": list(atoms)}
        for (chain, resid, name), atoms in candidates.items()
    ]
    return {
        "policy_version": INPUT_VALIDATION_VERSION,
        "can_proceed": not errors,
        "errors": errors,
        "warnings": warnings,
        "repairable_residues": repairs,
    }


def validation_message(report: dict) -> str:
    return "\n".join(issue["message"] for issue in report.get("errors", []))


def _residue_ranges(records):
    """Compact author numbering for display; full residue records remain saved."""
    if len(records) <= 5 or any(row.get("insertion_code") for row in records):
        return ", ".join(
            f"{row['resname']} {row['resid']}{row.get('insertion_code', '')}" for row in records
        )
    numbers = sorted({record["resid"] for record in records})
    ranges = []
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in ranges)

"""Working polymer identities and explicitly accepted coordinate fragments."""

from __future__ import annotations

import re
import string

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES
from gmxbuilder.core.polymer import MAX_PEPTIDE_BOND_NM, peptide_connectivity_issues

OPTIONS = {
    "allow_incomplete_protein",
    "renumber_residues",
    "chain_names",
    "fragment_names",
    "exclude_fragments",
}


def chain_labels():
    """Use PDB-compatible labels first; canonical storage supports larger assemblies."""
    yield from string.ascii_uppercase + string.ascii_lowercase + string.digits
    index = 1
    while True:
        yield f"CHAIN{index}"
        index += 1


def normalize_chain_ids(structure):
    """Assign deterministic working IDs without changing deposited atom identities."""
    if structure.source_info.get("working_chain_ids"):
        return
    proteins = list(
        dict.fromkeys(
            c
            for c, r in zip(structure.chain_ids, structure.resnames, strict=True)
            if r in PROTEIN_RESNAMES
        )
    )
    ordered = proteins + [c for c in dict.fromkeys(structure.chain_ids) if c not in proteins]
    mapping = dict(zip(ordered, chain_labels()))
    structure.chain_ids = [mapping[c] for c in structure.chain_ids]
    structure.source_info["working_chain_ids"] = mapping


def validate_options(config):
    for name in ("allow_incomplete_protein", "renumber_residues"):
        if name in config and not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a boolean")
    names = config.get("chain_names", {})
    if not isinstance(names, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9]", v)
        for k, v in names.items()
    ):
        raise ValueError("Chain names must be unique single letters or digits")
    names = config.get("fragment_names", {})
    if not isinstance(names, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9]", v)
        for k, v in names.items()
    ):
        raise ValueError("Fragment names must be unique single letters or digits")
    excluded = config.get("exclude_fragments", [])
    if not isinstance(excluded, list) or any(not isinstance(k, str) for k in excluded):
        raise ValueError("exclude_fragments must be an array of strings")


def reconstruct_input(structure, config):
    """Split long peptide gaps on consent; never invent a loop or connect distant atoms."""
    validate_options(config)
    normalize_chain_ids(structure)
    from gmxbuilder.modules.nucleic_acid.support import nucleic_polymer_residues

    nucleic = nucleic_polymer_residues(structure)
    polymer_atoms = [
        name in PROTEIN_RESNAMES or (chain, resid) in nucleic
        for chain, resid, name in zip(
            structure.chain_ids, structure.resids, structure.resnames, strict=True
        )
    ]
    original = list(zip(structure.chain_ids, structure.resids, structure.resnames, strict=True))
    names = config.get("chain_names", {})
    chains = set(structure.chain_ids)
    unknown = set(names) - chains
    if unknown:
        raise ValueError(f"Unknown chain names: {sorted(unknown)}")
    effective = {c: names.get(c, c) for c in chains}
    if len(set(effective.values())) != len(effective):
        raise ValueError("Chain names must be unique; renaming must not merge chains")
    split_records = []
    if config.get("allow_incomplete_protein", False):
        indices = [i for i, r in enumerate(structure.resnames) if r in PROTEIN_RESNAMES]
        # Missing C/N, coincident atoms and non-finite geometry remain blocking damage.
        breaks = [
            x
            for x in peptide_connectivity_issues(structure, indices)
            if x["code"] == "peptide_break"
            and x["distance_nm"] is not None
            and x["distance_nm"] > MAX_PEPTIDE_BOND_NM
        ]
        boundaries = {(x["chain"], x["right_resid"]): x for x in breaks}
        used = set(effective.values())
        labels = (label for label in chain_labels() if label not in used)
        residues = list(dict.fromkeys(original))
        residue_chains = {}
        current = dict(effective)
        for chain, resid, resname in residues:
            boundary = boundaries.get((chain, resid))
            if boundary is not None:
                current[chain] = next(labels)
                used.add(current[chain])
                split_records.append({**boundary, "new_chain": current[chain]})
            residue_chains[(chain, resid, resname)] = current[chain]
        structure.chain_ids = [residue_chains[key] for key in original]
    else:
        structure.chain_ids = [effective[c] for c in structure.chain_ids]
    # Keys describe source-chain fragment order, independent of user labels.
    # Allocate all fragments before exclusion so rechecking cannot shift keys.
    fragments = {}
    source_fragments = {}
    for old, chain in zip(original, structure.chain_ids, strict=True):
        if chain not in fragments:
            siblings = source_fragments.setdefault(old[0], [])
            siblings.append(chain)
            fragments[chain] = {
                "fragment_key": f"{old[0]}:{len(siblings)}",
                "source_chain": old[0],
                "fragment_index": len(siblings),
            }
    fragment_names = config.get("fragment_names", {})
    excluded = set(config.get("exclude_fragments", []))
    keys = {row["fragment_key"] for row in fragments.values()}
    unknown = (set(fragment_names) | excluded) - keys
    if unknown:
        raise ValueError(f"Unknown fragments: {sorted(unknown)}; check the source selection again")
    final_names = {
        chain: fragment_names.get(row["fragment_key"], chain) for chain, row in fragments.items()
    }
    # Even excluded labels stay reserved, making re-inclusion unambiguous.
    if len(set(final_names.values())) != len(final_names):
        raise ValueError("Fragment names must be unique; renaming must not merge chains")
    for chain, row in fragments.items():
        row["fragment_count"] = len(source_fragments[row["source_chain"]])
        row["chain_id"] = final_names[chain]
        row["included"] = row["fragment_key"] not in excluded
    atom_keys = [fragments[c]["fragment_key"] for c in structure.chain_ids]
    structure.chain_ids = [final_names[c] for c in structure.chain_ids]
    for split in split_records:
        split["new_chain"] = final_names[split["new_chain"]]
    if config.get("renumber_residues", False):
        counters = {}
        mapping = {}
        for i, (chain, resid, resname) in enumerate(
            zip(structure.chain_ids, structure.resids, structure.resnames, strict=True)
        ):
            key = (chain, resid, resname)
            if key not in mapping:
                counters[chain] = counters.get(chain, 0) + 1
                mapping[key] = counters[chain]
            structure.resids[i] = mapping[key]
    residue_map = {}
    source_atoms = structure.source_info.get("atoms", {})
    for old, chain, resid, source_id in zip(
        original, structure.chain_ids, structure.resids, structure.source_ids, strict=True
    ):
        if old in residue_map:
            continue
        atom = source_atoms.get(source_id, {})
        residue_map[old] = {
            "original_chain": old[0],
            "original_resid": old[1],
            "author_chain": atom.get("author_chain", atom.get("_atom_site.auth_asym_id", old[0])),
            "author_resid": atom.get("resid", atom.get("_atom_site.auth_seq_id", old[1])),
            "insertion_code": atom.get("icode", atom.get("_atom_site.pdbx_PDB_ins_code", "")),
            "resname": old[2],
            "chain": chain,
            "resid": int(resid),
        }
    report = {
        "allow_incomplete_protein": config.get("allow_incomplete_protein", False),
        "renumber_residues": config.get("renumber_residues", False),
        "chain_names": names,
        "fragment_names": fragment_names,
        "exclude_fragments": sorted(excluded),
        "fragments": list(fragments.values()),
        "splits": split_records,
        "residue_mapping": list(residue_map.values()),
    }
    structure.source_info["input_reconstruction"] = report
    if excluded:
        keep = [i for i, key in enumerate(atom_keys) if key not in excluded or not polymer_atoms[i]]
        if not keep:
            raise ValueError("Select at least one input component")
        structure.select_atoms(keep)
        retained = set(zip(structure.chain_ids, structure.resids, strict=True))
        report["residue_mapping"] = [
            row for row in report["residue_mapping"] if (row["chain"], row["resid"]) in retained
        ]
    return report

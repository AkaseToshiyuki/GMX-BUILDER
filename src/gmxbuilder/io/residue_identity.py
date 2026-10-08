"""Reversible deposited residue identities behind integer working indices."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from gmxbuilder.core.exceptions import ParseError


def clean_identifier(value):
    return "" if value in (None, ".", "?") else str(value).strip()


def working_residue_ids(keys):
    """Map (logical chain, label chain, author number, insertion) to integers.

    Preserve ordinary numbering. Renumber an entire affected chain in deposited
    sequence order if insertion codes or namespace collisions require it; this
    avoids collisions with both preceding and following author numbers.
    """
    unique = dict.fromkeys(keys)
    by_number = defaultdict(set)
    remap = set()
    for chain, label, number, insertion in unique:
        try:
            numeric = int(number)
        except (ValueError, TypeError) as exc:
            raise ParseError(f"Invalid residue number {number!r} in chain {chain!r}") from exc
        if not np.iinfo(np.int64).min <= numeric <= np.iinfo(np.int64).max:
            raise ParseError("Residue identifier exceeds the supported signed 64-bit range")
        by_number[(chain, numeric)].add((label, insertion))
        if insertion or len(by_number[(chain, numeric)]) > 1:
            remap.add(chain)
    counters = defaultdict(int)
    assigned = {}
    mapping = []
    for key in unique:
        chain, label, number, insertion = key
        counters[chain] += 1
        assigned[key] = counters[chain] if chain in remap else int(number)
        if chain in remap:
            mapping.append(
                {
                    "chain": chain,
                    "resid": assigned[key],
                    "author_resid": str(number),
                    "label_chain": label,
                    "insertion_code": insertion,
                }
            )
    return [assigned[key] for key in keys], mapping


def source_residue(structure, index):
    """Recover author identity from immutable atom evidence after selections."""
    atom = structure.source_info.get("atoms", {}).get(structure.source_ids[index], {})
    return {
        "chain": atom.get(
            "author_chain", atom.get("_atom_site.auth_asym_id", structure.chain_ids[index])
        ),
        "resid": atom.get("resid", atom.get("_atom_site.auth_seq_id", structure.resids[index])),
        "insertion_code": clean_identifier(
            atom.get("icode", atom.get("_atom_site.pdbx_PDB_ins_code", ""))
        ),
        "label_chain": clean_identifier(atom.get("_atom_site.label_asym_id", "")),
        "label_seq_id": clean_identifier(atom.get("_atom_site.label_seq_id", "")),
    }

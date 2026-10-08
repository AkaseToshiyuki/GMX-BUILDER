"""Authoritative input storage, independent of fixed-width viewer exports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from gmxbuilder.core.structure import PER_ATOM_FIELDS, Structure

_FIELDS = PER_ATOM_FIELDS


def canonical_path(path):
    return Path(str(path) + ".input.npz")


def read_input(path):
    path = Path(path)
    if path.name.endswith(".input.npz"):
        with np.load(path, allow_pickle=False) as arrays:
            schema = int(arrays["input_schema"])
            if schema not in {1, 2}:
                raise ValueError("Unsupported canonical input version; upload the source again")
            source_json = (
                arrays["source_info"].tobytes().decode("utf-8")
                if schema == 2
                else str(arrays["source_info"])
            )
            structure = Structure(
                arrays["coordinates"],
                arrays["box_vectors"],
                **{key: arrays[key].tolist() for key in _FIELDS},
                source_info=json.loads(source_json),
            )
            from gmxbuilder.io.cell import classify_cell, display_envelope

            info = structure.source_info
            if (
                info.get("cell_status", classify_cell(info.get("cell_vectors_nm"))) == "placeholder"
                and classify_cell(structure.box_vectors) == "placeholder"
            ):
                info["cell_status"] = "placeholder"
                structure.box_vectors = display_envelope(structure.coordinates)
                info["box_source"] = "estimated"
            return structure
    from gmxbuilder.io.cif import CIFParser
    from gmxbuilder.io.cif_document import is_cif
    from gmxbuilder.io.pdb import PDBParser
    from gmxbuilder.io.source_document import source_scope, source_text
    from gmxbuilder.modules.input.validation import read_polymer_metadata

    with source_scope(path):
        raw = source_text(path, errors="strict")
        structure = (CIFParser() if is_cif(raw) else PDBParser()).parse(path)
        structure.source_info["polymer_metadata"] = read_polymer_metadata(path)
    return structure


def write_input(structure, path):
    """Atomically publish lossless arrays and JSON; never pickle input data."""
    structure.validate_atom_fields()
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            arrays = {key: np.asarray(getattr(structure, key)) for key in _FIELDS}
            np.savez_compressed(
                handle,
                coordinates=structure.coordinates,
                box_vectors=structure.box_vectors,
                input_schema=np.array(2),
                # UTF-8 avoids NumPy's four-byte-per-character Unicode array and
                # its large transient copies for per-atom deposition records.
                source_info=np.frombuffer(
                    json.dumps(structure.source_info, separators=(",", ":")).encode("utf-8"),
                    dtype=np.uint8,
                ),
                **arrays,
            )
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_mmcif(structure, path, *, source_identifiers=False):
    """Write working IDs for scientific adapters, or deposited IDs for export.

    Canonical input remains authoritative. Source-ID export preserves insertion
    codes after selection/renumbering; newly built atoms retain working IDs.
    """
    import gemmi

    from gmxbuilder.io.residue_identity import source_residue

    document = gemmi.cif.Document()
    block = document.add_new_block("gmxbuilder")
    tags = [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    ]
    table = block.init_loop("_atom_site.", tags)
    chains = {chain: str(i + 1) for i, chain in enumerate(dict.fromkeys(structure.chain_ids))}
    residue_order = {}
    for i in range(structure.num_atoms):
        name, residue, chain = (
            structure.atom_names[i],
            structure.resnames[i],
            structure.chain_ids[i],
        )
        number = str(structure.resids[i])
        key = (chain, number, residue)
        residue_order.setdefault(key, len(residue_order) + 1)
        label_number = str(residue_order[key])
        author_chain, author_number, insertion = chain, number, "?"
        if source_identifiers and structure.source_ids[i]:
            identity = source_residue(structure, i)
            author_chain = identity["chain"]
            author_number = str(identity["resid"])
            insertion = identity["insertion_code"] or "?"
        row = [
            "ATOM",
            str(i + 1),
            structure.elements[i],
            gemmi.cif.quote(name),
            ".",
            gemmi.cif.quote(residue),
            gemmi.cif.quote(chain),
            chains[chain],
            label_number,
            gemmi.cif.quote(insertion) if insertion != "?" else "?",
            *(format(float(x) * 10, ".17g") for x in structure.coordinates[i]),
            str(structure.occupancies[i]),
            str(structure.tempfactors[i]),
            author_number,
            gemmi.cif.quote(residue),
            gemmi.cif.quote(author_chain),
            gemmi.cif.quote(name),
            "1",
        ]
        table.add_row(row)
    Path(path).write_text(document.as_string())

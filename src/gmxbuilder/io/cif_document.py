"""CIF syntax handling shared by coordinates and deposited metadata."""

from __future__ import annotations

import gemmi

from gmxbuilder.core.exceptions import ParseError

# PDBx tags are case insensitive; retain historical spellings for API callers.
_MIXED = {
    x.lower(): x
    for x in (
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "B_iso_or_equiv",
        "pdbx_PDB_model_num",
        "pdbx_PDB_ins_code",
        "group_PDB",
        "PDB_model_num",
        "PDB_ins_code",
    )
}


def structure_block(raw: str):
    from gmxbuilder.io.source_document import parsed_documents

    cache = parsed_documents()
    if cache is not None and raw in cache:
        return cache[raw]
    try:
        document = gemmi.cif.read_string(raw.lstrip("\ufeff"))
    except (ValueError, RuntimeError) as exc:
        raise ParseError(f"Invalid CIF syntax: {exc}") from exc
    blocks = [b for b in document if b.find_mmcif_category("_atom_site.")]
    if len(blocks) == 1:
        if cache is not None:
            cache[raw] = blocks[0]
        return blocks[0]
    if not blocks and len(document) == 1:
        return document[0]
    raise ParseError(
        "CIF must identify exactly one coordinate data block; select a structure explicitly"
    )


def category(raw: str, prefix: str) -> tuple[list[str], list[str]]:
    table = structure_block(raw).find_mmcif_category(prefix.lower())
    if not table:
        return [], []
    tags = []
    for tag in table.tags:
        cat, name = tag.lower().split(".", 1)
        tags.append(cat + "." + _MIXED.get(name, name))
    values = [
        value if value in {".", "?"} else gemmi.cif.as_string(value)
        for row in table
        for value in row
    ]
    return tags, values


def is_cif(raw: str) -> bool:
    # Comments and empty lines may precede the first CIF control token.
    lines = (line.strip() for line in raw.lstrip("\ufeff").splitlines())
    first = next((line for line in lines if line and not line.startswith("#")), "")
    return first.lower().startswith("data_")

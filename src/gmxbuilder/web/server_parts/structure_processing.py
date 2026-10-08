"""Bounded structure parsing used by upload and resume endpoints."""

from __future__ import annotations

import gzip
import io
import logging
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np

from gmxbuilder.core.chemistry import is_hydrogen
from gmxbuilder.io.pdb import (
    _LIPID_DETERGENT,
    _PROTEIN_RESNAMES,
    _SOLVENT_IONS,
    PDBValidator,
    PDBWriter,
)
from gmxbuilder.modules.nucleic_acid.support import (
    classify_nucleic_residue,
    nucleic_polymer_residues,
)
from gmxbuilder.web.server_parts.input_limits import (
    StructureInputLimits,
    enforce_parsed_atom_limit,
    inspect_structure_payload,
)

logger = logging.getLogger("gmxbuilder.web")

# Kept at module scope so server.py can re-export the legacy private names.
NONSTANDARD_AA_MAP = {
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "CYM": "CYS",
    "CYX": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    "MSE": "MET",
}

WATER_RESNAMES = {"HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD"}

STRUCTURE_SUFFIX_FORMATS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "cif",
    ".mmcif": "cif",
}


def filter_pdb_for_display(pdb_text: str) -> str:
    """Strip water and hydrogen records from text used by the 3D viewer."""
    lines: list[str] = []
    for line in pdb_text.split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            lines.append(line)
            continue
        resname = line[17:20].strip() if len(line) > 20 else ""
        atom_name = line[12:16].strip() if len(line) > 16 else ""
        element = line[76:78].strip() if len(line) >= 78 else ""
        if resname.upper() in WATER_RESNAMES:
            continue
        if is_hydrogen(atom_name, element):
            continue
        lines.append(line)
    return "\n".join(lines)


def detect_disulfides(pdb_path: Path) -> list[tuple[str, int, str, str, int, str]]:
    """Find CYS SG-SG pairs within disulfide bond distance (< 2.5 Å)."""
    cys_atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            if resname == "CYS" and atom_name == "SG":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chain = line[21:22].strip()
                    resid = int(line[22:26].strip())
                    cys_atoms.append((x, y, z, chain, resid))
                except (ValueError, IndexError):
                    continue

    pairs = []
    for i in range(len(cys_atoms)):
        for j in range(i + 1, len(cys_atoms)):
            xi, yi, zi, chi, ridi = cys_atoms[i]
            xj, yj, zj, chj, ridj = cys_atoms[j]
            distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            if distance < 2.5:
                pairs.append((chi, ridi, "CYS", chj, ridj, "CYS"))
    return pairs


def apply_disulfide_rename(pdb_path: Path, pairs: list) -> None:
    """Rename paired CYS residues to CYX in the PDB file."""
    rename_set = set()
    for chain, resid, _, other_chain, other_resid, _ in pairs:
        rename_set.add((chain, resid))
        rename_set.add((other_chain, other_resid))

    with open(pdb_path) as fh:
        lines = fh.readlines()

    with open(pdb_path, "w") as fh:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                try:
                    resid = int(line[22:26].strip())
                except ValueError:
                    resid = 0
                if resname == "CYS" and (chain, resid) in rename_set:
                    line = line[:17] + "CYX" + line[20:]
            fh.write(line)


def auto_clean_pdb(input_path: Path, output_path: Path) -> None:
    """Clean, center, and identify disulfides in a PDB file."""
    renamed = 0
    stripped_water = 0
    stripped_h = 0
    kept = 0

    with open(input_path) as fh_in, open(output_path, "w") as fh_out:
        for line in fh_in:
            if not line.startswith(("ATOM", "HETATM")):
                fh_out.write(line)
                continue

            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if resname.upper() in WATER_RESNAMES:
                stripped_water += 1
                continue
            if is_hydrogen(atom_name, element):
                stripped_h += 1
                continue
            new_name = NONSTANDARD_AA_MAP.get(resname.upper())
            if new_name and line.startswith("ATOM"):
                line = line[:17] + f"{new_name:>3s}" + line[20:]
                renamed += 1
            fh_out.write(line)
            kept += 1

    if kept == 0:
        output_path.unlink(missing_ok=True)
        return

    atom_coords = []
    with open(output_path) as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except (ValueError, IndexError):
                pass
    if atom_coords:
        centroid = np.array(atom_coords).mean(axis=0)
        with open(output_path, "w") as fh:
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38]) - centroid[0]
                        y = float(line[38:46]) - centroid[1]
                        z = float(line[46:54]) - centroid[2]
                        line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                    except (ValueError, IndexError):
                        pass
                fh.write(line)

    disulfide_pairs = detect_disulfides(output_path)
    if disulfide_pairs:
        apply_disulfide_rename(output_path, disulfide_pairs)
        logger.info(
            "PDB cleanup: %d residues renamed, %d waters removed, "
            "%d hydrogens removed, %d atoms kept, centered at origin, "
            "%d disulfide bond(s) detected",
            renamed,
            stripped_water,
            stripped_h,
            kept,
            len(disulfide_pairs),
        )
    else:
        logger.info(
            "PDB cleanup: %d residues renamed, %d waters removed, "
            "%d hydrogens removed, %d atoms kept, centered at origin",
            renamed,
            stripped_water,
            stripped_h,
            kept,
        )


def structure_upload_suffix(filename: str) -> tuple[str, bool]:
    """Return the declared structure format and whether it is gzip-compressed."""
    safe_name = Path(filename or "").name
    lower = safe_name.lower()
    compressed = lower.endswith(".gz")
    inner_name = safe_name[:-3] if compressed else safe_name
    declared = STRUCTURE_SUFFIX_FORMATS.get(Path(inner_name).suffix.lower())
    if declared is None:
        raise ValueError(
            "Accepted structure formats are .pdb, .ent, .cif, .mmcif, and their .gz variants"
        )
    return declared, compressed


def prepare_structure_upload(
    filename: str,
    content: bytes,
    max_bytes: int,
) -> tuple[str, bytes, str, list[str]]:
    """Bound decompression, detect content format, and choose a canonical name."""
    declared, compressed = structure_upload_suffix(filename)
    if compressed:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as stream:
                content = stream.read(max_bytes + 1)
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise ValueError("The uploaded .gz file is not a valid gzip stream") from exc
        if len(content) > max_bytes:
            raise ValueError("Decompressed structure exceeds the upload size limit")
    if not content:
        raise ValueError("The uploaded structure file is empty")
    if b"\x00" in content[: 1024 * 1024]:
        raise ValueError("The uploaded structure contains binary data")

    sample = content[: 2 * 1024 * 1024].decode("utf-8-sig", errors="replace")
    normalized = sample.replace("\r\n", "\n").replace("\r", "\n")
    pdb_records = any(
        line[:6].strip().upper() in {"ATOM", "HETATM", "MODEL", "CRYST1"}
        for line in normalized.splitlines()
    )
    from gmxbuilder.io.cif_document import is_cif

    cif_records = is_cif(content.decode("utf-8-sig", errors="strict"))
    # mmCIF wins whenever its markers are present, and is tested first. Its
    # _atom_site loop rows begin with ATOM or HETATM -- that is the group_PDB
    # column -- so every real mmCIF also looks like it holds PDB records.
    # Requiring their absence made the CIF branch unreachable for any file that
    # actually contained coordinates, and the PDB parser was handed mmCIF.
    # `_atom_site.` category tags are the discriminator: a PDB file never has
    # them.
    detected = "cif" if cif_records else "pdb" if pdb_records else declared
    warnings: list[str] = []
    if detected != declared:
        warnings.append(
            f"File content was detected as {detected.upper()} despite its filename; "
            "content detection was used."
        )

    original = Path(filename[:-3] if compressed else filename).name
    stem = Path(original).stem or "structure"
    canonical_name = f"{stem}.{'cif' if detected == 'cif' else 'pdb'}"
    return canonical_name, content, detected, warnings


def prepare_and_inspect_structure_upload(
    filename: str,
    content: bytes,
    limits: StructureInputLimits,
) -> tuple[str, bytes, str, list[str]]:
    """Decompress, identify and complexity-scan untrusted structure text."""
    prepared = prepare_structure_upload(filename, content, limits.max_bytes)
    _stored_name, payload, structure_format, _warnings = prepared
    inspect_structure_payload(payload, structure_format, limits)
    return prepared


def extract_sequences(structure) -> list[dict]:
    """Extract ordered protein and nucleic-acid residue sequences by chain."""
    non_protein = _SOLVENT_IONS | _LIPID_DETERGENT
    chain_data: dict[str, list[dict]] = OrderedDict()
    seen: dict[str, set] = {}

    nucleic_residues = nucleic_polymer_residues(structure)
    for index in range(structure.num_atoms):
        chain = structure.chain_ids[index] if index < len(structure.chain_ids) else "?"
        resname = structure.resnames[index] if index < len(structure.resnames) else "UNK"
        resid = structure.resids[index] if index < len(structure.resids) else index + 1
        if resname in non_protein:
            continue
        is_protein = resname in _PROTEIN_RESNAMES
        residue_key = (str(chain), int(resid))
        is_nucleic = residue_key in nucleic_residues
        nucleic_type = (
            nucleic_residues[residue_key] if is_nucleic else classify_nucleic_residue(resname)
        )
        if not is_protein and not is_nucleic:
            continue

        if chain not in chain_data:
            chain_data[chain] = []
            seen[chain] = set()
        key = (resname, resid)
        if key in seen[chain]:
            continue
        seen[chain].add(key)
        from gmxbuilder.io.residue_identity import source_residue

        record = {"resname": resname, "resid": resid, "is_protein": is_protein}
        if structure.source_ids[index]:
            source_identity = source_residue(structure, index)
            record.update(
                {
                    "author_resid": source_identity["resid"],
                    "author_chain": source_identity["chain"],
                    "insertion_code": source_identity["insertion_code"],
                }
            )
        if is_nucleic:
            record["is_nucleic"] = True
            record["polymer_type"] = nucleic_type or "modified"
        chain_data[chain].append(record)

    sequences = [
        {"chain_id": chain_id, "length": len(residues), "residues": residues}
        for chain_id, residues in chain_data.items()
        if residues
    ]
    # Canonical lineage survives splitting, renumbering and checkpoint resume.
    mapping = {
        (row["chain"], row["resid"]): row
        for row in structure.source_info.get("input_reconstruction", {}).get("residue_mapping", [])
    }
    groups = {}
    for chain in sequences:
        rows = []
        for residue in chain["residues"]:
            source = mapping.get((chain["chain_id"], residue["resid"]))
            if source:
                residue["author_resid"] = source["author_resid"]
                residue["author_chain"] = source["author_chain"]
                rows.append(source)
        if rows:
            chain["source_chain"] = rows[0]["original_chain"]
            chain["author_chain"] = rows[0]["author_chain"]
            groups.setdefault(chain["source_chain"], []).append(chain)
    for fragments in groups.values():
        for index, chain in enumerate(fragments):
            chain["fragment_index"] = index + 1
            chain["fragment_count"] = len(fragments)
            chain["internal_n_terminus"] = index > 0
            chain["internal_c_terminus"] = index < len(fragments) - 1
    fragment_records = {
        row["chain_id"]: row
        for row in structure.source_info.get("input_reconstruction", {}).get("fragments", [])
    }
    for chain in sequences:
        if record := fragment_records.get(chain["chain_id"]):
            chain.update(record)
            chain["internal_n_terminus"] = record["fragment_index"] > 1
            chain["internal_c_terminus"] = record["fragment_index"] < record["fragment_count"]
    return sequences


def _pdb_summary(
    pdb_path: Path,
    limits: StructureInputLimits,
    display_filter: Callable[[str], str],
    sequence_extractor: Callable[[object], list[dict]],
    protein_resnames: Collection[str],
    *,
    validate: bool,
    source_metadata: dict | None = None,
    structure=None,
    preview_text: str | None = None,
    assess_readiness: bool = True,
) -> dict:
    from gmxbuilder.io.input_document import canonical_path, read_input

    checkpoint = pdb_path.name == "viewer.pdb" and (pdb_path.parent / "system.npz").is_file()
    payload = b"" if checkpoint else pdb_path.read_bytes()
    if payload:
        inspect_structure_payload(payload, "pdb", limits)
    canonical = canonical_path(pdb_path)
    validation = (
        PDBValidator.validate(pdb_path)
        if validate and not checkpoint and not canonical.exists()
        else {"valid": True, "errors": [], "warnings": []}
    )
    if not validation["valid"]:
        return {"valid": False, "validation": validation, "pdb_path": str(pdb_path)}

    if structure is not None:
        pass
    elif checkpoint:
        from gmxbuilder.core.system import System

        structure = System.load_checkpoint(pdb_path.parent).structure
    else:
        structure = read_input(canonical if canonical.exists() else pdb_path)
    from gmxbuilder.modules.input.reconstruction import normalize_chain_ids

    normalize_chain_ids(structure)
    enforce_parsed_atom_limit(structure.num_atoms, limits)
    validation["warnings"].extend(structure.source_info.get("format_warnings", []))
    if not PDBWriter.identifiers_fit(structure):
        validation["warnings"].append(
            "The PDB preview abbreviates identifiers. Full identities are retained in canonical "
            "input; PDB-only scientific tools require a lossless adapter."
        )
    info = structure.source_info
    if info.get("model_count", 1) > 1:
        validation["warnings"].append(
            f"Selected model {info.get('selected_model_number')} "
            f"(ordinal {info.get('selected_model_ordinal')}) of {info['model_count']} models."
        )
    from gmxbuilder.io.cell import classify_cell

    cell_status = info.get("cell_status", classify_cell(info.get("cell_vectors_nm")))
    if cell_status == "placeholder":
        validation["warnings"].append(
            "The deposited 1-Angstrom cell is a placeholder, not a simulation box. "
            "An estimated display envelope is shown; the construction step determines the box."
        )
    if info.get("residue_mapping"):
        validation["warnings"].append(
            "Residues with insertion codes or colliding author numbers have unique working "
            "numbers. Original residue identities are retained in the source mapping."
        )
    if not assess_readiness:
        validation["warnings"].append(
            "File read successfully. Coarse-grained preparation has not been checked; "
            "run the workflow input check before continuing."
        )
    if preview_text is None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "viewer.pdb"
            PDBWriter.write(structure, preview, wrap_ids_for_viewer=True)
            preview_text = preview.read_text()
    pdb_text = preview_text
    sequences = sequence_extractor(structure)
    # Selection and preview use working IDs; deposited identities stay in source_info.
    chains = [item["chain_id"] for item in sequences]
    small_molecules = PDBValidator.detect_small_molecules(structure)
    from gmxbuilder.modules.input.validation import assess_input_structure, read_polymer_metadata

    if source_metadata is None:
        source_metadata = structure.source_info.get("polymer_metadata")
        if source_metadata is None:
            source_metadata = read_polymer_metadata(pdb_path)
    atom_counts = Counter(structure.resnames)
    residue_counts = Counter(
        name
        for _, _, name in dict.fromkeys(
            zip(structure.chain_ids, structure.resids, structure.resnames, strict=True)
        )
    )
    readiness = assess_input_structure(structure, source_metadata) if assess_readiness else None
    return {
        "valid": True,
        "validation": validation,
        "input_source_metadata": source_metadata,
        "input_validation": readiness,
        "input_status": {
            "parsing": "complete",
            "readiness": ("passed" if readiness["can_proceed"] else "blocked")
            if readiness is not None
            else "not_checked",
            "scope": "all_atom_input" if assess_readiness else "coarse_grained",
        },
        "cell_info": {
            "status": cell_status,
            "deposited_vectors_nm": info.get("cell_vectors_nm"),
            "display_vectors_nm": structure.box_vectors.tolist(),
            "box_source": info.get("box_source", "estimated"),
        },
        "chain_identity_version": 4,
        "pdb_path": str(pdb_path),
        "pdb_content": display_filter(pdb_text) or pdb_text,
        "num_atoms": structure.num_atoms,
        # Retain the legacy atom-count field for older clients.
        "residues": dict(atom_counts.most_common(20)),
        "atom_counts_by_resname": dict(atom_counts),
        "residue_counts_by_resname": dict(residue_counts),
        "protein_residues": sorted(
            residue for residue in residue_counts if residue in protein_resnames
        ),
        "chains": chains,
        "chain_mapping": structure.source_info.get("working_chain_ids", {}),
        "box_nm": [round(value, 3) for value in structure.dimensions().tolist()],
        "sequences": sequences,
        "small_molecules": small_molecules,
    }


def process_uploaded_structure(
    uploaded_path: Path,
    task_dir: Path,
    structure_format: str,
    format_warnings: list[str],
    limits: StructureInputLimits,
    display_filter: Callable[[str], str],
    sequence_extractor: Callable[[object], list[dict]],
    protein_resnames: Collection[str],
    assess_readiness: bool = True,
) -> dict:
    """Convert if needed and parse a new upload entirely outside the event loop."""
    from gmxbuilder.io.input_document import canonical_path, read_input, write_input

    structure = read_input(uploaded_path)
    from gmxbuilder.modules.input.reconstruction import normalize_chain_ids

    normalize_chain_ids(structure)
    enforce_parsed_atom_limit(structure.num_atoms, limits)
    if structure_format == "pdb":
        structure.source_info["format_warnings"] = PDBValidator.validate(uploaded_path).get(
            "warnings", []
        )
    pdb_path = task_dir / "converted.pdb"
    PDBWriter.write(structure, pdb_path, title="Normalized input preview", wrap_ids_for_viewer=True)

    write_input(structure, canonical_path(pdb_path))
    result = _pdb_summary(
        pdb_path,
        limits,
        display_filter,
        sequence_extractor,
        protein_resnames,
        validate=True,
        structure=structure,
        preview_text=pdb_path.read_text(),
        assess_readiness=assess_readiness,
    )
    validation = dict(result["validation"])
    validation["warnings"] = list(format_warnings) + list(validation.get("warnings", []))
    result["validation"] = validation
    return result


def summarize_resume_structure(
    pdb_path: Path,
    limits: StructureInputLimits,
    display_filter: Callable[[str], str],
    sequence_extractor: Callable[[object], list[dict]],
    protein_resnames: Collection[str],
    source_metadata: dict | None = None,
    source_path: Path | None = None,
    assess_readiness: bool = True,
) -> dict:
    """Rebuild a legacy or changed-checkpoint summary under the same limits."""
    if (
        source_metadata is None
        and source_path is not None
        and source_path.is_file()
        and not source_path.is_symlink()
    ):
        from gmxbuilder.modules.input.validation import read_polymer_metadata

        source_metadata = read_polymer_metadata(source_path)
    return _pdb_summary(
        pdb_path,
        limits,
        display_filter,
        sequence_extractor,
        protein_resnames,
        validate=False,
        source_metadata=source_metadata,
        assess_readiness=assess_readiness,
    )


def read_bounded_pdb_display(
    pdb_path: Path,
    limits: StructureInputLimits,
    display_filter: Callable[[str], str],
) -> str:
    """Read viewer text while re-applying byte/record complexity limits."""
    payload = pdb_path.read_bytes()
    inspect_structure_payload(payload, "pdb", limits)
    text = payload.decode("utf-8", errors="replace")
    return display_filter(text) or text

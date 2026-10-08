"""Synchronous structure-file transforms used by web routes."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Collection
from pathlib import Path

import numpy as np

from gmxbuilder.io.pdb import PDBParser, PDBWriter


def write_preview_pdb(
    oriented_pdb: str,
    preview_path: Path,
    box_dimensions_nm: object,
) -> None:
    """Write frontend-oriented coordinates with the requested preview box."""
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(suffix=".pdb")
        os.close(descriptor)
        Path(temporary_name).write_text(oriented_pdb)
        structure = PDBParser().parse(temporary_name)
        if box_dimensions_nm and len(box_dimensions_nm) == 3:  # type: ignore[arg-type]
            structure.box_vectors = np.diag(
                [
                    float(box_dimensions_nm[0]),  # type: ignore[index]
                    float(box_dimensions_nm[1]),  # type: ignore[index]
                    float(box_dimensions_nm[2]),  # type: ignore[index]
                ]
            )
        PDBWriter.write(structure, preview_path, title="GMXBUILDER Preview")
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def normalise_small_molecule_labels(
    raw_labels: object,
    allowed_resnames: set[str],
) -> dict[str, str]:
    """Validate user-facing molecule labels without changing structural IDs."""
    if not isinstance(raw_labels, dict):
        raise ValueError("small_molecule_labels must be an object")

    labels: dict[str, str] = {}
    for raw_key, raw_label in raw_labels.items():
        if not isinstance(raw_key, str) or not isinstance(raw_label, str):
            raise ValueError("Small-molecule label keys and values must be strings")
        key = raw_key.strip().upper()
        label = raw_label.strip()
        if key not in allowed_resnames:
            raise ValueError(f"Unknown small-molecule key {key!r}")
        if not label:
            raise ValueError(f"Display label for {key} must not be empty")
        if len(label) > 64:
            raise ValueError(f"Display label for {key} must be at most 64 characters")
        if any(ord(character) < 32 or ord(character) == 127 for character in label):
            raise ValueError(f"Display label for {key} contains control characters")
        labels[key] = label

    effective = {key: labels.get(key, key) for key in sorted(allowed_resnames)}
    seen: dict[str, str] = {}
    for key, label in effective.items():
        folded = label.casefold()
        if folded in seen:
            raise ValueError(
                f"Small-molecule display label {label!r} is used for both {seen[folded]} and {key}"
            )
        seen[folded] = key
    return labels


def filter_pdb_file(
    source: Path,
    destination: Path,
    include_chains: Collection[str] | None,
    exclude_resnames: Collection[str],
) -> tuple[int, int]:
    """Publish a nonempty canonical selection and its separate display adapter.

    None selects all chains. An explicit empty collection selects nothing.
    Never filter a prior selection in place: deselected atoms must be recoverable.
    """
    if source.resolve() == destination.resolve() or (
        destination.exists() and os.path.samefile(source, destination)
    ):
        raise ValueError("Filtering requires distinct original and output files")
    if include_chains is not None and not include_chains:
        raise ValueError("Select at least one chain or retained molecule")
    from gmxbuilder.io.input_document import canonical_path, read_input, write_input

    canonical_source = canonical_path(source)
    structure = read_input(canonical_source if canonical_source.exists() else source)
    from gmxbuilder.modules.input.reconstruction import normalize_chain_ids

    normalize_chain_ids(structure)
    found_chains = set(structure.chain_ids)
    unknown = set(include_chains or ()) - found_chains
    if unknown:
        raise ValueError(f"Selected chain IDs are absent from the upload: {sorted(unknown)!r}")
    indices = [
        i
        for i in range(structure.num_atoms)
        if (include_chains is None or structure.chain_ids[i] in include_chains)
        and structure.resnames[i] not in exclude_resnames
    ]
    if not indices:
        raise ValueError("Selection contains no atoms; retain at least one solute")
    selected = structure.take(indices)
    descriptor, name = tempfile.mkstemp(dir=destination.parent, suffix=".pdb")
    os.close(descriptor)
    try:
        PDBWriter.write(selected, name, title="Selected input preview", wrap_ids_for_viewer=True)
        # Canonical arrays are authoritative. The PDB is only a display adapter.
        write_input(selected, canonical_path(destination))
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return len(indices), structure.num_atoms - len(indices)

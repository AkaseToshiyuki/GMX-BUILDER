"""Conservative repair of missing heavy atoms in standard amino acids.

The input step may repair a residue only when its backbone is complete and
the observed side-chain atoms form an unbroken path from the backbone.  This
keeps routine coordinate omissions automatic while leaving backbone gaps,
non-standard chemistry, and disconnected partial side chains for explicit
user review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure


_BACKBONE = {"N", "CA", "C", "O"}

# Parent trees are used only to decide whether an incomplete side chain is
# unambiguous enough to repair.  Ring closure bonds are intentionally omitted:
# every observed atom must still have at least one intact path to CA.
_SIDECHAIN_PARENTS: dict[str, dict[str, str]] = {
    "ALA": {"CB": "CA"},
    "ARG": {"CB": "CA", "CG": "CB", "CD": "CG", "NE": "CD", "CZ": "NE",
            "NH1": "CZ", "NH2": "CZ"},
    "ASN": {"CB": "CA", "CG": "CB", "OD1": "CG", "ND2": "CG"},
    "ASP": {"CB": "CA", "CG": "CB", "OD1": "CG", "OD2": "CG"},
    "CYS": {"CB": "CA", "SG": "CB"},
    "GLN": {"CB": "CA", "CG": "CB", "CD": "CG", "OE1": "CD", "NE2": "CD"},
    "GLU": {"CB": "CA", "CG": "CB", "CD": "CG", "OE1": "CD", "OE2": "CD"},
    "GLY": {},
    "HIS": {"CB": "CA", "CG": "CB", "ND1": "CG", "CD2": "CG",
            "CE1": "ND1", "NE2": "CE1"},
    "ILE": {"CB": "CA", "CG1": "CB", "CG2": "CB", "CD1": "CG1"},
    "LEU": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG"},
    "LYS": {"CB": "CA", "CG": "CB", "CD": "CG", "CE": "CD", "NZ": "CE"},
    "MET": {"CB": "CA", "CG": "CB", "SD": "CG", "CE": "SD"},
    "PHE": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG",
            "CE1": "CD1", "CE2": "CD2", "CZ": "CE1"},
    "PRO": {"CB": "CA", "CG": "CB", "CD": "CG"},
    "SER": {"CB": "CA", "OG": "CB"},
    "THR": {"CB": "CA", "OG1": "CB", "CG2": "CB"},
    "TRP": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG",
            "NE1": "CD1", "CE2": "NE1", "CE3": "CD2", "CZ2": "CE2",
            "CZ3": "CE3", "CH2": "CZ2"},
    "TYR": {"CB": "CA", "CG": "CB", "CD1": "CG", "CD2": "CG",
            "CE1": "CD1", "CE2": "CD2", "CZ": "CE1", "OH": "CZ"},
    "VAL": {"CB": "CA", "CG1": "CB", "CG2": "CB"},
}


@dataclass(frozen=True)
class RepairRecord:
    chain: str
    resid: int
    resname: str
    added_atoms: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "chain": self.chain or "?",
            "resid": self.resid,
            "resname": self.resname,
            "added_atoms": list(self.added_atoms),
        }


def _atom_key(structure: Structure, index: int) -> tuple[str, int, str, str]:
    return (
        str(structure.chain_ids[index]).strip(),
        int(structure.resids[index]),
        str(structure.resnames[index]).strip().upper(),
        str(structure.atom_names[index]).strip().upper(),
    )


def _protein_residue_groups(
    structure: Structure,
) -> dict[tuple[str, int, str], list[int]]:
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index in range(structure.num_atoms):
        resname = str(structure.resnames[index]).strip().upper()
        if resname not in _SIDECHAIN_PARENTS:
            continue
        key = (
            str(structure.chain_ids[index]).strip(),
            int(structure.resids[index]),
            resname,
        )
        groups.setdefault(key, []).append(index)
    return groups


def find_repairable_missing_atoms(
    structure: Structure,
) -> dict[tuple[str, int, str], tuple[str, ...]]:
    """Return conservative repair candidates or raise for ambiguous damage."""
    candidates: dict[tuple[str, int, str], tuple[str, ...]] = {}
    blockers: list[str] = []

    for key, indices in sorted(_protein_residue_groups(structure).items()):
        chain, resid, resname = key
        names = [str(structure.atom_names[index]).strip().upper() for index in indices]
        observed = set(names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            blockers.append(
                f"{chain or '?'}:{resid} {resname} has duplicate atom names "
                f"{','.join(duplicates)}"
            )
            continue

        missing_backbone = sorted(_BACKBONE - observed)
        if missing_backbone:
            blockers.append(
                f"{chain or '?'}:{resid} {resname} missing backbone "
                f"{','.join(missing_backbone)}"
            )
            continue

        parents = _SIDECHAIN_PARENTS[resname]
        missing = sorted(set(parents) - observed)
        if not missing:
            continue

        # A present distal atom separated from CA by a missing parent cannot
        # be placed without choosing between incompatible conformations.
        disconnected: list[str] = []
        for atom in sorted(observed & set(parents)):
            parent = parents[atom]
            if parent not in observed:
                disconnected.append(f"{atom} (missing parent {parent})")
        if disconnected:
            blockers.append(
                f"{chain or '?'}:{resid} {resname} has disconnected partial side chain: "
                + ", ".join(disconnected)
            )
            continue
        candidates[key] = tuple(missing)

    if blockers:
        preview = "; ".join(blockers[:10])
        suffix = f"; and {len(blockers) - 10} more" if len(blockers) > 10 else ""
        raise ModuleConfigError(
            "Protein damage requires user review: " + preview + suffix
            + ". Automatic repair is limited to complete backbones with an unbroken "
              "partial side chain. Upload a repaired model for ambiguous residues."
        )
    return candidates


def _structure_from_fixer(fixer, original: Structure) -> Structure:
    from openmm import unit

    coordinates = np.asarray(
        fixer.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    atoms = list(fixer.topology.atoms())
    if len(atoms) != len(coordinates):
        raise ModuleConfigError("PDBFixer returned inconsistent atom and coordinate counts")

    original_metadata = {
        _atom_key(original, index): (
            original.segids[index],
            float(original.occupancies[index]),
            float(original.tempfactors[index]),
            original.coordinates[index].copy(),
        )
        for index in range(original.num_atoms)
    }
    atom_names: list[str] = []
    resnames: list[str] = []
    resids: list[int] = []
    chain_ids: list[str] = []
    segids: list[str] = []
    elements: list[str] = []
    occupancies: list[float] = []
    tempfactors: list[float] = []

    for atom in atoms:
        residue = atom.residue
        chain = str(residue.chain.id or "").strip()
        try:
            resid = int(residue.id)
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError(
                f"PDBFixer changed residue identifier {residue.id!r} to a non-integer value"
            ) from exc
        resname = str(residue.name).strip().upper()
        atom_name = str(atom.name).strip().upper()
        key = (chain, resid, resname, atom_name)
        metadata = original_metadata.get(key)
        if metadata is None:
            segid, occupancy, tempfactor = "", 1.0, 0.0
        else:
            segid, occupancy, tempfactor, original_coordinate = metadata
            # PDB is limited to 0.001 Angstrom precision.  Restore every
            # retained coordinate exactly after PDBFixer has placed the new
            # atoms so CIF and high-precision PDB inputs are not rounded.
            coordinates[len(atom_names)] = original_coordinate
        atom_names.append(atom_name)
        resnames.append(resname)
        resids.append(resid)
        chain_ids.append(chain)
        segids.append(segid)
        elements.append(atom.element.symbol.upper() if atom.element is not None else atom_name[:1])
        occupancies.append(occupancy)
        tempfactors.append(tempfactor)

    return Structure(
        coordinates=coordinates,
        box_vectors=original.box_vectors.copy(),
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=chain_ids,
        segids=segids,
        elements=elements,
        occupancies=occupancies,
        tempfactors=tempfactors,
    )


def _validate_repair(
    original: Structure,
    repaired: Structure,
    candidates: dict[tuple[str, int, str], tuple[str, ...]],
) -> list[RepairRecord]:
    original_by_key = {_atom_key(original, index): index for index in range(original.num_atoms)}
    repaired_by_key = {_atom_key(repaired, index): index for index in range(repaired.num_atoms)}
    if len(original_by_key) != original.num_atoms or len(repaired_by_key) != repaired.num_atoms:
        raise ModuleConfigError("Automatic repair produced duplicate atom identities")

    missing_original = sorted(set(original_by_key) - set(repaired_by_key))
    if missing_original:
        raise ModuleConfigError(
            "Automatic repair unexpectedly removed existing atoms: "
            + ", ".join(":".join(map(str, key)) for key in missing_original[:10])
        )
    for key, old_index in original_by_key.items():
        new_index = repaired_by_key[key]
        if not np.allclose(
            original.coordinates[old_index], repaired.coordinates[new_index], atol=1e-5, rtol=0.0
        ):
            raise ModuleConfigError(
                "Automatic repair moved an existing atom; the input was left unchanged"
            )

    expected_added = {
        (*residue_key, atom)
        for residue_key, atoms in candidates.items()
        for atom in atoms
    }
    actual_added = set(repaired_by_key) - set(original_by_key)
    if actual_added != expected_added:
        missing = sorted(expected_added - actual_added)
        extra = sorted(actual_added - expected_added)
        details = []
        if missing:
            details.append("not added=" + ",".join(key[-1] for key in missing[:10]))
        if extra:
            details.append("unexpected=" + ",".join(key[-1] for key in extra[:10]))
        raise ModuleConfigError("Automatic repair atom-set mismatch (" + "; ".join(details) + ")")

    # Reject gross external overlaps.  Same-residue atoms are excluded because
    # their covalent geometry is checked separately below.
    for key in sorted(actual_added):
        new_index = repaired_by_key[key]
        for old_key, old_index in original_by_key.items():
            if key[:3] == old_key[:3]:
                continue
            distance = float(np.linalg.norm(
                repaired.coordinates[new_index] - repaired.coordinates[repaired_by_key[old_key]]
            ))
            if distance < 0.075:
                raise ModuleConfigError(
                    f"Automatic repair created a severe clash: {key[0] or '?'}:{key[1]} "
                    f"{key[2]} {key[3]} is {distance:.3f} nm from "
                    f"{old_key[0] or '?'}:{old_key[1]} {old_key[2]} {old_key[3]}"
                )

    records: list[RepairRecord] = []
    for residue_key, atoms in sorted(candidates.items()):
        chain, resid, resname = residue_key
        parents = _SIDECHAIN_PARENTS[resname]
        for atom in atoms:
            parent = parents[atom]
            atom_key = (*residue_key, atom)
            parent_key = (*residue_key, parent)
            if parent_key not in repaired_by_key:
                raise ModuleConfigError(
                    f"Automatic repair could not validate parent {parent} for {resname} {atom}"
                )
            distance = float(np.linalg.norm(
                repaired.coordinates[repaired_by_key[atom_key]]
                - repaired.coordinates[repaired_by_key[parent_key]]
            ))
            if not 0.08 <= distance <= 0.25:
                raise ModuleConfigError(
                    f"Automatic repair produced invalid {resname} {parent}-{atom} bond "
                    f"({distance:.3f} nm) at {chain or '?'}:{resid}"
                )
        records.append(RepairRecord(chain, resid, resname, tuple(sorted(atoms))))
    return records


def repair_standard_protein_heavy_atoms(
    structure: Structure,
) -> tuple[Structure, list[RepairRecord]]:
    """Repair safe standard-residue omissions with PDBFixer/OpenMM."""
    candidates = find_repairable_missing_atoms(structure)
    if not candidates:
        return structure, []

    try:
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ModuleConfigError(
            "Missing standard protein side-chain atoms were detected, but the "
            "PDBFixer/OpenMM repair backend is not installed. Install the project "
            "dependencies or upload a complete model."
        ) from exc

    from tempfile import TemporaryDirectory
    from gmxbuilder.io.pdb import PDBWriter

    with TemporaryDirectory(prefix="gmxbuilder-repair-") as tmpdir:
        input_path = f"{tmpdir}/input.pdb"
        PDBWriter.write(structure, input_path, title="GMXBUILDER repair input")
        try:
            fixer = PDBFixer(filename=input_path)
            fixer.findMissingResidues()
            fixer.missingResidues = {}  # Never synthesize absent sequence/loop residues.
            fixer.findMissingAtoms()
        except Exception as exc:
            raise ModuleConfigError(
                f"PDBFixer could not analyse the incomplete protein: {exc}"
            ) from exc

        allowed = {key: set(atoms) for key, atoms in candidates.items()}
        selected = {}
        selected_names: set[tuple[str, int, str, str]] = set()
        for residue, atoms in fixer.missingAtoms.items():
            try:
                residue_key = (
                    str(residue.chain.id or "").strip(),
                    int(residue.id),
                    str(residue.name).strip().upper(),
                )
            except (TypeError, ValueError):
                continue
            wanted = allowed.get(residue_key)
            if not wanted:
                continue
            chosen = [atom for atom in atoms if str(atom.name).strip().upper() in wanted]
            if chosen:
                selected[residue] = chosen
                selected_names.update(
                    (*residue_key, str(atom.name).strip().upper()) for atom in chosen
                )

        expected_names = {
            (*key, atom) for key, atoms in candidates.items() for atom in atoms
        }
        if selected_names != expected_names:
            unavailable = sorted(expected_names - selected_names)
            detail = ", ".join(
                f"{key[0] or '?'}:{key[1]} {key[2]} {key[3]}" for key in unavailable[:10]
            )
            raise ModuleConfigError(
                "PDBFixer has no matching standard template for: " + detail
            )

        fixer.missingAtoms = selected
        fixer.missingTerminals = {}
        try:
            fixer.addMissingAtoms()
        except Exception as exc:
            raise ModuleConfigError(
                f"PDBFixer could not place the missing protein heavy atoms: {exc}"
            ) from exc
        repaired = _structure_from_fixer(fixer, structure)

    records = _validate_repair(structure, repaired, candidates)
    return repaired, records


def repair_report(records: Iterable[RepairRecord]) -> dict:
    records = list(records)
    return {
        "status": "repaired" if records else "not_needed",
        "backend": "PDBFixer/OpenMM" if records else None,
        "residues_repaired": len(records),
        "atoms_added": sum(len(record.added_atoms) for record in records),
        "residues": [record.as_dict() for record in records],
        "validation": (
            "Existing coordinates preserved; atom identities, covalent distances, "
            "and severe external clashes checked."
            if records else "No missing standard protein heavy atoms detected."
        ),
    }

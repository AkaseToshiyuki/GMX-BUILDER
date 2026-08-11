"""Nucleic-acid identity and force-field capability contracts.

Only canonical DNA/RNA polymers are enabled.  Modified nucleotides are
identified as nucleic-acid material so they can never fall through to the
general small-molecule parameterization path, but remain explicitly blocked
until a residue-specific polymer topology is available.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CANONICAL_DNA_RESNAMES = frozenset({
    "DA", "DC", "DG", "DT",
    "DA5", "DC5", "DG5", "DT5",
    "DA3", "DC3", "DG3", "DT3",
    "DAN", "DCN", "DGN", "DTN",
})
CANONICAL_RNA_RESNAMES = frozenset({
    "A", "C", "G", "U", "RA", "RC", "RG", "RU",
    "RA5", "RC5", "RG5", "RU5",
    "RA3", "RC3", "RG3", "RU3",
    "RAN", "RCN", "RGN", "RUN",
})

# Frequent PDB Chemical Component Dictionary identifiers.  They are detected
# as polymer residues but intentionally not mapped to a canonical base: doing
# so would discard real chemistry (methylation, pseudouridine, oxidation,
# caps, etc.).
KNOWN_MODIFIED_NUCLEOTIDES = frozenset({
    "1MA", "1MG", "2MA", "2MG", "5MC", "5MU", "6MA", "7MG",
    "H2U", "M2G", "M5C", "MIA", "OMG", "OMC", "PSU", "YG",
    "I", "DI", "DU", "BRU", "FHU", "UR3",
})


def classify_nucleic_residue(resname: object) -> str | None:
    """Return ``DNA``, ``RNA``, ``modified`` or ``None``."""
    name = str(resname).strip().upper()
    if name in CANONICAL_DNA_RESNAMES:
        return "DNA"
    if name in CANONICAL_RNA_RESNAMES:
        return "RNA"
    if name in KNOWN_MODIFIED_NUCLEOTIDES:
        return "modified"
    return None


def is_nucleic_like_residue(resname: object, atom_names: Iterable[object]) -> bool:
    """Conservatively recognize a nucleotide from its sugar/backbone atoms."""
    if classify_nucleic_residue(resname) is not None:
        return True
    names = {str(name).strip().upper().replace("*", "'") for name in atom_names}
    sugar = {"C1'", "C2'", "C3'", "C4'", "O3'"}
    return sugar.issubset(names) and ("P" in names or "O5'" in names)


def nucleic_polymer_residues(structure) -> dict[tuple[str, int], str]:
    """Identify residues that belong to a covalent DNA/RNA polymer.

    Canonical residue names are authoritative.  Modified/unknown nucleotide-
    like residues require an observed O3'-P link to an adjacent nucleotide;
    this keeps free ATP/AMP-like ligands in the small-molecule workflow.
    """
    residue_order: list[tuple[str, int]] = []
    atoms: dict[tuple[str, int], dict[str, int]] = {}
    resnames: dict[tuple[str, int], str] = {}
    for index in range(structure.num_atoms):
        key = (
            str(structure.chain_ids[index]),
            int(structure.resids[index]),
        )
        if key not in atoms:
            residue_order.append(key)
            atoms[key] = {}
            resnames[key] = str(structure.resnames[index]).strip().upper()
        name = str(structure.atom_names[index]).strip().upper().replace("*", "'")
        atoms[key][name] = index

    result: dict[tuple[str, int], str] = {}
    candidates: set[tuple[str, int]] = set()
    for key in residue_order:
        classification = classify_nucleic_residue(resnames[key])
        if classification in {"DNA", "RNA"}:
            result[key] = classification
            candidates.add(key)
        elif classification == "modified" or is_nucleic_like_residue(
            resnames[key], atoms[key]
        ):
            candidates.add(key)

    for left, right in zip(residue_order, residue_order[1:]):
        if left[0] != right[0] or left not in candidates or right not in candidates:
            continue
        left_o3 = atoms[left].get("O3'")
        right_p = atoms[right].get("P")
        if left_o3 is None or right_p is None:
            continue
        distance = float(np.linalg.norm(
            structure.coordinates[left_o3] - structure.coordinates[right_p]
        ))
        if distance <= 0.25:
            result.setdefault(left, classify_nucleic_residue(resnames[left]) or "modified")
            result.setdefault(right, classify_nucleic_residue(resnames[right]) or "modified")
    return result


def validate_nucleic_backbone(structure, component) -> list[str]:
    """Return actionable O3'-P continuity defects for one polymer component."""
    residues: list[tuple[tuple[str, int], dict[str, int]]] = []
    lookup: dict[tuple[str, int], dict[str, int]] = {}
    for raw_index in component.atom_indices:
        index = int(raw_index)
        key = (str(structure.chain_ids[index]), int(structure.resids[index]))
        if key not in lookup:
            lookup[key] = {}
            residues.append((key, lookup[key]))
        name = str(structure.atom_names[index]).strip().upper().replace("*", "'")
        lookup[key][name] = index

    issues: list[str] = []
    for (left_key, left_atoms), (right_key, right_atoms) in zip(
        residues, residues[1:]
    ):
        left_o3 = left_atoms.get("O3'")
        right_p = right_atoms.get("P")
        label = (
            f"chain {left_key[0] or '?'} residues {left_key[1]}-{right_key[1]}"
        )
        if left_o3 is None or right_p is None:
            missing = []
            if left_o3 is None:
                missing.append(f"O3' at {left_key[1]}")
            if right_p is None:
                missing.append(f"P at {right_key[1]}")
            issues.append(f"{label} lacks {' and '.join(missing)}")
            continue
        distance = float(np.linalg.norm(
            structure.coordinates[left_o3] - structure.coordinates[right_p]
        ))
        if not 0.12 <= distance <= 0.25:
            issues.append(
                f"{label} has O3'-P distance {distance:.3f} nm "
                "(expected 0.12-0.25 nm)"
            )
    if len(residues) > 1:
        first_key, first_atoms = residues[0]
        last_key, last_atoms = residues[-1]
        first_p = first_atoms.get("P")
        last_o3 = last_atoms.get("O3'")
        if first_p is not None and last_o3 is not None:
            closing_distance = float(np.linalg.norm(
                structure.coordinates[last_o3] - structure.coordinates[first_p]
            ))
            if closing_distance <= 0.20:
                issues.append(
                    f"chain {first_key[0] or '?'} has a {closing_distance:.3f} nm "
                    f"closing O3'-P link between residues {last_key[1]} and "
                    f"{first_key[1]}; circular nucleic acids are not supported"
                )
    return issues


def nucleic_force_field_capability(force_field: str) -> tuple[bool, str]:
    """Return the validated canonical nucleic-acid capability."""
    name = str(force_field).strip().lower()
    if name == "charmm36m":
        return True, (
            "canonical DNA/RNA use the bundled CHARMM36 nucleic-acid "
            "parameters through native GROMACS pdb2gmx topology generation"
        )
    if name == "charmm36":
        return False, (
            "the legacy monolithic CHARMM36 port is not exposed for DNA/RNA; "
            "use CHARMM36m, which bundles the validated CHARMM36 nucleic-acid "
            "parameters in GROMACS-native split databases"
        )
    if name == "amber14sb":
        return False, (
            "Amber ff14SB is a protein force field; modern Amber nucleic-acid "
            "support requires a separately validated DNA OL15/bsc1 or RNA OL3 "
            "parameter set, which is not bundled"
        )
    if name in {"amber99sb", "amber99sb-ildn"}:
        return False, (
            f"{name} bundles legacy AMBER94 nucleic-acid templates; GMXBUILDER "
            "does not expose them as a production-quality DNA/RNA backend"
        )
    return False, f"{force_field} has no validated GMXBUILDER DNA/RNA backend"

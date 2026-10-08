"""Checks on peptide connectivity before chemistry or topology expansion."""

from collections.abc import Iterable

import numpy as np

from gmxbuilder.core.structure import Structure

# Normal peptide C--N is about 0.133 nm. This deliberately generous upper
# bound detects unresolved breaks; it is not a bond-length optimization target.
MAX_PEPTIDE_BOND_NM = 0.20


def validate_peptide_connectivity(structure: Structure, indices: Iterable[int]) -> None:
    """Reject breaks within a declared chain instead of leaving uncapped fragments."""
    issues = peptide_connectivity_issues(structure, indices)
    if issues:
        raise ValueError(issues[0]["message"])


def peptide_connectivity_issues(structure: Structure, indices: Iterable[int]) -> list[dict]:
    """Collect every peptide break, without treating numbering gaps as bonds."""
    issues = []
    chains: dict[str, dict[tuple[int, str], dict[str, int]]] = {}
    for index in indices:
        chain = str(structure.chain_ids[index])
        residue = (int(structure.resids[index]), str(structure.resnames[index]))
        chains.setdefault(chain, {}).setdefault(residue, {})[
            structure.atom_names[index].strip()
        ] = index
    for chain, residues in chains.items():
        ordered = list(residues.items())
        for (left_key, left), (right_key, right) in zip(ordered, ordered[1:]):
            if "C" not in left or "N" not in right:
                issues.append(
                    {
                        "code": "missing_peptide_atoms",
                        "chain": chain,
                        "message": (
                            f"Protein chain {chain or '<unnamed>'} lacks peptide C/N atoms between "
                            f"{left_key[1]} {left_key[0]} and {right_key[1]} {right_key[0]}; "
                            "repair the backbone or define separate chains with explicit termini"
                        ),
                    }
                )
                continue
            distance = float(
                np.linalg.norm(structure.coordinates[left["C"]] - structure.coordinates[right["N"]])
            )
            if not np.isfinite(distance) or distance > MAX_PEPTIDE_BOND_NM or distance < 0.08:
                issues.append(
                    {
                        "code": "peptide_break",
                        "chain": chain,
                        "left_resid": left_key[0],
                        "right_resid": right_key[0],
                        "distance_nm": float(distance) if np.isfinite(distance) else None,
                        "message": (
                            f"Protein chain {chain or '<unnamed>'} is broken between "
                            f"{left_key[1]} {left_key[0]} and {right_key[1]} {right_key[0]} "
                            f"(C-N {distance:.3f} nm, maximum {MAX_PEPTIDE_BOND_NM:.2f} nm). "
                            "Repair the missing segment or define separate chains "
                            "with explicit termini; "
                            "internal residue templates cannot represent this break."
                        ),
                    }
                )
    return issues

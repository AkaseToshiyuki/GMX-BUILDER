"""Conservative normalization of modified protein residues at input.

Recognized single-residue modifications are reduced to their standard parent
residue before heavy-atom repair.  The original chemistry is recorded so the
Structure Processing step can offer the corresponding force-field patch.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.modules.input.protein_repair import _BACKBONE, _SIDECHAIN_PARENTS
from gmxbuilder.modules.modifications.patches import ALL_PATCHES, patch_capability


_STANDARD_RESIDUES = frozenset(_SIDECHAIN_PARENTS)
_PROTONATION_ALIASES = frozenset(
    {
        "ASH",
        "GLH",
        "CYM",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
        "LYN",
    }
)
_TERMINAL_PRODUCTS = frozenset({"ACE", "NME", "FOR"})


def _recognized_product_patches() -> dict[str, tuple[str, str]]:
    """Return unambiguous product-name → (parent, patch-id) mappings."""
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for patch_id, patch in ALL_PATCHES.items():
        # Step 1 may only auto-normalize chemistry that Step 3 can reconstruct
        # in at least one bundled force field.  Catalogue-only placeholders
        # must remain untouched: several deposited names collide with a
        # different force-field residue (notably CHARMM MLY = dimethyllysine).
        if not patch_capability(patch_id)[0]:
            continue
        if len(patch.target_residues) != 1:
            continue
        parent = patch.target_residues[0].strip().upper()
        product = patch.product_name.strip().upper()
        if parent not in _STANDARD_RESIDUES:
            continue
        # Standard products (ASP/GLU) cannot reveal whether deamidation
        # occurred. MSE is conventionally selenomethionine in deposited PDBs,
        # so it must not be inferred as methionine sulfoxide from its name.
        if product in _STANDARD_RESIDUES or product == "MSE" or product in _TERMINAL_PRODUCTS:
            continue
        candidates[product].append((parent, patch_id))
    return {product: entries[0] for product, entries in candidates.items() if len(entries) == 1}


RECOGNIZED_PRODUCT_PATCHES = _recognized_product_patches()


def _slice_structure(structure: Structure, keep: np.ndarray) -> Structure:
    indices = np.flatnonzero(keep)
    return Structure(
        coordinates=structure.coordinates[indices].copy(),
        box_vectors=structure.box_vectors.copy(),
        atom_names=[structure.atom_names[index] for index in indices],
        resnames=[structure.resnames[index] for index in indices],
        resids=[structure.resids[index] for index in indices],
        chain_ids=[structure.chain_ids[index] for index in indices],
        segids=[structure.segids[index] for index in indices],
        elements=([structure.elements[index] for index in indices] if structure.elements else []),
        occupancies=(
            [structure.occupancies[index] for index in indices] if structure.occupancies else []
        ),
        tempfactors=(
            [structure.tempfactors[index] for index in indices] if structure.tempfactors else []
        ),
    )


def _residue_groups(structure: Structure) -> list[tuple[tuple[str, int, str], list[int]]]:
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index in range(structure.num_atoms):
        key = (
            str(structure.chain_ids[index]).strip(),
            int(structure.resids[index]),
            str(structure.resnames[index]).strip().upper(),
        )
        groups.setdefault(key, []).append(index)
    return list(groups.items())


def normalize_detected_modifications(
    structure: Structure,
    protein_resnames: set[str] | frozenset[str],
) -> tuple[Structure, dict]:
    """Normalize recognizable PTMs and return a task-safe detection report.

    Only residue-name mappings that identify one registered single-residue
    patch are transformed. Unknown or ambiguous non-standard residues are left
    unchanged and reported; guessing their parent chemistry would be unsafe.
    """
    keep = np.ones(structure.num_atoms, dtype=bool)
    records: list[dict] = []
    warnings: list[str] = []

    for (chain, resid, original), indices in _residue_groups(structure):
        mapping = RECOGNIZED_PRODUCT_PATCHES.get(original)
        if mapping is not None:
            parent, patch_id = mapping
            allowed = _BACKBONE | set(_SIDECHAIN_PARENTS[parent]) | {"OXT"}
            observed = {str(structure.atom_names[index]).strip().upper() for index in indices}
            retained = observed & allowed
            blockers = sorted(_BACKBONE - retained)
            disconnected = sorted(
                atom
                for atom, parent_atom in _SIDECHAIN_PARENTS[parent].items()
                if atom in retained and parent_atom not in retained
            )
            if blockers or disconnected:
                details = []
                if blockers:
                    details.append("missing standard backbone " + ",".join(blockers))
                if disconnected:
                    details.append("disconnected parent atoms for " + ",".join(disconnected))
                warning = (
                    f"{chain or '?'}:{resid} {original} matches {patch_id}, but it "
                    "cannot be converted conservatively (" + "; ".join(details) + "); "
                    "it was left unchanged and must be repaired or handled manually."
                )
                warnings.append(warning)
                records.append(
                    {
                        "chain": chain or "?",
                        "resid": resid,
                        "original_resname": original,
                        "standard_resname": parent,
                        "patch_id": patch_id,
                        "status": "recognized_unconvertible",
                        "normalized": False,
                        "removed_atoms": [],
                        "warning": warning,
                    }
                )
                continue
            removed_atoms: list[str] = []
            for index in indices:
                atom_name = str(structure.atom_names[index]).strip().upper()
                if atom_name not in allowed:
                    keep[index] = False
                    removed_atoms.append(atom_name)
                    continue
                structure.resnames[index] = parent
            record = {
                "chain": chain or "?",
                "resid": resid,
                "original_resname": original,
                "standard_resname": parent,
                "patch_id": patch_id,
                "status": "recognized",
                "normalized": True,
                "removed_atoms": sorted(set(removed_atoms)),
            }
            records.append(record)
            continue

        if original == "MSE":
            # Deposited MSE denotes selenomethionine in normal PDB usage.
            # Normalize the selenium atom explicitly, but do not claim that
            # the MSO_MET oxidation patch was present.
            for index in indices:
                structure.resnames[index] = "MET"
                if str(structure.atom_names[index]).strip().upper() == "SE":
                    structure.atom_names[index] = "SD"
                    if structure.elements:
                        structure.elements[index] = "S"
            warning = (
                f"{chain or '?'}:{resid} MSE was normalized to MET as "
                "selenomethionine; no oxidation patch was inferred from the ambiguous MSE label."
            )
            warnings.append(warning)
            records.append(
                {
                    "chain": chain or "?",
                    "resid": resid,
                    "original_resname": original,
                    "standard_resname": "MET",
                    "patch_id": None,
                    "status": "normalized_only",
                    "normalized": True,
                    "removed_atoms": [],
                    "warning": warning,
                }
            )
            continue

        if (
            original in protein_resnames
            and original not in _STANDARD_RESIDUES
            and original not in _PROTONATION_ALIASES
            and original not in _TERMINAL_PRODUCTS
        ):
            warning = (
                f"{chain or '?'}:{resid} {original} is a non-standard protein residue, "
                "but no unambiguous reversible modification mapping is registered; "
                "it was left unchanged for user review."
            )
            warnings.append(warning)
            records.append(
                {
                    "chain": chain or "?",
                    "resid": resid,
                    "original_resname": original,
                    "standard_resname": None,
                    "patch_id": None,
                    "status": "unrecognized",
                    "normalized": False,
                    "removed_atoms": [],
                    "warning": warning,
                }
            )

    if not keep.all():
        structure = _slice_structure(structure, keep)

    residue_index: dict[tuple[str, int], int] = {}
    for chain_resid_name, _indices in _residue_groups(structure):
        chain, resid, resname = chain_resid_name
        if resname not in protein_resnames and resname not in _STANDARD_RESIDUES:
            continue
        residue_index.setdefault((chain or "?", resid), len(residue_index))
    for record in records:
        record["residue_index"] = residue_index.get((record["chain"], int(record["resid"])))

    recognized = sum(record["status"] == "recognized" for record in records)
    return structure, {
        "detected": len(records),
        "recognized": recognized,
        "records": records,
        "warnings": warnings,
    }

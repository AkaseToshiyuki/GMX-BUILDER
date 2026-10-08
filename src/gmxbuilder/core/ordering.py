"""Keep constrained atom groups contiguous without changing molecular placement.

Apply a final atom permutation so each hydrogen stays next to its bonded heavy
atom for GROMACS update groups. Preserve molecule and leaflet order: rebuilding
in a different order changes seeded random draws, and regrouping lipid species
would invalidate positional upper/lower-leaflet membership.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.enums import ComponentKind

# Compatibility alias; Structure owns the field classification.
from gmxbuilder.core.structure import PER_ATOM_FIELDS
from gmxbuilder.core.system import System

_PER_ATOM_FIELDS = PER_ATOM_FIELDS


class AtomOrderingError(RuntimeError):
    """The system is not shaped the way this reordering requires."""


def _molecule_runs(structure, span: range) -> list[list[int]]:
    """Split a span into molecules: maximal runs of one (resname, resid, chain)."""
    runs: list[list[int]] = []
    previous = None
    for index in span:
        key = (
            str(structure.resnames[index]),
            int(structure.resids[index]),
            str(structure.chain_ids[index]) if index < len(structure.chain_ids) else "",
        )
        if key != previous:
            runs.append([])
            previous = key
        runs[-1].append(index)
    return runs


def apply_permutation(system: System, permutation: np.ndarray) -> None:
    """Renumber atoms in place. `permutation[new] = old`."""
    structure = system.structure
    n_atoms = structure.num_atoms
    permutation = np.asarray(permutation, dtype=int)
    if permutation.shape != (n_atoms,):
        raise AtomOrderingError("Permutation length does not match the atom count")
    if not np.array_equal(np.sort(permutation), np.arange(n_atoms)):
        raise AtomOrderingError("Permutation is not a bijection over the atoms")

    try:
        structure.validate_atom_fields()
    except ValueError as exc:
        raise AtomOrderingError(str(exc)) from exc
    for component in system.components:
        indices = np.asarray(component.atom_indices)
        if np.any(indices < 0) or np.any(indices >= n_atoms):
            raise AtomOrderingError("Component indices lie outside the structure")
    structure.select_atoms(permutation)

    # inverse[old] = new
    inverse = np.empty(n_atoms, dtype=int)
    inverse[permutation] = np.arange(n_atoms)
    for component in system.components:
        component.atom_indices = np.sort(inverse[np.asarray(component.atom_indices, dtype=int)])

    # Bonds, exclusions and per-atom types all address atoms by index. Leaving
    # them behind produces a topology that is silently wrong rather than one
    # that fails, which a disulfide test caught the first time this ran.
    topology = getattr(system, "topology", None)
    if topology is not None and hasattr(topology, "remap"):
        topology.remap([int(value) for value in inverse])


# A hydrogen sits about 0.10 nm from the atom it is bonded to; the next heavy
# atom is at least 0.2 nm away. Anything beyond this is not a bond, and the
# hydrogen is left where it is rather than attached to a guess.
_MAX_BOND_NM = 0.16


def _is_hydrogen(structure, index: int) -> bool:
    element = str(structure.elements[index]).strip() if index < len(structure.elements) else ""
    if element:
        return element.upper() == "H"
    return str(structure.atom_names[index]).strip().upper().startswith("H")


def order_hydrogens_after_parents(system: System) -> dict:
    """Put protein hydrogens immediately after their bonded heavy atoms.

    GROMACS can only use update groups when every set of atoms coupled by
    constraints is a contiguous range of indices. A residue written as all its
    heavy atoms followed by all its hydrogens breaks that for every heavy atom
    but the last, which is exactly the message a membrane protein build
    produces: "atoms that are (in)directly constrained together are
    interdispersed with other atoms". Losing update groups costs GPU
    performance for the whole simulation.

    Lipids and water already come out in this order. Protein residues do not,
    because hydrogens are added after the heavy-atom template is placed.

    Only protein components may be permuted. External ligand and native
    nucleic-acid ITPs refer to a fixed atom order and are not remapped by
    ``apply_permutation``; moving their atoms would assign parameters to the
    wrong coordinates when those ITPs are copied into the export.

    Parentage is decided by distance rather than by atom name: names are a
    force-field convention and differ between them, whereas a hydrogen is
    always nearest to the atom it is bonded to by a factor of two.
    """
    structure = system.structure
    n_atoms = structure.num_atoms
    if n_atoms == 0:
        return {"reordered": False, "reason": "empty system"}

    permutation: list[int] = []
    moved_residues = 0
    unassigned = 0
    worst_bond = 0.0
    protein_indices = {
        int(index)
        for component in system.component_by_kind(ComponentKind.PROTEIN)
        for index in component.atom_indices
    }

    for run in _molecule_runs(structure, range(n_atoms)):
        if not all(index in protein_indices for index in run):
            permutation.extend(run)
            continue
        hydrogens = [i for i in run if _is_hydrogen(structure, i)]
        heavies = [i for i in run if not _is_hydrogen(structure, i)]
        # One heavy atom, or none, is already a contiguous group.
        if len(heavies) < 2 or not hydrogens:
            permutation.extend(run)
            continue

        heavy_xyz = structure.coordinates[heavies]
        attached: dict[int, list[int]] = {index: [] for index in heavies}
        orphans: list[int] = []
        for hydrogen in hydrogens:
            distances = np.linalg.norm(heavy_xyz - structure.coordinates[hydrogen], axis=1)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            if distance > _MAX_BOND_NM:
                orphans.append(hydrogen)
                unassigned += 1
                continue
            worst_bond = max(worst_bond, distance)
            attached[heavies[nearest]].append(hydrogen)

        ordered = [i for heavy in heavies for i in (heavy, *attached[heavy])] + orphans
        if ordered != run:
            moved_residues += 1
        permutation.extend(ordered)

    permutation_array = np.asarray(permutation, dtype=int)
    if np.array_equal(permutation_array, np.arange(n_atoms)):
        # Still report an unattached hydrogen: it is a fact about the
        # structure, not about whether anything needed moving.
        return {
            "reordered": False,
            "reason": "hydrogens already follow their parents",
            "unassigned_hydrogens": unassigned,
            "longest_bond_nm": round(worst_bond, 4),
        }

    apply_permutation(system, permutation_array)
    return {
        "reordered": True,
        "residues_reordered": moved_residues,
        "unassigned_hydrogens": unassigned,
        "longest_bond_nm": round(worst_bond, 4),
    }

"""Pipeline module: applies protonation, modifications, and termini capping.

Must run AFTER PDB input and BEFORE orientation.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.chemistry import is_hydrogen

from gmxbuilder.core.topology import Bond, Topology
from gmxbuilder.core.system import System
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.modules import register_module
from gmxbuilder.modules.modifications.geometry import (
    GeometryQuality,
    ModificationGeometryError,
    build_modified_heavy_atom_geometry,
)

from pathlib import Path


_SUPPORTED_ATOM_TRANSFORMS: dict[str, tuple[str, str, str]] = {
    "DEA_ASN": ("ND2", "OD2", "O"),
    "DEG_GLN": ("NE2", "OE2", "O"),
}

# Force-field product templates occasionally rename a chemically retained
# parent atom.  Preserve its uploaded coordinate before deciding which atoms
# truly need to be rebuilt.  Y1P calls the tyrosine phenolic oxygen OG rather
# than OH; deleting and reconstructing that atom would unnecessarily disturb
# the aromatic side-chain frame.
_PRODUCT_HEAVY_ATOM_ALIASES: dict[str, dict[str, str]] = {
    "Y1P": {"OH": "OG"},
}

_BACKBONE_OXYGEN_ALIASES = {"O", "OXT", "O1", "O2", "OT1", "OT2", "OC1", "OC2"}

_HISTIDINE_ALIASES: dict[str, dict[str, str]] = {
    "amber": {"HSD": "HID", "HSE": "HIE", "HSP": "HIP"},
    "charmm": {"HID": "HSD", "HIE": "HSE", "HIP": "HSP"},
    "opls": {
        "HID": "HISD", "HSD": "HISD",
        "HIE": "HISE", "HSE": "HISE",
        "HIP": "HISH", "HSP": "HISH",
    },
}


def _forcefield_family(force_field: str) -> str:
    name = force_field.strip().lower()
    if name.startswith("amber"):
        return "amber"
    if name.startswith("charmm"):
        return "charmm"
    if name.startswith("opls"):
        return "opls"
    return "unknown"


def _normalise_protein_names(system: System, force_field: str) -> list[str]:
    """Translate well-defined PDB aliases to the selected force-field dialect.

    This deliberately does not guess the tautomer of an unqualified ``HIS``.
    The frontend must submit a protonation assignment so that a scientifically
    meaningful choice cannot disappear behind a naming conversion.
    """
    family = _forcefield_family(force_field)
    residue_aliases = _HISTIDINE_ALIASES.get(family, {})
    protein_indices = {
        int(index)
        for component in system.components
        if component.kind == ComponentKind.PROTEIN
        for index in component.atom_indices
    }
    ambiguous_histidines: set[tuple[str, int]] = set()
    renamed_residues = 0
    renamed_atoms = 0

    # Atom aliases must follow the bundled RTP that will actually be used for
    # topology generation.  Force-field-family folklore is not reliable here:
    # the CHARMM36/36m and Amber ports bundled with GROMACS all use ILE:CD,
    # while deposited PDB structures normally call the same atom CD1.
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    ile_template = rtp.get_residue("ILE")
    ile_names = {
        str(atom[0]).strip().upper() for atom in ile_template.get("atoms", [])
    } if ile_template else set()
    if "CD" in ile_names and "CD1" not in ile_names:
        ile_alias = ("CD1", "CD")
    elif "CD1" in ile_names and "CD" not in ile_names:
        ile_alias = ("CD", "CD1")
    else:
        ile_alias = None

    for index in sorted(protein_indices):
        residue = str(system.structure.resnames[index]).strip().upper()
        if residue == "HIS" and family in {"amber", "charmm", "opls"}:
            ambiguous_histidines.add(
                (str(system.structure.chain_ids[index]), int(system.structure.resids[index]))
            )
        replacement = residue_aliases.get(residue)
        if replacement:
            system.structure.resnames[index] = replacement
            renamed_residues += 1

        if ile_alias is not None and residue == "ILE":
            atom_name = str(system.structure.atom_names[index]).strip().upper()
            if atom_name == ile_alias[0]:
                system.structure.atom_names[index] = ile_alias[1]
                renamed_atoms += 1

    if ambiguous_histidines:
        locations = ", ".join(
            f"{chain or '?'}:{resid}" for chain, resid in sorted(ambiguous_histidines)
        )
        raise ModuleConfigError(
            "Unassigned HIS protonation state for the selected force field at "
            f"{locations}. Choose a histidine tautomer/protonation state in Step 3; "
            "do not silently infer one."
        )

    messages = []
    if renamed_residues:
        messages.append(
            f"Force-field naming: translated {renamed_residues} histidine atom labels"
        )
    if renamed_atoms and ile_alias is not None:
        messages.append(
            f"Force-field naming: translated {renamed_atoms} ILE "
            f"{ile_alias[0]} atoms to {ile_alias[1]} for {force_field}"
        )
    return messages


def _validate_protein_heavy_atoms(system: System, force_field: str) -> None:
    """Reject incomplete protein residues before charges/topology are emitted."""
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    protein_indices = sorted({
        int(index)
        for component in system.components
        if component.kind == ComponentKind.PROTEIN
        for index in component.atom_indices
    })
    residues: dict[tuple[str, int], list[int]] = {}
    for index in protein_indices:
        key = (str(system.structure.chain_ids[index]), int(system.structure.resids[index]))
        residues.setdefault(key, []).append(index)

    incomplete = []
    for (chain, resid), indices in residues.items():
        resname = str(system.structure.resnames[indices[0]]).strip().upper()
        try:
            template = rtp.get_residue(resname)
        except KeyError:
            continue
        if template is None:
            raise ModuleConfigError(
                f"Protein residue {resname} at {chain}:{resid} has no {force_field} "
                "template; remove the non-protein molecule in Step 1 or choose "
                "a force field that explicitly supports this residue"
            )
        expected = {
            str(atom[0]).strip()
            for atom in template.get("atoms", [])
            if not _is_hydrogen_name(str(atom[0]))
        }
        observed = {
            str(system.structure.atom_names[index]).strip()
            for index in indices
            if str(system.structure.elements[index]).strip().upper() != "H"
        }
        missing = sorted(expected - observed)
        if missing:
            incomplete.append(
                f"{chain or '?'}:{resid} {resname} missing {','.join(missing)}"
            )
    if incomplete:
        preview = "; ".join(incomplete[:10])
        suffix = f"; and {len(incomplete) - 10} more" if len(incomplete) > 10 else ""
        raise ModuleConfigError(
            "Incomplete protein heavy atoms are not simulation-ready: "
            f"{preview}{suffix}. Repair the structure in Step 1 or upload a "
            "complete model; topology generation will not assign partial-residue charges."
        )


def _is_hydrogen_name(name: str) -> bool:
    return is_hydrogen(name)


def _remap_system_atoms(system: System, keep: list[int]) -> None:
    """Keep selected atoms and remap every component/topology index."""
    old_to_new = {old: new for new, old in enumerate(keep)}
    structure = system.structure
    structure.coordinates = structure.coordinates[keep]
    for attribute in (
        "atom_names", "resnames", "resids", "chain_ids", "segids",
        "elements", "occupancies", "tempfactors",
    ):
        values = getattr(structure, attribute)
        setattr(structure, attribute, [values[index] for index in keep])

    for component in system.components:
        component.atom_indices = np.asarray(
            sorted(
                old_to_new[int(index)]
                for index in component.atom_indices
                if int(index) in old_to_new
            ),
            dtype=np.int64,
        )

    topology = system.topology
    if topology is None:
        return
    topology.atom_types = [
        atom_type for old, atom_type in enumerate(topology.atom_types) if old in old_to_new
    ]
    indexed_terms = (
        (topology.bonds, ("i", "j")),
        (topology.angles, ("i", "j", "k")),
        (topology.dihedrals, ("i", "j", "k", "l")),
        (topology.impropers, ("i", "j", "k", "l")),
        (topology.pairs, ("i", "j")),
    )
    for terms, fields in indexed_terms:
        retained = []
        for term in terms:
            old_indices = [getattr(term, field) for field in fields]
            if not all(index in old_to_new for index in old_indices):
                continue
            for field, old_index in zip(fields, old_indices):
                setattr(term, field, old_to_new[old_index])
            retained.append(term)
        terms[:] = retained
    topology.exclusions = [
        {old_to_new[index] for index in exclusion if index in old_to_new}
        for exclusion in topology.exclusions
    ]
    for block in topology.molecule_blocks:
        block.atom_indices = [
            old_to_new[index] for index in block.atom_indices if index in old_to_new
        ]


def _group_protein_chains_before_other_molecules(system: System) -> bool:
    """Match coordinate order to the master topology's molecule order.

    HDB appends newly constructed protein hydrogens to the structure.  When a
    retained ligand is already present, that otherwise leaves protein atoms on
    both sides of the ligand even though GROMACS requires each ``[ molecules ]``
    entry to consume one contiguous coordinate block.
    """
    protein_indices = {
        int(index)
        for component in system.components
        if component.kind == ComponentKind.PROTEIN
        for index in component.atom_indices
    }
    by_chain: dict[str, list[int]] = {}
    for index in sorted(protein_indices):
        chain = str(system.structure.chain_ids[index]).strip()
        by_chain.setdefault(chain, []).append(index)
    ordered_protein: list[int] = []
    for chain in sorted(by_chain):
        residue_groups: dict[tuple[int, str], list[int]] = {}
        residue_order: list[tuple[int, str]] = []
        for index in by_chain[chain]:
            key = (
                int(system.structure.resids[index]),
                str(system.structure.resnames[index]).strip().upper(),
            )
            if key not in residue_groups:
                residue_groups[key] = []
                residue_order.append(key)
            residue_groups[key].append(index)
        # Cap atoms are appended during processing.  Put ACE before the first
        # amino acid and NME after the last one so topology residue order and
        # the physical cross-residue bonds agree.
        residue_order.sort(
            key=lambda key: 0 if key[1] == "ACE" else (2 if key[1] == "NME" else 1)
        )
        ordered_protein.extend(
            index for key in residue_order for index in residue_groups[key]
        )
    other = [index for index in range(system.num_atoms) if index not in protein_indices]
    order = ordered_protein + other
    if order == list(range(system.num_atoms)):
        return False
    _remap_system_atoms(system, order)
    return True


def _cap_capability(force_field: str, cap: str) -> tuple[bool, str]:
    """Return whether the bundled RTP can represent a requested terminal cap."""
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    name = cap.strip().upper()
    if name not in {"ACE", "NME", "FOR"}:
        return False, f"Unknown terminal cap {cap!r}"
    template = load_force_field_rtp(force_field).get_residue(name)
    if template is None:
        return False, f"{force_field} has no {name} residue topology"
    if name == "FOR":
        return False, "No bundled protein force field provides a validated FOR cap topology"
    return True, ""


def terminal_capabilities(force_field: str) -> dict[str, dict[str, object]]:
    """Expose force-field-specific terminal choices to the web client."""
    result: dict[str, dict[str, object]] = {}
    for cap, end in (("ACE", "N"), ("FOR", "N"), ("NME", "C")):
        supported, reason = _cap_capability(force_field, cap)
        result[cap] = {"end": end, "supported": supported, "reason": reason}
    return result


def _append_cap_residue(
    system: System,
    chain: str,
    terminal_key: tuple[str, int],
    cap: str,
    template: dict,
) -> int:
    """Build an explicit ACE or NME residue bonded to one protein terminus."""
    structure = system.structure
    terminal_indices = [
        index for index, (atom_chain, atom_resid) in enumerate(
            zip(structure.chain_ids, structure.resids)
        )
        if str(atom_chain) == terminal_key[0] and int(atom_resid) == terminal_key[1]
    ]
    terminal_atoms = {
        str(structure.atom_names[index]).strip(): index for index in terminal_indices
    }
    parent_component = next(
        component for component in system.components
        if component.kind == ComponentKind.PROTEIN
        and any(index in set(map(int, component.atom_indices)) for index in terminal_indices)
    )
    if not {"N", "CA", "C"}.issubset(terminal_atoms):
        raise ModuleConfigError(
            f"Cannot build {cap} at {chain}:{terminal_key[1]} without N, CA, and C"
        )

    def unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise ModuleConfigError(f"Degenerate terminal geometry at {chain}:{terminal_key[1]}")
        return vector / norm

    def perpendicular(vector: np.ndarray) -> np.ndarray:
        trial = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(vector, trial))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        return unit(np.cross(vector, trial))

    heavy_names = [
        str(atom[0]).strip() for atom in template["atoms"]
        if not _is_hydrogen_name(str(atom[0]))
    ]
    coordinates: dict[str, np.ndarray] = {}
    external_neighbours: dict[str, list[np.ndarray]] = {}
    if cap == "ACE":
        n = structure.coordinates[terminal_atoms["N"]]
        ca = structure.coordinates[terminal_atoms["CA"]]
        direction = unit(n - ca)
        side = perpendicular(direction)
        coordinates["C"] = n + 0.133 * direction
        coordinates["O"] = coordinates["C"] + 0.123 * unit(-direction + np.sqrt(3.0) * side)
        methyl_name = next(name for name in heavy_names if name not in {"C", "O"})
        coordinates[methyl_name] = (
            coordinates["C"] + 0.152 * unit(-direction - np.sqrt(3.0) * side)
        )
        external_neighbours["C"] = [n]
        new_resid = min(
            int(resid) for atom_chain, resid in zip(structure.chain_ids, structure.resids)
            if str(atom_chain) == chain
        ) - 1
    else:
        c = structure.coordinates[terminal_atoms["C"]]
        ca = structure.coordinates[terminal_atoms["CA"]]
        direction = unit(c - ca)
        side = perpendicular(direction)
        coordinates["N"] = c + 0.133 * direction
        methyl_name = next(name for name in heavy_names if name != "N")
        coordinates[methyl_name] = (
            coordinates["N"] + 0.145 * unit(direction + np.sqrt(3.0) * side)
        )
        external_neighbours["N"] = [c]
        new_resid = max(
            int(resid) for atom_chain, resid in zip(structure.chain_ids, structure.resids)
            if str(atom_chain) == chain
        ) + 1

    prototype = terminal_indices[0]
    name_to_index: dict[str, int] = {}
    element_by_name = {
        str(atom[0]).strip(): str(atom[0]).strip()[0] for atom in template["atoms"]
    }
    for name in heavy_names:
        index = _append_atom(
            system, coordinates[name], name, element_by_name[name], prototype, parent_component
        )
        structure.resnames[index] = cap
        structure.resids[index] = new_resid
        structure.chain_ids[index] = chain
        name_to_index[name] = index

    from gmxbuilder.modules.forcefield.hdb import _compute_h_positions

    hydrogen_names = [
        str(atom[0]).strip() for atom in template["atoms"]
        if _is_hydrogen_name(str(atom[0]))
    ]
    by_control: dict[str, list[str]] = {}
    for hydrogen in hydrogen_names:
        control = next((
            atom2 if atom1 == hydrogen else atom1
            for atom1, atom2 in template.get("bonds", [])
            if (atom1 == hydrogen and atom2 in name_to_index)
            or (atom2 == hydrogen and atom1 in name_to_index)
        ), None)
        if control is None:
            raise ModuleConfigError(f"Cannot identify {cap}:{hydrogen} parent atom")
        by_control.setdefault(control, []).append(hydrogen)

    for control, hydrogens in by_control.items():
        neighbours = list(external_neighbours.get(control, []))
        for atom1, atom2 in template.get("bonds", []):
            neighbour = atom2 if atom1 == control else (atom1 if atom2 == control else "")
            if neighbour in name_to_index and not _is_hydrogen_name(neighbour):
                neighbours.append(structure.coordinates[name_to_index[neighbour]])
        positions = _compute_h_positions(
            structure.coordinates[name_to_index[control]], neighbours, len(hydrogens), atom_name=control
        )
        for hydrogen, position in zip(hydrogens, positions):
            index = _append_atom(system, position, hydrogen, "H", prototype, parent_component)
            structure.resnames[index] = cap
            structure.resids[index] = new_resid
            structure.chain_ids[index] = chain
            name_to_index[hydrogen] = index
    return len(template["atoms"])


def _append_atom(
    system: System,
    coordinate: np.ndarray,
    name: str,
    element: str,
    prototype_index: int,
    component,
) -> int:
    structure = system.structure
    new_index = structure.num_atoms
    structure.coordinates = np.vstack([structure.coordinates, coordinate])
    structure.atom_names.append(name)
    structure.resnames.append(structure.resnames[prototype_index])
    structure.resids.append(structure.resids[prototype_index])
    structure.chain_ids.append(structure.chain_ids[prototype_index])
    structure.segids.append(structure.segids[prototype_index])
    structure.elements.append(element)
    structure.occupancies.append(1.0)
    structure.tempfactors.append(0.0)
    component.atom_indices = np.concatenate([
        component.atom_indices, np.asarray([new_index], dtype=np.int64)
    ])
    return new_index


def _element_from_atom_name(name: str) -> str:
    letters = "".join(character for character in name if character.isalpha()).upper()
    if not letters:
        raise ModuleConfigError(f"Cannot infer element for atom {name!r}")
    return letters[0]


def _synchronise_modified_residue(
    system: System,
    key: tuple[str, int],
    product_name: str,
    template: dict,
    stereo_constraints=(),
) -> tuple[int, GeometryQuality]:
    """Convert one residue to an atom-complete native force-field template.

    Existing common heavy atoms retain their uploaded coordinates.  Obsolete
    atoms and hydrogens are removed, then all missing heavy atoms and template
    hydrogens are grown from the RTP bond graph.  Building the hydrogens here
    is required for force-field supplements whose residue names are not in the
    base amino-acid HDB (for example AmberTools phosaa14SB).
    """
    chain, resid = key
    structure = system.structure
    indices = [
        index for index, (atom_chain, atom_resid) in enumerate(
            zip(structure.chain_ids, structure.resids)
        )
        if str(atom_chain) == chain and int(atom_resid) == resid
    ]
    if not indices:
        raise ModuleConfigError(f"Modification target {chain}:{resid} has no atoms")
    component = next((
        item for item in system.components
        if item.kind == ComponentKind.PROTEIN
        and any(index in set(map(int, item.atom_indices)) for index in indices)
    ), None)
    if component is None:
        raise ModuleConfigError(f"Modification target {chain}:{resid} is not protein")

    heavy_order = [
        str(atom[0]).strip() for atom in template["atoms"]
        if not _is_hydrogen_name(str(atom[0]))
    ]
    heavy_set = set(heavy_order)
    atom_aliases = _PRODUCT_HEAVY_ATOM_ALIASES.get(product_name, {})
    if product_name == "HYP" and "CD2" in heavy_set and "CD" not in heavy_set:
        atom_aliases = {**atom_aliases, "CD": "CD2"}
    for index in indices:
        atom_name = str(structure.atom_names[index]).strip()
        replacement = atom_aliases.get(atom_name)
        if replacement:
            structure.atom_names[index] = replacement
    remove = [
        index for index in indices
        if str(structure.atom_names[index]).strip() not in heavy_set
        or str(structure.elements[index]).strip().upper() == "H"
        or _is_hydrogen_name(str(structure.atom_names[index]))
    ]
    if remove:
        remove_set = set(remove)
        _remap_system_atoms(
            system, [index for index in range(structure.num_atoms) if index not in remove_set]
        )
        structure = system.structure

    indices = [
        index for index, (atom_chain, atom_resid) in enumerate(
            zip(structure.chain_ids, structure.resids)
        )
        if str(atom_chain) == chain and int(atom_resid) == resid
    ]
    for index in indices:
        structure.resnames[index] = product_name
    name_to_index = {
        str(structure.atom_names[index]).strip(): index for index in indices
    }
    prototype = indices[0]

    retained_coordinates = {
        name: structure.coordinates[index].copy()
        for name, index in name_to_index.items()
        if name in heavy_set
    }
    residue_index_set = set(indices)
    environment_indices = [
        index for index in range(structure.num_atoms)
        if index not in residue_index_set
        and str(structure.elements[index]).strip().upper() != "H"
        and not _is_hydrogen_name(str(structure.atom_names[index]))
    ]
    environment = np.empty((0, 3), dtype=float)
    if environment_indices and retained_coordinates:
        candidates = structure.coordinates[environment_indices]
        retained_array = np.vstack(list(retained_coordinates.values()))
        nearby = np.min(
            np.linalg.norm(candidates[:, None, :] - retained_array[None, :, :], axis=2),
            axis=1,
        ) <= 0.8
        environment = candidates[nearby]
    try:
        built_coordinates, geometry_quality = build_modified_heavy_atom_geometry(
            force_field=str(system.metadata.get("force_field", "")),
            template=template,
            retained_coordinates=retained_coordinates,
            environment_coordinates=environment,
            stereo_constraints=stereo_constraints,
        )
    except ModificationGeometryError as error:
        raise ModuleConfigError(
            f"Cannot construct force-field-consistent {product_name} geometry at "
            f"{chain}:{resid}: {error}"
        ) from error

    for name in heavy_order:
        if name in name_to_index:
            continue
        new_index = _append_atom(
            system,
            built_coordinates[name],
            name,
            _element_from_atom_name(name),
            prototype,
            component,
        )
        structure.resnames[new_index] = product_name
        name_to_index[name] = new_index

    observed = set(name_to_index)
    if observed != heavy_set:
        raise ModuleConfigError(
            f"Modification {product_name} atom mismatch at {chain}:{resid}: "
            f"expected {sorted(heavy_set)}, got {sorted(observed)}"
        )
    hydrogen_order = [
        str(atom[0]).strip() for atom in template["atoms"]
        if _is_hydrogen_name(str(atom[0]))
    ]
    by_control: dict[str, list[str]] = {}
    for hydrogen in hydrogen_order:
        control = next((
            atom2 if atom1 == hydrogen else atom1
            for atom1, atom2 in template.get("bonds", [])
            if (atom1 == hydrogen and atom2 in name_to_index)
            or (atom2 == hydrogen and atom1 in name_to_index)
        ), None)
        if control is None:
            raise ModuleConfigError(
                f"Cannot identify {product_name}:{hydrogen} parent atom"
            )
        by_control.setdefault(control, []).append(hydrogen)

    from gmxbuilder.modules.forcefield.hdb import _compute_h_positions

    n_hydrogens = 0
    for control, hydrogens in by_control.items():
        neighbours = []
        for atom1, atom2 in template.get("bonds", []):
            neighbour = atom2 if atom1 == control else (
                atom1 if atom2 == control else ""
            )
            if neighbour in name_to_index and not _is_hydrogen_name(neighbour):
                neighbours.append(structure.coordinates[name_to_index[neighbour]])
        positions = _compute_h_positions(
            structure.coordinates[name_to_index[control]], neighbours,
            len(hydrogens), atom_name=control,
        )
        if len(positions) != len(hydrogens):
            raise ModuleConfigError(
                f"Cannot construct {product_name} hydrogen geometry at {chain}:{resid}"
            )
        for hydrogen, position in zip(hydrogens, positions):
            index = _append_atom(
                system, position, hydrogen, "H", prototype, component
            )
            structure.resnames[index] = product_name
            name_to_index[hydrogen] = index
            n_hydrogens += 1

    expected = {str(atom[0]).strip() for atom in template["atoms"]}
    if set(name_to_index) != expected:
        raise ModuleConfigError(
            f"Modification {product_name} final atom mismatch at {chain}:{resid}"
        )
    return len(remove) + n_hydrogens, geometry_quality


def _prepare_terminal_residue(
    system: System,
    key: tuple[str, int],
    template: dict,
    end: str,
) -> int:
    """Synchronise one residue's atoms with a terminal RTP template."""
    chain, resid = key
    structure = system.structure
    indices = [
        index for index, (atom_chain, atom_resid) in enumerate(
            zip(structure.chain_ids, structure.resids)
        )
        if str(atom_chain) == chain and int(atom_resid) == resid
    ]
    if not indices:
        raise ModuleConfigError(f"Terminal residue {chain}:{resid} has no atoms")
    parent_component = next(
        (
            component for component in system.components
            if component.kind == ComponentKind.PROTEIN
            and any(index in set(map(int, component.atom_indices)) for index in indices)
        ),
        None,
    )
    if parent_component is None:
        raise ModuleConfigError(f"Terminal residue {chain}:{resid} has no protein component")

    target_order = [atom[0] for atom in template["atoms"]]
    target_names = set(target_order)
    names = structure.atom_names
    target_oxygens: list[str] = []

    if end == "C":
        for atom1, atom2 in template.get("bonds", []):
            if atom1 == "C" and atom2.startswith("O"):
                target_oxygens.append(atom2)
            elif atom2 == "C" and atom1.startswith("O"):
                target_oxygens.append(atom1)
        target_oxygens = list(dict.fromkeys(target_oxygens))
        existing_oxygens = [
            index for index in indices if names[index].strip() in _BACKBONE_OXYGEN_ALIASES
        ]
        existing_oxygens.sort(key=lambda index: names[index].strip() != "O")
        for index, target_name in zip(existing_oxygens, target_oxygens):
            names[index] = target_name

    extras = []
    for index in indices:
        name = names[index].strip()
        if name in target_names:
            continue
        element = str(structure.elements[index]).strip().upper()
        if element == "H" or _is_hydrogen_name(name) or name in _BACKBONE_OXYGEN_ALIASES:
            extras.append(index)
        else:
            raise ModuleConfigError(
                f"Unexpected heavy atom {name} in terminal residue {chain}:{resid}"
            )
    if extras:
        keep = [index for index in range(structure.num_atoms) if index not in set(extras)]
        _remap_system_atoms(system, keep)
        structure = system.structure

    indices = [
        index for index, (atom_chain, atom_resid) in enumerate(
            zip(structure.chain_ids, structure.resids)
        )
        if str(atom_chain) == chain and int(atom_resid) == resid
    ]
    prototype = indices[0]
    name_to_index = {structure.atom_names[index].strip(): index for index in indices}
    missing = [name for name in target_order if name not in name_to_index]

    # Add terminal/carboxyl oxygens before hydrogens so H geometry can use all
    # available heavy-atom neighbours.
    for name in [item for item in missing if item.startswith("O")]:
        if "C" not in name_to_index or "CA" not in name_to_index:
            raise ModuleConfigError(f"Cannot place terminal oxygen {name} at {chain}:{resid}")
        c_position = structure.coordinates[name_to_index["C"]]
        ca_position = structure.coordinates[name_to_index["CA"]]
        existing_oxygen = next(
            (candidate for candidate in target_oxygens if candidate in name_to_index), None
        )
        if existing_oxygen is None:
            raise ModuleConfigError(f"No reference oxygen at terminal residue {chain}:{resid}")
        reference = structure.coordinates[name_to_index[existing_oxygen]] - c_position
        axis = ca_position - c_position
        axis_norm = np.linalg.norm(axis)
        reference_norm = np.linalg.norm(reference)
        if axis_norm < 1e-8 or reference_norm < 1e-8:
            raise ModuleConfigError(f"Degenerate C-terminal geometry at {chain}:{resid}")
        unit_axis = axis / axis_norm
        reflected = 2.0 * np.dot(reference, unit_axis) * unit_axis - reference
        reflected = reflected / np.linalg.norm(reflected) * reference_norm
        new_index = _append_atom(
            system, c_position + reflected, name, "O", prototype, parent_component
        )
        name_to_index[name] = new_index

    missing_hydrogens = [name for name in target_order if name not in name_to_index and _is_hydrogen_name(name)]
    hydrogen_controls: dict[str, list[str]] = {}
    for hydrogen in missing_hydrogens:
        for atom1, atom2 in template.get("bonds", []):
            if atom1 == hydrogen and not atom2.startswith(("+", "-")):
                hydrogen_controls.setdefault(atom2, []).append(hydrogen)
                break
            if atom2 == hydrogen and not atom1.startswith(("+", "-")):
                hydrogen_controls.setdefault(atom1, []).append(hydrogen)
                break

    from gmxbuilder.modules.forcefield.hdb import _compute_h_positions
    for control, hydrogen_names in hydrogen_controls.items():
        if control not in name_to_index:
            raise ModuleConfigError(
                f"Cannot place terminal hydrogens: missing {control} at {chain}:{resid}"
            )
        neighbour_positions = []
        for atom1, atom2 in template.get("bonds", []):
            neighbour = None
            if atom1 == control:
                neighbour = atom2
            elif atom2 == control:
                neighbour = atom1
            if neighbour in name_to_index and not _is_hydrogen_name(neighbour):
                neighbour_positions.append(structure.coordinates[name_to_index[neighbour]])
        positions = _compute_h_positions(
            structure.coordinates[name_to_index[control]],
            neighbour_positions,
            len(hydrogen_names),
            atom_name=control,
        )
        if len(positions) != len(hydrogen_names):
            raise ModuleConfigError(
                f"Cannot construct terminal hydrogen geometry at {chain}:{resid}"
            )
        for hydrogen, position in zip(hydrogen_names, positions):
            new_index = _append_atom(
                system, position, hydrogen, "H", prototype, parent_component
            )
            name_to_index[hydrogen] = new_index

    still_missing = [name for name in target_order if name not in name_to_index]
    if still_missing:
        raise ModuleConfigError(
            f"Terminal template at {chain}:{resid} is missing atoms: {still_missing}"
        )
    return len(extras) + len(missing)


def _find_hdb(ff_name: str) -> Path | None:
    """Return the path to the HDB file for *ff_name*, or None.

    Tries several naming conventions used by different force-field bundles:
    ``charmm36/merged.hdb``, ``{ff}.ff/aminoacids.hdb``,
    ``{ff}/aminoacids.hdb``.  Falls back to ``charmm36/merged.hdb``.
    """
    base = Path(__file__).resolve().parent.parent.parent / "data" / "forcefields"

    candidates = [
        base / ff_name / "merged.hdb",
        base / (ff_name + ".ff") / "aminoacids.hdb",
        base / ff_name / "aminoacids.hdb",
        # Fallback — CHARMM36 merged file
        base / "charmm36" / "merged.hdb",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@register_module
class StructureProcessor(BaseModule):
    """Apply protonation state changes, PTM patches, and chain capping."""

    name = "structure"
    description = "Apply protonation, post-translational modifications, and termini capping"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {"protonation", "modifications", "termini", "pH", "skip_protonation",
             "prepare_standard_termini", "crosslinks", "seed"},
        )
        try:
            pH = float(config.get("pH", 7.0))
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError("pH must be a number") from exc
        if not np.isfinite(pH) or not 1.0 <= pH <= 13.0:
            raise ModuleConfigError(f"pH must be between 1.0 and 13.0, got {pH}")
        for key, expected in (
            ("protonation", list), ("modifications", list),
            ("crosslinks", list), ("termini", dict),
        ):
            if not isinstance(config.get(key, expected()), expected):
                raise ModuleConfigError(f"{key} must be a {expected.__name__}")
        for entry in config.get("protonation", []):
            if not isinstance(entry, dict):
                raise ModuleConfigError("Each protonation assignment must be an object")
            index = entry.get("index")
            assigned = entry.get("assigned_name")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ModuleConfigError(f"Invalid protonation residue index: {index!r}")
            if not isinstance(assigned, str) or not assigned.strip():
                raise ModuleConfigError("Protonation assigned_name must be non-empty")
        for entry in config.get("modifications", []):
            if not isinstance(entry, dict):
                raise ModuleConfigError("Each modification assignment must be an object")
            index = entry.get("index")
            patch_id = entry.get("patch_id")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ModuleConfigError(f"Invalid modification residue index: {index!r}")
            if not isinstance(patch_id, str) or not patch_id.strip():
                raise ModuleConfigError("Modification patch_id must be non-empty")
            unknown = set(entry) - {"index", "patch_id", "product_name"}
            if unknown:
                raise ModuleConfigError(
                    "Unknown modification option(s): " + ", ".join(sorted(unknown))
                )
            if "product_name" in entry and not isinstance(entry["product_name"], str):
                raise ModuleConfigError("Modification product_name must be a string")
        for entry in config.get("crosslinks", []):
            if not isinstance(entry, dict):
                raise ModuleConfigError("Each crosslink assignment must be an object")
            unknown = set(entry) - {"type", "first_index", "second_index"}
            if unknown:
                raise ModuleConfigError(
                    "Unknown crosslink option(s): " + ", ".join(sorted(unknown))
                )
            if entry.get("type") != "disulfide":
                raise ModuleConfigError(
                    f"Unsupported crosslink type: {entry.get('type')!r}"
                )
            for key in ("first_index", "second_index"):
                value = entry.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ModuleConfigError(f"Invalid crosslink {key}: {value!r}")
        for chain, caps in config.get("termini", {}).items():
            if not isinstance(chain, str) or not chain:
                raise ModuleConfigError("Each termini chain identifier must be non-empty")
            if not isinstance(caps, dict):
                raise ModuleConfigError(f"Termini for chain {chain!r} must be an object")
            unknown = set(caps) - {"nter", "cter"}
            if unknown:
                raise ModuleConfigError(
                    f"Unknown termini option(s) for chain {chain!r}: "
                    + ", ".join(sorted(unknown))
                )
            for end in ("nter", "cter"):
                if end in caps and not isinstance(caps[end], str):
                    raise ModuleConfigError(
                        f"Termini {end} cap for chain {chain!r} must be a string"
                    )
        for flag in ("skip_protonation", "prepare_standard_termini"):
            if flag in config and not isinstance(config[flag], bool):
                raise ModuleConfigError(f"{flag} must be a boolean")
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        protonation = config.get("protonation", [])
        modifications = config.get("modifications", [])
        crosslinks = config.get("crosslinks", [])
        termini = config.get("termini", {})
        pH = config.get("pH", 7.0)
        skip = config.get("skip_protonation", False)
        prepare_standard_termini = config.get("prepare_standard_termini", True)

        log = []
        n_atoms = system.structure.num_atoms

        if n_atoms == 0:
            return ModuleResult(success=True, system=system,
                              log=["No atoms to process"])

        # Work on a copy so any late chemistry/template error leaves the input
        # checkpoint untouched.
        system = system.copy()

        resnames = list(system.structure.resnames)
        resids = system.structure.resids
        chain_ids = system.structure.chain_ids

        # Build ordered per-residue lookup matching frontend _procResidues order.
        # _procResidues indices are per-residue, but resnames is per-atom —
        # we need to map residue index → all atom positions for that residue.
        _residue_order: list[tuple[str, int]] = []  # [(chain, resid), ...]
        _residue_atoms: dict[tuple[str, int], list[int]] = {}  # (chain, resid) → [atom_idx, ...]
        previous_key: tuple[str, int] | None = None
        closed_keys: set[tuple[str, int]] = set()
        residue_names_by_key: dict[tuple[str, int], str] = {}
        for i, (ch, rid) in enumerate(zip(chain_ids, resids)):
            key = (str(ch), int(rid))
            residue_name = str(resnames[i]).strip().upper()
            if key != previous_key:
                if previous_key is not None:
                    closed_keys.add(previous_key)
                if key in closed_keys:
                    raise ModuleConfigError(
                        "Non-contiguous repeated residue identifier "
                        f"{key[0] or '?'}:{key[1]}. Renumber the input structure so "
                        "each residue has one unambiguous contiguous atom block."
                    )
                previous_key = key
            if key not in _residue_atoms:
                _residue_atoms[key] = []
                _residue_order.append(key)
                residue_names_by_key[key] = residue_name
            elif residue_names_by_key[key] != residue_name:
                raise ModuleConfigError(
                    f"Residue identifier {key[0] or '?'}:{key[1]} contains both "
                    f"{residue_names_by_key[key]} and {residue_name}; renumber the input "
                    "before assigning chemistry."
                )
            _residue_atoms[key].append(i)

        # Validate the complete chemistry request before mutating the system.
        # Renaming a residue without changing its atoms produces an invalid RTP
        # match, so unsupported catalogue entries fail closed.
        from gmxbuilder.modules.modifications.patches import (
            disulfide_capability,
            get_patch,
            patch_capability,
        )

        ff_name = str(system.metadata.get("force_field", "charmm36"))
        validated_modifications: list[tuple[tuple[str, int], str, object]] = []
        modified_indices: set[int] = set()
        for mod in modifications:
            idx = mod.get("index")
            patch_id = str(mod.get("patch_id", ""))
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(_residue_order):
                raise ModuleConfigError(f"Invalid modification residue index: {idx!r}")
            if idx in modified_indices:
                raise ModuleConfigError(f"Multiple modifications target residue index {idx}")

            patch = get_patch(patch_id)
            if patch is None:
                raise ModuleConfigError(f"Unknown modification patch: {patch_id!r}")
            supported, reason = patch_capability(patch_id, ff_name)
            if not supported:
                raise ModuleConfigError(f"Patch {patch_id} is unavailable: {reason}")

            key = _residue_order[idx]
            atom_indices = _residue_atoms[key]
            original = str(resnames[atom_indices[0]]).strip().upper()
            if original not in patch.target_residues:
                raise ModuleConfigError(
                    f"Patch {patch_id} requires {patch.target_residues}, got {original} "
                    f"at residue index {idx}"
                )
            requested_product = str(mod.get("product_name", "")).strip().upper()
            if requested_product and requested_product != patch.product_name:
                raise ModuleConfigError(
                    f"Patch {patch_id} produces {patch.product_name}, not {requested_product}"
                )

            if patch_id in _SUPPORTED_ATOM_TRANSFORMS:
                old_atom, new_atom, _ = _SUPPORTED_ATOM_TRANSFORMS[patch_id]
                residue_atom_names = [str(system.structure.atom_names[i]).strip() for i in atom_indices]
                if residue_atom_names.count(old_atom) != 1:
                    raise ModuleConfigError(
                        f"Patch {patch_id} requires exactly one {old_atom} atom; "
                        f"found {residue_atom_names.count(old_atom)}"
                    )
                if new_atom in residue_atom_names:
                    raise ModuleConfigError(f"Patch {patch_id} cannot create duplicate atom {new_atom}")
            modified_indices.add(idx)
            validated_modifications.append((key, patch_id, patch))

        validated_disulfides: list[tuple[tuple[str, int], tuple[str, int], float]] = []
        crosslinked_indices: set[int] = set()
        for entry in crosslinks:
            first_index = entry["first_index"]
            second_index = entry["second_index"]
            if first_index == second_index:
                raise ModuleConfigError("A disulfide requires two distinct cysteine residues")
            if not 0 <= first_index < len(_residue_order) or not 0 <= second_index < len(_residue_order):
                raise ModuleConfigError(
                    f"Disulfide indices {first_index}, {second_index} are outside the "
                    f"{len(_residue_order)}-residue structure"
                )
            for index in (first_index, second_index):
                if index in modified_indices:
                    raise ModuleConfigError(
                        f"Residue index {index} cannot have both a single-residue modification "
                        "and a disulfide crosslink"
                    )
                if index in crosslinked_indices:
                    raise ModuleConfigError(
                        f"Residue index {index} is assigned to more than one disulfide"
                    )
            supported, reason, target_distance = disulfide_capability(ff_name)
            if not supported or target_distance is None:
                raise ModuleConfigError(
                    f"Disulfide crosslinks are unavailable for {ff_name}: {reason}"
                )
            keys = (_residue_order[first_index], _residue_order[second_index])
            sulphurs = []
            for index, key in zip((first_index, second_index), keys):
                atom_indices = _residue_atoms[key]
                residue_name = str(resnames[atom_indices[0]]).strip().upper()
                if residue_name != "CYS":
                    raise ModuleConfigError(
                        f"Disulfide residue index {index} must be CYS, got {residue_name}"
                    )
                matches = [
                    atom_index for atom_index in atom_indices
                    if str(system.structure.atom_names[atom_index]).strip() == "SG"
                ]
                if len(matches) != 1:
                    raise ModuleConfigError(
                        f"Disulfide CYS at {key[0] or '?'}:{key[1]} requires exactly one SG atom"
                    )
                sulphurs.append(matches[0])
            observed_distance = float(np.linalg.norm(
                system.structure.coordinates[sulphurs[0]]
                - system.structure.coordinates[sulphurs[1]]
            ))
            if abs(observed_distance - target_distance) > 0.04:
                raise ModuleConfigError(
                    f"Disulfide SG-SG distance is {observed_distance:.3f} nm, outside the "
                    f"validated starting range around the {target_distance:.3f} nm "
                    "force-field target. Supply a structure with an already formed bridge; "
                    "GMXBUILDER will not drag distant side chains together."
                )
            crosslinked_indices.update((first_index, second_index))
            validated_disulfides.append((keys[0], keys[1], target_distance))

        for entry in protonation:
            idx = entry["index"]
            if idx >= len(_residue_order):
                raise ModuleConfigError(
                    f"Protonation residue index {idx} is outside the "
                    f"{len(_residue_order)}-residue structure"
                )

        # A partial assignment list is unsafe: bare HIS names remain ambiguous
        # and other titratable residues can silently keep states calculated for
        # a previous pH. Require one assignment for every canonical titratable
        # protein residue when protonation is enabled.
        if not skip:
            from gmxbuilder.modules.modifications.protonation import get_titratable_residues

            assigned_indices = [entry["index"] for entry in protonation]
            seen_indices: set[int] = set()
            duplicate_indices: set[int] = set()
            for idx in assigned_indices:
                if idx in seen_indices:
                    duplicate_indices.add(idx)
                seen_indices.add(idx)
            duplicates = sorted(duplicate_indices)
            if duplicates:
                raise ModuleConfigError(
                    "Duplicate protonation assignments for residue index(es): "
                    + ", ".join(str(idx) for idx in duplicates)
                )

            protein_atoms = {
                int(index)
                for component in system.components
                if component.kind == ComponentKind.PROTEIN
                for index in component.atom_indices
            }
            titratable_states = get_titratable_residues()
            titratable_names = set(titratable_states)
            expected: list[tuple[int, tuple[str, int], str]] = []
            for idx, key in enumerate(_residue_order):
                atom_indices = _residue_atoms[key]
                if not any(atom_index in protein_atoms for atom_index in atom_indices):
                    continue
                resname = str(resnames[atom_indices[0]]).strip().upper()
                if resname in titratable_names:
                    expected.append((idx, key, resname))

            assigned_set = set(assigned_indices)
            expected_by_index = {idx: resname for idx, _, resname in expected}
            for entry in protonation:
                idx = entry["index"]
                original = expected_by_index.get(idx)
                if original is None:
                    raise ModuleConfigError(
                        f"Protonation assignment targets non-titratable protein residue index {idx}"
                    )
                allowed_names = {
                    state.residue_name for state in titratable_states[original]
                }
                assigned_name = entry["assigned_name"].strip().upper()
                if assigned_name not in allowed_names:
                    raise ModuleConfigError(
                        f"Invalid protonation state {assigned_name!r} for {original} "
                        f"at residue index {idx}; expected one of {sorted(allowed_names)}"
                    )
            missing = [item for item in expected if item[0] not in assigned_set]
            if missing:
                locations = ", ".join(
                    f"{chain or '?'}:{resid} {resname}"
                    for _, (chain, resid), resname in missing[:12]
                )
                if len(missing) > 12:
                    locations += f", ... ({len(missing)} total)"
                raise ModuleConfigError(
                    "Incomplete protonation assignments at "
                    f"{locations}. Run Compute again in Step 3 before Check Structure."
                )

        ff_name = str(system.metadata.get("force_field", "charmm36"))
        requested_caps = [
            f"{chain}:{end}={cap}"
            for chain, caps in termini.items()
            for end, cap in caps.items()
            if cap
        ]
        for chain, caps in termini.items():
            if not isinstance(caps, dict):
                raise ModuleConfigError(f"Termini for chain {chain!r} must be an object")
            for end, expected in (("nter", {"", "ACE", "FOR"}), ("cter", {"", "NME"})):
                cap = str(caps.get(end, "")).strip().upper()
                if cap not in expected:
                    raise ModuleConfigError(f"Invalid {end} cap {cap!r} for chain {chain!r}")
                if cap:
                    supported, reason = _cap_capability(ff_name, cap)
                    if not supported:
                        raise ModuleConfigError(
                            f"{cap} cap is unavailable for {ff_name}: {reason}"
                        )

        # ---- 1. Protonation renaming ----
        renamed_residues: set[tuple[str, int]] = set()
        if not skip and protonation:
            renamed_atoms = 0
            for entry in protonation:
                idx = entry.get("index")
                new_name = entry.get("assigned_name", "")
                key = _residue_order[idx]
                for atom_idx in _residue_atoms.get(key, []):
                    if resnames[atom_idx] != new_name:
                        resnames[atom_idx] = new_name
                        renamed_atoms += 1
                        renamed_residues.add(key)
            if renamed_atoms:
                log.append(
                    f"Protonation: renamed {len(renamed_residues)} residues "
                    f"({renamed_atoms} atom labels) at pH {pH}"
                )
        elif skip:
            log.append("Protonation skipped by user request")

        system.structure.resnames = list(resnames)

        # ---- 2. Modifications (patch application) ----
        geometry_metadata: list[dict] = []
        if validated_modifications:
            applied = []
            geometry_reports: list[tuple[str, tuple[str, int], GeometryQuality]] = []
            for key, patch_id, patch in validated_modifications:
                current_indices = [
                    index for index, (chain, resid) in enumerate(
                        zip(system.structure.chain_ids, system.structure.resids)
                    )
                    if (str(chain), int(resid)) == key
                ]
                old = str(system.structure.resnames[current_indices[0]])
                if patch_id in _SUPPORTED_ATOM_TRANSFORMS:
                    old_atom, new_atom, new_element = _SUPPORTED_ATOM_TRANSFORMS[patch_id]
                    for atom_idx in current_indices:
                        system.structure.resnames[atom_idx] = patch.product_name
                        if str(system.structure.atom_names[atom_idx]).strip() == old_atom:
                            system.structure.atom_names[atom_idx] = new_atom
                            system.structure.elements[atom_idx] = new_element
                else:
                    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
                    template = load_force_field_rtp(ff_name).get_residue(patch.product_name)
                    if template is None:
                        raise ModuleConfigError(
                            f"{ff_name} has no {patch.product_name} residue topology"
                        )
                    _changes, quality = _synchronise_modified_residue(
                        system, key, patch.product_name, template,
                        patch.stereo_constraints,
                    )
                    geometry_reports.append((patch_id, key, quality))
                applied.append(f"{old}→{patch.product_name}")
            if applied:
                log.append(f"Modifications: {len(applied)} applied ({', '.join(applied[:5])}{'...' if len(applied) > 5 else ''})")
            for patch_id, (chain, resid), quality in geometry_reports:
                geometry_metadata.append({
                    "patch_id": patch_id,
                    "chain": chain or "?",
                    "resid": resid,
                    "added_atoms": list(quality.added_atoms),
                    "max_bond_error_nm": quality.max_bond_error_nm,
                    "max_angle_error_deg": quality.max_angle_error_deg,
                    "min_nonbonded_distance_nm": quality.min_nonbonded_distance_nm,
                    "stereo_centres": list(quality.stereo_centres),
                    "status": "passed",
                })
                clash = (
                    f", minimum non-bonded distance {quality.min_nonbonded_distance_nm:.3f} nm"
                    if quality.min_nonbonded_distance_nm is not None else ""
                )
                log.append(
                    f"Modification geometry {patch_id} at {chain or '?'}:{resid}: "
                    f"force-field bond error <= {quality.max_bond_error_nm:.4f} nm, "
                    f"angle error <= {quality.max_angle_error_deg:.2f} degrees{clash}"
                )

        disulfide_metadata: list[dict] = []
        if validated_disulfides:
            from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

            # Uploaded coordinate-only systems legitimately have no topology
            # yet.  Preserve the explicit covalent crosslink now so the later
            # topology writer can merge linked chains into one molecule and
            # emit the SG--SG bond instead of silently losing it.
            if system.topology is None:
                system.topology = Topology(force_field=ff_name)
            cyx_template = load_force_field_rtp(ff_name).get_residue("CYX")
            if cyx_template is None:
                raise ModuleConfigError(f"{ff_name} has no CYX residue topology")
            for first_key, second_key, target_distance in validated_disulfides:
                _synchronise_modified_residue(
                    system, first_key, "CYX", cyx_template
                )
                _synchronise_modified_residue(
                    system, second_key, "CYX", cyx_template
                )
                sulphur_indices = []
                for chain, resid in (first_key, second_key):
                    matches = [
                        index for index, (atom_chain, atom_resid, atom_name) in enumerate(zip(
                            system.structure.chain_ids,
                            system.structure.resids,
                            system.structure.atom_names,
                        ))
                        if str(atom_chain) == chain and int(atom_resid) == resid
                        and str(atom_name).strip() == "SG"
                    ]
                    if len(matches) != 1:
                        raise ModuleConfigError(
                            f"Converted CYX at {chain or '?'}:{resid} has invalid SG atoms"
                        )
                    sulphur_indices.append(matches[0])
                pair = tuple(sorted(sulphur_indices))
                if not any(tuple(sorted((bond.i, bond.j))) == pair for bond in system.topology.bonds):
                    system.topology.bonds.append(Bond(pair[0], pair[1]))
                observed = float(np.linalg.norm(
                    system.structure.coordinates[pair[0]]
                    - system.structure.coordinates[pair[1]]
                ))
                record = {
                    "type": "disulfide",
                    "first": {"chain": first_key[0], "resid": first_key[1]},
                    "second": {"chain": second_key[0], "resid": second_key[1]},
                    "target_distance_nm": target_distance,
                    "observed_distance_nm": observed,
                    "status": "passed",
                }
                disulfide_metadata.append(record)
                log.append(
                    f"Disulfide: {first_key[0] or '?'}:{first_key[1]} SG — "
                    f"{second_key[0] or '?'}:{second_key[1]} SG at {observed:.3f} nm"
                )

        # ---- 3. Termini capping ----
        # Empty/default termini entries intentionally retain the standard
        # charged termini.  Non-empty caps were rejected during preflight.

        # Convert equivalent residue/atom aliases to the naming dialect used
        # by the selected force field before HDB and RTP template lookup.
        log.extend(_normalise_protein_names(system, ff_name))
        resnames = list(system.structure.resnames)
        if "requested_force_field" in system.metadata:
            _validate_protein_heavy_atoms(system, ff_name)

        # ---- 4. Add missing hydrogens using HDB rules ----
        from gmxbuilder.modules.forcefield.hdb import HDBHydrogenAdder
        hdb_path = _find_hdb(ff_name)
        if hdb_path is not None:
            adder = HDBHydrogenAdder(hdb_path)
            new_names, new_coords, new_resnames, new_resids, new_chains = adder.add_hydrogens(
                system.structure.atom_names,
                system.structure.coordinates,
                resnames,
                system.structure.resids,
                system.structure.chain_ids,
            )
            n_old = len(resnames)
            n_added = len(new_names) - n_old
            if n_added > 0:
                system.structure.atom_names = new_names
                system.structure.coordinates = new_coords
                system.structure.resnames = new_resnames
                system.structure.resids = new_resids
                system.structure.chain_ids = new_chains
                # Pad other attributes
                system.structure.elements = list(system.structure.elements) + ["H"] * n_added
                system.structure.occupancies = list(system.structure.occupancies) + [1.0] * n_added
                system.structure.tempfactors = list(system.structure.tempfactors) + [0.0] * n_added
                system.structure.segids = list(system.structure.segids) + [""] * n_added
                # New H atoms belong to the PROTEIN component containing their
                # parent residue.  Adding every H to every protein component
                # duplicates indices for multi-component structures.
                new_h_indices = set(range(n_old, n_old + n_added))
                assigned_h_indices: set[int] = set()
                protein_components = [
                    comp for comp in system.components if comp.kind == ComponentKind.PROTEIN
                ]
                for comp in protein_components:
                    parent_residues = {
                        (str(system.structure.chain_ids[int(idx)]), int(system.structure.resids[int(idx)]))
                        for idx in comp.atom_indices
                        if int(idx) < n_old
                    }
                    component_h = sorted(
                        idx for idx in new_h_indices
                        if (str(new_chains[idx]), int(new_resids[idx])) in parent_residues
                    )
                    if component_h:
                        comp.atom_indices = np.concatenate([
                            comp.atom_indices,
                            np.array(component_h, dtype=np.int64),
                        ])
                        assigned_h_indices.update(component_h)

                # HDB rules should only add protein hydrogens.  If an unusual
                # input prevents residue matching, retain complete coverage by
                # assigning the remaining H atoms to the sole/first component.
                unassigned_h = sorted(new_h_indices - assigned_h_indices)
                if unassigned_h and protein_components:
                    protein_components[0].atom_indices = np.concatenate([
                        protein_components[0].atom_indices,
                        np.array(unassigned_h, dtype=np.int64),
                    ])
                    log.append(
                        f"HDB: assigned {len(unassigned_h)} unmatched hydrogen atoms "
                        "to the primary protein component"
                    )
                log.append(f"HDB: added {n_added} hydrogen atoms")

        # ---- 5. Prepare standard charged protein termini ----
        # HDB first builds the complete internal-residue hydrogen set.  We then
        # reconcile only the first/last residue of each protein chain with the
        # selected force field's NH3+/COO- terminal template.
        from gmxbuilder.modules.forcefield.rtp_parser import get_terminal_residue

        chain_residues: dict[str, list[tuple[str, int]]] = {}
        seen_residues: set[tuple[str, int]] = set()
        protein_indices = sorted({
            int(index)
            for component in system.components
            if component.kind == ComponentKind.PROTEIN
            for index in component.atom_indices
            if int(index) < system.structure.num_atoms
        })
        for index in protein_indices:
            key = (
                str(system.structure.chain_ids[index]),
                int(system.structure.resids[index]),
            )
            if key not in seen_residues:
                seen_residues.add(key)
                chain_residues.setdefault(key[0], []).append(key)

        prepared_termini = 0
        built_caps: list[str] = []
        terminal_chains = (
            chain_residues.items() if (prepare_standard_termini or requested_caps) else []
        )
        for chain, residues in terminal_chains:
            chain_caps = termini.get(chain, {})
            n_cap = str(chain_caps.get("nter", "")).strip().upper()
            c_cap = str(chain_caps.get("cter", "")).strip().upper()
            if len(residues) == 1 and not (n_cap and c_cap):
                raise ModuleConfigError(
                    f"Single-residue protein chain {chain!r} requires a combined "
                    "terminal template; choose both ACE and NME caps or provide a longer chain"
                )
            first_key, last_key = residues[0], residues[-1]
            first_index = next(
                index for index in range(system.structure.num_atoms)
                if str(system.structure.chain_ids[index]) == first_key[0]
                and int(system.structure.resids[index]) == first_key[1]
            )
            first_resname = str(system.structure.resnames[first_index]).strip().upper()
            if n_cap:
                from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
                base_template = load_force_field_rtp(ff_name).get_residue(first_resname)
                cap_template = load_force_field_rtp(ff_name).get_residue(n_cap)
                if base_template is None or cap_template is None:
                    raise ModuleConfigError(
                        f"Cannot prepare {n_cap} cap for {first_resname} in {ff_name}"
                    )
                _prepare_terminal_residue(system, first_key, base_template, "N")
                _append_cap_residue(system, chain, first_key, n_cap, cap_template)
                built_caps.append(f"{chain}:N={n_cap}")
            else:
                _variant, n_template = get_terminal_residue(ff_name, first_resname, "N")
                _prepare_terminal_residue(system, first_key, n_template, "N")
                prepared_termini += 1

            last_index = next(
                index for index in range(system.structure.num_atoms)
                if str(system.structure.chain_ids[index]) == last_key[0]
                and int(system.structure.resids[index]) == last_key[1]
            )
            last_resname = str(system.structure.resnames[last_index]).strip().upper()
            if c_cap:
                from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
                base_template = load_force_field_rtp(ff_name).get_residue(last_resname)
                cap_template = load_force_field_rtp(ff_name).get_residue(c_cap)
                if base_template is None or cap_template is None:
                    raise ModuleConfigError(
                        f"Cannot prepare {c_cap} cap for {last_resname} in {ff_name}"
                    )
                _prepare_terminal_residue(system, last_key, base_template, "C")
                _append_cap_residue(system, chain, last_key, c_cap, cap_template)
                built_caps.append(f"{chain}:C={c_cap}")
            else:
                _variant, c_template = get_terminal_residue(ff_name, last_resname, "C")
                _prepare_terminal_residue(system, last_key, c_template, "C")
                prepared_termini += 1

        if prepared_termini:
            log.append(
                f"Standard termini: prepared {prepared_termini} NH3+/COO- residue templates"
            )
        if built_caps:
            log.append("Terminal caps: built explicit " + ", ".join(built_caps))

        if _group_protein_chains_before_other_molecules(system):
            log.append(
                "Coordinate order: grouped protein chains before retained molecules "
                "to match the GROMACS topology"
            )

        # Store processing metadata for downstream modules
        system.metadata["protonation_pH"] = pH
        system.metadata["n_residues_renamed"] = len(renamed_residues)
        system.metadata["n_modifications"] = (
            len(validated_modifications) + 2 * len(validated_disulfides)
        )
        system.metadata["modification_geometry"] = geometry_metadata
        system.metadata["crosslinks"] = disulfide_metadata
        system.metadata["standard_termini_prepared"] = prepared_termini
        system.metadata["terminal_caps"] = built_caps

        return ModuleResult(success=True, system=system, log=log)

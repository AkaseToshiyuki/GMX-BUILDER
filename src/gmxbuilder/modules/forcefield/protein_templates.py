"""Resolve the same protein chemistry for assignment and GROMACS export."""

from functools import lru_cache

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import TopologyError
from gmxbuilder.core.topology import AtomType
from gmxbuilder.modules.forcefield.rtp_parser import (
    _force_field_path,
    get_terminal_residue,
    load_force_field_rtp,
)


def protein_template_names(structure, indices, force_field):
    """Recognize prepared terminal atom inventories without changing residue IDs."""
    parser = load_force_field_rtp(force_field)
    observed = {}
    chains = {}
    for i in indices:
        key = (str(structure.chain_ids[i]), str(structure.resnames[i]), int(structure.resids[i]))
        if key not in observed:
            observed[key] = set()
            chains.setdefault(key[0], []).append(key)
        observed[key].add(str(structure.atom_names[i]).strip())
    resolved = {}
    for residues in chains.values():
        if len(residues) <= 1:
            continue
        for key, end in ((residues[0], "N"), (residues[-1], "C")):
            name = key[1]
            if (end == "N" and name == "ACE") or (end == "C" and name == "NME"):
                continue
            base = parser.get_residue(name)
            if base is None:
                raise TopologyError(f"No {force_field} protein residue template for {name}")
            base_atoms = {row[0] for row in base["atoms"]}
            # An untouched internal template is permitted for explicit uncapped probes.
            if observed[key] == base_atoms:
                continue
            variant_name, variant = get_terminal_residue(force_field, name, end)
            variant_atoms = {row[0] for row in variant["atoms"]}
            added, removed = variant_atoms - base_atoms, base_atoms - variant_atoms
            if added and added.issubset(observed[key]) and not removed.intersection(observed[key]):
                resolved[key] = variant_name
    return resolved


def _parameter_lines(path, defines, stack=()):
    """Follow bundled includes with the default (ordinary hydrogen) macro state."""
    if path in stack:
        raise TopologyError(f"Recursive force-field include: {path}")
    active = [True]
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if not line.startswith("#"):
            if active[-1]:
                yield line
            continue
        parts = line[1:].split(maxsplit=1)
        directive, argument = parts[0], parts[1] if len(parts) > 1 else ""
        if directive in {"ifdef", "ifndef"}:
            enabled = argument in defines
            active.append(active[-1] and (enabled if directive == "ifdef" else not enabled))
        elif directive == "else":
            active[-1] = active[-2] and not active[-1]
        elif directive == "endif":
            active.pop()
        elif active[-1]:
            if directive == "define":
                defines.add(argument.split()[0])
            elif directive == "undef":
                defines.discard(argument)
            elif directive == "include":
                include = path.parent / argument.strip('"<>')
                yield from _parameter_lines(include, defines, (*stack, path))
            else:
                raise TopologyError(f"Unsupported force-field directive: {line}")
    if len(active) != 1:
        raise TopologyError(f"Unclosed force-field conditional: {path}")


@lru_cache(maxsize=16)
def _nonbonded_types(force_field):
    """Read the bundled sigma/epsilon atom types; these protein FFs use rules 2/3."""
    path = _force_field_path(force_field) / "forcefield.itp"
    atomtypes = {}
    section = ""
    for line in _parameter_lines(path, set()):
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section != "atomtypes":
            continue
        parts = line.split()
        if len(parts) not in {6, 7, 8} or parts[-3] not in {"A", "S", "V", "D"}:
            raise TopologyError(f"Unsupported atom type definition in {path.name}: {line}")
        # Optional bonded-type and atomic-number columns precede the final five fields.
        atomtypes[parts[0]] = (
            float(parts[-5]),
            float(parts[-2]),
            float(parts[-1]),
            parts[1] if len(parts) == 8 else parts[0],
        )
    return atomtypes


def assign_protein_atoms(system, topology, force_field):
    """Replace generic placeholders with exact residue and terminal atom parameters."""
    indices = sorted(
        {
            int(i)
            for component in system.components
            if component.kind == ComponentKind.PROTEIN
            for i in component.atom_indices
        }
    )
    if not indices:
        return
    names = protein_template_names(system.structure, indices, force_field)
    parser = load_force_field_rtp(force_field)
    nonbonded = _nonbonded_types(force_field)
    for i in indices:
        structure = system.structure
        key = (str(structure.chain_ids[i]), str(structure.resnames[i]), int(structure.resids[i]))
        template = names.get(key, key[1])
        atom = str(structure.atom_names[i]).strip()
        parameter = parser.get_atom_type(template, atom)
        if parameter is None or parameter[0] not in nonbonded:
            raise TopologyError(f"No exact {force_field} protein parameters for {template}:{atom}")
        atom_type, charge = parameter
        mass, sigma, epsilon, atom_class = nonbonded[atom_type]
        topology.atom_types[i] = AtomType(atom_type, mass, charge, sigma, epsilon, atom_class)
    system.metadata["native_rtp_unassigned_atom_count"] = sum(
        record.name == "UNASSIGNED" for record in topology.atom_types
    )


def native_atom_types(system, force_field):
    """Resolve native RTP atoms; explicitly mark separately parameterized atoms.

    Ligand/native-polymer/lipid ITPs are authoritative at export. Missing RTP
    entries must never masquerade as zero-charge carbon parameters in a
    checkpoint. NaN records cannot be consumed as partial charges or LJ values.
    Protein termini are filled by assign_protein_atoms immediately afterwards.
    """
    parser = load_force_field_rtp(force_field)
    nonbonded = _nonbonded_types(force_field)
    records = []
    unresolved = 0
    for residue, atom in zip(system.structure.resnames, system.structure.atom_names, strict=True):
        parameter = parser.get_atom_type(str(residue).strip(), str(atom).strip())
        if parameter is not None:
            name, charge = parameter
            if name not in nonbonded:
                raise TopologyError(f"No {force_field} nonbonded parameters for {name}")
            mass, sigma, epsilon, atom_class = nonbonded[name]
            records.append(AtomType(name, mass, charge, sigma, epsilon, atom_class))
        else:
            records.append(AtomType("UNASSIGNED", *(float("nan"),) * 4))
            unresolved += 1
    system.metadata["native_rtp_unassigned_atom_count"] = unresolved
    return records

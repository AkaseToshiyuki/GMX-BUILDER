"""Native GROMACS topology preparation for canonical DNA/RNA polymers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.io.pdb import PDBWriter
from gmxbuilder.runtime.hardware import find_gromacs_executable


_SUPPORTED_FORCE_FIELDS = {"charmm36m"}


def _force_field_directory(force_field: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "data" / "forcefields"
    for candidate in (root / force_field, root / f"{force_field}.ff"):
        if candidate.is_dir():
            return candidate
    raise ModuleConfigError(f"Bundled force-field directory is missing for {force_field}")


def _subset(structure: Structure, indices: list[int]) -> Structure:
    return Structure(
        coordinates=structure.coordinates[indices].copy(),
        box_vectors=structure.box_vectors.copy(),
        atom_names=[structure.atom_names[index] for index in indices],
        resnames=[structure.resnames[index] for index in indices],
        resids=[structure.resids[index] for index in indices],
        chain_ids=[structure.chain_ids[index] for index in indices],
        segids=[structure.segids[index] for index in indices],
        elements=[structure.elements[index] for index in indices],
        occupancies=[structure.occupancies[index] for index in indices],
        tempfactors=[structure.tempfactors[index] for index in indices],
    )


def _section_body(text: str, section: str) -> list[str]:
    lines = text.splitlines()
    start = None
    result: list[str] = []
    for line in lines:
        match = re.match(r"\s*\[\s*([^]]+)\s*]\s*$", line)
        if match:
            current = match.group(1).strip().lower()
            if start is not None and current != section.lower():
                break
            if current == section.lower():
                start = True
                continue
        elif start is not None:
            result.append(line)
    return result


def _molecule_type(itp_text: str) -> str:
    for line in _section_body(itp_text, "moleculetype"):
        clean = line.split(";", 1)[0].strip()
        if clean:
            return clean.split()[0]
    raise ModuleConfigError("GROMACS nucleic-acid ITP has no [ moleculetype ] name")


def _itp_charge(itp_text: str) -> float:
    total = 0.0
    atoms = 0
    for line in _section_body(itp_text, "atoms"):
        clean = line.split(";", 1)[0].strip()
        if not clean:
            continue
        fields = clean.split()
        if len(fields) < 7:
            continue
        try:
            total += float(fields[6])
        except ValueError as exc:
            raise ModuleConfigError("Invalid charge in native nucleic-acid ITP") from exc
        atoms += 1
    if atoms == 0:
        raise ModuleConfigError("Native nucleic-acid ITP contains no atoms")
    rounded = round(total)
    if abs(total - rounded) > 1e-3:
        raise ModuleConfigError(
            f"Native nucleic-acid molecule has non-integral net charge {total:.6f}"
        )
    return float(rounded)


def _sanitize_itp(text: str) -> str:
    """Remove generator headers that expose host paths, users, and timestamps."""
    match = re.search(r"(?m)^\s*\[\s*moleculetype\s*]\s*$", text)
    if not match:
        raise ModuleConfigError("Generated nucleic-acid ITP is incomplete")
    body = text[match.start():]
    # A single-chain pdb2gmx run writes the molecule directly into the master
    # topology.  Remove its subsequent water/ion/system sections while
    # retaining the molecule's POSRES include.
    water_include = re.search(
        r'(?m)^\s*#include\s+"[^"]*(?:tip3p|tip4p|spc|spce)\.itp"\s*$',
        body,
    )
    system_section = re.search(r"(?m)^\s*\[\s*system\s*]\s*$", body)
    stops = [item.start() for item in (water_include, system_section) if item]
    if stops:
        body = body[:min(stops)]
    return "; Native GROMACS/CHARMM36 nucleic-acid topology\n" + body.rstrip() + "\n"


def _sanitize_posre(text: str, itp_text: str) -> str:
    match = re.search(r"(?m)^\s*\[\s*position_restraints\s*]\s*$", text)
    if not match:
        raise ModuleConfigError("Generated nucleic-acid position restraints are incomplete")
    atom_names: dict[int, str] = {}
    for line in _section_body(itp_text, "atoms"):
        clean = line.split(";", 1)[0].strip()
        fields = clean.split()
        if len(fields) >= 5:
            try:
                atom_names[int(fields[0])] = fields[4].strip()
            except ValueError:
                continue
    backbone = {
        "P", "O1P", "O2P", "OP1", "OP2", "O5'", "C5'", "C4'",
        "O4'", "C1'", "C2'", "O2'", "C3'", "O3'",
    }
    lines = text[match.start():].splitlines()
    rewritten: list[str] = []
    for line in lines:
        clean = line.split(";", 1)[0].strip()
        fields = clean.split()
        if len(fields) >= 5 and fields[0].isdigit() and fields[1].isdigit():
            atom_index = int(fields[0])
            macro = (
                "POSRES_FC_BB"
                if atom_names.get(atom_index, "") in backbone
                else "POSRES_FC_SC"
            )
            rewritten.append(
                f"{atom_index:6d}    1  {macro:>16s} {macro:>16s} {macro:>16s}"
            )
        else:
            rewritten.append(line)
    return (
        "; Native nucleic-acid heavy-atom restraints\n"
        + "\n".join(rewritten).rstrip()
        + "\n"
    )


def _replace_molecule_type(text: str, old: str, new: str) -> str:
    lines = text.splitlines()
    in_section = False
    replaced = False
    for index, line in enumerate(lines):
        match = re.match(r"\s*\[\s*([^]]+)\s*]\s*$", line)
        if match:
            in_section = match.group(1).strip().lower() == "moleculetype"
            continue
        if in_section and not replaced:
            clean = line.split(";", 1)[0].strip()
            if clean:
                fields = line.split()
                if fields[0] != old:
                    raise ModuleConfigError("Generated nucleic-acid molecule name changed unexpectedly")
                lines[index] = line.replace(old, new, 1)
                replaced = True
    if not replaced:
        raise ModuleConfigError("Could not normalize nucleic-acid molecule name")
    return "\n".join(lines) + "\n"


def _element(atom_name: str) -> str:
    name = str(atom_name).strip().upper().lstrip("0123456789")
    return "H" if name.startswith("H") else name[:1] or "C"


def _make_polymer_molecules_contiguous(system: System) -> None:
    """Group protein chains and native NA chains into coordinate molecules.

    Protein hydrogen construction appends atoms to the structure.  In a
    protein--nucleic-acid input that can otherwise leave one protein molecule
    split around a DNA/RNA block, which violates GROMACS [ molecules ] order.
    """
    blocks: list[tuple[int, list[int]]] = []
    owned: set[int] = set()
    for component in system.components:
        if component.kind == ComponentKind.PROTEIN:
            by_chain: dict[str, list[int]] = {}
            for raw_index in component.atom_indices:
                index = int(raw_index)
                by_chain.setdefault(str(system.structure.chain_ids[index]), []).append(index)
            polymer_blocks = by_chain.values()
        elif component.kind == ComponentKind.NUCLEIC_ACID:
            polymer_blocks = ([int(index) for index in component.atom_indices],)
        else:
            continue
        for raw_block in polymer_blocks:
            block = sorted(set(map(int, raw_block)))
            if not block:
                continue
            if owned.intersection(block):
                raise ModuleConfigError("Polymer components have overlapping atom indices")
            owned.update(block)
            blocks.append((min(block), block))

    blocks.extend((index, [index]) for index in range(system.num_atoms) if index not in owned)
    order = [index for _first, block in sorted(blocks) for index in block]
    if order == list(range(system.num_atoms)):
        return
    if sorted(order) != list(range(system.num_atoms)):
        raise ModuleConfigError("Could not establish a complete polymer coordinate order")

    old_to_new = {old: new for new, old in enumerate(order)}
    system.structure.coordinates = system.structure.coordinates[order]
    for name in (
        "atom_names", "resnames", "resids", "chain_ids", "segids",
        "elements", "occupancies", "tempfactors",
    ):
        values = getattr(system.structure, name)
        setattr(system.structure, name, [values[index] for index in order])
    for component in system.components:
        component.atom_indices = np.asarray(
            sorted(old_to_new[int(index)] for index in component.atom_indices),
            dtype=int,
        )
        native = component.metadata.get("native_topology")
        if component.kind == ComponentKind.NUCLEIC_ACID and isinstance(native, dict):
            native["atom_indices"] = component.atom_indices.tolist()

    # Structure Processing may hold transient index-based terms.  Final
    # force-field assignment reconstructs them from residue/crosslink metadata.
    system.topology = None


def _prepare_component(
    structure: Structure,
    component: Component,
    force_field: str,
    ordinal: int,
) -> tuple[Structure, dict]:
    gmx = find_gromacs_executable()
    if not gmx:
        raise ModuleConfigError(
            "GROMACS is required to construct canonical DNA/RNA hydrogens, "
            "termini, and polymer topology"
        )
    indices = [int(index) for index in component.atom_indices]
    source = _subset(structure, indices)
    chain = str(component.metadata.get("chain_id", "")) or "A"
    polymer = str(component.metadata.get("polymer_type", "NA"))
    safe_chain = re.sub(r"[^A-Za-z0-9_]", "_", chain) or "chain"
    canonical_moltype = f"{polymer}_chain_{safe_chain}_{ordinal}"

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-nucleic-") as temp_name:
        work = Path(temp_name)
        ff_target = work / f"{force_field}.ff"
        shutil.copytree(_force_field_directory(force_field), ff_target)
        input_pdb = work / "nucleic.pdb"
        PDBWriter.write(source, input_pdb, title="Canonical nucleic acid")
        command = [
            gmx, "pdb2gmx", "-f", str(input_pdb), "-o", "processed.gro",
            "-p", "native.top", "-ff", force_field, "-water", "tip3p",
            "-ignh", "-ter",
        ]
        env = os.environ.copy()
        env["GMXLIB"] = str(work)
        try:
            completed = subprocess.run(
                command,
                cwd=work,
                env=env,
                # The bundled CHARMM36m TDB fixes these menu entries to the
                # 5TER and 3TER hydroxyl patches; output is verified below.
                input="4\n6\n",
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModuleConfigError(
                f"Native GROMACS preparation timed out for {polymer} chain {chain}"
            ) from exc
        except OSError as exc:
            raise ModuleConfigError(
                f"Native GROMACS preparation could not start for {polymer} chain {chain}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stdout + "\n" + completed.stderr).strip()
            detail = detail.replace(str(work), "<temporary-workdir>").replace(gmx, "gmx")
            detail = "\n".join(detail.splitlines()[-18:])
            raise ModuleConfigError(
                f"Native GROMACS preparation failed for {polymer} chain {chain}: {detail}"
            )
        terminal_report = completed.stdout + "\n" + completed.stderr
        if not re.search(r"Start terminus .*:\s*5TER", terminal_report) or not re.search(
            r"End terminus .*:\s*3TER", terminal_report
        ):
            raise ModuleConfigError(
                f"Native GROMACS did not confirm 5TER/3TER hydroxyl termini for "
                f"{polymer} chain {chain}"
            )
        processed_path = work / "processed.gro"
        candidates = sorted(
            path for path in work.glob("*.itp")
            if not path.name.startswith("posre")
        )
        if not processed_path.is_file() or len(candidates) > 1:
            raise ModuleConfigError(
                f"Native GROMACS output for {polymer} chain {chain} is incomplete"
            )
        itp_path = candidates[0] if candidates else work / "native.top"
        if not itp_path.is_file():
            raise ModuleConfigError(
                f"Native GROMACS topology for {polymer} chain {chain} is missing"
            )
        posre_candidates = sorted(work.glob("posre*.itp"))
        if len(posre_candidates) != 1:
            raise ModuleConfigError(
                f"Native GROMACS restraints for {polymer} chain {chain} are incomplete"
            )
        itp_text = _sanitize_itp(itp_path.read_text())
        old_moltype = _molecule_type(itp_text)
        itp_text = _replace_molecule_type(itp_text, old_moltype, canonical_moltype)
        posre_name = f"posre_{canonical_moltype}.itp"
        itp_text = itp_text.replace(posre_candidates[0].name, posre_name)
        posre_text = _sanitize_posre(posre_candidates[0].read_text(), itp_text)
        charge = _itp_charge(itp_text)
        processed = GROReader().read(processed_path)

    processed.chain_ids = [chain] * processed.num_atoms
    processed.segids = [safe_chain[:4]] * processed.num_atoms
    processed.elements = [_element(name) for name in processed.atom_names]
    processed.occupancies = [1.0] * processed.num_atoms
    processed.tempfactors = [0.0] * processed.num_atoms
    native = {
        "molecule_type": canonical_moltype,
        "itp_filename": f"topol_{canonical_moltype}.itp",
        "itp_text": itp_text,
        "posre_filename": posre_name,
        "posre_text": posre_text,
        "net_charge": charge,
        "polymer_type": polymer,
        "chain_id": chain,
        "atom_count": processed.num_atoms,
        "residue_count": len(set(processed.resids)),
        "backend": "gromacs-pdb2gmx-charmm36",
    }
    return processed, native


def prepare_nucleic_acids(system: System) -> tuple[System, list[str]]:
    """Add canonical DNA/RNA hydrogens and persist exact native ITP content."""
    components = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
    if not components:
        return system, []
    force_field = str(system.metadata.get("force_field", "")).strip().lower()
    if force_field not in _SUPPORTED_FORCE_FIELDS:
        raise ModuleConfigError(
            f"Native nucleic-acid preparation is unavailable for {force_field or 'no force field'}"
        )

    prepared: dict[int, tuple[Component, Structure, dict]] = {}
    replaced: set[int] = set()
    for ordinal, component in enumerate(
        sorted(components, key=lambda item: min(map(int, item.atom_indices))), start=1
    ):
        result, native = _prepare_component(
            system.structure, component, force_field, ordinal
        )
        first = min(map(int, component.atom_indices))
        prepared[first] = (component, result, native)
        replaced.update(map(int, component.atom_indices))

    coordinates: list[np.ndarray] = []
    fields = {name: [] for name in (
        "atom_names", "resnames", "resids", "chain_ids", "segids",
        "elements", "occupancies", "tempfactors",
    )}
    old_to_new: dict[int, int] = {}
    new_nucleic: dict[int, Component] = {}

    def append_old(index: int) -> None:
        old_to_new[index] = len(coordinates)
        coordinates.append(system.structure.coordinates[index].copy())
        for name in fields:
            fields[name].append(getattr(system.structure, name)[index])

    for old_index in range(system.num_atoms):
        if old_index in prepared:
            old_component, new_structure, native = prepared[old_index]
            start = len(coordinates)
            coordinates.extend(new_structure.coordinates.copy())
            for name in fields:
                fields[name].extend(list(getattr(new_structure, name)))
            atom_indices = np.arange(start, len(coordinates), dtype=int)
            native["atom_indices"] = atom_indices.tolist()
            metadata = dict(old_component.metadata)
            metadata.update({
                "native_topology": native,
                "net_charge": native["net_charge"],
                "prepared": True,
            })
            new_nucleic[id(old_component)] = Component(
                name=old_component.name,
                kind=ComponentKind.NUCLEIC_ACID,
                atom_indices=atom_indices,
                metadata=metadata,
            )
        if old_index not in replaced:
            append_old(old_index)

    system.structure = Structure(
        coordinates=np.asarray(coordinates, dtype=float),
        box_vectors=system.structure.box_vectors.copy(),
        **fields,
    )
    rebuilt: list[Component] = []
    for component in system.components:
        replacement = new_nucleic.get(id(component))
        if replacement is not None:
            rebuilt.append(replacement)
            continue
        try:
            mapped = [old_to_new[int(index)] for index in component.atom_indices]
        except KeyError as exc:
            raise ModuleConfigError("Component indices overlap a nucleic-acid polymer") from exc
        rebuilt.append(Component(
            name=component.name,
            kind=component.kind,
            atom_indices=np.asarray(mapped, dtype=int),
            metadata=dict(component.metadata),
        ))
    system.components = rebuilt
    _make_polymer_molecules_contiguous(system)
    native_records = [
        component.metadata["native_topology"]
        for component in rebuilt if component.kind == ComponentKind.NUCLEIC_ACID
    ]
    system.metadata["native_nucleic_topologies"] = native_records
    log = [
        f"Native nucleic-acid topology: {record['polymer_type']} chain "
        f"{record['chain_id']} ({record['residue_count']} residues, "
        f"{record['atom_count']} atoms, charge {record['net_charge']:+.0f} e)"
        for record in native_records
    ]
    return system, log

"""Private helpers used only inside the Martini 3 module."""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.modules.coarse_grained.assets import (
    lipid_viewer_topologies,
    load_manifest,
)


STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HID",
    "HIE",
    "HIP",
    "HSD",
    "HSE",
    "HSP",
    "ASH",
    "GLH",
    "LYN",
    "CYX",
}


def strict_bool(config: dict, name: str, default: bool) -> bool:
    """Return a JSON boolean without accepting truthy strings or numbers."""
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ModuleConfigError(f"{name} must be true or false")
    return value


def task_step_dir(config: dict) -> Path:
    value = config.get("_step_dir")
    if not value:
        raise ModuleConfigError("Internal task step directory is missing")
    path = Path(str(value)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_root(config: dict) -> Path:
    value = config.get("_task_dir")
    if not value:
        raise ModuleConfigError("Internal task directory is missing")
    return Path(str(value)).resolve()


def run_checked(
    args: list[str],
    *,
    cwd: Path,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run one pinned scientific tool without a shell or inherited oversubscription."""
    if not args or not Path(args[0]).is_file():
        raise ModuleConfigError(f"Required executable is unavailable: {args[0] if args else '?'}")
    env = os.environ.copy()
    try:
        configured_threads = int(env.get("GMXBUILDER_TASK_THREADS", "1"))
    except (TypeError, ValueError):
        configured_threads = 1
    threads = str(max(1, configured_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = threads
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Martinize2 can launch DSSP and other helpers. Terminating the private
        # process group prevents timed-out Web tasks from leaving workers behind.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.communicate()
        raise ModuleConfigError(
            f"{Path(args[0]).name} exceeded the {timeout:g} second limit"
        ) from exc
    result = subprocess.CompletedProcess(args, process.returncode, stdout, None)
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-25:])
        raise ModuleConfigError(
            f"{Path(args[0]).name} failed with exit code {result.returncode}:\n{tail}"
        )
    return result


def martinize_executable() -> Path:
    candidate = Path(sys.executable).with_name("martinize2")
    if candidate.is_file():
        return candidate
    found = shutil.which("martinize2")
    if found:
        return Path(found)
    raise ModuleConfigError("Martinize2 is not installed in the GMXBUILDER environment")


def normalize_composition(value: object, *, label: str) -> list[dict]:
    """Validate a leaflet composition and normalize positive ratios."""
    if not isinstance(value, list) or not value:
        raise ModuleConfigError(f"{label} leaflet must contain at least one lipid")
    manifest = load_manifest()["lipids"]
    merged: dict[str, float] = defaultdict(float)
    for entry in value:
        if not isinstance(entry, dict):
            raise ModuleConfigError(f"{label} leaflet entries must be objects")
        name = str(entry.get("name", "")).strip().upper()
        if name not in manifest:
            raise ModuleConfigError(
                f"{name or 'Unnamed lipid'} is unavailable in the validated Martini 3 bundle"
            )
        try:
            ratio = float(entry.get("ratio", 0.0))
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError(f"{label} {name} ratio must be numeric") from exc
        if not math.isfinite(ratio) or ratio <= 0:
            raise ModuleConfigError(f"{label} {name} ratio must be positive")
        merged[name] += ratio
    total = sum(merged.values())
    return [
        {"name": name, "ratio": ratio / total, **manifest[name]}
        for name, ratio in sorted(merged.items())
    ]


def weighted_apl(upper: list[dict], lower: list[dict]) -> float:
    values = upper + lower
    return float(sum(item["ratio"] * item["apl_nm2"] for item in values) / 2.0)


def coby_lipid_tokens(entries: list[dict]) -> list[str]:
    tokens: list[str] = []
    for entry in entries:
        ratio = max(1, int(round(float(entry["ratio"]) * 1000)))
        suffix = ":params:default" if entry["builder_params"] == "default" else ""
        tokens.append(f"lipid:{entry['name']}:{ratio}{suffix}")
    return tokens


def write_cg_viewer_pdb(system: System, path: Path, *, task_dir: Path) -> None:
    """Write CG coordinates with authoritative protein and lipid connections.

    Martini bead separations are commonly longer than atomistic covalent-bond
    guessing thresholds.  A coordinate-only PDB therefore appears as isolated
    beads in browser viewers even though the simulation topology is intact.
    Martinize2 and the bundled lipid ITP files already contain the correct
    connection graphs.  Retain those graphs after coordinate transformations
    instead of reconstructing bonds by distance.
    """
    system.write_viewer_pdb(path)
    if system.num_atoms > 99999:
        return

    root = Path(task_dir).resolve()
    relative_sources = [
        system.metadata.get("cg_connectivity_pdb"),
        # Checkpoints created before the dedicated metadata key still have the
        # authoritative task-private Martinize2 output at this stable path.
        "steps/cg_mapping/martinize/cg_protein.pdb",
        system.metadata.get("cg_protein_pdb"),
    ]
    source_lines: list[str] | None = None
    for value in relative_sources:
        if not value:
            continue
        raw_candidate = root / str(value)
        if raw_candidate.is_symlink():
            continue
        candidate = raw_candidate.resolve()
        if candidate != root and root in candidate.parents and candidate.is_file():
            candidate_lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(line.startswith("CONECT") for line in candidate_lines):
                source_lines = candidate_lines
                break
    connections: set[tuple[int, int]] = set()
    if source_lines is not None:
        atom_serials: list[int] = []
        source_connections: list[list[int]] = []
        for line in source_lines:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM"}:
                try:
                    atom_serials.append(int(line[6:11]))
                except ValueError:
                    atom_serials = []
                    break
            elif record == "CONECT":
                try:
                    serials = [int(value) for value in line[6:].split()]
                except ValueError:
                    source_connections = []
                    break
                if len(serials) >= 2:
                    source_connections.append(serials)

        protein_atoms = len(atom_serials)
        protein_indices = sorted(
            {
                int(index)
                for component in system.components
                if component.kind == ComponentKind.PROTEIN
                for index in component.atom_indices
            }
        )
        if (
            source_connections
            and atom_serials == list(range(1, protein_atoms + 1))
            and protein_atoms <= system.num_atoms
            and protein_indices[:protein_atoms] == list(range(protein_atoms))
            and all(
                1 <= serial <= protein_atoms for serials in source_connections for serial in serials
            )
        ):
            for serials in source_connections:
                source = serials[0]
                for target in serials[1:]:
                    if source != target:
                        connections.add(tuple(sorted((source, target))))

    lipid_topologies = lipid_viewer_topologies()
    for component in system.components:
        if component.kind != ComponentKind.MEMBRANE:
            continue
        current_key: tuple[str, int, str] | None = None
        residue_indices: list[int] = []

        def add_lipid_residue() -> None:
            if not residue_indices:
                return
            first = residue_indices[0]
            name = str(system.structure.resnames[first]).strip().upper()
            topology = lipid_topologies.get(name)
            if topology is None:
                return
            observed = tuple(
                str(system.structure.atom_names[index]).strip().upper() for index in residue_indices
            )
            if observed != topology["atom_names"]:
                return
            for local_a, local_b in topology["edges"]:
                if local_a <= len(residue_indices) and local_b <= len(residue_indices):
                    serial_a = residue_indices[local_a - 1] + 1
                    serial_b = residue_indices[local_b - 1] + 1
                    connections.add(tuple(sorted((serial_a, serial_b))))

        for raw_index in sorted(int(index) for index in component.atom_indices):
            key = (
                str(system.structure.resnames[raw_index]).strip().upper(),
                int(system.structure.resids[raw_index]),
                str(system.structure.chain_ids[raw_index]),
            )
            if current_key is not None and key != current_key:
                add_lipid_residue()
                residue_indices = []
            current_key = key
            residue_indices.append(raw_index)
        add_lipid_residue()

    if not connections:
        return

    adjacency: dict[int, list[int]] = defaultdict(list)
    for source, target in sorted(connections):
        adjacency[source].append(target)
    connect_lines: list[str] = []
    for source, targets in sorted(adjacency.items()):
        for offset in range(0, len(targets), 4):
            connect_lines.append(
                f"CONECT{source:5d}"
                + "".join(f"{target:5d}" for target in targets[offset : offset + 4])
            )

    viewer_lines = path.read_text(encoding="utf-8").splitlines()
    end_index = next(
        (index for index, line in enumerate(viewer_lines) if line.strip() == "END"),
        len(viewer_lines),
    )
    viewer_lines[end_index:end_index] = connect_lines
    path.write_text("\n".join(viewer_lines) + "\n", encoding="utf-8")


def molecule_types_from_topology(text: str) -> list[str]:
    names: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().replace(" ", "").lower() != "[moleculetype]":
            continue
        for candidate in lines[index + 1 :]:
            candidate = candidate.split(";", 1)[0].strip()
            if candidate:
                names.append(candidate.split()[0])
                break
    return names


def molecules_table(text: str) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    active = False
    for line in text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("["):
            active = stripped.replace(" ", "").lower() == "[molecules]"
            continue
        if not active or not stripped:
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            try:
                entries.append((fields[0], int(fields[1])))
            except ValueError:
                continue
    return entries


def molecule_type_charges(texts: Iterable[str]) -> dict[str, float]:
    """Extract net charges from GROMACS molecule-type atom tables."""
    charges: dict[str, float] = {}
    for text in texts:
        current: str | None = None
        section = ""
        awaiting_type = False
        for raw in str(text).splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                section = line.strip("[] ").lower()
                awaiting_type = section == "moleculetype"
                continue
            if awaiting_type:
                current = line.split()[0]
                charges.setdefault(current, 0.0)
                awaiting_type = False
                continue
            if section == "atoms" and current:
                fields = line.split()
                if len(fields) >= 7:
                    try:
                        charges[current] += float(fields[6])
                    except ValueError:
                        pass
    return charges


def topology_texts_from_dir(directory: Path) -> dict[str, str]:
    allowed = {".itp", ".top"}
    result: dict[str, str] = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in allowed:
            result[path.name] = path.read_text(encoding="utf-8", errors="strict")
    return result


def write_topology_texts(texts: dict[str, str], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in sorted(texts.items()):
        if Path(name).name != name or not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
            raise ModuleConfigError(f"Unsafe topology filename in checkpoint: {name!r}")
        path = directory / name
        path.write_text(str(text), encoding="utf-8")
        written.append(path)
    return written


def system_from_gro(path: Path, topology_text: str, *, metadata: dict) -> System:
    structure = GROReader().read(path)
    lipids = set(load_manifest()["lipids"])
    groups: dict[ComponentKind, list[int]] = defaultdict(list)
    for index, resname in enumerate(structure.resnames):
        name = str(resname).strip().upper()
        if name in lipids:
            groups[ComponentKind.MEMBRANE].append(index)
        elif name == "W":
            groups[ComponentKind.SOLVENT].append(index)
        elif name in {"NA", "CL"}:
            groups[ComponentKind.IONS].append(index)
        elif name in STANDARD_PROTEIN_RESIDUES:
            groups[ComponentKind.PROTEIN].append(index)
        else:
            groups[ComponentKind.UNKNOWN].append(index)

    components: list[Component] = []
    labels = {
        ComponentKind.PROTEIN: "CG Protein",
        ComponentKind.MEMBRANE: "Martini 3 Membrane",
        ComponentKind.SOLVENT: "Martini Water",
        ComponentKind.IONS: "Martini Ions",
        ComponentKind.UNKNOWN: "Unclassified CG Beads",
    }
    molecule_counts: Counter[str] = Counter()
    for molecule_name, count in molecules_table(topology_text):
        molecule_counts[molecule_name] += count
    for kind, indices in groups.items():
        if not indices:
            continue
        component_metadata: dict = {}
        if kind == ComponentKind.SOLVENT:
            component_metadata = {
                "water_model": "W",
                "n_molecules": int(molecule_counts.get("W", len(indices))),
            }
        elif kind == ComponentKind.IONS:
            counts = {
                name: int(molecule_counts[name])
                for name in ("NA", "CL")
                if molecule_counts.get(name, 0)
            }
            component_metadata = {
                "n_molecules": int(sum(counts.values()) or len(indices)),
                "counts": counts,
            }
        elif kind == ComponentKind.MEMBRANE:
            counts = {
                name: int(molecule_counts[name])
                for name in sorted(lipids)
                if molecule_counts.get(name, 0)
            }
            component_metadata = {
                "n_molecules": int(sum(counts.values())),
                "counts": counts,
            }
        components.append(
            Component(
                name=labels[kind],
                kind=kind,
                atom_indices=np.asarray(indices, dtype=np.int64),
                metadata=component_metadata,
            )
        )
    metadata = dict(metadata)
    metadata["cg_molecule_counts"] = dict(molecule_counts)
    metadata["resolution"] = "coarse-grained"
    metadata["force_field"] = "martini3"
    return System(structure=structure, components=components, metadata=metadata)


def rotation_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    x, y, z = np.radians([x_deg, y_deg, z_deg])
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def residue_groups(system: System, indices: Iterable[int]) -> list[np.ndarray]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for raw in indices:
        index = int(raw)
        groups[(str(system.structure.resnames[index]), int(system.structure.resids[index]))].append(
            index
        )
    return [np.asarray(value, dtype=np.int64) for value in groups.values()]

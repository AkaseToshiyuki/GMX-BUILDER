"""Cached AmberTools/ACPYPE GAFF2 parameterization for non-RTP molecules."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Iterator

import numpy as np


DEFAULT_GAFF_ENV = Path.home() / ".local" / "share" / "gmxbuilder" / "gaff-env"
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_task_cache_root: ContextVar[tuple[Path, frozenset[str]] | None] = ContextVar(
    "gmxbuilder_task_gaff_cache_root", default=None
)


def _run_external(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a tool and clean up its complete process group on timeout.

    AmberTools wrappers launch ``sqm`` through several shell processes.  A
    timeout applied only to the wrapper leaves those children consuming CPU
    after the temporary working directory has been removed.  A separate
    session gives the complete command tree one process group that can be
    terminated atomically.
    """
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
        detail = ((stdout or "") + "\n" + (stderr or ""))[-4000:]
        raise RuntimeError(
            f"Command timed out after {timeout} seconds ({' '.join(args)}):\n{detail}"
        ) from exc
    return subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _run_smiles_acpype(
    args: list[str],
    *,
    work: Path,
    env: dict[str, str],
    timeout: int,
    attempts: int = 3,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Retry only stochastic SMILES coordinate-generation failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    last_result: subprocess.CompletedProcess[str] | None = None
    last_work = work
    for attempt in range(1, attempts + 1):
        attempt_work = work / f"acpype_attempt_{attempt}"
        attempt_work.mkdir()
        result = _run_external(args, cwd=attempt_work, env=env, timeout=timeout)
        last_result = result
        last_work = attempt_work
        if result.returncode == 0:
            return result, attempt_work
        detail = result.stdout + "\n" + result.stderr
        coordinate_failure = (
            "Atoms TOO close" in detail or "Coordinates issues with your system" in detail
        )
        if not coordinate_failure:
            break
    if last_result is None:  # Defensive guard; attempts is validated above.
        raise RuntimeError("ACPYPE was not started")
    return last_result, last_work


@dataclass(frozen=True)
class GAFFTemplate:
    name: str
    atom_names: tuple[str, ...]
    coordinates: np.ndarray
    itp_path: Path
    atomtypes_path: Path
    charge_method: str


@dataclass(frozen=True)
class GAFFChargeSuggestion:
    """Coordinate-derived integer charge proposal for one retained molecule."""

    net_charge: int
    pH: float
    formula: str
    atom_count: int
    method: str = "Open Babel pH model with coordinate-based bond perception"


def gaff_environment_path() -> Path:
    return Path(os.environ.get("GMXBUILDER_GAFF_ENV", DEFAULT_GAFF_ENV))


def _gaff_tool_environment() -> dict[str, str]:
    """Return an isolated environment for AmberTools/Open Babel children."""
    env = os.environ.copy()
    env["PATH"] = str(gaff_environment_path() / "bin") + os.pathsep + env.get("PATH", "")
    configured = (
        os.environ.get("GMXBUILDER_GAFF_THREADS", "").strip()
        or os.environ.get("GMXBUILDER_LIPID_THREADS", "").strip()
    )
    if configured:
        try:
            threads = int(configured)
        except ValueError as exc:
            raise ValueError("GMXBUILDER_GAFF_THREADS must be a positive integer") from exc
        if threads <= 0:
            raise ValueError("GMXBUILDER_GAFF_THREADS must be a positive integer")
        from gmxbuilder.runtime.hardware import configured_task_threads

        threads = min(threads, configured_task_threads())
        # Scope OpenMP only to GAFF tools.  Exporting this on the parent
        # GMXBUILDER process conflicts with GROMACS' explicit ``-ntomp``.
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_THREAD_LIMIT"] = str(threads)
    return env


def gaff_available() -> bool:
    env_path = gaff_environment_path()
    return all(
        (env_path / "bin" / executable).is_file()
        for executable in ("acpype", "antechamber", "parmchk2", "tleap", "obabel")
    )


def _mol2_integer_charge(path: Path, name: str) -> int:
    """Return the integer charge encoded by one Open Babel MOL2 file."""
    lines = path.read_text(errors="replace").splitlines()
    try:
        start = (
            next(
                index for index, line in enumerate(lines) if line.strip().upper() == "@<TRIPOS>ATOM"
            )
            + 1
        )
    except StopIteration as exc:
        raise RuntimeError(f"Charge estimation produced an invalid MOL2 for {name}") from exc
    charges = []
    for line in lines[start:]:
        if line.strip().startswith("@<TRIPOS>"):
            break
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 9:
            raise RuntimeError(f"Charge estimation produced a malformed MOL2 for {name}")
        charges.append(float(fields[8]))
    if not charges or not np.isfinite(charges).all():
        raise RuntimeError(f"Charge estimation produced no finite charges for {name}")
    charge_sum = float(sum(charges))
    net_charge = int(round(charge_sum))
    if abs(charge_sum - net_charge) > 0.05:
        raise RuntimeError(f"Charge estimate for {name} sums to {charge_sum:+.3f}, not an integer")
    return net_charge


def _restore_mol2_heavy_atom_names(
    path: Path,
    input_names: tuple[str, ...],
    input_elements: tuple[str, ...],
) -> None:
    """Restore PDB heavy-atom names after Open Babel pH protonation."""
    lines = path.read_text(errors="replace").splitlines()
    in_atoms = False
    heavy_index = 0
    output = []
    for line in lines:
        marker = line.strip().upper()
        if marker == "@<TRIPOS>ATOM":
            in_atoms = True
            output.append(line)
            continue
        if in_atoms and marker.startswith("@<TRIPOS>"):
            in_atoms = False
        if in_atoms and line.split():
            fields = line.split()
            if len(fields) < 6:
                raise RuntimeError("pH-dependent protonation produced a malformed MOL2 atom")
            element = fields[5].split(".", 1)[0].upper()
            if element != "H":
                if heavy_index >= len(input_names):
                    raise RuntimeError("pH-dependent protonation added an unexpected heavy atom")
                expected_element = str(input_elements[heavy_index]).strip().upper()
                if element != expected_element:
                    raise RuntimeError(
                        "pH-dependent protonation changed heavy-atom order or elements"
                    )
                fields[1] = input_names[heavy_index]
                heavy_index += 1
                line = " ".join(fields)
        output.append(line)
    if heavy_index != len(input_names):
        raise RuntimeError(
            f"pH-dependent protonation retained {heavy_index} of "
            f"{len(input_names)} uploaded heavy atoms"
        )
    path.write_text("\n".join(output) + "\n")


def _acpype_failure_detail(work: Path, result: subprocess.CompletedProcess[str]) -> str:
    """Include ACPYPE/SQM files when a wrapper exits without console output."""
    details = [result.stdout or "", result.stderr or ""]
    for pattern in ("*.acpype/acpype.log", "*.acpype/sqm.out"):
        for path in sorted(work.glob(pattern)):
            details.append(f"\n--- {path.name} ---\n{path.read_text(errors='replace')}")
    detail = "\n".join(details).strip()
    return detail[-8000:] if detail else f"ACPYPE exited with status {result.returncode}"


def estimate_gaff_net_charge(
    name: str,
    structure,
    atom_indices: list[int] | tuple[int, ...],
    pH: float,
) -> GAFFChargeSuggestion:
    """Suggest an integer formal charge after pH-dependent protonation.

    This is deliberately a suggestion: PDB coordinates do not encode bond
    orders, so the web UI preserves an explicit user override.  Open Babel's
    protonation model adds the pH-appropriate hydrogens; its MOL2 partial
    charges are accepted only when they sum closely to one integer.
    """
    if not gaff_available():
        raise RuntimeError(f"GAFF2 environment is unavailable at {gaff_environment_path()}")
    target_pH = float(pH)
    if not 1.0 <= target_pH <= 13.0:
        raise ValueError("Ligand charge estimation pH must be between 1.0 and 13.0")
    indices = [int(index) for index in atom_indices]
    if not indices:
        raise ValueError(f"No atoms supplied for molecule {name}")

    from collections import Counter
    from gmxbuilder.core.structure import Structure
    from gmxbuilder.io.pdb import PDBWriter

    molecule_name = _safe_name(name)
    ligand_structure = Structure(
        coordinates=structure.coordinates[indices].copy(),
        box_vectors=structure.box_vectors.copy(),
        atom_names=[structure.atom_names[index] for index in indices],
        resnames=[molecule_name] * len(indices),
        resids=[1] * len(indices),
        chain_ids=["L"] * len(indices),
        elements=[structure.elements[index] for index in indices],
    )
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-charge-") as temporary:
        work = Path(temporary)
        pdb_path = work / "molecule.pdb"
        mol2_path = work / "protonated.mol2"
        PDBWriter.write(ligand_structure, pdb_path, title=f"Charge estimate {molecule_name}")
        result = _run_external(
            [
                str(gaff_environment_path() / "bin" / "obabel"),
                "-ipdb",
                str(pdb_path),
                "-omol2",
                "-O",
                str(mol2_path),
                "-p",
                f"{target_pH:.3f}",
            ],
            cwd=work,
            env=_gaff_tool_environment(),
            timeout=300,
        )
        if result.returncode != 0 or not mol2_path.is_file():
            detail = (result.stdout + "\n" + result.stderr)[-4000:]
            raise RuntimeError(f"Charge estimation failed for {name}: {detail}")

        lines = mol2_path.read_text(errors="replace").splitlines()
        try:
            start = (
                next(
                    index
                    for index, line in enumerate(lines)
                    if line.strip().upper() == "@<TRIPOS>ATOM"
                )
                + 1
            )
        except StopIteration as exc:
            raise RuntimeError(f"Charge estimation produced an invalid MOL2 for {name}") from exc
        charges = []
        elements = []
        for line in lines[start:]:
            if line.strip().startswith("@<TRIPOS>"):
                break
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 9:
                raise RuntimeError(f"Charge estimation produced a malformed MOL2 for {name}")
            raw_element = fields[5].split(".", 1)[0]
            element = raw_element[:1].upper() + raw_element[1:].lower()
            elements.append(element)
            charges.append(float(fields[8]))
        if not charges or not np.isfinite(charges).all():
            raise RuntimeError(f"Charge estimation produced no finite charges for {name}")
        net_charge = _mol2_integer_charge(mol2_path, name)
        counts = Counter(elements)
        ordered_elements = ["C"] if counts.get("C") else []
        if counts.get("H"):
            ordered_elements.append("H")
        ordered_elements.extend(sorted(set(counts) - set(ordered_elements)))
        formula = "".join(
            element + (str(counts[element]) if counts[element] != 1 else "")
            for element in ordered_elements
        )
        if net_charge > 0:
            formula += "+" if net_charge == 1 else f"{net_charge}+"
        elif net_charge < 0:
            formula += "-" if net_charge == -1 else f"{abs(net_charge)}-"
        return GAFFChargeSuggestion(
            net_charge=net_charge,
            pH=target_pH,
            formula=formula,
            atom_count=len(charges),
        )


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name.upper())
    if not safe or not safe[0].isalpha():
        safe = f"L_{safe}"
    return safe[:20]


def _cache_root(molecule_name: str | None = None) -> Path:
    scoped = _task_cache_root.get()
    if scoped is not None:
        root, isolated_names = scoped
        if molecule_name is not None and molecule_name.upper() in isolated_names:
            return root
    from gmxbuilder.runtime.prebuilt_assets import ensure_prebuilt_assets

    ensure_prebuilt_assets()
    return Path(
        os.environ.get(
            "GMXBUILDER_GAFF_CACHE",
            Path.home() / ".cache" / "gmxbuilder" / "gaff2",
        )
    )


@contextmanager
def task_gaff_cache(
    root: str | Path,
    isolated_names: set[str] | frozenset[str],
) -> Iterator[None]:
    """Route GAFF artifacts to one task-owned cache for this execution."""
    token = _task_cache_root.set(
        (
            Path(root).expanduser().resolve(),
            frozenset(str(name).upper() for name in isolated_names),
        )
    )
    try:
        yield
    finally:
        _task_cache_root.reset(token)


def _cache_key(name: str, smiles: str, net_charge: int, charge_method: str) -> str:
    payload = f"v2\0{name}\0{smiles}\0{net_charge}\0{charge_method}\0gaff2"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _normalize_itp(
    source: Path, destination: Path, atomtypes_destination: Path, molecule_name: str
) -> None:
    """Namespace GAFF atom types and normalize molecule/residue names."""
    prefix = f"g_{molecule_name.lower()}_"
    section = ""
    type_map: dict[str, str] = {}
    moleculetype_written = False
    output = []
    atomtypes_output = []
    for raw in source.read_text().splitlines():
        code, separator, comment = raw.partition(";")
        stripped = code.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[] ").lower()
            (atomtypes_output if section == "atomtypes" else output).append(raw)
            continue
        if not stripped or stripped.startswith("#"):
            (atomtypes_output if section == "atomtypes" else output).append(raw)
            continue
        fields = stripped.split()
        if section == "atomtypes" and len(fields) >= 2:
            for index in (0, 1):
                original = fields[index]
                fields[index] = type_map.setdefault(original, prefix + original)
        elif section == "moleculetype" and not moleculetype_written:
            fields[0] = molecule_name
            moleculetype_written = True
        elif section == "atoms" and len(fields) >= 5:
            fields[1] = type_map.get(fields[1], prefix + fields[1])
            fields[3] = molecule_name
        else:
            output.append(raw)
            continue
        rebuilt = " ".join(fields)
        if separator:
            rebuilt += f" ;{comment}"
        (atomtypes_output if section == "atomtypes" else output).append(rebuilt)
    destination.write_text("\n".join(output) + "\n")
    atomtypes_destination.write_text("\n".join(atomtypes_output) + "\n")


def _read_gro(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    from gmxbuilder.io.gro import GROReader

    structure = GROReader().read(path)
    return structure.coordinates.copy(), tuple(name.strip() for name in structure.atom_names)


def _itp_charges(path: Path) -> list[float]:
    charges: list[float] = []
    section = ""
    for raw in path.read_text().splitlines():
        code = raw.split(";", 1)[0].strip()
        if code.startswith("[") and code.endswith("]"):
            section = code.strip("[] ").lower()
            continue
        if section == "atoms" and code and not code.startswith("#"):
            fields = code.split()
            if len(fields) >= 7:
                charges.append(float(fields[6]))
    return charges


def _load_cached(directory: Path) -> GAFFTemplate | None:
    metadata_path = directory / "metadata.json"
    itp_path = directory / "lipid.itp"
    atomtypes_path = directory / "atomtypes.itp"
    gro_path = directory / "lipid.gro"
    if not all(path.is_file() for path in (metadata_path, itp_path, atomtypes_path, gro_path)):
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        coordinates, atom_names = _read_gro(gro_path)
        charges = _itp_charges(itp_path)
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        return None
    if (
        list(atom_names) != metadata.get("atom_names")
        or len(atom_names) != metadata.get("num_atoms")
        or not atom_names
        or len(set(atom_names)) != len(atom_names)
        or not np.isfinite(coordinates).all()
        or len(charges) != len(atom_names)
        or abs(sum(charges) - float(metadata.get("net_charge", 0))) > 0.02
    ):
        return None
    return GAFFTemplate(
        name=metadata["name"],
        atom_names=atom_names,
        coordinates=coordinates,
        itp_path=itp_path,
        atomtypes_path=atomtypes_path,
        charge_method=metadata["charge_method"],
    )


def prepare_gaff_lipid(
    name: str,
    smiles: str,
    net_charge: int,
    *,
    charge_method: str | None = None,
    timeout: int = 10800,
) -> GAFFTemplate:
    """Return a persistent, atom-order-consistent GAFF2 template."""
    if not gaff_available():
        raise RuntimeError(f"GAFF2 environment is unavailable at {gaff_environment_path()}")
    molecule_name = _safe_name(name)
    charge_method = (
        charge_method or os.environ.get("GMXBUILDER_GAFF_CHARGE_METHOD", "bcc")
    ).lower()
    if charge_method not in {"bcc", "gas"}:
        raise ValueError(f"Unsupported GAFF charge method: {charge_method}")
    key = _cache_key(molecule_name, smiles, int(net_charge), charge_method)
    directory = _cache_root(molecule_name) / f"{molecule_name}-{key}"
    cached = _load_cached(directory)
    if cached is not None:
        return cached

    with _lock_for(key):
        cached = _load_cached(directory)
        if cached is not None:
            return cached
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
            work = Path(temporary)
            staging = work / "cache"
            staging.mkdir()
            env = _gaff_tool_environment()
            result, generated_work = _run_smiles_acpype(
                [
                    str(gaff_environment_path() / "bin" / "acpype"),
                    "-i",
                    smiles,
                    "-b",
                    molecule_name,
                    "-c",
                    charge_method,
                    "-n",
                    str(int(net_charge)),
                    "-a",
                    "gaff2",
                    "-o",
                    "gmx",
                    "-w",
                ],
                work=work,
                env=env,
                timeout=timeout,
            )
            if result.returncode != 0:
                detail = (result.stdout + "\n" + result.stderr)[-4000:]
                raise RuntimeError(f"GAFF2 parameterization failed for {name}: {detail}")
            generated = generated_work / f"{molecule_name}.acpype"
            source_itp = generated / f"{molecule_name}_GMX.itp"
            source_gro = generated / f"{molecule_name}_GMX.gro"
            if not (source_itp.is_file() and source_gro.is_file()):
                raise RuntimeError(f"ACPYPE did not produce GROMACS files for {name}")
            _normalize_itp(
                source_itp,
                staging / "lipid.itp",
                staging / "atomtypes.itp",
                molecule_name,
            )
            shutil.copy2(source_gro, staging / "lipid.gro")
            coordinates, atom_names = _read_gro(staging / "lipid.gro")
            (staging / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": molecule_name,
                        "smiles": smiles,
                        "net_charge": int(net_charge),
                        "charge_method": charge_method,
                        "atom_names": list(atom_names),
                        "num_atoms": len(atom_names),
                    },
                    indent=2,
                )
            )
            if directory.exists():
                shutil.rmtree(directory)
            staging.rename(directory)
        cached = _load_cached(directory)
        if cached is None:
            raise RuntimeError(f"Invalid GAFF2 cache generated for {name}")
        return cached


def prepare_gaff_molecule(
    name: str,
    structure,
    atom_indices: list[int] | tuple[int, ...],
    net_charge: int,
    *,
    charge_method: str | None = None,
    target_pH: float = 7.0,
    timeout: int = 10800,
) -> GAFFTemplate:
    """Parameterize one coordinate-defined molecule with GAFF2.

    Open Babel infers connectivity and adds hydrogens before ACPYPE.  The
    generated topology is accepted only when its heavy-atom prefix preserves
    the input PDB atom order and names, allowing the hydrogens to be appended
    without changing the user-supplied heavy-atom coordinates.
    """
    if not gaff_available():
        raise RuntimeError(f"GAFF2 environment is unavailable at {gaff_environment_path()}")
    indices = [int(index) for index in atom_indices]
    if not indices:
        raise ValueError(f"No atoms supplied for molecule {name}")
    target_pH = float(target_pH)
    if not 1.0 <= target_pH <= 13.0:
        raise ValueError("GAFF2 ligand protonation pH must be between 1.0 and 13.0")
    molecule_name = _safe_name(name)
    charge_method = (
        charge_method or os.environ.get("GMXBUILDER_GAFF_CHARGE_METHOD", "bcc")
    ).lower()
    if charge_method not in {"bcc", "gas"}:
        raise ValueError(f"Unsupported GAFF charge method: {charge_method}")
    input_names = tuple(str(structure.atom_names[index]).strip() for index in indices)
    input_elements = tuple(str(structure.elements[index]).strip() for index in indices)
    if len(set(input_names)) != len(input_names):
        raise ValueError(f"Molecule {name} has duplicate atom names")
    signature = json.dumps(
        {
            "names": input_names,
            "elements": [str(structure.elements[index]).strip() for index in indices],
            "distances": np.round(
                np.linalg.norm(
                    structure.coordinates[indices][:, None, :]
                    - structure.coordinates[indices][None, :, :],
                    axis=2,
                ),
                3,
            ).tolist(),
            "protonation_pH": round(target_pH, 3),
        },
        sort_keys=True,
    )
    key = _cache_key(molecule_name, signature, int(net_charge), charge_method)
    directory = _cache_root(molecule_name) / f"MOL_{molecule_name}-{key}"
    cached = _load_cached(directory)
    if cached is not None:
        if cached.atom_names[: len(input_names)] != input_names:
            raise RuntimeError(f"Cached GAFF2 atom order mismatch for {name}")
        return cached

    with _lock_for(key):
        cached = _load_cached(directory)
        if cached is not None:
            return cached
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
            work = Path(temporary)
            staging = work / "cache"
            staging.mkdir()
            from gmxbuilder.core.structure import Structure
            from gmxbuilder.io.pdb import PDBWriter

            ligand_structure = Structure(
                coordinates=structure.coordinates[indices].copy(),
                box_vectors=structure.box_vectors.copy(),
                atom_names=[structure.atom_names[index] for index in indices],
                resnames=[molecule_name] * len(indices),
                resids=[1] * len(indices),
                chain_ids=["L"] * len(indices),
                elements=[structure.elements[index] for index in indices],
            )
            pdb_path = work / "molecule.pdb"
            mol2_path = work / "molecule_h.mol2"
            PDBWriter.write(ligand_structure, pdb_path, title=f"GAFF2 input {molecule_name}")
            env = _gaff_tool_environment()
            obabel = _run_external(
                [
                    str(gaff_environment_path() / "bin" / "obabel"),
                    "-ipdb",
                    str(pdb_path),
                    "-omol2",
                    "-O",
                    str(mol2_path),
                    "-h",
                ],
                cwd=work,
                env=env,
                timeout=300,
            )
            if obabel.returncode != 0 or not mol2_path.is_file():
                detail = (obabel.stdout + "\n" + obabel.stderr)[-4000:]
                raise RuntimeError(f"Bond perception failed for {name}: {detail}")
            explicit_charge = _mol2_integer_charge(mol2_path, name)
            if explicit_charge != int(net_charge):
                protonated_path = work / "molecule_ph.mol2"
                protonated = _run_external(
                    [
                        str(gaff_environment_path() / "bin" / "obabel"),
                        "-ipdb",
                        str(pdb_path),
                        "-omol2",
                        "-O",
                        str(protonated_path),
                        "-p",
                        f"{target_pH:.3f}",
                    ],
                    cwd=work,
                    env=env,
                    timeout=300,
                )
                if protonated.returncode != 0 or not protonated_path.is_file():
                    detail = (protonated.stdout + "\n" + protonated.stderr)[-4000:]
                    raise RuntimeError(f"pH-dependent protonation failed for {name}: {detail}")
                inferred_charge = _mol2_integer_charge(protonated_path, name)
                if inferred_charge != int(net_charge):
                    raise ValueError(
                        f"Requested net charge {int(net_charge):+d} for {name} does not match "
                        f"the uploaded structure ({explicit_charge:+d}) or the pH "
                        f"{target_pH:.1f} protonation model ({inferred_charge:+d}). "
                        "Adjust the target pH/net charge or upload the intended explicit "
                        "protonation state."
                    )
                _restore_mol2_heavy_atom_names(
                    protonated_path,
                    input_names,
                    input_elements,
                )
                mol2_path = protonated_path
            result = _run_external(
                [
                    str(gaff_environment_path() / "bin" / "acpype"),
                    "-i",
                    str(mol2_path),
                    "-b",
                    molecule_name,
                    "-c",
                    charge_method,
                    "-n",
                    str(int(net_charge)),
                    "-a",
                    "gaff2",
                    "-o",
                    "gmx",
                    "-w",
                ],
                cwd=work,
                env=env,
                timeout=timeout,
            )
            if result.returncode != 0:
                detail = _acpype_failure_detail(work, result)
                raise RuntimeError(f"GAFF2 parameterization failed for {name}: {detail}")
            generated = work / f"{molecule_name}.acpype"
            source_itp = generated / f"{molecule_name}_GMX.itp"
            source_gro = generated / f"{molecule_name}_GMX.gro"
            if not (source_itp.is_file() and source_gro.is_file()):
                raise RuntimeError(f"ACPYPE did not produce GROMACS files for {name}")
            _normalize_itp(
                source_itp,
                staging / "lipid.itp",
                staging / "atomtypes.itp",
                molecule_name,
            )
            shutil.copy2(source_gro, staging / "lipid.gro")
            coordinates, atom_names = _read_gro(staging / "lipid.gro")
            if atom_names[: len(input_names)] != input_names:
                raise RuntimeError(
                    f"GAFF2 changed heavy-atom order for {name}: "
                    f"expected {input_names}, got {atom_names[: len(input_names)]}"
                )
            (staging / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": molecule_name,
                        "identity_signature": signature,
                        "net_charge": int(net_charge),
                        "charge_method": charge_method,
                        "atom_names": list(atom_names),
                        "num_atoms": len(atom_names),
                    },
                    indent=2,
                )
            )
            if directory.exists():
                shutil.rmtree(directory)
            staging.rename(directory)
        cached = _load_cached(directory)
        if cached is None:
            raise RuntimeError(f"Invalid GAFF2 cache generated for {name}")
        return cached

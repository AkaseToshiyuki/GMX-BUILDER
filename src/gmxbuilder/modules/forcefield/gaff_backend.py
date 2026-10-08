"""Cached AmberTools/ACPYPE GAFF2 parameterization for non-RTP molecules."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

DEFAULT_GAFF_ENV = Path.home() / ".local" / "share" / "gmxbuilder" / "gaff-env"
MAX_GAFF_LIGAND_ATOMS = 2_048
_COORDINATE_SIGNATURE_SCHEMA = "gaff-coordinate-v4-ph-first-relative-0.0001nm"
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


def _bounded_atom_indices(name: str, atom_indices: list[int] | tuple[int, ...]) -> list[int]:
    """Normalize one ligand selection while enforcing the supported size."""
    if len(atom_indices) > MAX_GAFF_LIGAND_ATOMS:
        raise ValueError(
            f"Molecule {name} contains more than {MAX_GAFF_LIGAND_ATOMS} atoms; "
            "GAFF2 parameterization supports one small molecule at a time"
        )
    indices = [int(index) for index in atom_indices]
    if not indices:
        raise ValueError(f"No atoms supplied for molecule {name}")
    return indices


def _coordinate_identity_signature(
    names: tuple[str, ...],
    elements: tuple[str, ...],
    coordinates: np.ndarray,
    target_pH: float,
) -> str:
    """Return a versioned, translation-invariant O(N)-memory cache identity.

    Coordinate order and orientation remain significant.  A rigid translation
    therefore reuses the same parameterization, while any atom identity,
    ordering, relative-coordinate, or protonation-pH change gets a new key.
    """
    coordinate_array = np.asarray(coordinates, dtype=float)
    if (
        coordinate_array.shape != (len(names), 3)
        or len(elements) != len(names)
        or not names
        or not np.isfinite(coordinate_array).all()
    ):
        raise ValueError("GAFF2 ligand coordinates must be a finite N×3 array")

    digest = hashlib.sha256()
    digest.update((_COORDINATE_SIGNATURE_SCHEMA + "\0").encode("ascii"))
    digest.update(f"{float(target_pH):.3f}\0".encode("ascii"))
    origin = coordinate_array[0]
    for name, element, coordinate in zip(names, elements, coordinate_array, strict=True):
        relative = np.round(coordinate - origin, 4)
        # Canonicalize signed zero so harmless translations do not create a
        # platform-dependent cache miss.
        relative[relative == 0.0] = 0.0
        record = json.dumps(
            [str(name), str(element), *(float(value) for value in relative)],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return json.dumps(
        {
            "schema": _COORDINATE_SIGNATURE_SCHEMA,
            "atom_count": len(names),
            "protonation_pH": round(float(target_pH), 3),
            "sha256": digest.hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def gaff_environment_path() -> Path:
    return Path(os.environ.get("GMXBUILDER_GAFF_ENV", DEFAULT_GAFF_ENV))


#: Threads given to AmberTools when nothing else says otherwise.
#:
#: With no setting at all OpenMP takes every logical core, and AM1-BCC does not
#: repay that. Measured on this project's own charge fits: sqm held 96 threads
#: open on a 96-core host and drew 114% CPU -- one core of work and ninety-five
#: threads contending for it. The charge fit is the long pole in building a
#: lipid entry, so it is worth bounding rather than leaving to the default.
DEFAULT_GAFF_THREADS = 48


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
    else:
        threads = DEFAULT_GAFF_THREADS
    # Clamp to the task's budget only where one is actually in force. Outside a
    # task -- the offline library queue, a test, a script -- the deployment-wide
    # default is 1, and clamping to that would run the charge fit on a single
    # thread, which is far worse than the unbounded default this replaces.
    from gmxbuilder.runtime.hardware import scoped_task_threads

    budget = scoped_task_threads()
    ceiling = budget if budget is not None else (os.cpu_count() or threads)
    threads = max(1, min(threads, ceiling))
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
    target_pH = float(pH)
    if not 1.0 <= target_pH <= 13.0:
        raise ValueError("Ligand charge estimation pH must be between 1.0 and 13.0")
    indices = _bounded_atom_indices(name, atom_indices)
    if not gaff_available():
        raise RuntimeError(f"GAFF2 environment is unavailable at {gaff_environment_path()}")

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


def _cache_root(molecule_name: str | None = None, *, install: bool = True) -> Path:
    scoped = _task_cache_root.get()
    if scoped is not None:
        root, isolated_names = scoped
        if molecule_name is not None and molecule_name.upper() in isolated_names:
            return root
    if install:
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


@contextmanager
def _exclusive_fit(key: str, directory: Path) -> Iterator[None]:
    """Hold this molecule's cache entry against every other fitter.

    A thread lock is not enough. The library queue runs both replicas of an
    entry as separate processes, one per GPU, and they reach the same molecule
    at the same moment: with only ``_lock_for`` each runs the same half-hour
    AM1-BCC fit, and then each removes the other's cache directory before
    renaming its own into place. The work is wasted, and the window where the
    directory does not exist is visible to anything else reading it.

    The thread lock serialises threads; a file lock serialises processes. The
    lock file is kept apart from the entry, because the entry is what gets
    replaced.
    """
    import fcntl

    locks = directory.parent / ".locks"
    with _lock_for(key):
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"{directory.name}.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _make_atom_names_unique(itp_path: Path, gro_path: Path) -> None:
    """Give every atom its own name, in both files, when ACPYPE did not.

    ACPYPE names atoms element-first -- C, C1, C2, O, O1 -- and its counter can
    restart part way through a molecule, so CER16 comes back with two atoms
    called C, two called O and two called H out of 105. GROMACS does not mind;
    this project does, because the cached template's names are the handles that
    tie a conformer to a topology, and a repeated name makes that mapping
    ambiguous. The entry was rejected outright before, which left the lipid
    unbuildable under GAFF2 rather than merely awkward.

    The renaming is by position, applied to the topology and the coordinates
    together, so the two stay in step.
    """
    coordinate_lines = gro_path.read_text().splitlines()
    if len(coordinate_lines) < 3:
        return
    body = coordinate_lines[2:-1]
    names = [line[10:15].strip() for line in body]
    if len(set(names)) == len(names):
        return

    taken = set()
    renamed: list[str] = []
    for position, name in enumerate(names, start=1):
        candidate = name
        if candidate in taken:
            element = "".join(character for character in name if character.isalpha()) or "X"
            candidate = f"{element[:3]}{position}"
            while candidate in taken:
                position += 1
                candidate = f"{element[:3]}{position}"
        taken.add(candidate)
        renamed.append(candidate)

    gro_path.write_text(
        "\n".join(
            coordinate_lines[:2]
            + [f"{line[:10]}{name:>5s}{line[15:]}" for line, name in zip(body, renamed)]
            + [coordinate_lines[-1]]
        )
        + "\n"
    )

    output = []
    section = ""
    index = 0
    for raw in itp_path.read_text().splitlines():
        code, separator, comment = raw.partition(";")
        stripped = code.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[] ").lower()
            output.append(raw)
            continue
        fields = stripped.split()
        if section == "atoms" and len(fields) >= 5 and index < len(renamed):
            fields[4] = renamed[index]
            index += 1
            output.append(" ".join(fields) + (separator + comment if separator else ""))
            continue
        output.append(raw)
    itp_path.write_text("\n".join(output) + "\n")


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


def _load_cached(
    directory: Path,
    *,
    expected_identity_signature: str | None = None,
    expected_smiles: str | None = None,
) -> GAFFTemplate | None:
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
        or (
            expected_identity_signature is not None
            and (
                metadata.get("schema_version") != 2
                or metadata.get("identity_signature") != expected_identity_signature
            )
        )
    ):
        return None
    template = GAFFTemplate(
        name=metadata["name"],
        atom_names=atom_names,
        coordinates=coordinates,
        itp_path=itp_path,
        atomtypes_path=atomtypes_path,
        charge_method=metadata["charge_method"],
    )
    if expected_smiles is not None:
        from gmxbuilder.modules.forcefield.gaff_lipid_identity import validate_template

        if metadata.get("smiles") != expected_smiles:
            return None
        try:
            validate_template(expected_smiles, template)
        except (OSError, ValueError, IndexError, KeyError):
            return None
    return template


def cached_gaff_template(
    name: str,
    smiles: str,
    net_charge: int,
    *,
    charge_method: str | None = None,
) -> GAFFTemplate | None:
    """The fitted template for this molecule, if one is already on disk.

    Never fits anything, and never installs anything either. Callers that only
    need to know what the topology will look like should not trigger half an
    hour of AM1-BCC -- and must not trigger an asset installation, because the
    library asks this question from inside one: recognising a stale entry is
    part of installing over it. Installing from here made the process wait for
    a file lock it already held, and everything else queued behind it.
    """
    molecule_name = _safe_name(name)
    key = _cache_key(molecule_name, smiles, int(net_charge), gaff_charge_method(charge_method))
    root = _cache_root(molecule_name, install=False)
    return _load_cached(root / f"{molecule_name}-{key}", expected_smiles=smiles)


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
    charge_method = gaff_charge_method(charge_method)
    key = _cache_key(molecule_name, smiles, int(net_charge), charge_method)
    directory = _cache_root(molecule_name) / f"{molecule_name}-{key}"
    cached = _load_cached(directory, expected_smiles=smiles)
    if cached is not None:
        return cached

    with _exclusive_fit(key, directory):
        cached = _load_cached(directory, expected_smiles=smiles)
        if cached is not None:
            return cached
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
            work = Path(temporary)
            staging = work / "cache"
            staging.mkdir()
            env = _gaff_tool_environment()
            from gmxbuilder.modules.forcefield.gaff_lipid_identity import write_lipid_sdf

            input_sdf = work / "lipid.sdf"
            write_lipid_sdf(smiles, input_sdf)
            result, generated_work = _run_smiles_acpype(
                [
                    str(gaff_environment_path() / "bin" / "acpype"),
                    "-i",
                    str(input_sdf),
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
            _make_atom_names_unique(staging / "lipid.itp", staging / "lipid.gro")
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
            # Validate before publishing: a failed fit must preserve the old
            # cache as evidence, rather than replacing it with another bad one.
            if _load_cached(staging, expected_smiles=smiles) is None:
                raise RuntimeError(f"GAFF2 fitted coordinates fail identity checks for {name}")
            if directory.exists():
                from datetime import datetime, timezone

                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                directory.rename(directory.with_name(f"{directory.name}.replaced-{stamp}"))
            staging.rename(directory)
        cached = _load_cached(directory, expected_smiles=smiles)
        if cached is None:
            raise RuntimeError(f"Invalid GAFF2 cache generated for {name}")
        return cached


class _MoleculeCacheTarget(NamedTuple):
    """Where a coordinate-defined molecule's parameters live, and its inputs."""

    molecule_name: str
    charge_method: str
    signature: str
    key: str
    directory: Path
    input_names: tuple[str, ...]
    input_elements: tuple[str, ...]
    input_coordinates: np.ndarray


# What each charge method is, and what waiting for it means. Used to tell the
# user what a parameterization is doing: promising "several minutes" of
# quantum chemistry for a method that does neither would be simply untrue.
_CHARGE_METHOD_DESCRIPTIONS = {
    "bcc": (
        "AM1-BCC",
        "a one-time quantum-chemistry calculation that takes several minutes",
    ),
    "gas": ("Gasteiger", "a one-time empirical charge assignment"),
}


def gaff_charge_method(charge_method: str | None = None) -> str:
    """Return the charge method a parameterization would actually use."""
    resolved = (charge_method or os.environ.get("GMXBUILDER_GAFF_CHARGE_METHOD", "bcc")).lower()
    if resolved not in _CHARGE_METHOD_DESCRIPTIONS:
        raise ValueError(f"Unsupported GAFF charge method: {resolved}")
    return resolved


def describe_gaff_charge_method(charge_method: str | None = None) -> tuple[str, str]:
    """Return the method's name and what its wait consists of."""
    return _CHARGE_METHOD_DESCRIPTIONS[gaff_charge_method(charge_method)]


def _molecule_cache_target(
    name: str,
    structure,
    indices: tuple[int, ...] | list[int],
    net_charge: int,
    *,
    charge_method: str | None,
    target_pH: float,
) -> _MoleculeCacheTarget:
    """Resolve where a coordinate-defined molecule's parameters live.

    Extracted so that asking whether a molecule is already parameterized uses
    exactly the key the parameterization itself will use. A second copy of this
    derivation would answer that question about a different cache entry as soon
    as either copy changed.
    """
    if not gaff_available():
        raise RuntimeError(f"GAFF2 environment is unavailable at {gaff_environment_path()}")
    target_pH = float(target_pH)
    if not 1.0 <= target_pH <= 13.0:
        raise ValueError("GAFF2 ligand protonation pH must be between 1.0 and 13.0")
    molecule_name = _safe_name(name)
    charge_method = gaff_charge_method(charge_method)
    input_names = tuple(str(structure.atom_names[index]).strip() for index in indices)
    input_elements = tuple(str(structure.elements[index]).strip() for index in indices)
    if len(set(input_names)) != len(input_names):
        raise ValueError(f"Molecule {name} has duplicate atom names")
    input_coordinates = np.asarray(structure.coordinates[indices], dtype=float)
    signature = _coordinate_identity_signature(
        input_names,
        input_elements,
        input_coordinates,
        target_pH,
    )
    key = _cache_key(molecule_name, signature, int(net_charge), charge_method)
    directory = _cache_root(molecule_name) / f"MOL_{molecule_name}-{key}"
    return _MoleculeCacheTarget(
        molecule_name,
        charge_method,
        signature,
        key,
        directory,
        input_names,
        input_elements,
        input_coordinates,
    )


def gaff_molecule_is_cached(
    name: str,
    structure,
    atom_indices: list[int] | tuple[int, ...],
    net_charge: int,
    *,
    charge_method: str | None = None,
    target_pH: float = 7.0,
) -> bool:
    """Return whether parameterizing this molecule would be a cache hit.

    A hit takes a fraction of a second; a miss is an AM1-BCC calculation that
    takes minutes. The difference is worth telling the user before it starts,
    and worth knowing before deciding to precompute one in the background.
    Answering is best-effort: a molecule the parameterizer would reject is
    reported as not cached, so the caller learns that from the real call.
    """
    try:
        indices = _bounded_atom_indices(name, atom_indices)
        target = _molecule_cache_target(
            name,
            structure,
            indices,
            net_charge,
            charge_method=charge_method,
            target_pH=target_pH,
        )
    except Exception:  # noqa: BLE001 - a question, not an operation
        return False
    cached = _load_cached(target.directory, expected_identity_signature=target.signature)
    return cached is not None


class MoleculeJob(NamedTuple):
    """One molecule to parameterize, with everything the call needs."""

    name: str
    structure: object
    atom_indices: tuple[int, ...] | list[int]
    net_charge: int


def parameterize_molecules(
    jobs: list[MoleculeJob],
    *,
    target_pH: float = 7.0,
    charge_method: str | None = None,
    max_workers: int | None = None,
    on_progress=None,
) -> dict[str, GAFFTemplate]:
    """Parameterize independent molecules, concurrently where it pays.

    Each molecule is a separate `sqm` run that links no OpenMP and takes
    minutes, so a system with four ligands spent four times as long as it
    needed to. The molecules share nothing: distinct cache keys, distinct
    directories, and `_exclusive_fit` is per key, so they contend only for CPU.

    Concurrency is capped by the task's own thread budget. One `sqm` is one
    thread, so that cap is also the number of molecules worth starting.

    `on_progress(completed, total, name)` is called on the *calling* thread as
    each finishes, so a caller may report progress without being thread-safe.

    Failures are raised in job order rather than completion order: which
    molecule is reported must not depend on which happened to fail first.
    """
    if not jobs:
        return {}

    def run_one(job: MoleculeJob) -> GAFFTemplate:
        return prepare_gaff_molecule(
            job.name,
            job.structure,
            job.atom_indices,
            job.net_charge,
            charge_method=charge_method,
            target_pH=target_pH,
        )

    # One molecule is the common case and gains nothing from a pool.
    if len(jobs) == 1:
        template = run_one(jobs[0])
        if on_progress is not None:
            on_progress(1, 1, jobs[0].name)
        return {jobs[0].name: template}

    from gmxbuilder.runtime.hardware import current_task_threads

    workers = max_workers or min(len(jobs), max(1, current_task_threads()))
    results: dict[str, GAFFTemplate] = {}
    failures: dict[str, BaseException] = {}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gaff") as pool:
        # A worker thread starts with an empty context, so without this it
        # would see neither the task's scoped GAFF cache nor its custom lipid
        # registry -- writing a task-private molecule into the shared cache.
        # Contexts cannot be entered twice at once, hence one copy per job.
        futures = {pool.submit(contextvars.copy_context().run, run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job.name] = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised in job order below
                failures[job.name] = exc
            if on_progress is not None:
                on_progress(len(results) + len(failures), len(jobs), job.name)

    for job in jobs:
        if job.name in failures:
            raise failures[job.name]
    # Rebuilt in job order: the caller's downstream iteration order is theirs,
    # not whichever molecule finished first.
    return {job.name: results[job.name] for job in jobs}


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
    indices = _bounded_atom_indices(name, atom_indices)
    target = _molecule_cache_target(
        name,
        structure,
        indices,
        net_charge,
        charge_method=charge_method,
        target_pH=target_pH,
    )
    molecule_name = target.molecule_name
    charge_method = target.charge_method
    signature = target.signature
    key = target.key
    directory = target.directory
    input_names = target.input_names
    input_elements = target.input_elements
    input_coordinates = target.input_coordinates
    cached = _load_cached(directory, expected_identity_signature=signature)
    if cached is not None:
        if cached.atom_names[: len(input_names)] != input_names:
            raise RuntimeError(f"Cached GAFF2 atom order mismatch for {name}")
        return cached

    with _exclusive_fit(key, directory):
        cached = _load_cached(directory, expected_identity_signature=signature)
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
                coordinates=input_coordinates.copy(),
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
            # Net charge alone cannot identify a microstate: NH2/COOH and
            # NH3+/COO- can both sum to zero. Apply the selected pH model
            # before validating the requested charge, even when it is zero.
            protonated = _run_external(
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
                env=env,
                timeout=300,
            )
            if protonated.returncode != 0 or not mol2_path.is_file():
                detail = (protonated.stdout + "\n" + protonated.stderr)[-4000:]
                raise RuntimeError(f"pH-dependent protonation failed for {name}: {detail}")
            inferred_charge = _mol2_integer_charge(mol2_path, name)
            if inferred_charge != int(net_charge):
                raise ValueError(
                    f"Requested net charge {int(net_charge):+d} for {name} does not match "
                    f"the pH {target_pH:.1f} protonation model ({inferred_charge:+d}). "
                    "Review the molecular identity and target pH/net charge. "
                    "A net-charge override alone cannot specify a different microstate."
                )
            _restore_mol2_heavy_atom_names(mol2_path, input_names, input_elements)
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
            _make_atom_names_unique(staging / "lipid.itp", staging / "lipid.gro")
            coordinates, atom_names = _read_gro(staging / "lipid.gro")
            if atom_names[: len(input_names)] != input_names:
                raise RuntimeError(
                    f"GAFF2 changed heavy-atom order for {name}: "
                    f"expected {input_names}, got {atom_names[: len(input_names)]}"
                )
            (staging / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
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
        cached = _load_cached(directory, expected_identity_signature=signature)
        if cached is None:
            raise RuntimeError(f"Invalid GAFF2 cache generated for {name}")
        return cached

"""Compact, component-aware display artifacts tied to immutable checkpoint bytes."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = 3


def _display_bonds(system, selected, membership):
    """Keep saved bonds; fill missing AA residue graphs for display only.

    Coordinate checkpoints often predate topology assembly. Restrict geometric
    links to one component/chain/residue, never between neighbouring molecules.
    CG bonds remain authoritative: bead spacing is not an atomistic bond rule.
    """
    structure = system.structure
    reverse = np.full(system.num_atoms, -1, dtype=np.int32)
    reverse[selected] = np.arange(len(selected))
    bonds = set()
    covered = set()
    groups = {}
    for index in selected:
        key = (int(membership[index]), structure.chain_ids[index], structure.resids[index])
        groups.setdefault(key, []).append(int(index))
    if system.topology is not None:
        for bond in system.topology.bonds:
            a, b = int(bond.i), int(bond.j)
            if 0 <= a < system.num_atoms and 0 <= b < system.num_atoms:
                if reverse[a] >= 0 and reverse[b] >= 0:
                    bonds.add(tuple(sorted((int(reverse[a]), int(reverse[b])))))
                    first = (int(membership[a]), structure.chain_ids[a], structure.resids[a])
                    last = (int(membership[b]), structure.chain_ids[b], structure.resids[b])
                    if first == last:
                        covered.add(first)
    explicit_count = len(bonds)
    if system.metadata.get("resolution") != "coarse-grained":
        from rdkit import Chem
        from scipy.spatial import cKDTree

        table = Chem.GetPeriodicTable()
        organic = {"B", "C", "N", "O", "F", "Si", "P", "S", "Se", "Cl", "Br", "I"}
        radii = {symbol: table.GetRcovalent(symbol) for symbol in organic}
        for key, indices in groups.items():
            if key in covered:
                continue
            component = system.components[key[0]] if key[0] < len(system.components) else None
            if component is not None and component.kind.name in {"SOLVENT", "IONS"}:
                continue
            ids = [i for i in indices if structure.elements[i].strip().title() in organic]
            if len(ids) < 2:
                continue
            coords = structure.coordinates[ids] * 10
            radius = np.array([radii[structure.elements[i].strip().title()] for i in ids])
            pairs = cKDTree(coords).query_pairs(2.5 * max(radius), output_type="ndarray")
            if not len(pairs):
                continue
            distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
            limit = 1.25 * (radius[pairs[:, 0]] + radius[pairs[:, 1]])
            for a, b in pairs[(distances > 0.4) & (distances <= limit)]:
                bonds.add(tuple(sorted((int(reverse[ids[a]]), int(reverse[ids[b]])))))
    return sorted(bonds), {
        "topology": explicit_count,
        "residue_geometry": len(bonds) - explicit_count,
    }


def checkpoint_stamp(directory: Path) -> list:
    values = []
    for name in ("system.json", "system.npz"):
        path = directory / name
        if path.is_symlink():
            raise ValueError("Checkpoint must not be a symbolic link")
        stat = path.stat()
        values.append([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    return values


def checkpoint_revision(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("system.json", "system.npz"):
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def cached_viewer(directory: Path) -> Path | None:
    try:
        index = directory / "display-index.json"
        if index.is_symlink():
            return None
        record = json.loads(index.read_text())
        if not isinstance(record, dict):
            return None
        revision = record.get("revision", "")
        if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
            return None
        path = directory / f"display-{revision}.json.gz"
        if (
            record.get("schema") == SCHEMA
            and record.get("stamp") == checkpoint_stamp(directory)
            and path.is_file()
            and not path.is_symlink()
        ):
            return path
    except (OSError, ValueError, TypeError):
        pass
    return None


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".display-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_viewer(directory: Path) -> Path:
    from gmxbuilder.core.system import System

    hit = cached_viewer(directory)
    if hit:
        return hit
    before = checkpoint_stamp(directory)
    revision = checkpoint_revision(directory)
    system = System.load_checkpoint(directory)
    structure = system.structure
    n = system.num_atoms
    # Display one oxygen per AA water, all CG beads and all other heavy atoms.
    # Original indices and complete component counts remain explicit.
    selected = np.flatnonzero(~np.isin(np.char.upper(structure.elements), ["H", "D"]))
    membership = np.full(n, -1, dtype=np.int32)
    components = []
    for index, component in enumerate(system.components):
        ids = np.asarray(component.atom_indices, dtype=int)
        if (
            np.any(ids < 0)
            or np.any(ids >= n)
            or np.any(membership[ids] >= 0)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Invalid or overlapping checkpoint component membership")
        membership[ids] = index
        components.append(
            {
                "name": component.name,
                "kind": component.kind.name,
                "atoms": len(ids),
                "molecules": component.metadata.get("n_molecules"),
                "lipids": (
                    component.metadata.get("n_lipids_upper", 0)
                    + component.metadata.get("n_lipids_lower", 0)
                )
                or None,
                "display_atoms": int(np.isin(selected, ids).sum()),
            }
        )
    if np.any(membership < 0):
        missing = membership < 0
        membership[missing] = len(components)
        components.append(
            {
                "name": "Other atoms",
                "kind": "OTHER",
                "atoms": int(missing.sum()),
                "display_atoms": int(missing[selected].sum()),
                "molecules": None,
                "lipids": None,
            }
        )

    def binary(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")

    def dictionary(values):
        labels, codes = np.unique(values, return_inverse=True)
        return {"labels": labels.tolist(), "codes": binary(codes, "<u4")}

    bonds, bond_sources = _display_bonds(system, selected, membership)
    payload = {
        "schema": SCHEMA,
        "revision": revision,
        "source_step": directory.name,
        "resolution": system.metadata.get("resolution", "atomistic"),
        "atom_count": n,
        "display_count": len(selected),
        "components": components,
        "box_nm": structure.box_vectors.tolist(),
        "coordinates_A": binary(structure.coordinates[selected] * 10, "<f4"),
        "original_indices": binary(selected, "<u4"),
        "component_indices": binary(membership[selected], "<u4"),
        "resids": dictionary(np.asarray(structure.resids, dtype=str)[selected]),
        "names": dictionary(np.asarray(structure.atom_names)[selected]),
        "resnames": dictionary(np.asarray(structure.resnames)[selected]),
        "chains": dictionary(np.asarray(structure.chain_ids)[selected]),
        "elements": dictionary(np.asarray(structure.elements)[selected]),
        "bonds": bonds,
        "bond_sources": bond_sources,
    }
    if checkpoint_stamp(directory) != before:
        raise ValueError("Checkpoint changed while preparing its viewer; retry")
    target = directory / f"display-{revision}.json.gz"
    _atomic_write(
        target,
        gzip.compress(
            json.dumps(payload, separators=(",", ":")).encode(), compresslevel=3, mtime=0
        ),
    )
    _atomic_write(
        directory / "display-index.json",
        json.dumps({"schema": SCHEMA, "stamp": before, "revision": revision}).encode(),
    )
    return target


def final_source(task_state: dict) -> str:
    task_type = (task_state.get("task_type") or {}).get("id") or task_state.get("task_type_id")
    if str(task_type).startswith("martini3-"):
        return "cg_system"
    if (
        task_type == "pure-membrane"
        and (task_state.get("step_solvation_config") or {}).get("enabled") is False
    ):
        return "membrane"
    return "ions"


def confirmed_revision(task_state: dict, directory: Path) -> bool:
    record = task_state.get("final_review") or {}
    return (
        record.get("source_step") == directory.name
        and record.get("confirmed") is True
        and record.get("revision") == checkpoint_revision(directory)
    )


def require_final_review(task_state: dict, directory: Path, submitted_revision: str | None) -> None:
    """Recheck approval after queue wait, immediately before package generation."""
    record = task_state.get("final_review") or {}
    if (
        not submitted_revision
        or record.get("revision") != submitted_revision
        or not confirmed_revision(task_state, directory)
    ):
        raise ValueError(
            "The final checkpoint changed or is unconfirmed; repeat Final Structure Review"
        )

"""Independent membrane positioning for mapped Martini 3 proteins.

This module deliberately owns its scoring and coordinate handling.  It does
not call the atomistic Bilayer Builder orientation module, so changes to one
workflow cannot silently alter the other.
"""

from __future__ import annotations

import math

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.geometry.align import compute_principal_axes
from gmxbuilder.geometry.transforms import (
    rotation_matrix_from_axis_angle,
    rotation_matrix_from_vectors,
)
from gmxbuilder.modules.coarse_grained.common import (
    task_root,
    task_step_dir,
    write_cg_viewer_pdb,
)
from gmxbuilder.pipeline.base import BaseModule, ModuleResult


# Wimley-White whole-residue water-to-interface transfer free energies.
# The CG pose is scored per Martini backbone bead, retaining residue identity.
_TRANSFER = {
    "ALA": 0.17,
    "ARG": 0.81,
    "ASN": 0.42,
    "ASP": 1.23,
    "CYS": -0.24,
    "GLN": 0.58,
    "GLU": 0.11,
    "GLY": 0.01,
    "HIS": 0.96,
    "ILE": -0.31,
    "LEU": -0.56,
    "LYS": 0.99,
    "MET": -0.22,
    "PHE": -1.13,
    "PRO": 0.45,
    "SER": 0.13,
    "THR": 0.14,
    "TRP": -1.85,
    "TYR": -0.94,
    "VAL": -0.07,
    "ASH": 1.23,
    "GLH": 0.11,
    "CYX": -0.24,
    "HID": 0.96,
    "HIE": 0.96,
    "HIP": 0.96,
    "HSD": 0.96,
    "HSE": 0.96,
    "HSP": 0.96,
    "LYN": 0.99,
}
_HYDROPHOBIC = {"ALA", "CYS", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "VAL"}


def _finite(value: object, label: str, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModuleConfigError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ModuleConfigError(f"{label} must be between {low:g} and {high:g}")
    return result


def _backbone_rows(structure) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, str]]]:
    coordinates: list[np.ndarray] = []
    energies: list[float] = []
    keys: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, atom_name in enumerate(structure.atom_names):
        if str(atom_name).strip().upper() != "BB":
            continue
        key = (
            str(structure.chain_ids[index]),
            int(structure.resids[index]),
            str(structure.resnames[index]).strip().upper(),
        )
        if key in seen:
            continue
        seen.add(key)
        coordinates.append(np.asarray(structure.coordinates[index], dtype=float))
        energies.append(float(_TRANSFER.get(key[2], 0.0)))
        keys.append(key)
    if len(coordinates) < 3:
        raise ModuleConfigError(
            "Mapped membrane protein needs at least three Martini BB beads for orientation"
        )
    return np.asarray(coordinates), np.asarray(energies), keys


def _profile_score(
    coords: np.ndarray, energies: np.ndarray, z_shift: float, half_thickness: float
) -> float:
    distance = np.abs(coords[:, 2] + z_shift) - half_thickness
    weights = 1.0 / (1.0 + np.exp(np.clip(distance / 0.30, -100.0, 100.0)))
    score = float(np.sum(energies * weights))
    hydrophobic = energies < -0.5
    polar = energies > 0.3
    if np.any(hydrophobic):
        score += 0.5 * float(np.sum(np.abs(energies[hydrophobic]) * (1.0 - weights[hydrophobic])))
    if np.any(polar):
        score += 2.0 * float(np.sum(energies[polar] * weights[polar]))
    overpacked = max(0.0, float(np.mean(weights)) - 0.50)
    return score + 200.0 * overpacked * overpacked


def _candidate_normals(
    coords: np.ndarray, keys: list[tuple[str, int, str]]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return PCA axes plus axes of hydrophobic sequence windows.

    Soluble domains can dominate whole-protein PCA.  Hydrophobic backbone
    windows provide candidate transmembrane axes for single-pass proteins and
    multi-helix bundles before transfer-energy ranking.
    """
    candidates = [np.asarray(axis, dtype=float) for axis in compute_principal_axes(coords)]
    tm_residues: set[int] = set()
    by_chain: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        by_chain.setdefault(key[0], []).append(index)
    for indices in by_chain.values():
        for width in (11, 15, 19, 23):
            if len(indices) < width:
                continue
            for start in range(0, len(indices) - width + 1, 2):
                selection = indices[start : start + width]
                names = [keys[index][2] for index in selection]
                hydrophobic_fraction = sum(name in _HYDROPHOBIC for name in names) / width
                if hydrophobic_fraction < 0.48:
                    continue
                segment = coords[selection]
                centered = segment - segment.mean(axis=0)
                covariance = centered.T @ centered
                values, vectors = np.linalg.eigh(covariance)
                axial_fraction = float(values[-1] / max(values.sum(), 1e-12))
                if float(values[-1]) <= 1e-12 or axial_fraction < 0.55:
                    continue
                candidates.append(vectors[:, -1])
                if width >= 15 and hydrophobic_fraction >= 0.60:
                    tm_residues.update(selection)
    unique: list[np.ndarray] = []
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-12:
            continue
        candidate = candidate / norm
        if candidate[int(np.argmax(np.abs(candidate)))] < 0:
            candidate = -candidate
        if any(abs(float(np.dot(candidate, old))) > 0.985 for old in unique):
            continue
        unique.append(candidate)
    return unique, np.asarray(sorted(tm_residues), dtype=np.int64)


def _best_automatic_pose(structure, half_thickness: float) -> tuple[np.ndarray, np.ndarray, dict]:
    residue_coords, energies, keys = _backbone_rows(structure)
    center = residue_coords.mean(axis=0)
    centered = residue_coords - center
    best: tuple[float, np.ndarray, float, float] | None = None
    z_axis = np.array([0.0, 0.0, 1.0])
    normals, tm_indices = _candidate_normals(centered, keys)
    for normal in normals:
        base = rotation_matrix_from_vectors(normal, z_axis)
        aligned = centered @ base.T
        # Refine the candidate normal in both directions around two in-plane axes.
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            for degrees in np.linspace(-30.0, 30.0, 13):
                refinement = rotation_matrix_from_axis_angle(axis, math.radians(float(degrees)))
                rotated = aligned @ refinement.T
                span = max(float(np.max(np.abs(rotated[:, 2]))), half_thickness * 1.5)
                for z_shift in np.linspace(-span, span, 81):
                    score = _profile_score(rotated, energies, float(z_shift), half_thickness)
                    if tm_indices.size:
                        tm_z = rotated[tm_indices, 2] + float(z_shift)
                        tm_center = float(np.median(tm_z))
                        tm_core_fraction = float(np.mean(np.abs(tm_z) <= half_thickness + 0.35))
                        # A confidently detected hydrophobic segment is direct
                        # evidence of a membrane anchor.  Whole-protein transfer
                        # energy alone can otherwise prefer placing a large
                        # single-pass protein entirely in water.
                        score += 25.0 * (tm_center / half_thickness) ** 2
                        score += 80.0 * (1.0 - tm_core_fraction) ** 2
                    if best is None or score < best[0]:
                        best = (score, refinement @ base, float(z_shift), float(degrees))
    if best is None:
        raise ModuleConfigError("Could not determine a finite Martini membrane orientation")
    score, rotation, z_shift, refinement_degrees = best
    transformed = centered @ rotation.T
    transformed[:, 2] += z_shift
    core = np.abs(transformed[:, 2]) <= half_thickness
    hydrophobic = np.asarray([key[2] in _HYDROPHOBIC for key in keys], dtype=bool)
    polar = energies > 0.3
    tm_z = transformed[tm_indices, 2] if tm_indices.size else np.asarray([], dtype=float)
    tm_core_fraction = (
        float(np.mean(np.abs(tm_z) <= half_thickness + 0.35)) if tm_indices.size else None
    )
    if tm_indices.size and (tm_core_fraction is None or tm_core_fraction < 0.65):
        raise ModuleConfigError(
            "Automatic orientation detected a transmembrane hydrophobic segment but "
            "could not place it reliably in the membrane core; use manual review"
        )
    metrics = {
        "method": "ppm",
        "half_thickness_nm": half_thickness,
        "z_offset_nm": z_shift,
        "refinement_degrees": refinement_degrees,
        "transfer_score": score,
        "core_residues": int(np.count_nonzero(core)),
        "hydrophobic_core_fraction": float(np.mean(hydrophobic[core])) if np.any(core) else 0.0,
        "polar_core_fraction": float(np.mean(polar[core])) if np.any(core) else 0.0,
        "tm_window_residues": int(tm_indices.size),
        "tm_core_fraction": tm_core_fraction,
        "tm_z_range_nm": [float(np.min(tm_z)), float(np.max(tm_z))] if tm_indices.size else None,
    }
    return rotation, np.array([0.0, 0.0, z_shift]), metrics


class CGOrientationModule(BaseModule):
    name = "cg_orientation"
    description = "Position a mapped Martini protein in a planar bilayer"

    _allowed = {
        "method",
        "half_thickness",
        "z_offset",
        "tilt",
        "phi",
        "_task_dir",
        "_step_dir",
        "seed",
    }

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, self._allowed)
        method = str(config.get("method", "ppm")).lower()
        if method not in {"ppm", "manual"}:
            raise ModuleConfigError("CG orientation method must be ppm or manual")
        _finite(config.get("half_thickness", 1.4), "Hydrophobic half-thickness", 0.8, 2.5)
        if method == "manual":
            _finite(config.get("z_offset", 0.0), "Protein Z offset adjustment", -10.0, 10.0)
            _finite(config.get("tilt", 0.0), "Protein tilt adjustment", 0.0, 45.0)
            _finite(config.get("phi", 0.0), "Protein tilt direction", 0.0, 360.0)
        elif any(key in config for key in ("z_offset", "tilt", "phi")):
            raise ModuleConfigError(
                "Manual Z offset, tilt, and direction are not accepted by automatic CG orientation"
            )
        return True

    def run(self, system, config: dict) -> ModuleResult:
        if system.metadata.get("cg_environment") != "bilayer":
            raise ModuleConfigError("CG membrane orientation is available only for bilayer tasks")
        output = system.copy()
        if not output.metadata.get("cg_include_protein", True):
            output.metadata["cg_orientation"] = {"status": "not_applicable"}
            return ModuleResult(
                True, output, ["Protein orientation skipped for protein-free bilayer"]
            )
        if not output.component_by_kind(ComponentKind.PROTEIN):
            raise ModuleConfigError("Mapped Martini protein component is missing")

        method = str(config.get("method", "ppm")).lower()
        half_thickness = float(config.get("half_thickness", 1.4))
        if method == "ppm":
            metrics = apply_automatic_pose(output, half_thickness)
            logs = [
                "Automatic PPM-like Martini membrane positioning completed",
                f"Hydrophobic half-thickness: {half_thickness:.2f} nm",
                f"Protein Z offset: {metrics['z_offset_nm']:.2f} nm",
                f"Hydrophobic residues in core: {metrics['hydrophobic_core_fraction']:.0%}",
                (
                    f"Detected TM-window residues in core: {metrics['tm_core_fraction']:.0%}"
                    if metrics["tm_core_fraction"] is not None
                    else (
                        "No confident transmembrane window detected; "
                        "review peripheral-protein placement"
                    )
                ),
                (
                    "Review the membrane-plane preview; this local method is not "
                    "the external OPM/PPM server"
                ),
            ]
        else:
            base_metrics = apply_automatic_pose(output, half_thickness)
            metrics = apply_manual_adjustment(output, config, base_metrics)
            logs = [
                "Applied manual Martini membrane positioning",
                f"Protein Z adjustment: {metrics['z_adjustment_nm']:.2f} nm",
                f"Protein tilt adjustment: {metrics['tilt_degrees']:.1f} degrees",
                "Manual pose requires visual review against both membrane interfaces",
            ]

        output.metadata["cg_orientation"] = metrics
        output.metadata["cg_orientation_method"] = method
        oriented_path = task_step_dir(config) / "oriented_protein.pdb"
        write_cg_viewer_pdb(output, oriented_path, task_dir=task_root(config))
        relative = oriented_path.resolve().relative_to(task_root(config))
        output.metadata["cg_protein_pdb"] = str(relative)
        return ModuleResult(True, output, logs)


def apply_automatic_pose(system, half_thickness: float) -> dict:
    """Apply the deterministic PPM-like base pose to a CG system in place."""
    center = system.structure.center_of_geometry()
    system.structure.translate(-center)
    rotation, translation, metrics = _best_automatic_pose(system.structure, half_thickness)
    system.structure.rotate(rotation, center=np.zeros(3))
    system.structure.translate(translation)
    return dict(metrics)


def apply_manual_adjustment(system, config: dict, base_metrics: dict) -> dict:
    """Apply responsive manual controls relative to the deterministic base pose.

    The automatic pose is the stable reference shared by preview and Check.
    Manual controls then adjust insertion depth and tilt without exposing Euler
    rotations whose order is unclear to users.
    """
    z_offset = float(config.get("z_offset", 0.0))
    tilt = float(config.get("tilt", 0.0))
    phi = float(config.get("phi", 0.0))
    if tilt > 1e-12:
        phi_radians = math.radians(phi)
        axis = np.array([-math.sin(phi_radians), math.cos(phi_radians), 0.0])
        rotation = rotation_matrix_from_axis_angle(axis, math.radians(tilt))
        system.structure.rotate(rotation)
    system.structure.translate(np.array([0.0, 0.0, z_offset]))
    metrics = dict(base_metrics)
    metrics.update(
        {
            "method": "manual",
            "z_adjustment_nm": z_offset,
            "z_offset_nm": float(base_metrics.get("z_offset_nm", 0.0)) + z_offset,
            "tilt_degrees": tilt,
            "phi_degrees": phi,
        }
    )
    return metrics

"""Pipeline module for protein orientation step."""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.geometry.transforms import rotation_matrix_from_axis_angle
from gmxbuilder.modules import register_module


_HYDROPHOBIC_RESIDUES = {
    "ILE", "LEU", "VAL", "PHE", "TRP", "TYR", "MET", "CYS", "CYX",
}
_CHARGED_RESIDUES = {
    "ARG", "LYS", "ASP", "GLU", "HIP", "SEP", "TPO", "PTR", "S1P", "T1P", "Y1P",
    "MLZ", "MLY", "M3L", "2MR", "DA2", "OCS", "KCX",
}


def assess_membrane_orientation(
    system: System,
    half_thickness: float = 1.4,
) -> dict:
    """Return transparent, non-destructive checks for a membrane pose.

    The checks intentionally produce warnings rather than rejecting a pose:
    peripheral membrane proteins and channels with polar cores are legitimate.
    The report makes those scientifically meaningful exceptions visible to the
    user instead of allowing a numeric range check to imply physical validity.
    """
    protein_components = system.component_by_kind(ComponentKind.PROTEIN)
    if not protein_components:
        return {
            "status": "not_applicable",
            "warnings": ["No protein component is available for orientation checks."],
        }

    atom_indices = np.unique(np.concatenate([
        np.asarray(component.atom_indices, dtype=int)
        for component in protein_components
    ]))
    structure = system.structure
    residue_atoms: dict[tuple[str, int, str], list[int]] = {}
    for index in atom_indices:
        if index < 0 or index >= structure.num_atoms:
            continue
        key = (
            structure.chain_ids[index],
            int(structure.resids[index]),
            structure.resnames[index].strip().upper(),
        )
        residue_atoms.setdefault(key, []).append(int(index))

    residue_rows: list[tuple[tuple[str, int, str], str, np.ndarray]] = []
    for key, indices in residue_atoms.items():
        _chain, _resid, resname = key
        ca_index = next(
            (
                index for index in indices
                if structure.atom_names[index].strip().upper() == "CA"
            ),
            None,
        )
        coordinate = (
            structure.coordinates[ca_index]
            if ca_index is not None
            else structure.coordinates[indices].mean(axis=0)
        )
        residue_rows.append((key, resname, coordinate))

    if not residue_rows:
        return {
            "status": "warning",
            "warnings": ["No protein residue coordinates could be assessed."],
        }

    core_rows = [
        row for row in residue_rows if abs(float(row[2][2])) <= half_thickness
    ]
    hydrophobic_core = [
        row for row in core_rows if row[1] in _HYDROPHOBIC_RESIDUES
    ]
    charged_core = [
        row for row in core_rows if row[1] in _CHARGED_RESIDUES
    ]
    n_residues = len(residue_rows)
    n_core = len(core_rows)
    core_fraction = n_core / n_residues
    hydrophobic_core_fraction = len(hydrophobic_core) / max(n_core, 1)
    charged_core_fraction = len(charged_core) / max(n_core, 1)
    z_values = np.asarray([float(row[2][2]) for row in residue_rows])
    spans_bilayer = bool(
        z_values.min() <= -half_thickness and z_values.max() >= half_thickness
    )

    warnings: list[str] = []
    if n_core < 3:
        warnings.append(
            "Fewer than three residues occupy the hydrophobic core; the protein "
            "may not be inserted into the membrane."
        )
    elif hydrophobic_core_fraction < 0.35:
        warnings.append(
            f"Only {hydrophobic_core_fraction:.0%} of core residues are strongly "
            "hydrophobic; review the insertion depth and membrane-normal axis."
        )
    if charged_core_fraction > 0.15:
        warnings.append(
            f"{charged_core_fraction:.0%} of core residues are charged; verify "
            "that these residues form a pore or a known polar transmembrane site."
        )
    if core_fraction > 0.60:
        warnings.append(
            f"{core_fraction:.0%} of protein residues lie inside the hydrophobic "
            "core; this may indicate over-embedding of a globular domain."
        )
    if not spans_bilayer:
        warnings.append(
            "The protein does not span both membrane interfaces. This is valid "
            "for peripheral proteins but should be reviewed for a transmembrane protein."
        )

    # For alpha-helical membrane proteins, check the actual TM bundle rather
    # than reporting only the optional post-alignment refinement angle.  This
    # catches whole-protein PCA solutions that look numerically valid while a
    # GPCR bundle is visibly diagonal in the bilayer.
    from gmxbuilder.modules.membrane.orient import (
        _analyze_tm_helix_bundle,
        _get_residue_coords_and_energies,
        _membrane_transfer_score,
    )

    tm_analysis = _analyze_tm_helix_bundle(structure)
    tm_confident = bool(
        tm_analysis["axis"] is not None
        and tm_analysis["confidence"] >= 0.72
        and tm_analysis["window_count"] >= 3
    )
    tm_bundle_tilt = None
    median_tm_window_tilt = None
    non_tm_rows: list[tuple[tuple[str, int, str], str, np.ndarray]] = []
    non_tm_core_rows: list[tuple[tuple[str, int, str], str, np.ndarray]] = []
    if tm_confident:
        axis = np.asarray(tm_analysis["axis"], dtype=float)
        tm_bundle_tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
        window_axes = np.asarray(tm_analysis["window_axes"], dtype=float)
        window_tilts = np.degrees(
            np.arccos(np.clip(np.abs(window_axes[:, 2]), 0.0, 1.0))
        )
        median_tm_window_tilt = float(np.median(window_tilts))
        covered_keys = tm_analysis["covered_residue_keys"]
        non_tm_rows = [row for row in residue_rows if row[0] not in covered_keys]
        non_tm_core_rows = [
            row for row in non_tm_rows
            if abs(float(row[2][2])) <= half_thickness
        ]
        non_tm_core_fraction = len(non_tm_core_rows) / max(len(non_tm_rows), 1)
        if tm_bundle_tilt > 20.0:
            warnings.append(
                f"The transmembrane-helix bundle is tilted {tm_bundle_tilt:.1f}° "
                "from the membrane normal; review the protein angle."
            )
        if len(non_tm_rows) >= 5 and non_tm_core_fraction > 0.20:
            warnings.append(
                f"{non_tm_core_fraction:.0%} of non-transmembrane loop/domain "
                "residues lie inside the hydrophobic core; the protein may be "
                "embedded at the wrong depth."
            )

    applied_transfer_score = None
    optimal_transfer_score = system.metadata.get("_orientation_optimal_score")
    residue_coords, transfer_energies = _get_residue_coords_and_energies(structure)
    if residue_coords is not None and transfer_energies is not None:
        applied_transfer_score = float(
            _membrane_transfer_score(
                residue_coords,
                transfer_energies,
                0.0,
                half_thickness,
            )
        )
    if applied_transfer_score is not None and optimal_transfer_score is not None:
        optimal_transfer_score = float(optimal_transfer_score)
        score_delta = applied_transfer_score - optimal_transfer_score
        if score_delta > max(1.0, 0.10 * abs(optimal_transfer_score)):
            warnings.append(
                "The applied coordinates do not reproduce the optimizer's "
                "membrane-transfer score; rerun automatic orientation."
            )

    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "half_thickness_nm": round(float(half_thickness), 3),
        "residue_count": n_residues,
        "core_residue_count": n_core,
        "core_fraction": round(float(core_fraction), 4),
        "hydrophobic_core_fraction": round(float(hydrophobic_core_fraction), 4),
        "charged_core_fraction": round(float(charged_core_fraction), 4),
        "z_range_nm": [round(float(z_values.min()), 3), round(float(z_values.max()), 3)],
        "spans_bilayer": spans_bilayer,
        "tm_helix_window_count": int(tm_analysis["window_count"]),
        "tm_axis_confidence": round(float(tm_analysis["confidence"]), 4),
        "tm_bundle_tilt_degrees": (
            round(tm_bundle_tilt, 2) if tm_bundle_tilt is not None else None
        ),
        "median_tm_window_tilt_degrees": (
            round(median_tm_window_tilt, 2)
            if median_tm_window_tilt is not None else None
        ),
        "non_tm_residue_count": len(non_tm_rows) if tm_confident else None,
        "non_tm_core_residue_count": len(non_tm_core_rows) if tm_confident else None,
        "non_tm_core_fraction": (
            round(len(non_tm_core_rows) / max(len(non_tm_rows), 1), 4)
            if tm_confident else None
        ),
        "applied_transfer_score": (
            round(applied_transfer_score, 4)
            if applied_transfer_score is not None else None
        ),
        "optimal_transfer_score": (
            round(float(optimal_transfer_score), 4)
            if optimal_transfer_score is not None else None
        ),
    }


@register_module
class OrientModule(BaseModule):
    """Orient protein in membrane using PPM-like or manual settings."""

    name = "orient"
    description = "Determine protein orientation in membrane (PPM-like or manual)"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {"method", "z_offset", "tilt", "phi", "half_thickness", "seed"},
        )
        method = config.get("method", "ppm")
        if method not in ("ppm", "hmoment", "tmd", "pca", "com", "manual"):
            raise ModuleConfigError(f"Unknown orientation method: {method}")
        if config.get("half_thickness") is not None:
            try:
                half_thickness = float(config["half_thickness"])
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError("half_thickness must be a finite number") from exc
            if not np.isfinite(half_thickness) or not 0.5 <= half_thickness <= 3.0:
                raise ModuleConfigError(
                    "half_thickness must be between 0.5 and 3.0 nm"
                )
        if method == "manual":
            values: dict[str, float] = {}
            for key, default in (("tilt", 0.0), ("z_offset", 0.0), ("phi", 0.0)):
                try:
                    value = float(config.get(key, default))
                except (TypeError, ValueError) as exc:
                    raise ModuleConfigError(f"{key} must be a finite number") from exc
                if not np.isfinite(value):
                    raise ModuleConfigError(f"{key} must be a finite number")
                values[key] = value
            tilt = values["tilt"]
            if tilt < 0.0 or tilt > 45.0:
                raise ModuleConfigError(f"Tilt angle must be 0–45°, got {tilt:.1f}°")
            z_off = values["z_offset"]
            if z_off < -10.0 or z_off > 10.0:
                raise ModuleConfigError(f"Z-offset must be ±10 nm, got {z_off:.2f} nm")
            phi = values["phi"]
            if phi < 0.0 or phi > 360.0:
                raise ModuleConfigError(f"Phi angle must be 0–360°, got {phi:.1f}°")
        elif any(key in config for key in ("z_offset", "tilt", "phi")):
            raise ModuleConfigError(
                "z_offset, tilt and phi are manual-orientation inputs; "
                "automatic algorithms compute them and do not accept overrides"
            )
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        method = config.get("method", "ppm")
        half_thickness = (
            float(config["half_thickness"])
            if config.get("half_thickness") is not None
            else None
        )
        has_protein = bool(system.component_by_kind(ComponentKind.PROTEIN))
        log = []
        optimal_score = None

        if has_protein:
            if method == "manual":
                z_off = float(config.get("z_offset", 0.0))
                tilt = float(config.get("tilt", 0.0))
                phi  = float(config.get("phi", 0.0))
                # Use PPM to find the best membrane-normal axis (matching the
                # orientation the user sees in the 3D viewer).  Only the axis
                # rotation is applied — manual z_offset/tilt/phi REPLACE PPM's
                # computed values rather than adding on top of them.
                from gmxbuilder.modules.membrane.orient import _find_best_ppm_orientation
                from gmxbuilder.geometry.transforms import rotation_matrix_from_vectors
                best_axis, _, _, _, _, _ = _find_best_ppm_orientation(
                    system.structure, half_thickness=half_thickness)
                rot = rotation_matrix_from_vectors(best_axis, np.array([0, 0, 1]))
                system.structure.rotate(rot)
                # Apply manual Z offset (replaces PPM's z_offset)
                system.structure.translate(np.array([0.0, 0.0, z_off]))
                # Apply manual tilt (replaces PPM's tilt)
                if tilt > 0.01:
                    phi_rad = np.radians(phi)
                    axis = np.array([-np.sin(phi_rad), np.cos(phi_rad), 0.0])
                    R = rotation_matrix_from_axis_angle(axis, np.radians(tilt))
                    system.structure.rotate(R)
                # Store orient params for downstream steps (MembraneBuilder)
                system.metadata["_orient_params"] = {
                    "z_offset": float(z_off),
                    "tilt": float(tilt),
                    "phi": float(phi),
                }
                log.append(
                    f"Manual orientation: Z-offset={z_off:.2f} nm, "
                    f"tilt={tilt:.1f}°, phi={phi:.0f}°"
                )
            else:
                from gmxbuilder.modules.membrane.orient import apply_auto_orientation

                orientation = apply_auto_orientation(
                    system.structure,
                    method=method,
                    half_thickness=half_thickness,
                )
                z_off = orientation["z_offset"]
                tilt_angle = orientation["tilt_radians"]
                tilt_phi = orientation["tilt_phi_radians"]
                if np.isfinite(orientation["score"]):
                    optimal_score = orientation["score"]
                system.metadata["_orient_params"] = {
                    "z_offset": float(z_off),
                    "tilt": float(np.degrees(tilt_angle)),
                    # Manual phi describes the tilt-axis direction, not the
                    # optional azimuthal rotation around membrane-normal Z.
                    "phi": float(np.degrees(tilt_phi)),
                }
                system.metadata["_orientation_azimuth_degrees"] = float(
                    np.degrees(orientation["azimuth_radians"])
                )
                log.append(
                    f"Auto orientation ({method}): "
                    f"Z-offset={z_off:.2f} nm, tilt={np.degrees(tilt_angle):.1f}°"
                )
            # Mark as oriented — MembraneBuilder will skip re-orientation
            system.metadata["_oriented"] = True
            system.metadata["_orientation_method"] = method
            system.metadata["_orientation_half_thickness_nm"] = (
                float(half_thickness) if half_thickness is not None else 1.4
            )
            if optimal_score is None:
                system.metadata.pop("_orientation_optimal_score", None)
            else:
                system.metadata["_orientation_optimal_score"] = optimal_score
            quality = assess_membrane_orientation(
                system,
                half_thickness=(half_thickness if half_thickness is not None else 1.4),
            )
            system.metadata["_orientation_quality"] = quality
            if quality.get("tm_bundle_tilt_degrees") is not None:
                log.append(
                    "Measured TM-bundle tilt: "
                    f"{quality['tm_bundle_tilt_degrees']:.1f}° from membrane normal"
                )
            for warning in quality.get("warnings", []):
                log.append(f"Orientation warning: {warning}")
        else:
            log.append("No protein detected; orientation skipped")

        return ModuleResult(success=True, system=system, log=log)

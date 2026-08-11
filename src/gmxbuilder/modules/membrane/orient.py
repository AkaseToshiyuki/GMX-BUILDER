"""Protein orientation for membrane embedding — PPM 2.0-like algorithm.

Implements an automatic membrane positioning method inspired by the
OPM/PPM (Positioning of Proteins in Membranes) approach:
  1. Assign per-residue transfer free energy (ΔG) from the
     Wimley-White whole-residue hydrophobicity scale.
  2. Scan Z-offsets and tilt angles to find the orientation that
     minimizes the total transfer energy of the protein in the membrane.
  3. Return the optimal rotation matrix and Z-translation.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.geometry.transforms import (
    rotation_matrix_from_axis_angle,
    rotation_matrix_from_vectors,
)
from gmxbuilder.geometry.align import compute_principal_axes, orient_protein_to_membrane


# ---------------------------------------------------------------------------
# Wimley-White (1996) whole-residue water→POPC transfer free energies
# (kcal/mol).  Negative = favours the membrane (hydrophobic).
# Values from Wimley, Hristova & White, Biochemistry 35:5109 (1996) and
# Jayasinghe, Hristova & White, JMB 312:927 (2001).
# ---------------------------------------------------------------------------
_WW_TRANSFER: dict[str, float] = {
    "ALA":  0.17,  "ARG":  0.81,  "ASN":  0.42,  "ASP":  1.23,
    "CYS": -0.24,  "GLN":  0.58,  "GLU":  0.11,  "GLY":  0.01,
    "HIS":  0.96,  "ILE": -0.31,  "LEU": -0.56,  "LYS":  0.99,
    "MET": -0.22,  "PHE": -1.13,  "PRO":  0.45,  "SER":  0.13,
    "THR":  0.14,  "TRP": -1.85,  "TYR": -0.94,  "VAL": -0.07,
    # Protonated variants
    "ASH":  1.23,  "GLH":  0.11,  "CYX": -0.24,
    "HID":  0.96,  "HIE":  0.96,  "HIP":  0.96,
    "LYN":  0.99,
    # PTM / modified residues (estimated from parent + group hydrophobicity)
    "SEP":  0.63,  "TPO":  0.64,  "PTR": -0.44,  # phosphorylated → strongly hydrophilic
    "S1P": 0.63, "T1P": 0.64, "Y1P": -0.44,
    "ALY":  0.49,  "SLY":  0.49,  "BLY":  0.49,  # acylated LYS → less hydrophilic
    "CLY":  0.49,  "CRY":  0.49,  "PLY":  0.49,  "GRY":  0.49,
    "KME":  0.99,  "KM2":  0.99,  "KM3":  0.99,  # legacy catalogue labels
    "MLZ":  0.99,  "MLY":  0.99,  "M3L":  0.99,  # methylated LYS retains +1
    "RME":  0.81,  "RM2":  0.81, "2MR": 0.81, "DA2": 0.81,
    "CSO":  0.26,  "CSD":  0.76,  "CSX":  1.26,  # oxidized CYS → progressively hydrophilic
    "CSN": -0.24,  "SNC": -0.24, "SMC": -0.24,
    "OCS": 1.26,  # anionic cysteinesulfonate is strongly water-facing
    "CIR":  0.81,  # citrulline → similar to ARG
    "TYS": -1.44,  # sulfated TYR → more hydrophilic than TYR
    "SAC":  0.13,  "OAS": 0.13, "TAC":  0.14,
    "GCS":  0.63,  "GCT":  0.64,  # O-GlcNAc → hydrophilic
    "MSE": -0.22,  # selenomethionine → similar to MET
    "WOH": -1.35,  # oxidized TRP → slightly less hydrophobic
    "PCA":  0.01,  # pyroglutamate → similar to GLN
    "KCX": 1.49,  # anionic N-zeta-carboxylysine
    "NIY": -0.44,  # neutral nitro group reduces TYR membrane preference
}

# Default membrane hydrophobic half-thickness for PPM scoring.
#
# This is the distance from the bilayer midplane to the boundary between
# the hydrophobic core and the headgroup region — the interface where
# residue side-chains transition from "in membrane" to "in water" in the
# Wimley-White transfer-free-energy model.
#
# Value derived from experimental OPM database hydrophobic thicknesses
# (Kučerka et al., BBA 1808:2761, 2011) averaged over 9 common
# phospholipids (DLPC–POPS): mean = 1.40 nm, POPC = 1.42 nm.
# Rounded to 1.4 nm as a representative value for a generic lipid
# bilayer.  Individual lipid-specific values (±0.1–0.2 nm) have only
# minor effects on the predicted Z-offset and axis selection.
_DEFAULT_MEMBRANE_HALF_THICKNESS = 1.4  # nm

# Transition width for the continuous sigmoid membrane profile.
# Controls how abruptly the environment changes from "membrane core" to
# "bulk water" across the interface.  PPM 2.0 uses a comparable smooth
# dielectric profile; 0.3 nm gives a ~1.2 nm transition zone (10%–90%),
# matching the headgroup-region thickness of common phospholipids.
_MEMBRANE_TRANSITION_WIDTH = 0.3  # nm


def _get_residue_coords_and_energies(structure: Structure):
    """Extract per-residue Cα (or centroid) coordinates and ΔG values."""
    n = structure.num_atoms
    if n == 0:
        return None, None

    chain_ids = getattr(structure, "chain_ids", None) or []

    res_groups: dict[tuple[str, int, str], list[int]] = {}
    for i in range(n):
        rname = structure.resnames[i] if i < len(structure.resnames) else "UNK"
        rid = structure.resids[i] if i < len(structure.resids) else i
        cid = chain_ids[i] if i < len(chain_ids) else ""
        key = (rname, rid, cid)
        if key not in res_groups:
            res_groups[key] = []
        res_groups[key].append(i)

    coords_list = []
    energies = []
    for (rname, rid, _cid), indices in res_groups.items():
        # Use CA if available, else centroid
        ca_idx = None
        for idx in indices:
            aname = structure.atom_names[idx] if idx < len(structure.atom_names) else ""
            if aname.strip().upper() == "CA":
                ca_idx = idx
                break
        if ca_idx is not None:
            pos = structure.coordinates[ca_idx]
        else:
            pos = structure.coordinates[indices].mean(axis=0)
        coords_list.append(pos)
        energies.append(_WW_TRANSFER.get(rname, 0.0))

    return np.array(coords_list), np.array(energies)


def _membrane_transfer_score(
    coords: np.ndarray,
    energies: np.ndarray,
    z_offset: float,
    half_thickness: float,
    transition_width: float | None = None,
) -> float:
    """Total transfer score for a given Z-offset using a continuous membrane profile.

    Instead of a binary "inside / outside" boundary, each residue receives
    a membrane-proximity weight w(z) ∈ [0, 1] via a sigmoid centred at
    *half_thickness*.  This makes the energy landscape smooth and prevents
    the axis selection from flipping on borderline proteins when the
    membrane thickness changes by only 0.1 nm.

    Three penalties guard against non-physical orientations:

    1. Hydrophobic residues (ΔG < −0.5) left in water are penalised.
    2. Hydrophilic residues (ΔG > 0.3) penetrating the core are penalised
       (amplified for the low-dielectric environment).
    3. Over‑packing penalises configurations where >50 % of the total
       residue weight is inside the membrane.
    """
    if transition_width is None:
        transition_width = _MEMBRANE_TRANSITION_WIDTH

    z_rel = coords[:, 2] + z_offset

    # Continuous sigmoid weight: 1 = deep core, 0 = bulk water
    # Clip the exponent argument to avoid overflow at extreme |z|.
    d = np.abs(z_rel) - half_thickness
    d_scaled = np.clip(d / transition_width, -100.0, 100.0)
    weights = 1.0 / (1.0 + np.exp(d_scaled))

    # Core score: transfer energy weighted by membrane proximity
    score = float((energies * weights).sum())

    # Penalty 1 — strongly hydrophobic residues exposed to water
    hphob = energies < -0.5
    if hphob.any():
        water_weight = 1.0 - weights
        score += float((np.abs(energies[hphob]) * water_weight[hphob]).sum()) * 0.5

    # Penalty 2 — hydrophilic / charged residues forced into the
    # low-dielectric core (amplified ×2)
    hphil = energies > 0.3
    if hphil.any():
        score += float((energies[hphil] * weights[hphil]).sum()) * 2.0

    # Penalty 3 — over‑pack guard
    frac_inside = float(weights.sum()) / max(float(len(energies)), 1.0)
    overpack = max(0.0, frac_inside - 0.50)
    score += overpack * overpack * 200.0

    return score


def _scan_ppm_z_and_tilt(
    coords: np.ndarray,
    energies: np.ndarray,
    half_thickness: float,
    max_tilt: float,
    n_scans: int,
    tilt_improvement_threshold: float,
    rotation_center: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float, float]:
    """Run Z-offset + tilt scan on residue coords (already axis-aligned).

    Parameters
    ----------
    coords : (N, 3) ndarray — residue CA/centroid coords in aligned frame.
    energies : (N,) ndarray — per-residue ΔG values.
    half_thickness : float — membrane half-thickness (nm).
    max_tilt : float — max tilt angle (degrees).
    n_scans : int — grid points for scans.
    tilt_improvement_threshold : float — score improvement fraction for tilt.

    Returns
    -------
    z_offset : float          nm (translation applied directly to coordinates)
    tilt_vec : (3,) ndarray
    tilt_angle : float        radians
    best_score : float
    """
    # ---- 1. Z-offset scan ----
    # Scan must cover the full protein Z-extent after axis alignment,
    # otherwise the membrane mid-plane cannot reach the TM region when
    # the protein COM is far from z=0 (common for asymmetric GPCRs).
    z_all = coords[:, 2]
    z_span = max(abs(z_all.min()), abs(z_all.max()), half_thickness * 1.5)
    z_values = np.linspace(-z_span, z_span, n_scans)
    best_z = 0.0
    best_score = float("inf")
    for z in z_values:
        s = _membrane_transfer_score(coords, energies, z, half_thickness)
        if s < best_score:
            best_score = s
            best_z = z

    # _membrane_transfer_score evaluates ``coords[:, 2] + z_offset``.
    # Therefore the minimising scan value is already the translation to apply;
    # negating it moves the protein to the opposite side of the membrane.
    z_offset = best_z

    # ---- 2. Tilt refinement ----
    best_tilt_deg = 0.0
    best_tilt_score = best_score
    tilt_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    if rotation_center is None:
        rotation_center = coords.mean(axis=0)

    for axis in [np.array([1.0, 0.0, 0.0], dtype=float),
                 np.array([0.0, 1.0, 0.0], dtype=float)]:
        for deg in np.linspace(0, max_tilt, n_scans // 2):
            angle = np.radians(deg)
            R = rotation_matrix_from_axis_angle(axis, angle)
            tilted = (coords - rotation_center) @ R.T + rotation_center
            s = _membrane_transfer_score(tilted, energies, best_z, half_thickness)
            required_improvement = max(
                abs(best_tilt_score) * (1.0 - tilt_improvement_threshold),
                1e-6,
            )
            if best_tilt_score - s > required_improvement:
                best_tilt_score = s
                best_tilt_deg = deg
                tilt_vec = axis.copy()

    return z_offset, tilt_vec, np.radians(best_tilt_deg), best_tilt_score


def _find_best_ppm_orientation(
    structure: Structure,
    half_thickness: float | None = None,
    max_tilt: float = 30.0,
    n_scans: int = 50,
    tilt_improvement_threshold: float = 0.95,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float]:
    """Try all 3 principal axes, returning the best PPM orientation.

    For each principal axis of the protein, aligns it to Z and runs
    the full PPM Z-offset + tilt + Z-rotation scan.  Returns the
    axis (and its associated transforms) that minimises the membrane
    transfer energy.

    The Z-rotation (azimuthal) scan is critical for GPCRs: after
    aligning the TM axis to Z, the protein can still be rotated
    around Z to optimise which residues face the membrane lipids.
    Each TM helix has a hydrophobic face (outward) and a polar face
    (inward), and the correct azimuthal angle aligns the hydrophobic
    faces with the membrane.

    Returns
    -------
    best_axis : (3,) ndarray
        Principal axis (in *original* coordinates) to align to Z.
    z_offset : float
        Z translation (nm) to apply AFTER aligning best_axis to Z.
    tilt_vec : (3,) ndarray
        Tilt axis (in the aligned frame).
    tilt_angle : float
        Tilt angle (radians).
    phi_z : float
        Azimuthal Z-rotation angle (radians).
    best_score : float
        Membrane transfer energy (lower = better).
    """
    if half_thickness is None:
        half_thickness = _DEFAULT_MEMBRANE_HALF_THICKNESS

    # Standard PPM 2.0: per-residue CA coords with Wimley-White energies
    residue_coords, energies = _get_residue_coords_and_energies(structure)
    if residue_coords is None or len(residue_coords) < 3:
        com_z = structure.center_of_geometry()[2]
        return (np.array([0.0, 0.0, 1.0], dtype=float),
                -com_z, np.array([1.0, 0.0, 0.0], dtype=float),
                0.0, 0.0, float("inf"))

    # A confident hydrophobic alpha-helical bundle defines the membrane normal
    # more directly than whole-protein PCA, which is biased by soluble domains.
    tm_analysis = _analyze_tm_helix_bundle(structure)
    if (
        tm_analysis["axis"] is not None
        and tm_analysis["confidence"] >= 0.72
        and tm_analysis["window_count"] >= 3
    ):
        axes = np.asarray([tm_analysis["axis"]])
    else:
        axes = compute_principal_axes(structure.coordinates)
    target_z = np.array([0.0, 0.0, 1.0], dtype=float)
    rotation_center = structure.center_of_geometry()

    best_overall_score = float("inf")
    best_result = (axes[0].copy(), 0.0,
                   np.array([1.0, 0.0, 0.0], dtype=float), 0.0, 0.0)

    for axis_idx in range(len(axes)):
        cand_axis = axes[axis_idx].copy()
        if cand_axis[2] < 0:
            cand_axis = -cand_axis

        # Align candidate axis to Z
        rot = rotation_matrix_from_vectors(cand_axis, target_z)
        rotated_res = (
            (residue_coords - rotation_center) @ rot.T + rotation_center
        )

        # Z-offset + tilt scan
        z_off, tilt_vec, tilt_angle, score = _scan_ppm_z_and_tilt(
            rotated_res, energies, half_thickness,
            max_tilt, n_scans, tilt_improvement_threshold,
            rotation_center=rotation_center,
        )

        if score < best_overall_score:
            best_overall_score = score
            # NOTE: phi_z hardcoded to 0.0 — azimuthal Z-rotation scan
            # (which face of the protein faces the membrane) is not yet
            # implemented. A hydrophobic-moment azimuthal scan can be added
            # as a follow-on refinement.
            best_result = (cand_axis, z_off, tilt_vec, tilt_angle, 0.0)

    return (*best_result, best_overall_score)


def compute_ppm_orientation(
    structure: Structure,
    half_thickness: float | None = None,
    max_tilt: float = 30.0,
    n_scans: int = 50,
    tilt_improvement_threshold: float = 0.95,
) -> tuple[float, np.ndarray]:
    """Find optimal Z-offset and tilt angle for the protein in a membrane.

    Tries all 3 principal axes as candidates for the membrane normal
    and selects the orientation that minimises the Wimley-White
    transfer energy.  This is robust for GPCRs and other proteins
    where the longest PCA axis may not coincide with the membrane
    normal.

    Parameters
    ----------
    structure : Structure
        Protein coordinates (in any orientation).
    half_thickness : float or None
        Half the hydrophobic thickness of the membrane (nm).  If None,
        uses 1.5 nm (POPC).
    max_tilt : float
        Maximum tilt angle to scan (degrees).
    n_scans : int
        Number of grid points for Z and angle scans.
    tilt_improvement_threshold : float
        Fraction of current best score required to accept a new tilt
        (0.95 = 5% improvement). Lower values = more tilt-sensitive.

    Returns
    -------
    z_offset : float
        Optimal Z translation to apply (nm) — relative to the structure
        AFTER the best principal axis has been aligned to Z.
    tilt_vec : (3,) ndarray
        Unit vector for the optimal tilt axis (0 if no tilt needed).
    tilt_angle : float
        Optimal tilt angle (radians).
    """
    best_axis, z_offset, tilt_vec, tilt_angle, phi_z, _ = _find_best_ppm_orientation(
        structure,
        half_thickness=half_thickness,
        max_tilt=max_tilt,
        n_scans=n_scans,
        tilt_improvement_threshold=tilt_improvement_threshold,
    )
    return z_offset, tilt_vec, tilt_angle


def orient_protein(
    structure: Structure,
    method: str = "ppm",
    target_axis: np.ndarray = np.array([0, 0, 1]),
    half_thickness: float | None = None,
) -> Structure:
    """Orient protein so its membrane-spanning region aligns with Z=0.

    For the "ppm" and "tmd" methods this now tests **all three principal
    axes** as candidates for the membrane normal — PCA alone can pick
    the wrong axis for GPCRs where the largest-variance direction may
    lie in the membrane plane.

    Parameters
    ----------
    structure : Structure
        Protein structure to orient. Modified in-place.
    method : str
        "ppm"     — PPM 2.0-like multi-axis scan (default).
        "hmoment" — Hydrophobic-moment vector (Eisenberg scale) → Z,
                    then Z-offset scan.
        "tmd"     — Kyte-Doolittle TM-helix detection with multi-axis
                    scan.
        "pca"     — PCA-based alignment (longest axis → Z, fallback).
        "com"     — simple COM alignment.
    target_axis : (3,) ndarray
        Desired direction for the membrane normal (default Z).

    Returns
    -------
    structure : Structure (same object, mutated)
    """
    coords = structure.coordinates
    if len(coords) < 2:
        return structure

    log = []

    if method == "ppm":
        # Multi-axis PPM: try all 3 principal axes, Z-scan, tilt, Z-rotation
        best_axis, z_off, tilt_vec, tilt_angle, phi_z, score = \
            _find_best_ppm_orientation(
                structure, half_thickness=half_thickness,
            )

        # 1. Align the best principal axis to Z
        rot = rotation_matrix_from_vectors(best_axis, target_axis)
        structure.rotate(rot)

        # 2. Apply Z-offset
        if abs(z_off) > 0.001:
            structure.translate(np.array([0.0, 0.0, z_off]))

        # 3. Apply tilt refinement
        if tilt_angle > 0.01:
            R = rotation_matrix_from_axis_angle(tilt_vec, tilt_angle)
            structure.rotate(R)

        # 4. Apply Z-rotation (azimuthal) refinement
        if phi_z > 0.01:
            cos_p, sin_p = np.cos(phi_z), np.sin(phi_z)
            Rz = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
            structure.rotate(Rz)

        log.append(
            f"PPM: axis→Z, z_off={z_off:.2f} nm, "
            f"tilt={np.degrees(tilt_angle):.1f}°, "
            f"phi={np.degrees(phi_z):.0f}°, score={score:.1f}"
        )

    elif method == "hmoment":
        z_off, moment_dir, _ = compute_hydrophobic_moment_orientation(
            structure, half_thickness=half_thickness,
        )

        # Align hydrophobic moment to Z
        rot = rotation_matrix_from_vectors(moment_dir, target_axis)
        structure.rotate(rot)

        if abs(z_off) > 0.001:
            structure.translate(np.array([0.0, 0.0, z_off]))

        log.append(
            f"H-Moment: μ→Z, z_off={z_off:.2f} nm, "
            f"|μ|={np.linalg.norm(moment_dir):.2f}"
        )

    elif method == "tmd":
        z_off, best_axis, _ = compute_tmd_orientation(
            structure, half_thickness=half_thickness,
        )

        # Align the best principal axis to Z
        rot = rotation_matrix_from_vectors(best_axis, target_axis)
        structure.rotate(rot)

        if abs(z_off) > 0.001:
            structure.translate(np.array([0.0, 0.0, z_off]))

        log.append(
            f"TMD: axis→Z, z_off={z_off:.2f} nm, "
            f"{best_axis[0]:.2f},{best_axis[1]:.2f},{best_axis[2]:.2f}"
        )

    elif method == "pca":
        rot = orient_protein_to_membrane(coords, method="pca", target_axis=target_axis)
        structure.rotate(rot)
        log.append("PCA: principal axis aligned to membrane normal (Z)")

    elif method == "com":
        com_z = coords[:, 2].mean()
        structure.translate(np.array([0.0, 0.0, -com_z]))
        log.append(f"COM: center-of-mass at z=0 (offset={-com_z:.2f} nm)")

    return structure


def compute_embedding_depth(
    protein_coords: np.ndarray,
    bilayer_thickness: float,
    method: str = "ppm",
    half_thickness: float | None = None,
) -> float:
    """Compute Z-translation needed to embed the protein in the membrane.

    Parameters
    ----------
    protein_coords : (N, 3) ndarray
    bilayer_thickness : float
        Bilayer hydrophobic thickness (nm).
    method : str
        "ppm" (default), "com", "hydrophobic".
    half_thickness : float or None
        Half the hydrophobic thickness.

    Returns
    -------
    z_offset : float
    """
    if method in ("ppm", "hydrophobic"):
        # Create a temporary structure for PPM
        from gmxbuilder.core.structure import Structure as S
        # NOTE: Dummy 10 nm box — fine for non-PBC scoring, but
        # coordinate wrapping may shift atoms if used in PBC context.
        tmp = S(coordinates=protein_coords, box_vectors=np.eye(3) * 10.0)
        if half_thickness is None:
            half_thickness = bilayer_thickness * 0.4  # approximate hydrophobic half
        z_off, _, _ = compute_ppm_orientation(tmp, half_thickness=half_thickness)
        return z_off
    if method == "com":
        return -protein_coords[:, 2].mean()
    raise ValueError(f"Unknown embedding method: {method}")


# =============================================================================
# Additional orientation algorithms
# =============================================================================

# Eisenberg consensus hydrophobicity scale (Eisenberg et al., JMB 179:125, 1984).
# Positive = hydrophobic.  Used for the hydrophobic-moment method.
_EISENBERG: dict[str, float] = {
    "ALA": 0.62, "ARG": -2.53, "ASN": -0.78, "ASP": -0.90,
    "CYS": 0.29, "GLN": -0.85, "GLU": -0.74, "GLY": 0.48,
    "HIS": -0.40, "ILE": 1.38, "LEU": 1.06, "LYS": -1.50,
    "MET": 0.64, "PHE": 1.19, "PRO": 0.12, "SER": -0.18,
    "THR": -0.05, "TRP": 0.81, "TYR": 0.26, "VAL": 1.08,
}

# Kyte-Doolittle hydropathy index (Kyte & Doolittle, JMB 157:105, 1982).
# Used for transmembrane-domain detection via sliding-window scan.
_KD: dict[str, float] = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5,
    "CYS": 2.5, "GLN": -3.5, "GLU": -3.5, "GLY": -0.4,
    "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8,
    "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

# KD threshold: window average ≥ this → predicted TM helix
_TMD_KD_THRESHOLD = 1.6
# Sliding-window width (residues) for TM-helix detection
_TMD_WINDOW = 19


def _residue_records(
    structure: Structure,
) -> list[tuple[str, int, str, np.ndarray]]:
    """Return ordered ``(resname, resid, chain, CA/centroid)`` records."""
    chain_ids = getattr(structure, "chain_ids", None) or []
    groups: dict[tuple[str, int, str], list[int]] = {}
    for i in range(structure.num_atoms):
        rname = structure.resnames[i] if i < len(structure.resnames) else "UNK"
        rid = structure.resids[i] if i < len(structure.resids) else i
        cid = chain_ids[i] if i < len(chain_ids) else ""
        key = (rname, rid, cid)
        groups.setdefault(key, []).append(i)

    records: list[tuple[str, int, str, np.ndarray]] = []
    for (rname, rid, cid), indices in groups.items():
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
        records.append((rname, int(rid), cid, coordinate))
    return records


def _residue_sequence(structure: Structure) -> list[tuple[str, int, np.ndarray]]:
    """Return ordered list of (resname, resid, CA/centroid) by sequence."""
    return [
        (rname, resid, coordinate)
        for rname, resid, _chain, coordinate in _residue_records(structure)
    ]


def _analyze_tm_helix_bundle(
    structure: Structure,
    *,
    window_size: int = 13,
    hydropathy_threshold: float = 1.0,
    linearity_threshold: float = 0.70,
) -> dict:
    """Estimate a common membrane normal from hydrophobic helical windows.

    Whole-protein PCA is easily biased by extracellular and cytoplasmic
    domains.  This analysis instead selects continuous, hydrophobic and
    approximately linear C-alpha windows, then obtains their unoriented common
    axis from a weighted orientation tensor.  It works for alpha-helical
    single- and multi-pass proteins; beta barrels and proteins without a
    confident helical signal deliberately fall back to the generic PPM scan.
    """
    records = _residue_records(structure)
    if len(records) < window_size:
        return {
            "axis": None,
            "confidence": 0.0,
            "window_count": 0,
            "covered_residue_keys": set(),
            "window_axes": np.empty((0, 3), dtype=float),
        }

    axes: list[np.ndarray] = []
    weights: list[float] = []
    covered_keys: set[tuple[str, int, str]] = set()
    by_chain: dict[str, list[tuple[str, int, str, np.ndarray]]] = {}
    for record in records:
        by_chain.setdefault(record[2], []).append(record)

    for chain_records in by_chain.values():
        if len(chain_records) < window_size:
            continue
        coords = np.asarray([record[3] for record in chain_records])
        hydropathy = np.asarray([
            _KD.get(record[0].strip().upper(), 0.0)
            for record in chain_records
        ])
        for start in range(len(chain_records) - window_size + 1):
            stop = start + window_size
            window_coords = coords[start:stop]
            steps = np.linalg.norm(np.diff(window_coords, axis=0), axis=1)
            # Reject chain breaks and non-protein centroid fallbacks.
            if np.any(steps < 0.25) or np.any(steps > 0.50):
                continue
            mean_hydropathy = float(hydropathy[start:stop].mean())
            if mean_hydropathy < hydropathy_threshold:
                continue
            centered = window_coords - window_coords.mean(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
            total_variance = float(eigenvalues.sum())
            if total_variance <= 1e-12:
                continue
            linearity = float(eigenvalues[-1] / total_variance)
            end_to_end = float(np.linalg.norm(window_coords[-1] - window_coords[0]))
            if (
                linearity < linearity_threshold
                or end_to_end < 0.09 * (window_size - 1)
            ):
                continue
            axes.append(eigenvectors[:, -1])
            weights.append(linearity * mean_hydropathy)
            covered_keys.update(
                (record[2], record[1], record[0].strip().upper())
                for record in chain_records[start:stop]
            )

    if not axes:
        return {
            "axis": None,
            "confidence": 0.0,
            "window_count": 0,
            "covered_residue_keys": set(),
            "window_axes": np.empty((0, 3), dtype=float),
        }

    window_axes = np.asarray(axes)
    tensor = np.zeros((3, 3), dtype=float)
    for axis, weight in zip(window_axes, weights):
        tensor += float(weight) * np.outer(axis, axis)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    axis = eigenvectors[:, -1]
    if axis[2] < 0:
        axis = -axis
    confidence = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    return {
        "axis": axis,
        "confidence": confidence,
        "window_count": len(window_axes),
        "covered_residue_keys": covered_keys,
        "window_axes": window_axes,
    }


# ---------------------------------------------------------------------------
# Hydrophobic Moment
# ---------------------------------------------------------------------------

def compute_hydrophobic_moment_orientation(
    structure: Structure,
    half_thickness: float | None = None,
) -> tuple[float, np.ndarray, float]:
    """Orient via the 3D hydrophobic-moment vector (Eisenberg scale).

    Computes the hydrophobic-moment vector μ = Σ Hᵢ · ŝᵢ  where ŝᵢ is
    the unit vector from the geometric centre to residue i.  The protein
    is rotated so that μ aligns with the membrane normal (Z).  A
    subsequent Z-offset scan refines the embedding depth.

    Returns (z_offset_nm, tilt_vec, tilt_angle_rad).
    """
    if half_thickness is None:
        half_thickness = _DEFAULT_MEMBRANE_HALF_THICKNESS

    coords_all = structure.coordinates
    if len(coords_all) < 3:
        return 0.0, np.array([1.0, 0.0, 0.0]), 0.0

    # Per-residue CA coords + Eisenberg hydrophobicity
    seq = _residue_sequence(structure)
    if len(seq) < 3:
        return 0.0, np.array([1.0, 0.0, 0.0]), 0.0

    center = np.mean([c for _, _, c in seq], axis=0)
    moment = np.zeros(3, dtype=float)
    for rname, _, coord in seq:
        h = _EISENBERG.get(rname, 0.0)
        d = coord - center
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            moment += h * d / norm

    moment_norm = np.linalg.norm(moment)
    if moment_norm < 1e-8:
        # Fall back to COM
        return compute_com_orientation(structure, half_thickness)

    # Align hydrophobic moment to Z axis
    z_axis = np.array([0.0, 0.0, 1.0])
    moment_dir = moment / moment_norm
    if np.abs(np.dot(moment_dir, z_axis)) > 0.999:
        R = np.eye(3)
    else:
        from gmxbuilder.geometry.transforms import rotation_matrix_from_vectors
        R = rotation_matrix_from_vectors(moment_dir, z_axis)

    # Apply rotation to a copy for Z-scan
    rotation_center = structure.center_of_geometry()
    rotated = (coords_all - rotation_center) @ R.T + rotation_center
    tmp = Structure(coordinates=rotated, box_vectors=structure.box_vectors.copy(),
                    atom_names=structure.atom_names.copy() if len(structure.atom_names) else None,
                    resnames=structure.resnames.copy() if len(structure.resnames) else None,
                    resids=structure.resids.copy() if len(structure.resids) else None,
                    chain_ids=structure.chain_ids.copy() if len(structure.chain_ids) else None)

    # Scan Z-offset using Wimley-White score — full protein Z-extent
    z_all = tmp.coordinates[:, 2]
    z_range = max(abs(z_all.min()), abs(z_all.max()), half_thickness * 1.5)
    n_scans = 50
    z_values = np.linspace(-z_range, z_range, n_scans)
    best_z = 0.0
    best_score = float("inf")
    res_coords, energies = _get_residue_coords_and_energies(tmp)
    if energies is not None and res_coords is not None:
        for z in z_values:
            s = _membrane_transfer_score(res_coords, energies, z, half_thickness)
            if s < best_score:
                best_score = s
                best_z = z

    z_offset = best_z
    return z_offset, moment_dir, 0.0


# ---------------------------------------------------------------------------
# TMD (Trans-Membrane Domain) detection
# ---------------------------------------------------------------------------

def _score_tmd_for_axis(
    structure: Structure,
    axis: np.ndarray,
    half_thickness: float,
) -> tuple[float, float]:
    """Score how well a candidate principal axis serves as the membrane normal.

    Rotates the structure so *axis* aligns to Z, runs the Kyte-Doolittle
    sliding window, and returns (z_offset, quality_score) where a **lower**
    quality_score means TM residues are more tightly clustered in Z.

    Returns
    -------
    z_offset : float       nm — membrane-midplane placement
    quality  : float       lower = tighter TM-residue Z-clustering
    """
    from gmxbuilder.geometry.transforms import rotation_matrix_from_vectors
    target_z = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.abs(np.dot(axis, target_z)) > 0.999:
        rot = np.eye(3)
    else:
        rot = rotation_matrix_from_vectors(axis, target_z)

    rotation_center = structure.center_of_geometry()
    coords_all = (
        (structure.coordinates - rotation_center) @ rot.T + rotation_center
    )
    tmp = Structure(
        coordinates=coords_all,
        box_vectors=structure.box_vectors.copy(),
        atom_names=(structure.atom_names.copy() if len(structure.atom_names) else None),
        resnames=(structure.resnames.copy() if len(structure.resnames) else None),
        resids=(structure.resids.copy() if len(structure.resids) else None),
        chain_ids=(structure.chain_ids.copy() if len(structure.chain_ids) else None),
    )

    seq = _residue_sequence(tmp)
    n = len(seq)
    if n < _TMD_WINDOW:
        return compute_com_orientation(tmp, half_thickness)[0], float("inf")

    kd_values = np.array([_KD.get(rname, 0.0) for rname, _, _ in seq])
    window = np.ones(_TMD_WINDOW) / _TMD_WINDOW
    if n >= _TMD_WINDOW:
        profile = np.convolve(kd_values, window, mode="valid")
    else:
        profile = np.array([kd_values.mean()])

    tm_mask = profile >= _TMD_KD_THRESHOLD
    tm_indices = np.where(tm_mask)[0]

    if len(tm_indices) == 0:
        return compute_com_orientation(tmp, half_thickness)[0], float("inf")

    half_win = _TMD_WINDOW // 2
    tm_residue_indices: set[int] = set()
    for wi in tm_indices:
        for ri in range(wi, wi + _TMD_WINDOW):
            if ri < n:
                tm_residue_indices.add(ri)

    if not tm_residue_indices:
        return compute_com_orientation(tmp, half_thickness)[0], float("inf")

    # Membrane midplane at TM-residue Z-centre
    tm_z = np.array([seq[i][2][2] for i in tm_residue_indices])
    z_offset = -tm_z.mean()

    # Quality: negative total Z-extent (lower = more vertical = correct).
    #  std(TM_Z) alone fails when the protein is horizontal — ALL
    #  residues (TM and non-TM) are squeezed into a thin Z slice,
    #  making TM_Z artificially tight.  Maximising Z-extent directly
    #  counteracts the "flat protein" pathology.
    extent_z = float(coords_all[:, 2].max() - coords_all[:, 2].min())
    quality = -extent_z

    return z_offset, quality


def compute_tmd_orientation(
    structure: Structure,
    half_thickness: float | None = None,
) -> tuple[float, np.ndarray, float]:
    """Detect trans-membrane helices with a Kyte-Doolittle sliding window.

    Tries **all three principal axes** as candidates for the membrane
    normal.  For each axis the protein is rotated, a 19-residue KD
    sliding window identifies TM helices, and the Z-clustering of TM
    residues is scored.  The axis that gives the tightest TM-residue
    Z-clustering is selected — this guards against the case where PCA
    picks a non-membrane-normal axis (common for compact GPCRs).

    Returns (z_offset_nm, tilt_vec, tilt_angle_rad).
    """
    if half_thickness is None:
        half_thickness = _DEFAULT_MEMBRANE_HALF_THICKNESS

    if structure.num_atoms < 3:
        return 0.0, np.array([1.0, 0.0, 0.0]), 0.0

    tm_analysis = _analyze_tm_helix_bundle(structure)
    if (
        tm_analysis["axis"] is not None
        and tm_analysis["confidence"] >= 0.72
        and tm_analysis["window_count"] >= 3
    ):
        axes = np.asarray([tm_analysis["axis"]])
    else:
        axes = compute_principal_axes(structure.coordinates)
    best_axis = axes[0].copy()
    best_z_off = 0.0
    best_quality = float("inf")

    for i in range(len(axes)):
        axis = axes[i].copy()
        if axis[2] < 0:
            axis = -axis
        z_off, quality = _score_tmd_for_axis(structure, axis, half_thickness)
        if quality < best_quality:
            best_quality = quality
            best_z_off = z_off
            best_axis = axis

    return best_z_off, best_axis, 0.0  # tilt_angle = 0 (TMD doesn't refine tilt)


# ---------------------------------------------------------------------------
# COM (Centre-of-Mass)
# ---------------------------------------------------------------------------

def compute_com_orientation(
    structure: Structure,
    half_thickness: float | None = None,
) -> tuple[float, np.ndarray, float]:
    """Place the protein centre of mass at the membrane midplane (z=0).

    Returns (z_offset_nm, tilt_vec, tilt_angle_rad).
    """
    if len(structure.coordinates) == 0:
        return 0.0, np.array([1.0, 0.0, 0.0]), 0.0
    com_z = structure.coordinates[:, 2].mean()
    return -com_z, np.array([1.0, 0.0, 0.0]), 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ALGORITHMS = {
    "ppm": compute_ppm_orientation,
    "hmoment": compute_hydrophobic_moment_orientation,
    "tmd": compute_tmd_orientation,
    "com": compute_com_orientation,
}

_ALGORITHM_LABELS = {
    "ppm": "PPM-like — Wimley-White transfer free-energy/PCA scan",
    "hmoment": "Hydrophobic Moment — Eisenberg consensus scale",
    "tmd": "TMD Detection — Kyte-Doolittle sliding-window scan",
    "com": "Centre of Mass — simple geometric centre",
}


def list_orientation_algorithms() -> list[dict]:
    """Return available orientation algorithms with metadata."""
    return [
        {"id": aid, "label": _ALGORITHM_LABELS.get(aid, aid)}
        for aid in _ALGORITHMS
    ]


def compute_orientation(
    structure: Structure,
    algorithm: str = "ppm",
    half_thickness: float | None = None,
) -> tuple[float, np.ndarray, float]:
    """Run the named orientation algorithm.

    Parameters
    ----------
    structure : Structure
    algorithm : str
        One of "ppm", "hmoment", "tmd", "com".
    half_thickness : float or None

    Returns
    -------
    z_offset : float        nm
    tilt_vec  : (3,) ndarray
    tilt_angle : float      radians
    """
    func = _ALGORITHMS.get(algorithm)
    if func is None:
        raise ValueError(
            f"Unknown orientation algorithm {algorithm!r}. "
            f"Available: {list(_ALGORITHMS)}"
        )
    return func(structure, half_thickness=half_thickness)

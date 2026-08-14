"""Chemical head-to-tail invariants shared by membrane construction paths.

The force-field backends do not share atom-numbering conventions, so lipid
orientation must not depend on names such as ``O3`` or ``C40``.  This module
identifies the polar head region from elements and the hydrophobic region from
carbon atoms spatially separated from every polar heavy atom.  The resulting
axis works for phospholipids, glycolipids, sphingolipids, glycerolipids and
sterols while preserving each conformer's internal geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


POLAR_ELEMENTS = frozenset({"N", "O", "P", "S"})
HYDROPHOBIC_MIN_POLAR_DISTANCE_NM = 0.28
MIN_HEAD_TAIL_SEPARATION_NM = 0.15
MIN_INWARD_PROJECTION_NM = 0.10
MIN_INWARD_COSINE = 0.10
# Atom-centre separation at the bilayer midplane.  Removing two carbon van
# der Waals radii (~0.34 nm) leaves at most ~0.28 nm, below a water molecule's
# effective diameter, while allowing finite coordinate/percentile noise.
MAX_TAIL_CORE_GAP_NM = 0.62

# A leaflet change is an orientation change, not a mirror operation.  This
# 180-degree rotation about X maps +Z to -Z while keeping determinant +1, so
# stereochemistry and every internal distance are preserved.
_OPPOSITE_LEAFLET_ROTATION = np.diag([1.0, -1.0, -1.0])


class LipidOrientationError(ValueError):
    """Raised when a molecule cannot define a physical amphiphile axis."""


def rotate_to_opposite_leaflet(coordinates: np.ndarray) -> np.ndarray:
    """Return a chirality-preserving rigid rotation to the opposite leaflet.

    Coordinates are rotated around the origin.  Callers that need a specific
    anchor at the origin must translate to that anchor before this operation.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise LipidOrientationError("Lipid coordinates must have shape (N, 3)")
    if not np.isfinite(coords).all():
        raise LipidOrientationError("Lipid coordinates contain non-finite values")
    return coords @ _OPPOSITE_LEAFLET_ROTATION.T


@dataclass(frozen=True)
class LipidOrientation:
    """Chemically inferred head and hydrophobic-tail geometry."""

    polar_indices: np.ndarray
    tail_indices: np.ndarray
    head_centroid: np.ndarray
    tail_centroid: np.ndarray
    head_from_tail: np.ndarray
    separation: float


def atom_element(atom_name: str) -> str:
    """Return the element prefix used by GROMACS/GAFF atom names."""
    return next(
        (character for character in str(atom_name).strip().upper() if character.isalpha()),
        "",
    )


def _sterol_head_polar_indices(
    coordinates: np.ndarray, elements: np.ndarray, polar: np.ndarray, carbons: np.ndarray,
) -> np.ndarray:
    """Select the ring hydroxyl that defines a multiply oxygenated sterol head."""
    if (
        not 2 <= len(polar) <= 3
        or len(carbons) < 15
        or any(elements[index] != "O" for index in polar)
    ):
        return polar
    # This branch is specific to neutral sterols (only C/O heavy atoms).
    # Reconstruct the carbon skeleton with the smallest cutoff that closes the
    # fused rings. ACPYPE coordinates are rounded and individual C-C bonds can
    # vary by several hundredths of a nanometre, so one fixed cutoff either
    # misses a ring or creates false side-chain cycles in compact conformers.
    oxygen_neighbour: dict[int, int] = {}
    for oxygen in polar:
        distances = np.linalg.norm(coordinates[carbons] - coordinates[oxygen], axis=1)
        nearest = int(carbons[int(np.argmin(distances))])
        if float(distances.min()) <= 0.170:
            oxygen_neighbour[int(oxygen)] = nearest

    for cutoff in np.arange(0.175, 0.206, 0.0025):
        adjacency = {int(index): set() for index in carbons}
        for position, left in enumerate(carbons):
            for right in carbons[position + 1:]:
                if float(np.linalg.norm(coordinates[left] - coordinates[right])) <= cutoff:
                    adjacency[int(left)].add(int(right))
                    adjacency[int(right)].add(int(left))

        # Iteratively remove graph leaves. Nodes remaining in the 2-core
        # belong to rings; the canonical sterol head oxygen is bonded to one.
        cyclic_core = set(adjacency)
        degrees = {index: len(adjacency[index]) for index in cyclic_core}
        leaves = [index for index, degree in degrees.items() if degree < 2]
        while leaves:
            index = leaves.pop()
            if index not in cyclic_core:
                continue
            cyclic_core.remove(index)
            for neighbor in adjacency[index]:
                if neighbor in cyclic_core:
                    degrees[neighbor] -= 1
                    if degrees[neighbor] < 2:
                        leaves.append(neighbor)
        ring_polar = np.asarray([
            oxygen for oxygen, neighbour in oxygen_neighbour.items()
            if neighbour in cyclic_core
        ], dtype=int)
        if len(ring_polar):
            return ring_polar
    return polar


def infer_lipid_orientation(
    coordinates: np.ndarray,
    atom_names: list[str] | tuple[str, ...],
) -> LipidOrientation:
    """Infer an amphiphile's polar-head to hydrophobic-core axis.

    Hydrophobic carbons are selected by their distance from all polar heavy
    atoms.  If a very compact molecule has too few carbons beyond the normal
    cutoff, the farthest third is used.  Molecules without both a polar region
    and a carbon-rich region are rejected rather than silently placed as a
    membrane lipid.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(atom_names):
        raise LipidOrientationError("Lipid coordinates and atom names are inconsistent")
    if not np.isfinite(coords).all():
        raise LipidOrientationError("Lipid coordinates contain non-finite values")

    elements = np.asarray([atom_element(name) for name in atom_names])
    polar = np.flatnonzero(np.isin(elements, tuple(POLAR_ELEMENTS)))
    carbons = np.flatnonzero(elements == "C")
    if len(polar) == 0:
        raise LipidOrientationError(
            "Membrane lipid has no polar N/O/P/S headgroup atoms"
        )
    if len(carbons) < 3:
        raise LipidOrientationError(
            "Membrane lipid has fewer than three carbon atoms for a hydrophobic region"
        )
    head_polar = _sterol_head_polar_indices(coords, elements, polar, carbons)

    carbon_to_polar = np.linalg.norm(
        coords[carbons, None, :] - coords[polar][None, :, :], axis=2
    ).min(axis=1)
    tail = carbons[carbon_to_polar >= HYDROPHOBIC_MIN_POLAR_DISTANCE_NM]
    minimum_tail_atoms = min(len(carbons), max(3, int(np.ceil(len(carbons) / 3.0))))
    if len(tail) < minimum_tail_atoms:
        farthest = np.argsort(carbon_to_polar)[-minimum_tail_atoms:]
        tail = carbons[np.sort(farthest)]

    head_centroid = coords[head_polar].mean(axis=0)
    tail_centroid = coords[tail].mean(axis=0)
    axis = head_centroid - tail_centroid
    separation = float(np.linalg.norm(axis))
    if not np.isfinite(separation) or separation < MIN_HEAD_TAIL_SEPARATION_NM:
        raise LipidOrientationError(
            f"Polar head and hydrophobic region are not separated enough "
            f"({separation:.3f} nm < {MIN_HEAD_TAIL_SEPARATION_NM:.2f} nm)"
        )
    return LipidOrientation(
        polar_indices=head_polar,
        tail_indices=tail,
        head_centroid=head_centroid,
        tail_centroid=tail_centroid,
        head_from_tail=axis,
        separation=separation,
    )


def orient_lipid_to_outward_normal(
    coordinates: np.ndarray,
    atom_names: list[str] | tuple[str, ...],
    *,
    upper: bool,
) -> np.ndarray:
    """Rigidly align the chemical head-to-tail axis with a leaflet normal."""
    coords = np.asarray(coordinates, dtype=float)
    profile = infer_lipid_orientation(coords, atom_names)
    target = np.asarray([0.0, 0.0, 1.0 if upper else -1.0])
    source = profile.head_from_tail / profile.separation
    rotation, _ = Rotation.align_vectors([target], [source])
    centre = coords.mean(axis=0)
    oriented = rotation.apply(coords - centre) + centre

    verified = infer_lipid_orientation(oriented, atom_names)
    projection, cosine = outward_orientation(verified, upper=upper)
    if projection < MIN_INWARD_PROJECTION_NM or cosine < MIN_INWARD_COSINE:
        raise LipidOrientationError(
            "Could not align lipid polar head toward the solvent-facing normal"
        )
    return oriented


def outward_orientation(
    profile: LipidOrientation,
    *,
    upper: bool,
) -> tuple[float, float]:
    """Return outward head displacement and cosine for one leaflet."""
    sign = 1.0 if upper else -1.0
    projection = float(profile.head_from_tail[2] * sign)
    cosine = projection / profile.separation
    return projection, cosine

"""Structure data container — coordinates, box vectors, and per-atom metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Structure:
    """Molecular structure holding coordinates and per-atom metadata.

    All length units are nanometers.
    Coordinates shape is (N, 3). Box vectors shape is (3, 3).
    """

    coordinates: np.ndarray  # (N, 3) float64, nanometers
    box_vectors: np.ndarray  # (3, 3) float64, nanometers, triclinic

    atom_names: list[str] = field(default_factory=list)
    resnames: list[str] = field(default_factory=list)
    resids: list[int] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)
    segids: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    occupancies: list[float] = field(default_factory=list)
    tempfactors: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates, dtype=np.float64)
        self.box_vectors = np.asarray(self.box_vectors, dtype=np.float64)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (N, 3)")
        if self.box_vectors.shape != (3, 3):
            raise ValueError("box_vectors must have shape (3, 3)")
        n_atoms = len(self.coordinates)
        if not self.atom_names:
            self.atom_names = [""] * n_atoms
        if not self.resnames:
            self.resnames = [""] * n_atoms
        if not self.resids:
            self.resids = [0] * n_atoms
        if not self.chain_ids:
            self.chain_ids = [""] * n_atoms
        if not self.segids:
            self.segids = [""] * n_atoms
        if not self.elements:
            self.elements = [""] * n_atoms
        if not self.occupancies:
            self.occupancies = [1.0] * n_atoms
        if not self.tempfactors:
            self.tempfactors = [0.0] * n_atoms
        fields = {
            "atom_names": self.atom_names,
            "resnames": self.resnames,
            "resids": self.resids,
            "chain_ids": self.chain_ids,
            "segids": self.segids,
            "elements": self.elements,
            "occupancies": self.occupancies,
            "tempfactors": self.tempfactors,
        }
        mismatched = [name for name, values in fields.items() if len(values) != n_atoms]
        if mismatched:
            raise ValueError(
                f"Per-atom field length mismatch for {n_atoms} coordinates: "
                + ", ".join(mismatched)
            )

    @property
    def num_atoms(self) -> int:
        return len(self.coordinates)

    def center_of_mass(self, masses: np.ndarray | None = None) -> np.ndarray:
        if masses is None:
            return self.coordinates.mean(axis=0)
        return np.average(self.coordinates, axis=0, weights=masses)

    def center_of_geometry(self) -> np.ndarray:
        return self.coordinates.mean(axis=0)

    def translate(self, vector: np.ndarray) -> None:
        """Translate all coordinates by *vector* (nm)."""
        self.coordinates += vector

    def rotate(self, rotation_matrix: np.ndarray, center: np.ndarray | None = None) -> None:
        """Apply rotation matrix around an optional center point."""
        if center is None:
            center = self.center_of_geometry()
        centered = self.coordinates - center
        self.coordinates = centered @ rotation_matrix.T + center

    def dimensions(self) -> np.ndarray:
        """Return box lengths [a, b, c] in nm (diagonal of box_vectors)."""
        return np.sqrt((self.box_vectors ** 2).sum(axis=1))

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) for axis-aligned bounding box."""
        return self.coordinates.min(axis=0), self.coordinates.max(axis=0)

    def wrap_to_box(self) -> None:
        """Wrap coordinates into the primary periodic image."""
        # Use triclinic wrapping via the inverse box matrix
        box = self.box_vectors
        if np.linalg.norm(box - np.diag(np.diag(box))) < 1e-12:
            # Simple orthorhombic case
            dims = np.diag(box)
            self.coordinates = self.coordinates % dims
        else:
            # Triclinic case: fractional -> wrap -> real
            inv_box = np.linalg.inv(box)
            fractional = self.coordinates @ inv_box
            fractional = fractional % 1.0
            self.coordinates = fractional @ box

    def append(self, other: Structure) -> Structure:
        """Return new Structure by appending *other*.

        Uses numpy for vectorised concatenation — avoids O(N) Python list copies
        that become a bottleneck for large systems (>100k atoms).
        """
        new_coords = np.vstack([self.coordinates, other.coordinates])

        # Convert to numpy arrays for fast C-level concatenation, then back to lists
        def _cat(arr_a, arr_b, dtype=str):
            return np.concatenate([
                np.asarray(arr_a, dtype=dtype),
                np.asarray(arr_b, dtype=dtype),
            ]).tolist()

        if self.resids and other.resids:
            shift = max(self.resids) + 1 - min(other.resids)
        else:
            shift = 0
        shifted_resids = np.asarray(other.resids, dtype=int) + shift

        return Structure(
            coordinates=new_coords,
            box_vectors=self.box_vectors.copy(),
            atom_names=_cat(self.atom_names, other.atom_names),
            resnames=_cat(self.resnames, other.resnames),
            resids=self.resids + shifted_resids.tolist(),
            chain_ids=_cat(self.chain_ids, other.chain_ids),
            segids=_cat(self.segids, other.segids),
            elements=_cat(self.elements, other.elements),
            occupancies=_cat(self.occupancies, other.occupancies, dtype=float),
            tempfactors=_cat(self.tempfactors, other.tempfactors, dtype=float),
        )

    def copy(self) -> Structure:
        return Structure(
            coordinates=self.coordinates.copy(),
            box_vectors=self.box_vectors.copy(),
            atom_names=list(self.atom_names),
            resnames=list(self.resnames),
            resids=list(self.resids),
            chain_ids=list(self.chain_ids),
            segids=list(self.segids),
            elements=list(self.elements),
            occupancies=list(self.occupancies),
            tempfactors=list(self.tempfactors),
        )

"""Structure data container — coordinates, box vectors, and per-atom metadata."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

#: Every field that carries one value per atom.
#:
#: Authoritative, because it is used both to validate a Structure and to subset
#: one. A caller that subsets coordinates in place and then updates a
#: hand-written list of fields silently leaves the rest at the old length --
#: which is how ``source_ids`` came to be one release out of step and broke
#: every membrane build that merges two systems.
PER_ATOM_FIELDS = (
    "source_ids",
    "atom_names",
    "resnames",
    "resids",
    "chain_ids",
    "segids",
    "elements",
    "occupancies",
    "tempfactors",
)


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

    # Immutable deposition evidence; source_ids link surviving atoms to records.
    source_ids: list[str] = field(default_factory=list)
    source_info: dict = field(default_factory=dict)

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
        if not self.source_ids:
            self.source_ids = [""] * n_atoms
        self.validate_atom_fields()

    def validate_atom_fields(self) -> None:
        """Check mutated structures without padding away evidence of lost atoms."""
        n_atoms = self.num_atoms
        if self.coordinates.shape != (n_atoms, 3):
            raise ValueError("coordinates must have shape (N, 3)")
        fields = {name: getattr(self, name) for name in PER_ATOM_FIELDS}
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
        return np.sqrt((self.box_vectors**2).sum(axis=1))

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

        Every registered per-atom field follows the same concatenation.
        """
        self.validate_atom_fields()
        other.validate_atom_fields()
        fields = {
            name: list(getattr(self, name)) + list(getattr(other, name)) for name in PER_ATOM_FIELDS
        }
        shift = max(self.resids) + 1 - min(other.resids) if self.resids and other.resids else 0
        fields["resids"] = list(self.resids) + [number + shift for number in other.resids]
        return Structure(
            coordinates=np.vstack([self.coordinates, other.coordinates]),
            box_vectors=self.box_vectors.copy(),
            **fields,
            source_info=_merge_sources(self.source_info, other.source_info),
        )

    def copy(self) -> Structure:
        self.validate_atom_fields()
        return Structure(
            coordinates=self.coordinates.copy(),
            box_vectors=self.box_vectors.copy(),
            **{name: list(getattr(self, name)) for name in PER_ATOM_FIELDS},
            source_info=copy.deepcopy(self.source_info),
        )

    def take(self, indices) -> Structure:
        """Select atoms without converting or dropping deposition evidence."""
        self.validate_atom_fields()
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Atom selection must be one-dimensional")
        fields = {name: [getattr(self, name)[i] for i in indices] for name in PER_ATOM_FIELDS}
        return Structure(
            self.coordinates[indices].copy(),
            self.box_vectors.copy(),
            **fields,
            source_info=copy.deepcopy(self.source_info),
        )

    def select_atoms(self, indices) -> None:
        """Select in place only after all arrays have been validated and prepared."""
        selected = self.take(indices)
        self.coordinates = selected.coordinates
        for name in PER_ATOM_FIELDS:
            setattr(self, name, getattr(selected, name))


def _merge_sources(left, right):
    """Flatten source documents instead of recursively growing append trees."""
    documents = {}
    for source in (left, right):
        for item in source.get("documents", [source]) if source else []:
            documents[item["sha256"] if "sha256" in item else repr(item)] = item
    if len(documents) == 1:
        return copy.deepcopy(next(iter(documents.values())))
    return {"documents": copy.deepcopy(list(documents.values()))} if documents else {}

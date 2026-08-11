"""Topology data structures — atom types, bonds, angles, dihedrals, molecule blocks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AtomType:
    """Force field atom type with Lennard-Jones parameters.

    Attributes:
        name: Type name, e.g. "CTL3" (CHARMM) or "CT" (Amber).
        mass: Atomic mass in atomic units (u).
        charge: Partial charge in elementary charge units.
        sigma: Lennard-Jones sigma in nanometers.
        epsilon: Lennard-Jones epsilon in kJ/mol.
        atom_class: Optional class label for force field matching.
    """

    name: str
    mass: float
    charge: float
    sigma: float
    epsilon: float
    atom_class: str = ""


@dataclass
class Bond:
    i: int
    j: int
    funct: int = 1
    r0: float | None = None  # nm
    k_b: float | None = None  # kJ/(mol nm^2)


@dataclass
class Angle:
    i: int
    j: int
    k: int
    funct: int = 1
    theta0: float | None = None  # degrees
    k_theta: float | None = None  # kJ/(mol rad^2)


@dataclass
class Dihedral:
    i: int
    j: int
    k: int
    l: int
    funct: int
    phi: float | None = None  # degrees
    k_psi: float | None = None  # kJ/mol
    multiplicity: int | None = None


@dataclass
class Improper:
    i: int
    j: int
    k: int
    l: int
    funct: int = 2
    phi0: float | None = None  # degrees
    k_psi: float | None = None  # kJ/mol


@dataclass
class Pair:
    """1-4 non-bonded interaction pair."""
    i: int
    j: int
    funct: int = 1


@dataclass
class MoleculeBlock:
    """A [ moleculetype ] block in a GROMACS topology.

    Attributes:
        atom_indices: Global indices of atoms belonging to this molecule type.
        nrexcl: Number of bonds to exclude from non-bonded interactions.
        type_name: Molecule type name, e.g. "Protein", "POPC", "SOL".
        num_molecules: How many instances of this molecule type exist in the system.
    """

    atom_indices: list[int]
    nrexcl: int
    type_name: str
    num_molecules: int = 1


@dataclass
class Topology:
    """Complete molecular topology for a GROMACS system."""

    atom_types: list[AtomType] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    angles: list[Angle] = field(default_factory=list)
    dihedrals: list[Dihedral] = field(default_factory=list)
    impropers: list[Improper] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    exclusions: list[set[int]] = field(default_factory=list)
    molecule_blocks: list[MoleculeBlock] = field(default_factory=list)
    force_field: str = ""

    def assign_atom_types(self, types: list[AtomType]) -> None:
        self.atom_types = types

    def reindex(self, offset: int) -> Topology:
        """Shift all atom indices by *offset* (for merging). Returns self."""
        for bond in self.bonds:
            bond.i += offset
            bond.j += offset
        for angle in self.angles:
            angle.i += offset
            angle.j += offset
            angle.k += offset
        for dih in self.dihedrals:
            dih.i += offset
            dih.j += offset
            dih.k += offset
            dih.l += offset
        for imp in self.impropers:
            imp.i += offset
            imp.j += offset
            imp.k += offset
            imp.l += offset
        for pair in self.pairs:
            pair.i += offset
            pair.j += offset
        for exclusion in self.exclusions:
            shifted = {idx + offset for idx in exclusion}
            exclusion.clear()
            exclusion.update(shifted)
        for block in self.molecule_blocks:
            block.atom_indices = [idx + offset for idx in block.atom_indices]
        return self

    def merge(self, other: Topology) -> Topology:
        """Combine two topologies. Returns self with *other* appended."""
        offset = len(self.atom_types)
        other.reindex(offset)
        self.atom_types.extend(other.atom_types)
        self.bonds.extend(other.bonds)
        self.angles.extend(other.angles)
        self.dihedrals.extend(other.dihedrals)
        self.impropers.extend(other.impropers)
        self.pairs.extend(other.pairs)
        self.exclusions.extend(other.exclusions)
        self.molecule_blocks.extend(other.molecule_blocks)
        return self

    def num_atom_types(self) -> int:
        return len(self.atom_types)

    def copy(self) -> Topology:
        import copy
        return copy.deepcopy(self)

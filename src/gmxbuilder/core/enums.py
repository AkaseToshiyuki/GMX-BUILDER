"""Enumerations for GMXBUILDER."""

from enum import Enum, auto


class ComponentKind(Enum):
    """Type of a molecular component within a System."""

    PROTEIN = auto()
    MEMBRANE = auto()
    SOLVENT = auto()
    IONS = auto()
    LIGAND = auto()
    UNKNOWN = auto()
    # Appended to preserve the persisted integer values of all pre-v0.8.3
    # checkpoint component kinds.
    NUCLEIC_ACID = auto()


class BoxShape(Enum):
    """Supported simulation box shapes."""

    CUBIC = "cubic"
    RECTANGULAR = "rectangular"
    TRUNCATED_OCTAHEDRON = "truncated_octahedron"
    RHOMBIC_DODECAHEDRON = "rhombic_dodecahedron"
    HEXAGONAL = "hexagonal"

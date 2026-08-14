"""Protein embedding into the lipid bilayer."""

from __future__ import annotations

from gmxbuilder.core.system import System
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.modules.membrane.orient import compute_embedding_depth


def embed_protein(
    system: System,
    bilayer_thickness: float,
    method: str = "com",
) -> System:
    """Translate protein along Z to embed it in the bilayer.

    The protein is positioned so that its center of mass (or hydrophobic
    stretch) sits at z=0, i.e., the bilayer midplane.

    Parameters
    ----------
    system : System
        System containing the protein component.
    bilayer_thickness : float
        Bilayer hydrophobic thickness (nm).
    method : str
        "com" or "hydrophobic".

    Returns
    -------
    system : System (mutated in-place)
    """
    protein_comps = system.component_by_kind(ComponentKind.PROTEIN)
    if not protein_comps:
        return system

    for comp in protein_comps:
        coords = system.coordinates[comp.atom_indices]
        z_shift = compute_embedding_depth(coords, bilayer_thickness, method=method)
        system.structure.coordinates[comp.atom_indices, 2] += z_shift

    return system

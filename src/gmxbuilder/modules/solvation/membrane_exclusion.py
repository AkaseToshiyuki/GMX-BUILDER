"""Construction-only exclusion of solvent from the complete lipid Z envelope.

This is an initial-coordinate policy, not an equilibrium hydration model.
The slab covers all periodic XY positions, including pores and box edges.
"""

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError

# Saved-coordinate exclusion tolerance, applied after GRO serialization.
SURFACE_MARGIN_NM = 0.0011
# Each GRO coordinate is written to 0.001 nm and the box to 0.00001 nm.
# Reserve the combined rounding displacement when constructing the water box;
# saved-coordinate validation still uses the original surface margin.
GRO_ROUNDING_GUARD_NM = 0.0011


def membrane_slab(system):
    """Return an unwrapped (lower, upper, period) Z envelope, or None."""
    components = system.component_by_kind(ComponentKind.MEMBRANE)
    if not components:
        return None
    box = np.asarray(system.structure.box_vectors)
    if not np.isfinite(box).all() or not np.allclose(box, np.diag(np.diag(box))):
        raise ModuleConfigError("Membrane water exclusion requires an orthorhombic box")
    period = float(box[2, 2])
    z = np.concatenate([system.coordinates[c.atom_indices, 2] for c in components])
    if period <= 0 or not len(z) or not np.isfinite(z).all():
        raise ModuleConfigError("Invalid membrane coordinates for solvent exclusion")
    reference = system.metadata.get("solvation", {}).get("membrane_interface_z_nm")
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ModuleConfigError("Invalid membrane interface reference")
        center = float(reference.mean())
        z = center + (z - center + period / 2) % period - period / 2
    # Without a saved construction reference, use the full coordinate envelope
    # conservatively. Do not guess which of two empty gaps is aqueous solvent.
    lower, upper = float(z.min()), float(z.max())
    return lower, upper, period


def atoms_in_membrane(coordinates, slab, *, margin_nm=SURFACE_MARGIN_NM):
    if slab is None:
        return np.zeros(len(coordinates), dtype=bool)
    lower, upper, period = slab
    center = (lower + upper) / 2
    dz = (np.asarray(coordinates)[:, 2] - center + period / 2) % period - period / 2
    return np.abs(dz) <= (upper - lower) / 2 + margin_nm


def assert_membrane_water_free(system):
    """Reject stale or altered construction checkpoints before export/ion placement."""
    slab = membrane_slab(system)
    if slab is None:
        return
    # Inspect every solvent site, including hydrogens and virtual sites; also
    # catch imported waters whose component assignment was lost in an old file.
    indices = set()
    for component in system.component_by_kind(ComponentKind.SOLVENT):
        indices.update(map(int, component.atom_indices))
    indices.update(
        i
        for i, name in enumerate(system.structure.resnames)
        if str(name).strip().upper() in {"SOL", "HOH", "WAT", "TIP3", "TIP4", "TIP3P", "TIP4P", "W"}
    )
    if not indices:
        return
    count = int(atoms_in_membrane(system.coordinates[sorted(indices)], slab).sum())
    if count:
        raise ModuleConfigError(
            f"Membrane construction contains {count} solvent sites inside the lipid Z "
            "envelope. Rerun solvation and ion placement from the membrane checkpoint."
        )

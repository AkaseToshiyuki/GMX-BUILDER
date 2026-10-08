"""Apply the shared initial dry-membrane rule to COBY's single-bead solvent."""

from __future__ import annotations

import re

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.modules.coarse_grained.common import molecules_table, system_from_gro
from gmxbuilder.modules.solvation.membrane_exclusion import (
    assert_membrane_water_free,
    atoms_in_membrane,
    membrane_slab,
)


def exclude_membrane_water(gro, top, metadata, *, salt_molarity, seed):
    """Keep solute coordinates, reassign solvent slots, and synchronize topology.

    COBY supplies the neutralized zero-bulk-salt system. A regular Martini W,
    NA or CL molecule is one bead. Place neutralizing ions outside the full
    lipid slab before removing water there; then replace external water slots
    with integer NaCl pairs at the requested final W-bead concentration.
    """
    topology = top.read_text(encoding="utf-8")
    counts = molecules_table(topology)
    clean_metadata = dict(metadata)
    clean_metadata.pop("solvation", None)
    system = system_from_gro(gro, topology, metadata=clean_metadata)
    slab = membrane_slab(system)
    if slab is None:
        raise ModuleConfigError("Martini bilayer water exclusion requires membrane beads")
    structure = system.structure
    names = np.asarray([str(n).upper() for n in structure.resnames])
    fluid = np.isin(names, ["W", "NA", "CL"])
    for name in ("W", "NA", "CL"):
        if int((names == name).sum()) != sum(n for key, n in counts if key == name):
            raise ModuleConfigError("Martini solvent coordinates and molecule counts disagree")
    rng = np.random.default_rng(seed)
    inside = atoms_in_membrane(structure.coordinates, slab)
    internal_ions = np.flatnonzero(inside & np.isin(names, ["NA", "CL"]))
    external_water = rng.permutation(np.flatnonzero(~inside & (names == "W")))
    if len(external_water) < len(internal_ions):
        raise ModuleConfigError("Insufficient external water slots for neutralizing ions")
    for ion, water in zip(internal_ions, external_water, strict=False):
        structure.coordinates[[ion, water]] = structure.coordinates[[water, ion]]
    inside = atoms_in_membrane(structure.coordinates, slab)
    removed = inside & (names == "W")
    keep = ~removed

    # Retain net neutralization but normalize any paired ions to water before
    # applying the post-exclusion salt target. Counterion excess is unchanged.
    neutral_pairs = min(int((names == "NA").sum()), int((names == "CL").sum()))
    for name in ("NA", "CL"):
        convert = rng.permutation(np.flatnonzero(names == name))[:neutral_pairs]
        names[convert] = "W"
    water = rng.permutation(np.flatnonzero(keep & (names == "W")))
    bead_molarity = 55.5 / 4.0
    ratio = float(salt_molarity) / bead_molarity
    pairs = int(round(ratio * len(water) / (1.0 + 2.0 * ratio)))
    if 2 * pairs >= len(water):
        raise ModuleConfigError("Insufficient water for the requested Martini salt concentration")
    names[water[:pairs]] = "NA"
    names[water[pairs : 2 * pairs]] = "CL"
    for index in np.flatnonzero(fluid):
        structure.resnames[index] = str(names[index])
        structure.atom_names[index] = str(names[index])
    # The topology describes contiguous molecule blocks. Keep all protein/lipid
    # atoms in their original order, then emit W, NA, CL in matching blocks.
    order = np.concatenate(
        [np.flatnonzero(~fluid)]
        + [np.flatnonzero(keep & (names == name)) for name in ("W", "NA", "CL")]
    )
    structure.select_atoms(order)
    final_counts = {name: int((keep & (names == name)).sum()) for name in ("W", "NA", "CL")}
    lines = topology.splitlines()
    section = next(
        (
            i
            for i, line in enumerate(lines)
            if re.fullmatch(r"\s*\[\s*molecules\s*\]\s*(;.*)?", line)
        ),
        None,
    )
    if section is None or any(line.lstrip().startswith("[") for line in lines[section + 1 :]):
        raise ModuleConfigError("Cannot safely rewrite the Martini molecule table")
    retained = [(name, count) for name, count in counts if name not in final_counts]
    updated = "\n".join(lines[: section + 1]) + "\n"
    updated += (
        "\n".join(
            f"{name:<24} {count}" for name, count in [*retained, *final_counts.items()] if count
        )
        + "\n"
    )
    GROWriter.write(structure, gro, "GMXBUILDER Martini 3 dry membrane construction")
    top.write_text(updated, encoding="utf-8")
    reference = {"membrane_interface_z_nm": list(slab[:2])}
    checked = system_from_gro(gro, updated, metadata={**clean_metadata, "solvation": reference})
    assert_membrane_water_free(checked)
    metadata["solvation"] = reference
    metadata["cg_membrane_water_exclusion"] = {
        "policy": "complete-lipid-z-envelope-v1",
        "removed_water_beads": int(removed.sum()),
        "relocated_counterions": len(internal_ions),
        "salt_pairs": pairs,
        "water_sites_in_membrane": 0,
    }

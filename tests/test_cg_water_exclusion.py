"""Dry-slab solvent editing preserves solute and net countercharge."""

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.modules.coarse_grained.common import molecules_table, system_from_gro
from gmxbuilder.modules.coarse_grained.export import CGExportModule
from gmxbuilder.modules.coarse_grained.water_exclusion import exclude_membrane_water


@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_remove_internal_water_relocate_ions_and_preserve_solute(tmp_path, salt, monkeypatch):
    names = ["ALA", "POPC", "POPC"] + ["W"] * 104 + ["NA"] * 3 + ["CL"]
    z = [-1, 2, 6] + [4, 2, 6, 12] + [1] * 50 + [7] * 50 + [4, 4.5, 1] + [7]
    coords = np.array([[i * 0.001, 0, v] for i, v in enumerate(z)])
    structure = Structure(
        coordinates=coords,
        atom_names=["BB"] * len(z),
        resnames=names,
        resids=list(range(1, len(z) + 1)),
        elements=["C"] * len(z),
        box_vectors=np.eye(3) * 8,
    )
    gro, top = tmp_path / "input.gro", tmp_path / "topol.top"
    GROWriter.write(structure, gro)
    top.write_text("[ system ]\ntest\n[ molecules ]\nProtein 1\nPOPC 2\nW 104\nNA 3\nCL 1\n")
    metadata = {}
    exclude_membrane_water(gro, top, metadata, salt_molarity=salt, seed=6401)
    saved = GROReader().read(gro)
    np.testing.assert_allclose(saved.coordinates[:3], coords[:3])
    names = np.asarray(saved.resnames)
    fluid = np.isin(names, ["W", "NA", "CL"])
    assert not np.any((saved.coordinates[fluid, 2] >= 2) & (saved.coordinates[fluid, 2] <= 6))
    counts = dict(molecules_table(top.read_text()))
    for name in ["W", "NA", "CL"]:
        assert counts.get(name, 0) == int((names == name).sum())
    assert counts.get("NA", 0) - counts.get("CL", 0) == 2
    assert metadata["cg_membrane_water_exclusion"]["relocated_counterions"] == 2
    assert metadata["cg_membrane_water_exclusion"]["removed_water_beads"] == 6
    assert counts.get("CL", 0) == (1 if salt else 0)
    # A restored stale wet system must fail before export overwrites files.
    system = system_from_gro(gro, top.read_text(), metadata=metadata)
    system.metadata["system_confirmed"] = True
    system.coordinates[np.flatnonzero(names == "W")[0], 2] = 4
    exporter = CGExportModule()
    monkeypatch.setattr(exporter, "admit", lambda system, config: config)
    sentinel = gro.read_bytes()
    with pytest.raises(ModuleConfigError, match="solvent sites"):
        exporter.run(system, {"output_dir": str(tmp_path)})
    assert gro.read_bytes() == sentinel

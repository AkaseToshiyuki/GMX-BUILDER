"""Optional PDB limits must not prevent exporting simulation coordinates."""

import json
import zipfile

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.io.pdb import PDBParser, PDBWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.export.exporter import ExportModule


@pytest.mark.parametrize(
    "num_atoms,resid,expect_pdb",
    [
        (1, -1000, False),
        (1, -999, True),
        (1, 9999, True),
        (1, 10000, False),
        (99999, 1, True),
        (100000, 1, False),
    ],
)
def test_export_preserves_gro_when_optional_pdb_identifiers_overflow(
    tmp_path, monkeypatch, num_atoms, resid, expect_pdb
):
    structure = Structure(
        coordinates=np.tile([1.23456, 2.34567, 3.45678], (num_atoms, 1)),
        box_vectors=np.diag([5.0, 6.0, 7.0]),
        # Distinct identities keep this writer-limit fixture parseable. It is
        # synthetic export data, not a chemically valid protein residue.
        atom_names=["CA"]
        if num_atoms == 1
        else [np.base_repr(i, 36).zfill(4) for i in range(num_atoms)],
        resnames=["ALA"] * num_atoms,
        resids=[resid] * num_atoms,
        chain_ids=["A"] * num_atoms,
        elements=["C"] * num_atoms,
    )

    # Only topology generation is isolated: this test needs no installed force
    # field. Coordinate writers, manifest and archive generation remain real.
    def write_top(self, structure, path, **kwargs):
        path.write_text("; topology generation is outside this regression\n")

    monkeypatch.setattr(TopologyWriter, "write_top", write_top)
    pdb_path = tmp_path / "structure" / "input.pdb"
    pdb_path.parent.mkdir()
    pdb_path.write_text("stale PDB from a previous export\n")

    result = ExportModule().execute(
        System(structure=structure),
        {"output_dir": tmp_path, "system_name": "limits", "write_mdp": False},
    )

    assert result.success
    exported = GROReader().read(tmp_path / "structure" / "input.gro")
    assert exported.num_atoms == num_atoms
    np.testing.assert_allclose(exported.coordinates, structure.coordinates, rtol=0, atol=5.1e-4)
    np.testing.assert_array_equal(structure.resids, [resid] * num_atoms)
    assert pdb_path.exists() == expect_pdb
    with zipfile.ZipFile(tmp_path / "limits.zip") as archive:
        assert "structure/input.gro" in archive.namelist()
        assert ("structure/input.pdb" in archive.namelist()) == expect_pdb
        assert "topology/topol.top" in archive.namelist()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert ("structure/input.pdb" in json.dumps(manifest)) == expect_pdb
    if expect_pdb:
        assert PDBParser().parse(pdb_path).resids[0] == resid
    else:
        assert any("Skipped input.pdb" in line for line in result.log)
        # The general writer must still refuse lossy simulation exports.
        with pytest.raises(ValueError, match="fixed-width PDB"):
            PDBWriter.write(structure, tmp_path / "strict.pdb")

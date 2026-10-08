"""A fixed ligand ITP must match every coordinate instance, including old checkpoints."""

import numpy as np
import pytest

from gmxbuilder.core.exceptions import TopologyError
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.top import TopologyWriter
from tests.prerequisites import requires_forcefield


def _ligand_structure(second_names=("C1", "C2", "H1", "H2")):
    return Structure(
        coordinates=np.zeros((8, 3)),
        box_vectors=np.eye(3) * 4,
        atom_names=["C1", "C2", "H1", "H2", *second_names],
        resnames=["LIG"] * 8,
        resids=[1] * 8,
        chain_ids=["A"] * 4 + ["B"] * 4,
        elements=["C", "C", "H", "H"] * 2,
    )


def _itp(tmp_path):
    path = tmp_path / "external.itp"
    path.write_text(
        "; External molecule name intentionally differs from residue name\n"
        "[ moleculetype ]\nMOL_LIG 3\n[ atoms ]\n"
        "1 CT 1 LIG C1 1 0.0 12.0\n2 CT 1 LIG C2 2 0.0 12.0\n"
        "3 H  1 LIG H1 3 0.0 1.0\n4 H  1 LIG H2 4 0.0 1.0\n"
        "[ bonds ]\n1 3 1\n2 4 1\n"
    )
    return path


def test_all_external_ligand_instances_match_their_itp(tmp_path):
    TopologyWriter._validate_external_ligand_order(
        _ligand_structure(), "LIG", "MOL_LIG", _itp(tmp_path)
    )


@pytest.mark.parametrize(
    "second_names",
    [("C1", "H1", "C2", "H2"), ("C1", "C2", "H1", "H1"), ("C1", "C2", "H1", "HX")],
)
def test_second_ligand_instance_is_checked_even_with_same_residue_id(tmp_path, second_names):
    structure = _ligand_structure(second_names)
    original = structure.copy()
    with pytest.raises(TopologyError, match="External ligand LIG at B:1"):
        TopologyWriter._validate_external_ligand_order(structure, "LIG", "MOL_LIG", _itp(tmp_path))
    assert structure.atom_names == original.atom_names
    np.testing.assert_array_equal(structure.coordinates, original.coordinates)


@pytest.mark.parametrize("source", ["gaff2", "cgenff"])
@requires_forcefield("amber99sb-ildn")
def test_topology_export_refuses_a_stale_reordered_external_ligand(tmp_path, source):
    itp = _itp(tmp_path)
    atomtypes = tmp_path / "external_atomtypes.itp"
    atomtypes.write_text("[ atomtypes ]\n")
    writer = TopologyWriter(
        "amber99sb-ildn",
        ff_config={
            "ligand_parameters": {
                "LIG": {
                    "source": source,
                    "molecule_type": "MOL_LIG",
                    "itp_path": str(itp),
                    "atomtypes_path": str(atomtypes),
                }
            }
        },
    )
    with pytest.raises(TopologyError, match="Rebuild from the Force Field Check"):
        writer.write_top(_ligand_structure(("C1", "H1", "C2", "H2")), tmp_path / "topol.top")
    assert not (tmp_path / "topol.top").exists()

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.input.pdb_input import PDBInputModule
from gmxbuilder.modules.input.protein_repair import (
    assess_repairable_missing_atoms,
    find_repairable_missing_atoms,
    repair_standard_protein_heavy_atoms,
)


def _structure(atom_names, resname="ARG"):
    coordinates = np.asarray(
        [
            [0.12 * index, 0.03 * (index % 2), 0.02 * (index % 3)]
            for index in range(len(atom_names))
        ],
        dtype=np.float64,
    )
    return Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 5.0,
        atom_names=list(atom_names),
        resnames=[resname] * len(atom_names),
        resids=[16] * len(atom_names),
        chain_ids=["B"] * len(atom_names),
        elements=[name[0] for name in atom_names],
    )


def test_complete_backbone_with_truncated_sidechain_is_repairable():
    structure = _structure(["N", "CA", "C", "O", "CB"])

    candidates = find_repairable_missing_atoms(structure)

    assert candidates[("B", 16, "ARG")] == ("CD", "CG", "CZ", "NE", "NH1", "NH2")


def test_missing_backbone_requires_user_review():
    structure = _structure(["N", "CA", "C", "CB"])

    with pytest.raises(ModuleConfigError, match="missing backbone O"):
        find_repairable_missing_atoms(structure)


def test_disconnected_partial_sidechain_requires_user_review():
    structure = _structure(["N", "CA", "C", "O", "CB", "CD"])

    with pytest.raises(ModuleConfigError, match="disconnected partial side chain"):
        find_repairable_missing_atoms(structure)


def test_assessment_separates_blockers_from_safe_candidates():
    structure = _structure(["N", "CA", "C", "CB"])

    candidates, blockers = assess_repairable_missing_atoms(structure)

    assert candidates == {}
    assert len(blockers) == 1
    assert "missing backbone O" in blockers[0]


def test_input_preserves_unrepairable_damage_as_explicit_warning(tmp_path):
    pdb = tmp_path / "missing_backbone_oxygen.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   3       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   3       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   3       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  CB  ALA A   3       1.986  -0.752   1.247  1.00  0.00           C\n"
        "TER\nEND\n",
        encoding="utf-8",
    )
    initial = System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        )
    )

    result = PDBInputModule().run(initial, {"pdb": str(pdb)})

    assert result.success
    warnings = result.system.metadata["input_repair"]["unrepairable_warnings"]
    assert any("missing backbone O" in warning for warning in warnings)
    assert any("simulation topology remains blocked" in line for line in result.log)


def test_pdbfixer_repairs_arg_without_moving_existing_atoms():
    original = _structure(["N", "CA", "C", "O", "CB"])

    repaired, records = repair_standard_protein_heavy_atoms(original)

    assert len(records) == 1
    assert records[0].resname == "ARG"
    assert set(records[0].added_atoms) == {"CG", "CD", "NE", "CZ", "NH1", "NH2"}
    repaired_lookup = {
        name: repaired.coordinates[index] for index, name in enumerate(repaired.atom_names)
    }
    for index, name in enumerate(original.atom_names):
        assert np.array_equal(repaired_lookup[name], original.coordinates[index])


def test_input_module_records_repair_metadata(tmp_path):
    pdb = tmp_path / "truncated_ser.pdb"
    pdb.write_text(
        "ATOM      1  N   SER A   3       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  SER A   3       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   SER A   3       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   SER A   3       1.209   2.354   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  SER A   3       1.986  -0.752   1.247  1.00  0.00           C\n"
        "TER\nEND\n",
        encoding="utf-8",
    )
    initial = System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        )
    )

    result = PDBInputModule().run(initial, {"pdb": str(pdb)})

    assert result.success
    report = result.system.metadata["input_repair"]
    assert report["status"] == "repaired"
    assert report["residues_repaired"] == 1
    assert report["atoms_added"] == 1
    assert report["residues"][0]["added_atoms"] == ["OG"]
    assert any("Automatic protein heavy-atom repair" in line for line in result.log)

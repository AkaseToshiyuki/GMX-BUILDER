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
    from tests.structure_fixtures import peptide_structure

    reference = peptide_structure({"ARG": "R", "SER": "S"}[resname])
    structure = reference.take([reference.atom_names.index(name) for name in atom_names])
    structure.resids = [16] * structure.num_atoms
    structure.chain_ids = ["B"] * structure.num_atoms
    return structure


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


def test_input_blocks_unrepairable_backbone_damage_with_specific_reason(tmp_path):
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

    assert not result.success
    report = result.system.metadata["input_validation"]
    assert not report["can_proceed"]
    assert any("missing backbone O" in error["message"] for error in report["errors"])
    assert "missing backbone O" in result.log[0]


def test_pdbfixer_repairs_arg_without_moving_existing_atoms():
    original = _structure(["N", "CA", "C", "O", "CB"])
    original.source_ids = [f"input:{name}" for name in original.atom_names]
    original.source_info = {
        "atoms": {
            uid: {"name": name} for uid, name in zip(original.source_ids, original.atom_names)
        }
    }

    repaired, records = repair_standard_protein_heavy_atoms(original)
    from tests.test_atom_provenance import assert_surviving_heavy_sources

    assert_surviving_heavy_sources(original, repaired)

    assert len(records) == 1
    assert records[0].resname == "ARG"
    assert set(records[0].added_atoms) == {"CG", "CD", "NE", "CZ", "NH1", "NH2"}
    repaired_lookup = {
        name: repaired.coordinates[index] for index, name in enumerate(repaired.atom_names)
    }
    for index, name in enumerate(original.atom_names):
        assert np.array_equal(repaired_lookup[name], original.coordinates[index])


def test_bad_existing_geometry_is_blocked_before_automatic_repair(tmp_path, monkeypatch):
    from gmxbuilder.io.pdb import PDBWriter
    from gmxbuilder.modules.input import pdb_input

    damaged = _structure(["N", "CA", "C", "O", "CB"])
    damaged.coordinates[4] = damaged.coordinates[1] + [0.36, 0, 0]
    path = tmp_path / "invalid_sidechain.pdb"
    PDBWriter.write(damaged, path)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing geometric damage must block before PDBFixer is invoked")

    monkeypatch.setattr(pdb_input, "repair_standard_protein_heavy_atoms", forbidden)
    result = PDBInputModule().run(System(damaged), {"pdb": str(path)})
    assert not result.success
    assert any(
        issue["code"] == "protein_geometry"
        for issue in result.system.metadata["input_validation"]["errors"]
    )


def test_input_module_records_repair_metadata(tmp_path):
    pdb = tmp_path / "truncated_ser.pdb"
    from gmxbuilder.io.pdb import PDBWriter

    PDBWriter.write(_structure(["N", "CA", "C", "O", "CB"], "SER"), pdb)
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


def test_rotamer_proposal_moves_only_missing_atoms_and_preserves_ring_geometry():
    from gmxbuilder.modules.input.geometry_validation import added_atom_clashes
    from gmxbuilder.modules.input.protein_repair import _validate_repair
    from gmxbuilder.modules.input.sidechain_placement import relieve_added_sidechain_clashes
    from tests.structure_fixtures import peptide_structure

    # Independent RDKit geometry, with a neighbouring atom obstructing the missing ring.
    reference = peptide_structure("Y")
    obstacle = reference.coordinates[reference.atom_names.index("CE1")] + [0.02, 0, 0]
    repaired = Structure(
        coordinates=np.vstack([reference.coordinates, obstacle]),
        box_vectors=reference.box_vectors,
        atom_names=reference.atom_names + ["O1"],
        resnames=reference.resnames + ["LIG"],
        resids=reference.resids + [2],
        chain_ids=reference.chain_ids + ["B"],
        elements=reference.elements + ["O"],
    )
    missing = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")
    added = [repaired.atom_names.index(name) for name in missing]
    original = repaired.take([i for i in range(repaired.num_atoms) if i not in added])
    candidates = {(reference.chain_ids[0], 1, "TYR"): missing}
    assert added_atom_clashes(repaired, added)
    ring_before = repaired.coordinates[added].copy()

    relieve_added_sidechain_clashes(original, repaired, candidates)

    assert not added_atom_clashes(repaired, added)
    assert len(_validate_repair(original, repaired, candidates)) == 1
    ring_after = repaired.coordinates[added]
    np.testing.assert_allclose(
        np.linalg.norm(ring_before[:, None] - ring_before[None, :], axis=2),
        np.linalg.norm(ring_after[:, None] - ring_after[None, :], axis=2),
        atol=1e-12,
    )
    assert repaired.source_info["sidechain_rotamer_adjustments"]


@pytest.mark.parametrize("sequence", ["R", "Q", "E", "K", "S", "Y"])
def test_distorted_backend_sidechain_gets_valid_template_proposal(sequence, monkeypatch):
    from gmxbuilder.modules.input import protein_repair
    from gmxbuilder.modules.input.geometry_validation import _angle, protein_geometry_issues
    from tests.structure_fixtures import peptide_structure

    # RDKit supplies independent observed geometry. Simulate a backend placement
    # with a compressed new-atom subtree; never distort the input anchor atoms.
    complete = peptide_structure(sequence)
    keep = [i for i, atom in enumerate(complete.atom_names) if atom in {"N", "CA", "C", "O", "CB"}]
    original = complete.take(keep)
    candidates = find_repairable_missing_atoms(original)
    missing = next(iter(candidates.values()))
    added = [complete.atom_names.index(atom) for atom in missing]
    damaged = complete.copy()
    anchor = complete.coordinates[complete.atom_names.index("CB")]
    damaged.coordinates[added] = anchor + 0.2 * (damaged.coordinates[added] - anchor)
    assert protein_geometry_issues(damaged)
    monkeypatch.setattr(protein_repair, "_place_missing_atoms", lambda *args: damaged.copy())

    repaired, records = repair_standard_protein_heavy_atoms(original)

    assert len(records) == 1
    assert repaired.source_info["sidechain_template_replacements"]
    assert not protein_geometry_issues(repaired)
    for i, name in enumerate(original.atom_names):
        np.testing.assert_array_equal(
            original.coordinates[i], repaired.coordinates[repaired.atom_names.index(name)]
        )
    root = "OG" if sequence == "S" else "CG"
    positions = [repaired.coordinates[repaired.atom_names.index(a)] for a in ("CA", "CB", root)]
    # Independent tetrahedral-angle sanity bound, not read from the graft template.
    assert 100 < _angle(*positions) < 130


def test_template_fallback_does_not_hide_moved_observed_atoms(monkeypatch):
    from gmxbuilder.modules.input import protein_repair
    from tests.structure_fixtures import peptide_structure

    complete = peptide_structure("R")
    original = complete.take(
        [i for i, atom in enumerate(complete.atom_names) if atom in {"N", "CA", "C", "O", "CB"}]
    )
    complete.coordinates[complete.atom_names.index("CA")] += [0.01, 0, 0]
    calls = []

    def place(*args):
        calls.append(args[-1])
        return complete.copy()

    monkeypatch.setattr(protein_repair, "_place_missing_atoms", place)
    with pytest.raises(ModuleConfigError, match="moved an existing atom"):
        repair_standard_protein_heavy_atoms(original)
    assert calls == [1]


def test_template_fallback_leaves_two_attachment_ring_to_backend():
    from gmxbuilder.modules.input.sidechain_placement import restore_added_sidechain_templates
    from tests.structure_fixtures import peptide_structure

    complete = peptide_structure("P")
    original = complete.take([i for i, atom in enumerate(complete.atom_names) if atom != "CD"])
    candidates = find_repairable_missing_atoms(original)
    before = complete.coordinates.copy()

    restore_added_sidechain_templates(original, complete, candidates)

    # Proline CD is attached to both CG and backbone N. A single-anchor graft
    # would break ring closure; the fallback must leave this proposal untouched.
    np.testing.assert_array_equal(complete.coordinates, before)
    assert complete.source_info["sidechain_template_replacements"] == []


def test_anchor_frame_rejects_collinear_reference_points():
    from gmxbuilder.modules.input.sidechain_placement import _anchor_frame

    assert _anchor_frame(np.zeros(3), np.ones(3), 2 * np.ones(3)) is None

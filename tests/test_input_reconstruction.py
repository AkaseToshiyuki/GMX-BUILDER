"""Working identifiers must not create chemical connectivity or lose provenance."""

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.input_document import read_input
from gmxbuilder.io.pdb import PDBWriter
from gmxbuilder.modules.input.pdb_input import PDBInputModule
from gmxbuilder.modules.input.reconstruction import normalize_chain_ids, reconstruct_input
from gmxbuilder.modules.input.validation import assess_input_structure


def peptide(resids=(128, 135), chains=None):
    coords, numbers, chain_ids = [], [], []
    for i, resid in enumerate(resids):
        coords.extend(
            np.array([[0, 0, 0], [0.145, 0, 0], [0.2, 0.142, 0], [0.145, 0.245, 0]])
            + [i * 2.0, 0, 0]
        )
        numbers.extend([resid] * 4)
        chain_ids.extend([chains[i] if chains else "R"] * 4)
    return Structure(
        np.array(coords),
        np.eye(3) * 6,
        atom_names=["N", "CA", "C", "O"] * len(resids),
        resnames=["GLY"] * len(coords),
        resids=numbers,
        chain_ids=chain_ids,
        elements=["N", "C", "C", "O"] * len(resids),
    )


def deposited(tmp_path, connected=False):
    structure = peptide()
    if connected:
        structure.coordinates[4:] += (
            structure.coordinates[2] + [0.133, 0, 0] - structure.coordinates[4]
        )
    path = tmp_path / "input.pdb"
    PDBWriter.write(structure, path)
    path.write_text("REMARK 465     SER R   129\n" + path.read_text())
    return read_input(path), path


def test_default_names_are_stable_and_original_atoms_survive(tmp_path):
    structure, _ = deposited(tmp_path)
    original = structure.copy()
    normalize_chain_ids(structure)
    normalize_chain_ids(structure)
    assert structure.chain_ids == ["A"] * 8
    assert structure.source_info["working_chain_ids"] == {"R": "A"}
    assert structure.source_info["atoms"] == original.source_info["atoms"]
    assert structure.source_ids == original.source_ids
    np.testing.assert_array_equal(structure.coordinates, original.coordinates)


@pytest.mark.parametrize("renumber", [False, True])
def test_consent_splits_geometric_break_and_never_moves_atoms(tmp_path, renumber):
    structure, _ = deposited(tmp_path)
    original = structure.coordinates.copy()
    report = reconstruct_input(
        structure, {"allow_incomplete_protein": True, "renumber_residues": renumber}
    )
    assert structure.chain_ids == ["A"] * 4 + ["B"] * 4
    assert structure.resids == ([1] * 8 if renumber else [128] * 4 + [135] * 4)
    assert len(report["splits"]) == 1
    assert [r["author_chain"] for r in report["residue_mapping"]] == ["R", "R"]
    assert [r["author_resid"] for r in report["residue_mapping"]] == [128, 135]
    assert assess_input_structure(structure, structure.source_info["polymer_metadata"])[
        "can_proceed"
    ]
    np.testing.assert_array_equal(structure.coordinates, original)


def test_renumbering_alone_does_not_waive_geometry_or_missing_sequence(tmp_path):
    structure, _ = deposited(tmp_path)
    reconstruct_input(structure, {"renumber_residues": True})
    errors = assess_input_structure(structure, structure.source_info["polymer_metadata"])["errors"]
    assert {x["code"] for x in errors} >= {"peptide_break", "deposited_missing_segment"}
    assert (
        next(x for x in errors if x["code"] == "deposited_missing_segment")["residues"][0]["resid"]
        == 129
    )


def test_number_gap_with_intact_geometry_does_not_split(tmp_path):
    structure, _ = deposited(tmp_path, connected=True)
    report = reconstruct_input(
        structure, {"allow_incomplete_protein": True, "renumber_residues": True}
    )
    assert report["splits"] == []
    assert structure.chain_ids == ["A"] * 8
    assert structure.resids == [1] * 4 + [2] * 4
    assert assess_input_structure(structure, structure.source_info["polymer_metadata"])[
        "can_proceed"
    ]


@pytest.mark.parametrize("resids", [(54, 128), (128, 54)])
def test_numbering_alone_never_requires_fragment_consent(resids):
    structure = peptide(resids)
    structure.coordinates[4:] += structure.coordinates[2] + [0.133, 0, 0] - structure.coordinates[4]
    report = reconstruct_input(structure, {})
    assert report["splits"] == []
    assert assess_input_structure(structure)["can_proceed"]
    assert structure.resids == [resids[0]] * 4 + [resids[1]] * 4


def test_martini_input_exports_the_same_accepted_fragments(tmp_path):
    from gmxbuilder.io.pdb import PDBParser
    from gmxbuilder.modules.coarse_grained.input import CGInputModule

    structure, path = deposited(tmp_path)
    step = tmp_path / "steps" / "input"
    step.mkdir(parents=True)
    config = {
        "pdb": str(path),
        "allow_incomplete_protein": True,
        "renumber_residues": True,
        "chain_names": {"A": "Z"},
        "_task_dir": str(tmp_path),
        "_step_dir": str(step),
    }
    result = CGInputModule().run(System(structure), config)
    assert result.success
    exported = PDBParser().parse(step / "cg_input.pdb")
    assert exported.chain_ids == ["Z"] * 4 + ["A"] * 4
    assert exported.resids == [1] * 8
    np.testing.assert_allclose(exported.coordinates, structure.coordinates, atol=0.0001)


def test_fragment_consent_does_not_waive_missing_backbone_or_overlap(tmp_path):
    structure, _ = deposited(tmp_path)
    structure = structure.take([i for i, n in enumerate(structure.atom_names) if n != "C"])
    reconstruct_input(structure, {"allow_incomplete_protein": True})
    assert not assess_input_structure(structure)["can_proceed"]
    structure, _ = deposited(tmp_path)
    structure.coordinates[4] = structure.coordinates[2]
    reconstruct_input(structure, {"allow_incomplete_protein": True})
    assert not assess_input_structure(structure)["can_proceed"]


def test_rename_reaches_checked_structure_and_duplicate_names_are_rejected(tmp_path):
    structure = peptide((1, 2), chains=["X", "Y"])
    path = tmp_path / "two.pdb"
    PDBWriter.write(structure, path)
    result = PDBInputModule().run(
        System(structure=structure), {"pdb": str(path), "chain_names": {"A": "Z"}}
    )
    assert result.success
    assert set(result.system.structure.chain_ids) == {"Z", "B"}
    with pytest.raises(ValueError, match="merge chains"):
        reconstruct_input(read_input(path), {"chain_names": {"A": "B"}})
    with pytest.raises(ValueError, match="boolean"):
        reconstruct_input(read_input(path), {"allow_incomplete_protein": "false"})


@pytest.mark.parametrize("renumber", [False, True])
def test_internal_missing_sequence_is_reported_once_across_fragments(tmp_path, renumber):
    structure, _ = deposited(tmp_path)
    reconstruct_input(
        structure,
        {
            "allow_incomplete_protein": True,
            "renumber_residues": renumber,
            "chain_names": {"A": "Z"},
        },
    )
    report = assess_input_structure(structure, structure.source_info["polymer_metadata"])
    gaps = [row for row in report["warnings"] if row["code"] == "deposited_missing_segment"]
    assert len(gaps) == 1
    assert gaps[0]["chain"] == "A"
    assert gaps[0]["working_fragments"] == ["Z", "A"]
    assert [r["resid"] for r in gaps[0]["residues"]] == [129]
    assert not any(row["code"] == "deposited_missing_termini" for row in report["warnings"])
    assert report["can_proceed"]


def test_terminal_and_internal_omissions_keep_distinct_source_positions(tmp_path):
    structure, _ = deposited(tmp_path)
    source = structure.source_info["polymer_metadata"]
    source["missing_residues"].extend(
        [
            {"chain": "R", "resid": 127, "resname": "SER"},
            {"chain": "R", "resid": 136, "resname": "SER"},
        ]
    )
    reconstruct_input(structure, {"allow_incomplete_protein": True, "renumber_residues": True})
    report = assess_input_structure(structure, source)
    terminal = [row for row in report["warnings"] if row["code"] == "deposited_missing_termini"]
    assert len(terminal) == 1
    assert [r["resid"] for r in terminal[0]["residues"]] == [127, 136]


def test_checked_sequences_retain_source_identity_through_checkpoint(tmp_path):
    from gmxbuilder.web.server_parts.structure_processing import extract_sequences

    structure, _ = deposited(tmp_path)
    reconstruct_input(structure, {"allow_incomplete_protein": True, "renumber_residues": True})
    System(structure).save_checkpoint(tmp_path / "checkpoint")
    sequences = extract_sequences(System.load_checkpoint(tmp_path / "checkpoint").structure)
    assert [row["source_chain"] for row in sequences] == ["A", "A"]
    assert [row["author_chain"] for row in sequences] == ["R", "R"]
    assert [row["fragment_count"] for row in sequences] == [2, 2]
    assert [row["internal_n_terminus"] for row in sequences] == [False, True]
    assert [row["internal_c_terminus"] for row in sequences] == [True, False]
    assert [row["residues"][0]["author_resid"] for row in sequences] == [128, 135]
    assert [row["residues"][0]["resid"] for row in sequences] == [1, 1]


def test_assembly_copies_with_shared_author_chain_are_not_merged():
    structure = peptide((128, 135, 128, 135), ("R", "R", "S", "S"))
    structure.source_ids = [f"source:{i}" for i in range(structure.num_atoms)]
    structure.source_info["atoms"] = {
        uid: {"author_chain": "X", "resid": resid}
        for uid, resid in zip(structure.source_ids, structure.resids, strict=True)
    }
    reconstruct_input(structure, {"allow_incomplete_protein": True, "renumber_residues": True})
    report = assess_input_structure(
        structure,
        {
            "missing_residues": [{"chain": "X", "resid": 129, "resname": "SER"}],
        },
    )
    gaps = [row for row in report["warnings"] if row["code"] == "deposited_missing_segment"]
    assert {row["chain"] for row in gaps} == {"A", "B"}
    assert len(gaps) == 2
    assert all(len(row["working_fragments"]) == 2 for row in gaps)


def test_fragment_edit_preserves_atom_identity_and_gap_termini(tmp_path):
    from gmxbuilder.web.server_parts.structure_processing import extract_sequences

    structure, path = deposited(tmp_path)
    reference = structure.copy()
    config = {
        "allow_incomplete_protein": True,
        "renumber_residues": True,
        "fragment_names": {"A:1": "X", "A:2": "Y"},
        "exclude_fragments": ["A:1"],
    }
    reconstruct_input(structure, config)
    assert structure.chain_ids == ["Y"] * 4
    assert structure.resids == [1] * 4
    assert structure.source_ids == reference.source_ids[4:]
    np.testing.assert_array_equal(structure.coordinates, reference.coordinates[4:])
    sequence = extract_sequences(structure)[0]
    assert sequence["fragment_key"] == "A:2"
    assert sequence["fragment_index"] == sequence["fragment_count"] == 2
    assert sequence["internal_n_terminus"] and not sequence["internal_c_terminus"]
    assert sequence["residues"][0]["author_resid"] == 135
    restored = read_input(path)
    reconstruct_input(restored, {**config, "exclude_fragments": []})
    assert restored.chain_ids == ["X"] * 4 + ["Y"] * 4
    assert restored.source_ids == reference.source_ids
    np.testing.assert_array_equal(restored.coordinates, reference.coordinates)


def test_excluding_fragment_keeps_same_chain_small_molecule(tmp_path):
    structure, _ = deposited(tmp_path)
    # A deposited ligand sharing the source chain is an independent component.
    ligand = Structure(
        np.array([[4.0, 2.0, 1.0]]),
        np.eye(3) * 6,
        atom_names=["C1"],
        resnames=["LIG"],
        resids=[200],
        chain_ids=["R"],
        elements=["C"],
    )
    structure = structure.append(ligand)
    reconstruct_input(structure, {"allow_incomplete_protein": True, "exclude_fragments": ["A:2"]})
    assert structure.resnames == ["GLY"] * 4 + ["LIG"]
    assert structure.coordinates[-1].tolist() == [4.0, 2.0, 1.0]


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"fragment_names": {"A:1": "B"}}, "merge chains"),
        ({"fragment_names": {"A:9": "Z"}}, "Unknown fragments"),
        ({"exclude_fragments": ["A:9"]}, "Unknown fragments"),
        ({"exclude_fragments": ["A:1", "A:2"]}, "at least one"),
        ({"exclude_fragments": "A:1"}, "array of strings"),
    ],
)
def test_fragment_edit_rejects_ambiguous_or_empty_selection(tmp_path, extra, message):
    structure, _ = deposited(tmp_path)
    with pytest.raises(ValueError, match=message):
        reconstruct_input(structure, {"allow_incomplete_protein": True, **extra})

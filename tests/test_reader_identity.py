"""Independent source records exercise residue identity, cell semantics and upload reuse."""

import copy
from pathlib import Path

import gemmi
import numpy as np
import pytest

from gmxbuilder.io.input_document import read_input, write_input, write_mmcif
from gmxbuilder.io.residue_identity import source_residue
from gmxbuilder.modules.input.connections import connection_issues
from gmxbuilder.modules.input.reconstruction import reconstruct_input
from gmxbuilder.modules.input.validation import assess_input_structure, read_polymer_metadata


def source_file(tmp_path, file_format, *, occupancy=1, element="C"):
    # These three residues have the same author number; 27A has two altlocs.
    path = tmp_path / ("insert." + file_format)
    rows = [(27, "", "", 1, 0), (27, "A", "A", 0.4, 2), (27, "A", "B", 0.6, 3), (27, "B", "", 1, 5)]
    if file_format == "pdb":
        path.write_text(
            "".join(
                f"ATOM  {i:5d}  CA {alt:1}ALA A{number:4d}{ins:1}   "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occ * occupancy:6.2f}{20.0:6.2f}"
                f"          {element:>2}\n"
                for i, (number, ins, alt, occ, x) in enumerate(rows, 1)
            )
            + "END\n"
        )
    else:
        document = gemmi.cif.Document()
        block = document.add_new_block("identity")
        loop = block.init_loop(
            "_atom_site.",
            [
                "id",
                "type_symbol",
                "label_atom_id",
                "label_comp_id",
                "label_asym_id",
                "label_seq_id",
                "auth_asym_id",
                "auth_seq_id",
                "pdbx_PDB_ins_code",
                "label_alt_id",
                "Cartn_x",
                "Cartn_y",
                "Cartn_z",
                "occupancy",
            ],
        )
        for i, (number, ins, alt, occ, x) in enumerate(rows, 1):
            label = {"": "1", "A": "2", "B": "3"}[ins]
            loop.add_row(
                [
                    str(i),
                    element,
                    "CA",
                    "ALA",
                    "X",
                    label,
                    "A",
                    str(number),
                    ins or "?",
                    alt or ".",
                    str(x),
                    "0",
                    "0",
                    str(occ * occupancy),
                ]
            )
        path.write_text(document.as_string())
    return path


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
def test_insertion_altloc_selection_checkpoint_and_export(tmp_path, file_format):
    source = source_file(tmp_path, file_format)
    structure = read_input(source)
    assert structure.resids == [1, 2, 3]
    np.testing.assert_allclose(structure.coordinates[:, 0], [0, 0.3, 0.5])
    assert [source_residue(structure, i)["insertion_code"] for i in range(3)] == ["", "A", "B"]
    assert [str(source_residue(structure, i)["resid"]) for i in range(3)] == ["27"] * 3
    original = copy.deepcopy(structure.source_info["atoms"])
    selected = structure.take([2, 1])
    reconstruct_input(selected, {"renumber_residues": True, "chain_names": {"A": "Z"}})
    path = tmp_path / "retained.input.npz"
    write_input(selected, path)
    restored = read_input(path)
    assert restored.source_info["atoms"] == original
    assert restored.resids == [1, 2]
    assert [source_residue(restored, i)["insertion_code"] for i in range(2)] == ["B", "A"]
    exported = tmp_path / "source-identifiers.cif"
    write_mmcif(restored, exported, source_identifiers=True)
    block = gemmi.cif.read_file(str(exported)).sole_block()
    assert list(block.find_values("_atom_site.pdbx_PDB_ins_code")) == ["B", "A"]
    assert list(block.find_values("_atom_site.auth_seq_id")) == ["27", "27"]
    assert list(block.find_values("_atom_site.auth_asym_id")) == ["A", "A"]
    adapter = tmp_path / "working.cif"
    write_mmcif(restored, adapter)
    reparsed = read_input(adapter)
    assert reparsed.resids == [1, 2]
    assert reparsed.chain_ids == ["Z", "Z"]
    np.testing.assert_array_equal(reparsed.coordinates, restored.coordinates)


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
def test_occupancy_diagnostic_does_not_repair_bad_source(tmp_path, file_format):
    from gmxbuilder.core.exceptions import ParseError

    path = source_file(tmp_path, file_format, occupancy=-99)
    original = path.read_bytes()
    with pytest.raises(ParseError) as caught:
        read_input(path)
    assert len(caught.value.issues) == 4
    assert all(row["code"] == "invalid_occupancy" for row in caught.value.issues)
    assert path.read_bytes() == original


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
def test_connection_endpoints_distinguish_insertion_codes(tmp_path, file_format):
    structure = read_input(source_file(tmp_path, file_format))
    # A deliberately unsupported CA-CA link must resolve to two distinct residues.
    if file_format == "pdb":
        line = list(" " * 80)
        for start, text in [
            (0, "LINK  "),
            (12, " CA "),
            (21, "A"),
            (22, "  27"),
            (26, "A"),
            (42, " CA "),
            (51, "A"),
            (52, "  27"),
            (56, "B"),
        ]:
            line[start : start + len(text)] = text
        structure.source_info["link_records"] = ["".join(line)]
    else:
        connection = {"_struct_conn.conn_type_id": "covale"}
        for partner, insertion in [("ptnr1", "A"), ("ptnr2", "B")]:
            for field, value in [
                ("auth_asym_id", "A"),
                ("auth_seq_id", "27"),
                ("auth_atom_id", "CA"),
            ]:
                connection[f"_struct_conn.{partner}_{field}"] = value
            connection[f"_struct_conn.pdbx_{partner}_PDB_ins_code"] = insertion
        # Supply author atom IDs as a real deposition with this namespace would.
        for atom in structure.source_info["atoms"].values():
            atom["_atom_site.auth_atom_id"] = "CA"
        structure.source_info["connections"] = [connection]
    assert [row["code"] for row in connection_issues(structure)] == ["unsupported_connection"]
    assert [row["code"] for row in connection_issues(structure.take([0, 1]))] == [
        "dangling_connection"
    ]


def test_missing_insertion_is_not_hidden_by_observed_author_number(tmp_path):
    structure = read_input(source_file(tmp_path, "cif")).take([0, 2])
    metadata = {
        "missing_residues": [
            {
                "chain": "A",
                "resid": 27,
                "resname": "GLY",
                "insertion_code": "A",
                "label_seq_id": "2",
            }
        ]
    }
    report = assess_input_structure(structure, metadata)
    gap = next(row for row in report["errors"] if row["code"] == "deposited_missing_segment")
    assert gap["residues"][0]["insertion_code"] == "A"
    assert "27A" in gap["message"]
    path = tmp_path / "missing.pdb"
    path.write_text("REMARK 465     GLY A   27A\n")
    assert read_polymer_metadata(path)["missing_residues"][0]["insertion_code"] == "A"


@pytest.mark.parametrize("file_format", ["pdb", "cif"])
def test_placeholder_cell_preserves_deposition_but_estimates_envelope(tmp_path, file_format):
    path = source_file(tmp_path, file_format)
    if file_format == "pdb":
        path.write_text(
            "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
            + path.read_text()
        )
    else:
        path.write_text(
            path.read_text()
            + "\n"
            + "\n".join(
                f"_cell.{name} {value}"
                for name, value in [
                    ("length_a", 1),
                    ("length_b", 1),
                    ("length_c", 1),
                    ("angle_alpha", 90),
                    ("angle_beta", 90),
                    ("angle_gamma", 90),
                ]
            )
        )
    structure = read_input(path)
    assert structure.source_info["cell_status"] == "placeholder"
    np.testing.assert_allclose(
        structure.source_info["cell_vectors_nm"], np.eye(3) * 0.1, atol=1e-10
    )
    assert structure.source_info["box_source"] == "estimated"
    assert min(structure.dimensions()) >= 3


def test_assessment_leaves_input_and_source_untouched(tmp_path, monkeypatch):
    structure = read_input(source_file(tmp_path, "cif"))
    before = copy.deepcopy(structure)

    def no_deep_copy(*args, **kwargs):
        raise AssertionError("Assessment must not copy complete atom provenance")

    monkeypatch.setattr(type(structure), "copy", no_deep_copy)
    assess_input_structure(structure)
    assert structure.source_info == before.source_info
    assert structure.atom_names == before.atom_names
    np.testing.assert_array_equal(structure.coordinates, before.coordinates)


def test_upload_reuses_structure_and_metadata_and_defers_cg_readiness(tmp_path, monkeypatch):
    from gmxbuilder.io import input_document
    from gmxbuilder.modules.input import validation
    from gmxbuilder.web.server_parts import structure_processing as processing
    from gmxbuilder.web.server_parts.input_limits import StructureInputLimits

    path = source_file(tmp_path, "cif")
    reads = []
    original = input_document.read_input

    def tracked_read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(input_document, "read_input", tracked_read)
    calls = []
    original_metadata = validation.read_polymer_metadata

    def tracked_metadata(path):
        calls.append(Path(path))
        return original_metadata(path)

    monkeypatch.setattr(validation, "read_polymer_metadata", tracked_metadata)

    def unexpected_assessment(*args):
        raise AssertionError("AA readiness must not run for a CG upload")

    monkeypatch.setattr(validation, "assess_input_structure", unexpected_assessment)
    result = processing.process_uploaded_structure(
        path,
        tmp_path,
        "cif",
        [],
        StructureInputLimits.from_environment(),
        processing.filter_pdb_for_display,
        processing.extract_sequences,
        processing._PROTEIN_RESNAMES,
        False,
    )
    assert reads == [path]
    assert calls == [path]
    assert result["input_status"]["readiness"] == "not_checked"
    assert result["input_validation"] is None
    assert result["residue_counts_by_resname"] == {"ALA": 3}
    assert [row["insertion_code"] for row in result["sequences"][0]["residues"]] == ["", "A", "B"]


def test_canonical_utf8_and_legacy_unicode_both_preserve_source(tmp_path):
    import json

    from gmxbuilder.core.structure import PER_ATOM_FIELDS

    structure = read_input(source_file(tmp_path, "cif"))
    structure.source_info["note"] = "source μ chain — 原始编号"
    modern = tmp_path / "utf8.input.npz"
    write_input(structure, modern)
    with np.load(modern, allow_pickle=False) as arrays:
        assert int(arrays["input_schema"]) == 2
        assert arrays["source_info"].dtype == np.uint8
    legacy = tmp_path / "legacy.input.npz"
    np.savez_compressed(
        legacy,
        coordinates=structure.coordinates,
        box_vectors=structure.box_vectors,
        input_schema=np.array(1),
        source_info=np.array(json.dumps(structure.source_info)),
        **{field: np.asarray(getattr(structure, field)) for field in PER_ATOM_FIELDS},
    )
    for path in (modern, legacy):
        restored = read_input(path)
        assert restored.source_info == structure.source_info
        assert restored.source_ids == structure.source_ids
        np.testing.assert_array_equal(restored.coordinates, structure.coordinates)


def test_mixed_connection_namespace_uses_deposited_atom_mapping(tmp_path):
    structure = read_input(source_file(tmp_path, "cif"))
    for atom in structure.source_info["atoms"].values():
        atom["_atom_site.auth_atom_id"] = "AUTHOR_CA"
    row = {"_struct_conn.conn_type_id": "covale"}
    for partner, insertion in [("ptnr1", "A"), ("ptnr2", "B")]:
        row.update(
            {
                f"_struct_conn.{partner}_auth_asym_id": "A",
                f"_struct_conn.{partner}_auth_seq_id": "27",
                f"_struct_conn.{partner}_label_atom_id": "CA",
                f"_struct_conn.pdbx_{partner}_PDB_ins_code": insertion,
            }
        )
    structure.source_info["connections"] = [row]
    assert connection_issues(structure)[0]["code"] == "unsupported_connection"


def test_legacy_placeholder_input_is_migrated_without_changing_source_cell(tmp_path):
    import json

    from gmxbuilder.core.structure import PER_ATOM_FIELDS

    structure = read_input(source_file(tmp_path, "cif"))
    structure.box_vectors = np.eye(3) * 0.1
    structure.source_info.pop("cell_status")
    structure.source_info.update(
        cell_vectors_nm=structure.box_vectors.tolist(), box_source="deposited"
    )
    path = tmp_path / "old.input.npz"
    np.savez_compressed(
        path,
        coordinates=structure.coordinates,
        box_vectors=structure.box_vectors,
        input_schema=np.array(1),
        source_info=np.array(json.dumps(structure.source_info)),
        **{field: np.asarray(getattr(structure, field)) for field in PER_ATOM_FIELDS},
    )
    restored = read_input(path)
    assert restored.source_info["cell_status"] == "placeholder"
    assert restored.source_info["cell_vectors_nm"] == structure.box_vectors.tolist()
    assert restored.source_info["box_source"] == "estimated"
    assert min(restored.dimensions()) >= 3


def test_deposited_connection_insertion_tags_survive_cif_case_normalization(tmp_path):
    path = source_file(tmp_path, "cif")
    document = gemmi.cif.read_file(str(path))
    block = document.sole_block()
    connection = block.init_loop(
        "_struct_conn.",
        [
            "conn_type_id",
            "ptnr1_auth_asym_id",
            "ptnr1_auth_seq_id",
            "ptnr1_label_atom_id",
            "pdbx_ptnr1_PDB_ins_code",
            "ptnr2_auth_asym_id",
            "ptnr2_auth_seq_id",
            "ptnr2_label_atom_id",
            "pdbx_ptnr2_PDB_ins_code",
        ],
    )
    connection.add_row(["covale", "A", "27", "CA", "A", "A", "27", "CA", "B"])
    path.write_text(document.as_string())
    structure = read_input(path)
    assert [row["code"] for row in connection_issues(structure)] == ["unsupported_connection"]
    assert [row["code"] for row in connection_issues(structure.take([0, 1]))] == [
        "dangling_connection"
    ]


def test_canonical_read_does_not_add_metadata_to_generated_structures(tmp_path):
    from gmxbuilder.core.structure import Structure

    structure = Structure(np.zeros((1, 3)), np.eye(3) * 3, source_info={"note": "generated"})
    path = tmp_path / "generated.input.npz"
    write_input(structure, path)
    assert read_input(path).source_info == structure.source_info

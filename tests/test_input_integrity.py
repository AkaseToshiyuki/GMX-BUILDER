"""Input invariants independent of accession, filename and residue numbering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES
from gmxbuilder.core.exceptions import ModuleConfigError, ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.cif import CIFParser
from gmxbuilder.io.input_document import canonical_path, read_input, write_mmcif
from gmxbuilder.io.pdb import PDBParser, PDBValidator, PDBWriter
from gmxbuilder.modules.input.protein_repair import _validate_repair
from gmxbuilder.modules.input.validation import assess_input_structure, read_polymer_metadata
from gmxbuilder.web.server_parts.input_limits import StructureInputLimits
from gmxbuilder.web.server_parts.structure_files import filter_pdb_file
from gmxbuilder.web.server_parts.structure_processing import (
    extract_sequences,
    filter_pdb_for_display,
    process_uploaded_structure,
)


def test_canonical_input_reads_and_parses_one_immutable_document(
    tmp_path, two_residues, monkeypatch
):
    import gemmi

    path = tmp_path / "source.cif"
    write_mmcif(two_residues, path)
    read_bytes = Path.read_bytes
    read_string = gemmi.cif.read_string
    reads, parses = [], []

    def read(candidate):
        if candidate.resolve() == path.resolve():
            reads.append(candidate)
        return read_bytes(candidate)

    def parse(raw):
        parses.append(raw)
        return read_string(raw)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(gemmi.cif, "read_string", parse)
    result = read_input(path)
    assert len(reads) == len(parses) == 1
    np.testing.assert_allclose(result.coordinates, two_residues.coordinates)
    assert result.chain_ids == two_residues.chain_ids


@pytest.fixture
def two_residues():
    return Structure(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        np.eye(3) * 3,
        atom_names=["C1", "C1"],
        resnames=["LIG", "LIG"],
        resids=[1, 10000],
        chain_ids=["A", "A"],
        elements=["C", "C"],
    )


@pytest.mark.parametrize("variant", ["comment", "indent", "upper", "inline", "leading_comment"])
def test_legal_cif_layout_preserves_every_atom(tmp_path, two_residues, variant):
    path = tmp_path / "input.cif"
    write_mmcif(two_residues, path)
    raw = path.read_text()
    if variant == "comment":
        raw = raw.replace("\nATOM 2 ", "\n#\nATOM 2 ")
    elif variant == "indent":
        raw = raw.replace("loop_", "   loop_")
    elif variant == "upper":
        raw = raw.replace("_atom_site.", "_ATOM_SITE.")
    elif variant == "inline":
        raw = raw.replace(
            "_atom_site.group_PDB\n_atom_site.id", "_atom_site.group_PDB _atom_site.id"
        )
    else:
        raw = "# legal header comment\n" + raw
    path.write_text(raw)
    parsed = CIFParser().parse(path)
    assert parsed.resids == [1, 10000]
    np.testing.assert_allclose(parsed.coordinates, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], atol=1e-15)


@pytest.mark.parametrize(
    "chain,residue,numbers",
    [
        ("A", "LIG", [1, 10000]),
        ("long_chain", "LONG", [0, -2]),
        ("Q", "TEST", [9999, 19998]),
        ("Z", "LIG", [1, 2**31 + 1]),
    ],
)
def test_upload_selection_and_checkpoint_never_read_viewer_identities(
    tmp_path, two_residues, chain, residue, numbers
):
    two_residues.chain_ids = [chain] * 2
    two_residues.resnames = [residue] * 2
    two_residues.resids = numbers
    path = tmp_path / "source.cif"
    write_mmcif(two_residues, path)
    summary = process_uploaded_structure(
        path,
        tmp_path,
        "cif",
        [],
        StructureInputLimits.from_environment(),
        filter_pdb_for_display,
        extract_sequences,
        PROTEIN_RESNAMES,
    )
    assert summary["num_atoms"] == 2
    assert {m["resid"] for m in summary["small_molecules"]} == set(numbers)
    selected = tmp_path / "filtered.pdb"
    assert filter_pdb_file(path, selected, ["A"], []) == (2, 0)
    canonical = read_input(canonical_path(selected))
    assert canonical.chain_ids == ["A"] * 2 and canonical.resnames == [residue] * 2
    assert canonical.source_info["working_chain_ids"] == {chain: "A"}
    assert canonical.resids == numbers and canonical.num_atoms == 2
    np.testing.assert_array_equal(canonical.coordinates, two_residues.coordinates)
    System(canonical).save_checkpoint(tmp_path / "checkpoint")
    recovered = System.load_checkpoint(tmp_path / "checkpoint").structure
    assert recovered.resids == numbers
    assert recovered.source_ids == canonical.source_ids
    assert recovered.source_info == canonical.source_info


def test_duplicate_unlabelled_atoms_are_not_altloc(tmp_path, two_residues):
    two_residues.resids = [1, 1]
    path = tmp_path / "duplicate.pdb"
    PDBWriter.write(two_residues, path)
    with pytest.raises(ParseError, match="Duplicate atom identity"):
        PDBParser().parse(path)


def test_ter_separates_reused_author_chain_and_number(tmp_path, two_residues):
    two_residues.resids = [1, 1]
    path = tmp_path / "fragments.pdb"
    PDBWriter.write(two_residues, path)
    raw = path.read_text().replace("\nHETATM    2", "\nTER\nHETATM    2")
    path.write_text(raw)
    parsed = PDBParser().parse(path)
    assert parsed.num_atoms == 2 and len(set(parsed.chain_ids)) == 2
    assert {a["author_chain"] for a in parsed.source_info["atoms"].values()} == {"A"}


def test_ligand_summary_uses_selected_model(tmp_path, two_residues):
    path = tmp_path / "models.pdb"
    PDBWriter.write(two_residues.take([0]), path)
    atoms = "".join(
        line + "\n" for line in path.read_text().splitlines() if line.startswith(("ATOM", "HETATM"))
    )
    path.write_text(
        "MODEL        1\n" + atoms + "ENDMDL\nMODEL        2\n" + atoms + "ENDMDL\nEND\n"
    )
    assert PDBParser().parse(path).num_atoms == 1
    assert PDBValidator.detect_small_molecules(path)[0]["atom_count"] == 1


def test_missing_records_survive_leading_comments(tmp_path, two_residues):
    path = tmp_path / "sequence.cif"
    write_mmcif(two_residues, path)
    raw = (
        path.read_text()
        + """\nloop_
_pdbx_unobs_or_zero_occ_residues.PDB_model_num
_pdbx_unobs_or_zero_occ_residues.polymer_flag
_pdbx_unobs_or_zero_occ_residues.auth_asym_id
_pdbx_unobs_or_zero_occ_residues.auth_seq_id
_pdbx_unobs_or_zero_occ_residues.auth_comp_id
1 Y A 7 GLY
"""
    )
    path.write_text(raw)
    before = read_polymer_metadata(path)
    path.write_text("# comment\n" + raw)
    assert read_polymer_metadata(path) == before
    assert before["missing_residues"][0]["resid"] == 7


def test_cell_metric_survives_pdb_roundtrip(tmp_path, two_residues):
    two_residues.resids = [1, 2]
    two_residues.box_vectors = np.array([[3, 0, 0], [1, 3, 0], [0.2, 0.1, 4]])
    path = tmp_path / "cell.pdb"
    PDBWriter.write(two_residues, path)
    parsed = PDBParser().parse(path)
    np.testing.assert_allclose(
        parsed.box_vectors @ parsed.box_vectors.T,
        two_residues.box_vectors @ two_residues.box_vectors.T,
        atol=0.002,
    )
    assert parsed.source_info["box_source"] == "deposited"


def test_real_small_cell_is_not_replaced(tmp_path, two_residues):
    two_residues.resids = [1, 2]
    two_residues.box_vectors = np.eye(3) * 0.8
    path = tmp_path / "smallcell.pdb"
    PDBWriter.write(two_residues, path)
    np.testing.assert_allclose(PDBParser().parse(path).box_vectors, np.eye(3) * 0.8)


def test_all_coincident_complete_residue_is_blocked():
    s = Structure(
        np.zeros((5, 3)),
        np.eye(3) * 3,
        atom_names=["N", "CA", "C", "O", "CB"],
        resnames=["ALA"] * 5,
        resids=[1] * 5,
        chain_ids=["A"] * 5,
        elements=["N", "C", "C", "O", "C"],
    )
    assert not assess_input_structure(s)["can_proceed"]


def test_repair_rejects_inverted_chirality(small_pdb_file):
    complete = PDBParser().parse(small_pdb_file)
    original = complete.take([0, 1, 2, 3])
    assert len(_validate_repair(original, complete, {("A", 1, "ALA"): ("CB",)})) == 1
    complete.coordinates[4, 2] *= -1
    with pytest.raises(ModuleConfigError, match="stereochemistry"):
        _validate_repair(original, complete, {("A", 1, "ALA"): ("CB",)})


def test_coordinate_adapter_keeps_long_chains_and_precision(tmp_path, two_residues):
    two_residues.chain_ids = ["alpha", "alpha"]
    two_residues.coordinates[0, 0] = 0.123456789123
    path = tmp_path / "adapter.cif"
    write_mmcif(two_residues, path)
    parsed = CIFParser().parse(path)
    assert parsed.chain_ids == two_residues.chain_ids
    np.testing.assert_allclose(parsed.coordinates, two_residues.coordinates, rtol=0, atol=1e-15)


def test_declared_cross_residue_link_is_not_silently_ignored(tmp_path, two_residues):
    two_residues.resids = [1, 2]
    path = tmp_path / "linked.pdb"
    PDBWriter.write(two_residues, path)
    path.write_text(path.read_text().replace("TER\nEND", "CONECT    1    2\nTER\nEND"))
    report = assess_input_structure(PDBParser().parse(path))
    assert any(x["code"] == "unsupported_connection" for x in report["errors"])


def test_repeated_append_does_not_nest_provenance(two_residues):
    s = two_residues
    for _ in range(5):
        s = s.append(two_residues)
    assert s.source_info == {}


def test_repair_coordinate_adapter_is_readable_by_openmm(tmp_path, two_residues):
    from openmm.app import PDBxFile

    two_residues.chain_ids = ["long_chain"] * 2
    two_residues.resids = [0, -9]
    path = tmp_path / "backend.cif"
    write_mmcif(two_residues, path)
    parsed = PDBxFile(str(path))  # Parsing only: no force field, Context or dynamics.
    assert [r.id for r in parsed.topology.residues()] == ["0", "-9"]
    assert {c.id for c in parsed.topology.chains()} == {"long_chain"}


def test_cif_content_detection_is_case_insensitive(tmp_path, two_residues):
    from gmxbuilder.web.server_parts.structure_processing import (
        prepare_and_inspect_structure_upload,
    )

    path = tmp_path / "source.cif"
    write_mmcif(two_residues, path)
    payload = path.read_text().replace("data_", "DATA_").replace("_atom_site.", "_ATOM_SITE.")
    prepared = prepare_and_inspect_structure_upload(
        "misnamed.pdb", payload.encode(), StructureInputLimits.from_environment()
    )
    assert prepared[2] == "cif"


def test_actual_check_consumes_canonical_selection(
    tmp_path, monkeypatch, two_residues, empty_default_lipid_library
):
    from fastapi.testclient import TestClient

    from gmxbuilder.pipeline.step_executor import StepRunner
    from gmxbuilder.web import server
    from gmxbuilder.web.task_manager import TaskManager

    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_schedule_propka_precompute", lambda *args: None)
    monkeypatch.setattr(server.ligand_prep, "start", lambda *args: None)
    server._step_runners.clear()
    path = tmp_path / "source.cif"
    write_mmcif(two_residues, path)
    with TestClient(server.app) as client:
        upload = client.post(
            "/api/upload-pdb",
            data={"task_type": "solvator"},
            files={"file": ("source.cif", path.read_bytes(), "chemical/x-mmcif")},
        )
        assert upload.status_code == 200, upload.text
        task = upload.json()["task_id"]
        selection = client.post(
            f"/api/filter-pdb/{task}", json={"include_chains": ["A"], "exclude_resnames": []}
        )
        assert selection.status_code == 200, selection.text
        result = client.post(f"/api/step/{task}/input", json={"config": {}})
        assert result.json()["status"] == "ok", result.text
        recovered = (
            StepRunner(manager.get_task_dir(task), "solvator").load_system("input").structure
        )
        assert recovered.resids == [1, 10000] and recovered.num_atoms == 2
    server._step_runners.clear()


def test_blank_chain_ligand_identity_matches_filtering(tmp_path, two_residues):
    two_residues.chain_ids = ["", ""]
    two_residues.resids = [1, 2]
    path = tmp_path / "ligand.pdb"
    PDBWriter.write(two_residues, path)
    assert PDBValidator.detect_small_molecules(path)[0]["chain"] == ""
    assert filter_pdb_file(path, tmp_path / "filtered.pdb", ["A"], []) == (2, 0)


def test_new_atoms_are_checked_against_other_new_atoms(small_pdb_file):
    from gmxbuilder.modules.input.geometry_validation import added_atom_clashes

    left = PDBParser().parse(small_pdb_file)
    right = left.copy()
    right.chain_ids = ["B"] * right.num_atoms
    right.coordinates = right.coordinates @ np.diag([-1, -1, 1])
    right.coordinates += left.coordinates[4] + [0, 0, 0.12] - right.coordinates[4]
    combined = left.append(right)
    assert any(message.count("CB") == 2 for message in added_atom_clashes(combined, [4, 9]))


def test_legacy_selection_requires_reapplication(tmp_path):
    from types import SimpleNamespace

    from gmxbuilder.web.server_parts.task_resources import TaskResourceHelpers

    (tmp_path / "filtered.pdb").write_text("END\n")
    manager = SimpleNamespace(get_task_dir=lambda task: tmp_path)
    helper = TaskResourceHelpers(lambda: manager, lambda task: task)
    with pytest.raises(ValueError, match="Reapply the input selection"):
        helper.resolve_input_pdb("task")


def test_propka_uses_validated_canonical_checkpoint(tmp_path, two_residues):
    from types import SimpleNamespace

    from gmxbuilder.modules.input.validation import INPUT_VALIDATION_VERSION
    from gmxbuilder.web.server_parts.task_resources import TaskResourceHelpers

    two_residues.resids = [1, 2]
    system = System(
        two_residues,
        metadata={
            "input_validation": {
                "policy_version": INPUT_VALIDATION_VERSION,
                "can_proceed": True,
            }
        },
    )
    checkpoint = tmp_path / "steps" / "input"
    system.save_checkpoint(checkpoint)
    (tmp_path / "filtered.pdb").write_text("END\n")
    manager = SimpleNamespace(get_task_dir=lambda task: tmp_path)
    helper = TaskResourceHelpers(lambda: manager, lambda task: task)
    parsed = PDBParser().parse(helper.resolve_propka_pdb_path("task"))
    np.testing.assert_allclose(parsed.coordinates, two_residues.coordinates, atol=0.00005)
    system.metadata["input_validation"]["policy_version"] -= 1
    system.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="Repeat Input Check"):
        helper.resolve_propka_pdb_path("task")


def test_ligand_cn_link_is_not_assumed_to_be_a_peptide(two_residues):
    from gmxbuilder.modules.input.connections import connection_issues

    two_residues.atom_names = ["C", "N"]
    two_residues.elements = ["C", "N"]
    two_residues.coordinates = np.array([[0, 0, 0], [0.133, 0, 0]])
    two_residues.source_ids = ["a", "b"]
    two_residues.source_info = {
        "format": "pdb",
        "connections": [[1, 2]],
        "atoms": {
            "a": {"serial": 1, "chain": "A", "resid": 1, "name": "C"},
            "b": {"serial": 2, "chain": "A", "resid": 10000, "name": "N"},
        },
    }
    assert connection_issues(two_residues)[0]["code"] == "unsupported_connection"


@pytest.mark.parametrize(
    "case,code",
    [
        ("removed", "dangling_connection"),
        ("repeated", "ambiguous_connection"),
        ("symmetry", "unsupported_connection"),
    ],
)
def test_link_records_require_unambiguous_retained_endpoints(two_residues, case, code):
    from gmxbuilder.modules.input.connections import connection_issues

    record = list(" " * 80)
    for start, text in [
        (0, "LINK  "),
        (12, " C1 "),
        (21, "A"),
        (22, "   1"),
        (42, " C1 "),
        (51, "A"),
        (52, "   2"),
        (59, "  1555"),
        (66, "  2555" if case == "symmetry" else "  1555"),
    ]:
        record[start : start + len(text)] = text
    two_residues.source_ids = ["a", "b"]
    atoms = {
        "a": {"serial": 1, "chain": "A", "resid": 1, "name": "C1"},
        "b": {"serial": 2, "chain": "A", "resid": 1 if case == "repeated" else 2, "name": "C1"},
    }
    two_residues.source_info = {"format": "pdb", "link_records": ["".join(record)], "atoms": atoms}
    if case == "removed":
        two_residues = two_residues.take([0])
    assert connection_issues(two_residues)[0]["code"] == code

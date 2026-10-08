"""Regression tests for the first pipeline stage: PDB Loader."""

import numpy as np
import pytest

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.pdb import PDBParser, PDBWriter
from gmxbuilder.modules.input.pdb_input import PDBInputModule
from gmxbuilder.pipeline.step_executor import _compute_step_metrics


def _empty_system(metadata=None):
    return System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
        metadata=metadata or {},
    )


def test_blank_element_column_respects_pdb_atom_name_alignment(tmp_path):
    pdb = tmp_path / "blank-elements.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CD  GLU A   2       1.000   0.000   0.000  1.00  0.00\n"
        "ATOM      3 1HG1 VAL A   3       2.000   0.000   0.000  1.00  0.00\n"
        "HETATM    4 CL   CL  B   4       3.000   0.000   0.000  1.00  0.00\n"
        "HETATM    5 FE   HEM B   5       4.000   0.000   0.000  1.00  0.00\n"
        "END\n"
    )

    structure = PDBParser().parse(pdb)

    assert structure.elements == ["C", "C", "H", "CL", "FE"]


def test_pdb_writer_rejects_identifier_overflow_unless_viewer_wrap_is_explicit(
    tmp_path,
):
    structure = Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3),
        atom_names=["CA"],
        resnames=["ALA"],
        resids=[10000],
        chain_ids=["A"],
        elements=["C"],
    )
    output = tmp_path / "overflow.pdb"

    with pytest.raises(ValueError, match="fixed-width PDB"):
        PDBWriter.write(structure, output)
    assert not output.exists()

    PDBWriter.write(structure, output, wrap_ids_for_viewer=True)
    assert output.is_file()


def test_pdb_parser_selects_highest_occupancy_alternate_location(tmp_path):
    pdb = tmp_path / "altloc.pdb"
    pdb.write_text(
        "ATOM      1  CA AALA A   1       1.000   0.000   0.000  0.40  0.00           C\n"
        "ATOM      2  CA BALA A   1       2.000   0.000   0.000  0.60  0.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    structure = PDBParser().parse(pdb)

    assert structure.atom_names == ["CA", "C"]
    assert structure.coordinates[0, 0] == pytest.approx(0.2)


def test_pdb_parser_preserves_insertion_codes_without_merging_residues(tmp_path):
    pdb = tmp_path / "insertion.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A  10A      1.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )

    structure = PDBParser().parse(pdb)
    assert structure.resids == [1, 2]
    records = [structure.source_info["atoms"][uid] for uid in structure.source_ids]
    assert [(row["resid"], row["icode"]) for row in records] == [(10, ""), (10, "A")]
    np.testing.assert_allclose(structure.coordinates[:, 0], [0, 0.1])


def test_pdb_loader_preserves_build_metadata_and_centers_solute(small_pdb_file):
    initial = _empty_system({"seed": 8675309, "simparams": {"em_nsteps": 1234}})

    result = PDBInputModule().run(initial, {"pdb": str(small_pdb_file)})

    assert result.success
    assert result.system.metadata["seed"] == 8675309
    assert result.system.metadata["simparams"] == {"em_nsteps": 1234}
    assert result.system.metadata["pdb_path"] == str(small_pdb_file)
    assert result.system.num_atoms == 5
    assert np.allclose(result.system.structure.center_of_geometry(), np.zeros(3))
    assert result.system.component_by_name("PROTEIN") is not None


def test_pdb_loader_rejects_structure_empty_after_cleaning(tmp_path):
    water_only = tmp_path / "water_only.pdb"
    water_only.write_text(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  H1  HOH A   1       0.958   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )
    initial = _empty_system({"seed": 17})

    result = PDBInputModule().run(initial, {"pdb": str(water_only)})

    assert not result.success
    assert result.system is initial
    assert "no solute atoms" in result.log[0]


def test_pdb_loader_rejects_non_finite_structure_values(tmp_path, monkeypatch):
    input_path = tmp_path / "invalid.pdb"
    input_path.write_text("END\n")
    initial = _empty_system({"seed": 17})
    invalid = Structure(
        coordinates=np.array([[np.nan, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 10.0,
        atom_names=["CA"],
        resnames=["ALA"],
        resids=[1],
        chain_ids=["A"],
        elements=["C"],
    )
    monkeypatch.setattr(
        "gmxbuilder.io.pdb.PDBParser.parse",
        lambda _self, _path: invalid,
    )

    result = PDBInputModule().run(initial, {"pdb": str(input_path)})

    assert not result.success
    assert result.system is initial
    assert "non-finite" in result.log[0]


def test_same_chain_ions_and_buffers_are_not_promoted_to_protein():
    structure = Structure(
        coordinates=np.zeros((10, 3)),
        box_vectors=np.eye(3) * 5,
        atom_names=["N", "CA", "C", "O", "NA", "C1", "O1", "N", "CA", "C"],
        resnames=["ALA"] * 4 + ["NA", "ACT", "ACT"] + ["PLC"] * 3,
        resids=[1] * 4 + [2, 3, 3, 4, 4, 4],
        chain_ids=["A"] * 10,
        elements=["N", "C", "C", "O", "Na", "C", "O", "N", "C", "C"],
    )
    system = System(structure=structure)

    PDBInputModule()._detect_components(system)

    protein = system.component_by_kind(ComponentKind.PROTEIN)[0]
    ions = system.component_by_kind(ComponentKind.IONS)[0]
    unknown = system.component_by_kind(ComponentKind.UNKNOWN)[0]
    assert protein.atom_indices.tolist() == [0, 1, 2, 3]
    assert ions.atom_indices.tolist() == [4]
    assert unknown.atom_indices.tolist() == [5, 6, 7, 8, 9]


def test_input_normalizes_phosphoserine_and_records_reversible_patch(tmp_path):
    pdb = tmp_path / "phosphoserine.pdb"
    atoms = [
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
        ("OG", "O"),
        ("P", "P"),
        ("O1P", "O"),
        ("O2P", "O"),
        ("O3P", "O"),
    ]
    pdb.write_text(
        "".join(
            f"ATOM  {index:5d} {name:^4s} SEP A  10    "
            f"{index * 1.2:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
            for index, (name, element) in enumerate(atoms, 1)
        )
        + "TER\nEND\n"
    )

    _use_parent_scaffold(pdb, "SER", {})
    result = PDBInputModule().run(_empty_system(), {"pdb": str(pdb)})

    assert result.success
    assert set(result.system.structure.resnames) == {"SER"}
    assert set(result.system.structure.atom_names) == {"N", "CA", "C", "O", "CB", "OG"}
    report = result.system.metadata["input_modifications"]
    assert report["detected"] == report["recognized"] == 1
    assert len(report["records"]) == 1
    record = dict(report["records"][0])
    deposited = record.pop("original_atoms")
    input_lines = [line for line in pdb.read_text().splitlines() if line.startswith("ATOM")]
    assert [atom["name"] for atom in deposited] == [line[12:16].strip() for line in input_lines]
    assert [atom["element"] for atom in deposited] == [line[76:78].strip() for line in input_lines]
    np.testing.assert_allclose(
        [atom["coordinates_nm"] for atom in deposited],
        [
            [float(line[a:b]) / 10 for a, b in ((30, 38), (38, 46), (46, 54))]
            for line in input_lines
        ],
        atol=1e-12,
        rtol=0,
    )
    assert [record] == [
        {
            "chain": "A",
            "resid": 10,
            "original_resname": "SEP",
            "standard_resname": "SER",
            "patch_id": "PHOS_SER",
            "status": "recognized",
            "normalized": True,
            "removed_atoms": ["O1P", "O2P", "O3P", "P"],
            "residue_index": 0,
        }
    ]
    metrics = _compute_step_metrics(result.system, "input")
    assert metrics["input_sequences"] == [
        {
            "chain_id": "A",
            "length": 1,
            "residues": [{"resname": "SER", "resid": 10, "is_protein": True}],
        }
    ]


def test_input_recognizes_newly_validated_hydroxyproline(tmp_path):
    pdb = tmp_path / "hydroxyproline.pdb"
    pdb.write_text(
        "ATOM      1  N   HYP A   5       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  HYP A   5       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   HYP A   5       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   HYP A   5       1.209   2.354   0.000  1.00  0.00           O\n"
        "TER\nEND\n"
    )

    result = PDBInputModule().run(_empty_system(), {"pdb": str(pdb)})

    assert result.success
    report = result.system.metadata["input_modifications"]
    assert report["detected"] == report["recognized"] == 1
    assert report["records"][0]["patch_id"] == "HYP_PRO"
    assert report["records"][0]["status"] == "recognized"
    assert report["records"][0]["normalized"] is True
    assert set(result.system.structure.resnames) == {"PRO"}
    assert {"CB", "CG", "CD"}.issubset(set(result.system.structure.atom_names))
    assert report["warnings"] == []


def test_input_does_not_auto_normalize_catalogue_only_modification(tmp_path):
    pdb = tmp_path / "pyroglutamate.pdb"
    pdb.write_text(
        "ATOM      1  CA  PCA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C   PCA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      3  O   PCA A   1       1.209   2.354   0.000  1.00  0.00           O\n"
        "ATOM      4  CB  PCA A   1       1.986  -0.752   1.247  1.00  0.00           C\n"
        "TER\nEND\n"
    )

    result = PDBInputModule().run(_empty_system(), {"pdb": str(pdb)})

    assert result.success
    record = result.system.metadata["input_modifications"]["records"][0]
    assert record["patch_id"] is None
    assert record["status"] == "unrecognized"
    assert record["normalized"] is False
    assert "no unambiguous reversible modification mapping" in record["warning"]


def test_input_mly_is_dimethyllysine_not_malonyllysine(tmp_path):
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    template = load_force_field_rtp("charmm36m").get_residue("MLY")
    assert template is not None
    pdb = tmp_path / "dimethyllysine.pdb"
    heavy_atoms = [atom[0] for atom in template["atoms"] if not atom[0].startswith("H")]
    pdb.write_text(
        "".join(
            f"ATOM  {index:5d} {name:^4s} MLY A  10    "
            f"{index * 1.2:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {name[0]:>2s}\n"
            for index, name in enumerate(heavy_atoms, 1)
        )
        + "TER\nEND\n"
    )

    _use_parent_scaffold(pdb, "LYS", {})
    result = PDBInputModule().run(_empty_system(), {"pdb": str(pdb)})

    assert result.success
    record = result.system.metadata["input_modifications"]["records"][0]
    assert record["status"] == "recognized"
    assert record["standard_resname"] == "LYS"
    assert record["patch_id"] == "KME2_LYS"
    assert "MAL_LYS" not in str(record)


def test_input_normalizes_selenomethionine_without_inferring_oxidation(tmp_path):
    pdb = tmp_path / "selenomet.pdb"
    pdb.write_text(
        "ATOM      1  N   MSE A   8       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  MSE A   8       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   MSE A   8       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   MSE A   8       1.209   2.354   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  MSE A   8       1.986  -0.752   1.247  1.00  0.00           C\n"
        "ATOM      6  CG  MSE A   8       3.300  -0.900   1.400  1.00  0.00           C\n"
        "ATOM      7 SE   MSE A   8       4.400  -1.000   1.500  1.00  0.00          SE\n"
        "ATOM      8  CE  MSE A   8       5.500  -1.100   1.600  1.00  0.00           C\n"
        "TER\nEND\n"
    )

    _use_parent_scaffold(pdb, "MET", {"SE": "SD"})
    result = PDBInputModule().run(_empty_system(), {"pdb": str(pdb)})

    assert result.success
    assert set(result.system.structure.resnames) == {"MET"}
    assert "SD" in result.system.structure.atom_names
    assert "SE" not in result.system.structure.atom_names
    record = result.system.metadata["input_modifications"]["records"][0]
    assert record["status"] == "normalized_only"
    assert record["patch_id"] is None
    assert "no oxidation patch was inferred" in record["warning"]


def _use_parent_scaffold(path, parent, aliases=None):
    from importlib.metadata import distribution

    template = distribution("pdbfixer").locate_file(f"pdbfixer/templates/{parent}.pdb")
    records = {
        line[12:16].strip(): line[30:54]
        for line in template.read_text().splitlines()
        if line.startswith("ATOM")
    }
    aliases = aliases or {}
    lines = []
    for line in path.read_text().splitlines():
        name = aliases.get(line[12:16].strip(), line[12:16].strip())
        if line.startswith("ATOM") and name in records:
            line = line[:30] + records[name] + line[54:]
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")

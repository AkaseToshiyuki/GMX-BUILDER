"""Regression tests for force-field-specific RTP loading and peptide links."""

from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import TopologyError
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
from gmxbuilder.modules.forcefield.charmm36 import CHARMM36mForceField
from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
from gmxbuilder.modules.modifications.processor import StructureProcessor
from tests.test_structure_processor import _disulfide_system


def _ala_gly_structure() -> Structure:
    atom_names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    return Structure(
        coordinates=np.zeros((len(atom_names), 3)),
        box_vectors=np.eye(3) * 5.0,
        atom_names=atom_names,
        resnames=["ALA"] * 5 + ["GLY"] * 4,
        resids=[1] * 5 + [2] * 4,
        chain_ids=["A"] * len(atom_names),
        elements=["N", "C", "C", "O", "C", "N", "C", "C", "O"],
    )


def test_charmm_mixed_membrane_blocks_keep_each_lipid_identity():
    structure = Structure(
        coordinates=np.zeros((4, 3)),
        box_vectors=np.eye(3) * 5.0,
        atom_names=["C1", "C2", "C1", "O3"],
        resnames=["POPC", "POPC", "CHOL", "CHOL"],
        resids=[1, 1, 2, 2],
        elements=["C", "C", "C", "O"],
    )
    system = System(
        structure,
        components=[
            Component(
                "MEMBRANE",
                ComponentKind.MEMBRANE,
                np.arange(4),
                metadata={"n_lipids_upper": 1, "n_lipids_lower": 1, "lipid_sizes": [2, 2]},
            )
        ],
    )

    topology = CHARMM36mForceField().build_system_topology(system)

    assert [block.type_name for block in topology.molecule_blocks] == [
        "POPC",
        "CHOL",
    ]


def _bond_pairs(itp_path: Path) -> set[tuple[int, int]]:
    pairs = set()
    section = ""
    for raw in itp_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == "bonds" and line:
            fields = line.split()
            pairs.add((int(fields[0]), int(fields[1])))
    return pairs


def _section_rows(itp_path: Path, wanted: str) -> list[list[str]]:
    rows = []
    section = ""
    for raw in itp_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == wanted and line:
            rows.append(line.split())
    return rows


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m", "amber99sb", "oplsaa"])
def test_writer_resolves_rtp_peptide_bond_references(tmp_path, force_field):
    structure = _ala_gly_structure()
    path = tmp_path / f"{force_field}.itp"

    TopologyWriter(force_field)._write_protein_itp_for_indices(
        structure, path, list(range(structure.num_atoms))
    )

    # C of residue 1 is sequence atom 3; N of residue 2 is sequence atom 6.
    assert (3, 6) in _bond_pairs(path) or (6, 3) in _bond_pairs(path)
    assert _section_rows(path, "angles")
    assert _section_rows(path, "dihedrals")
    assert _section_rows(path, "pairs")


@pytest.mark.parametrize(
    "force_field,angle_funct,proper_funct",
    [
        ("charmm36", "5", "9"),
        ("charmm36m", "5", "9"),
        ("amber99sb", "1", "9"),
        ("amber99sb-ildn", "1", "9"),
        ("oplsaa", "1", "3"),
    ],
)
def test_generated_terms_use_force_field_bonded_functions(
    tmp_path, force_field, angle_funct, proper_funct
):
    structure = _ala_gly_structure()
    path = tmp_path / f"functions-{force_field}.itp"
    TopologyWriter(force_field)._write_protein_itp_for_indices(
        structure, path, list(range(structure.num_atoms))
    )
    assert {row[3] for row in _section_rows(path, "angles")} == {angle_funct}
    assert proper_funct in {row[4] for row in _section_rows(path, "dihedrals")}


def test_single_residue_does_not_resolve_missing_neighbour(tmp_path):
    structure = _ala_gly_structure()
    path = tmp_path / "single.itp"
    TopologyWriter("charmm36m")._write_protein_itp_for_indices(structure, path, list(range(5)))
    assert all(6 not in pair for pair in _bond_pairs(path))


def test_writer_does_not_bridge_a_geometric_chain_break(tmp_path):
    structure = _ala_gly_structure()
    structure.coordinates[5:] += np.array([2.0, 0.0, 0.0])
    path = tmp_path / "chain-break.itp"
    TopologyWriter("amber99sb")._write_protein_itp_for_indices(
        structure, path, list(range(structure.num_atoms))
    )
    assert (3, 6) not in _bond_pairs(path)
    assert (6, 3) not in _bond_pairs(path)


def test_charmm36m_loader_includes_its_extended_residue_templates():
    rtp = load_force_field_rtp("charmm36m")
    assert rtp.get_residue("ALY") is not None
    assert rtp.get_atom_type("ALY", "NZ") is not None


def test_missing_force_field_rtp_fails_explicitly():
    with pytest.raises(FileNotFoundError, match="force field directory"):
        load_force_field_rtp("does-not-exist")


def test_molecule_runs_preserve_mixed_coordinate_order():
    structure = Structure(
        coordinates=np.zeros((8, 3)),
        box_vectors=np.eye(3),
        atom_names=["A"] * 8,
        resnames=["POPC", "POPC", "CHOL", "POPC", "POPC", "POPC", "CHOL", "CHOL"],
        resids=[1, 1, 2, 3, 3, 4, 5, 5],
    )
    assert TopologyWriter._ordered_residue_runs(structure, {"POPC", "CHOL"}) == [
        ("POPC", 1),
        ("CHOL", 1),
        ("POPC", 2),
        ("CHOL", 1),
    ]


def test_lipid_without_force_field_parameters_fails_explicitly(tmp_path):
    structure = Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3),
        atom_names=["C1"],
        resnames=["20AHC"],
        resids=[1],
    )
    with pytest.raises(TopologyError, match="no charmm36m RTP.*Amber99SB-ILDN"):
        TopologyWriter("charmm36m")._write_lipid_itp("20AHC", structure, tmp_path / "20AHC.itp")


def test_cross_chain_disulfide_merges_molecule_and_persists_explicit_bond(tmp_path):
    system = (
        StructureProcessor()
        .run(
            _disulfide_system(cross_chain=True),
            {
                "skip_protonation": True,
                "prepare_standard_termini": False,
                "crosslinks": [{"type": "disulfide", "first_index": 0, "second_index": 1}],
            },
        )
        .system
    )
    checkpoint = tmp_path / "checkpoint"
    system.save_checkpoint(checkpoint)
    restored = type(system).load_checkpoint(checkpoint)
    sulphurs = [index for index, name in enumerate(restored.structure.atom_names) if name == "SG"]
    assert len(sulphurs) == 2
    assert any({bond.i, bond.j} == set(sulphurs) for bond in restored.topology.bonds)

    restored = ForceFieldAssigner().run(restored, {}).system
    assert any({bond.i, bond.j} == set(sulphurs) for bond in restored.topology.bonds), (
        "Final force-field assignment must not discard the saved SG-SG bond"
    )

    top_path = tmp_path / "topol.top"
    writer = TopologyWriter("amber14sb")
    writer.write_top(
        restored.structure,
        top_path,
        topology=restored.topology,
    )
    combined_itp = tmp_path / "topol_Protein_chain_A_B.itp"
    assert combined_itp.is_file()
    assert not (tmp_path / "topol_Protein_chain_A.itp").exists()
    assert not (tmp_path / "topol_Protein_chain_B.itp").exists()
    assert "Protein_chain_A_B" in top_path.read_text()

    merged_indices = writer._get_protein_chains(restored.structure, {"CYX"}, restored.topology)[0][
        1
    ]
    local_sulphurs = {merged_indices.index(index) + 1 for index in sulphurs}
    assert any(set(pair) == local_sulphurs for pair in _bond_pairs(combined_itp))

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.input.pdb_input import PDBInputModule
from gmxbuilder.modules.solution.forcefield import SolutionForceFieldSelector
from gmxbuilder.modules.solution.structure import SolutionStructureProcessor
from gmxbuilder.modules.nucleic_acid.native import _make_polymer_molecules_contiguous
from gmxbuilder.modules.forcefield.compatibility import compatibility_report
from gmxbuilder.runtime.hardware import find_gromacs_executable
from gmxbuilder.io.pdb import PDBParser, PDBValidator
from gmxbuilder.io.top import TopologyWriter


FIXTURE = Path(__file__).parent / "fixtures" / "dna_dinucleotide.pdb"


def _empty_system() -> System:
    return System(Structure(coordinates=np.empty((0, 3)), box_vectors=np.eye(3) * 10))


def _input_system() -> System:
    result = PDBInputModule().run(_empty_system(), {"pdb": str(FIXTURE)})
    assert result.success
    return result.system


def _force_field_config(name: str) -> dict:
    return {
        "name": name,
        "lipid_names": [],
        "lipid_ff": "none",
        "ligand_ff": "none",
        "water_model": "tip3p",
    }


def test_input_classifies_dna_as_polymer_not_ligand():
    system = _input_system()
    nucleic = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
    assert len(nucleic) == 1
    assert nucleic[0].metadata["polymer_type"] == "DNA"
    assert nucleic[0].metadata["n_residues"] == 2
    assert not system.component_by_kind(ComponentKind.UNKNOWN)
    assert not system.component_by_kind(ComponentKind.LIGAND)
    assert PDBValidator.detect_small_molecules(FIXTURE) == []


def test_web_sequence_extraction_keeps_nucleic_chain():
    from gmxbuilder.web.server import _extract_sequences

    sequences = _extract_sequences(PDBParser().parse(FIXTURE))
    assert len(sequences) == 1
    assert sequences[0]["chain_id"] == "A"
    assert sequences[0]["length"] == 2
    assert all(item["is_nucleic"] for item in sequences[0]["residues"])
    assert not any(item["is_protein"] for item in sequences[0]["residues"])


@pytest.mark.parametrize("force_field", ["amber14sb", "amber99sb-ildn", "charmm36"])
def test_unvalidated_nucleic_backends_are_explicitly_blocked(force_field):
    with pytest.raises(ModuleConfigError, match="Nucleic-acid force-field selection"):
        SolutionForceFieldSelector().run(_input_system(), _force_field_config(force_field))


def test_broken_nucleic_backbone_is_explicitly_blocked():
    system = _input_system()
    right_p = next(
        index
        for index, (resid, atom_name) in enumerate(
            zip(system.structure.resids, system.structure.atom_names)
        )
        if int(resid) == 2 and str(atom_name).strip() == "P"
    )
    system.structure.coordinates[right_p, 0] += 1.0
    with pytest.raises(ModuleConfigError, match="backbone continuity"):
        SolutionForceFieldSelector().run(system, _force_field_config("charmm36m"))


def test_modified_nucleotide_is_not_routed_as_small_molecule():
    system = _input_system()
    component = system.component_by_kind(ComponentKind.NUCLEIC_ACID)[0]
    first_residue = min(component.atom_indices)
    resid = system.structure.resids[first_residue]
    for index in component.atom_indices:
        if system.structure.resids[index] == resid:
            system.structure.resnames[index] = "PSU"
    # Re-run detection on the altered coordinate input contract.
    system.components = []
    PDBInputModule()._detect_components(system)
    nucleic = system.component_by_kind(ComponentKind.NUCLEIC_ACID)
    assert nucleic and "PSU" in nucleic[0].metadata["unsupported_residues"]
    report = compatibility_report(system, "charmm36m", [])
    assert report["nucleic_acid"]["enabled"] is False
    assert "modified/noncanonical" in report["nucleic_acid"]["reason"]
    with pytest.raises(ModuleConfigError, match="Modified or noncanonical"):
        SolutionForceFieldSelector().run(system, _force_field_config("charmm36m"))


def test_free_nucleotide_like_ligand_is_not_mistaken_for_polymer():
    names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'", "N9"]
    structure = Structure(
        coordinates=np.asarray([[index * 0.12, 0.0, 0.0] for index in range(len(names))]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=names,
        resnames=["AMP"] * len(names),
        resids=[1] * len(names),
        chain_ids=["L"] * len(names),
        elements=[name[0] for name in names],
    )
    system = System(structure)
    PDBInputModule()._detect_components(system)
    assert not system.component_by_kind(ComponentKind.NUCLEIC_ACID)
    assert system.component_by_kind(ComponentKind.UNKNOWN)


def test_molecule_runs_do_not_merge_across_an_excluded_polymer():
    structure = Structure(
        coordinates=np.zeros((3, 3)),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C1", "CA", "C1"],
        resnames=["LIG", "ALA", "LIG"],
        resids=[1, 2, 3],
        chain_ids=["L", "A", "L"],
        elements=["C", "C", "C"],
    )
    assert TopologyWriter._ordered_residue_run_records(
        structure, {"LIG"}, excluded_indices={1}
    ) == [(0, "LIG", 1), (2, "LIG", 1)]


def test_protein_and_nucleic_polymer_atoms_become_contiguous():
    structure = Structure(
        coordinates=np.asarray([[index, 0.0, 0.0] for index in range(4)]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["CA", "P", "O3'", "H"],
        resnames=["ALA", "DA", "DA", "ALA"],
        resids=[1, 2, 2, 1],
        chain_ids=["A", "D", "D", "A"],
        elements=["C", "P", "O", "H"],
    )
    native = {"atom_indices": [1, 2]}
    system = System(
        structure,
        components=[
            Component("PROTEIN", ComponentKind.PROTEIN, np.asarray([0, 3])),
            Component(
                "NUCLEIC_D",
                ComponentKind.NUCLEIC_ACID,
                np.asarray([1, 2]),
                metadata={"native_topology": native},
            ),
        ],
    )
    _make_polymer_molecules_contiguous(system)
    assert system.structure.atom_names == ["CA", "H", "P", "O3'"]
    assert system.components[0].atom_indices.tolist() == [0, 1]
    assert system.components[1].atom_indices.tolist() == [2, 3]
    assert native["atom_indices"] == [2, 3]


@pytest.mark.skipif(find_gromacs_executable() is None, reason="GROMACS unavailable")
def test_native_charmm36m_dna_topology_adds_hydrogens_and_exact_charge(tmp_path):
    system = (
        SolutionForceFieldSelector().run(_input_system(), _force_field_config("charmm36m")).system
    )
    prepared = SolutionStructureProcessor().run(system, {}).system
    component = prepared.component_by_kind(ComponentKind.NUCLEIC_ACID)[0]
    assert prepared.num_atoms > 38
    assert component.metadata["prepared"] is True
    assert component.metadata["net_charge"] == -1.0
    assert prepared.total_charge() == -1.0
    native = prepared.metadata["native_nucleic_topologies"][0]
    assert "O3'" in native["itp_text"]
    assert "POSRES_FC_BB" in native["posre_text"]
    assert "POSRES_FC_SC" in native["posre_text"]
    assert "/home/" not in native["itp_text"]
    assert "/tmp/" not in native["itp_text"]

    checkpoint = tmp_path / "checkpoint"
    prepared.save_checkpoint(checkpoint)
    loaded = System.load_checkpoint(checkpoint)
    assert loaded.total_charge() == -1.0
    assert loaded.metadata["native_nucleic_topologies"][0]["atom_count"] == prepared.num_atoms

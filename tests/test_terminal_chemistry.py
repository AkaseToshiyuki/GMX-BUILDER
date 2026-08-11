"""End-to-end standard terminal chemistry tests for all bundled protein FFs."""

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.rtp_parser import get_terminal_residue
from gmxbuilder.modules.modifications.processor import StructureProcessor


def _two_residue_system(force_field: str) -> System:
    names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    coordinates = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.145, 0.000, 0.000],
            [0.245, 0.100, 0.000],
            [0.225, 0.220, 0.000],
            [0.160, -0.150, 0.000],
            [0.370, 0.080, 0.000],
            [0.470, 0.160, 0.000],
            [0.590, 0.100, 0.000],
            [0.690, 0.160, 0.000],
        ]
    )
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 5.0,
        atom_names=names,
        resnames=["ALA"] * 5 + ["GLY"] * 4,
        resids=[1] * 5 + [2] * 4,
        chain_ids=["A"] * len(names),
        elements=["N", "C", "C", "O", "C", "N", "C", "C", "O"],
    )
    return System(
        structure=structure,
        components=[
            Component("PROTEIN_A", ComponentKind.PROTEIN, np.arange(len(names)))
        ],
        metadata={"force_field": force_field},
    )


def _residue_names(structure: Structure, resid: int) -> set[str]:
    return {
        structure.atom_names[index].strip()
        for index, atom_resid in enumerate(structure.resids)
        if atom_resid == resid
    }


def _itp_residue_charges(path) -> dict[int, float]:
    charges: dict[int, float] = {}
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section == "atoms" and line:
            fields = line.split()
            resid = int(fields[2])
            charges[resid] = charges.get(resid, 0.0) + float(fields[6])
    return charges


@pytest.mark.parametrize(
    "force_field,expected_n_type,expected_c_oxygens",
    [
        ("charmm36", "NH3", {"OT1", "OT2"}),
        ("charmm36m", "NH3", {"OT1", "OT2"}),
        ("amber99sb", "N3", {"OC1", "OC2"}),
        ("amber99sb-ildn", "N3", {"OC1", "OC2"}),
        ("oplsaa", "opls_287", {"O1", "O2"}),
    ],
)
def test_standard_termini_match_templates_and_itp_charges(
    tmp_path, force_field, expected_n_type, expected_c_oxygens
):
    result = StructureProcessor().run(
        _two_residue_system(force_field), {"skip_protonation": True}
    )
    assert result.success
    structure = result.system.structure

    _n_name, n_template = get_terminal_residue(force_field, "ALA", "N")
    _c_name, c_template = get_terminal_residue(force_field, "GLY", "C")
    assert _residue_names(structure, 1) == {atom[0] for atom in n_template["atoms"]}
    assert _residue_names(structure, 2) == {atom[0] for atom in c_template["atoms"]}
    assert {"H1", "H2", "H3"}.issubset(_residue_names(structure, 1))
    assert expected_c_oxygens.issubset(_residue_names(structure, 2))
    assert result.system.metadata["standard_termini_prepared"] == 2
    component_indices = list(map(int, result.system.components[0].atom_indices))
    assert len(component_indices) == len(set(component_indices)) == structure.num_atoms
    assert sorted(component_indices) == list(range(structure.num_atoms))
    assert np.isfinite(structure.coordinates).all()
    for attribute in (
        "atom_names", "resnames", "resids", "chain_ids", "segids",
        "elements", "occupancies", "tempfactors",
    ):
        assert len(getattr(structure, attribute)) == structure.num_atoms

    n_index = next(
        index for index, (resid, name) in enumerate(zip(structure.resids, structure.atom_names))
        if resid == 1 and name == "N"
    )
    for hydrogen in ("H1", "H2", "H3"):
        h_index = next(
            index for index, (resid, name) in enumerate(zip(structure.resids, structure.atom_names))
            if resid == 1 and name == hydrogen
        )
        bond_length = np.linalg.norm(
            structure.coordinates[h_index] - structure.coordinates[n_index]
        )
        assert bond_length == pytest.approx(0.101, abs=1e-4)

    path = tmp_path / f"{force_field}.itp"
    TopologyWriter(force_field)._write_protein_itp_for_indices(
        structure, path, list(map(int, result.system.components[0].atom_indices))
    )
    text = path.read_text()
    assert expected_n_type in text
    charges = _itp_residue_charges(path)
    assert charges[1] == pytest.approx(1.0, abs=1e-4)
    assert charges[2] == pytest.approx(-1.0, abs=1e-4)


def test_single_residue_chain_fails_without_partial_mutation():
    system = _two_residue_system("charmm36")
    keep = list(range(5))
    system.structure.coordinates = system.structure.coordinates[keep]
    for attribute in (
        "atom_names", "resnames", "resids", "chain_ids", "segids",
        "elements", "occupancies", "tempfactors",
    ):
        values = getattr(system.structure, attribute)
        setattr(system.structure, attribute, [values[index] for index in keep])
    system.components[0].atom_indices = np.arange(5)
    original = system.structure.copy()

    with pytest.raises(ModuleConfigError, match="Single-residue protein chain"):
        StructureProcessor().run(system, {"skip_protonation": True})

    assert system.structure.atom_names == original.atom_names
    np.testing.assert_array_equal(system.structure.coordinates, original.coordinates)


@pytest.mark.parametrize(
    "force_field", ["charmm36", "charmm36m", "amber99sb", "amber99sb-ildn"]
)
def test_explicit_ace_nme_caps_have_atoms_charges_and_cross_residue_bonds(
    tmp_path, force_field
):
    result = StructureProcessor().run(
        _two_residue_system(force_field),
        {
            "skip_protonation": True,
            "termini": {"A": {"nter": "ACE", "cter": "NME"}},
        },
    )
    structure = result.system.structure
    residue_order = list(dict.fromkeys(zip(structure.resnames, structure.resids)))
    assert [name for name, _ in residue_order] == ["ACE", "ALA", "GLY", "NME"]
    assert np.isfinite(structure.coordinates).all()
    assert result.system.metadata["terminal_caps"] == ["A:N=ACE", "A:C=NME"]

    path = tmp_path / f"capped-{force_field}.itp"
    TopologyWriter(force_field)._write_protein_itp_for_indices(
        structure, path, list(map(int, result.system.components[0].atom_indices))
    )
    charges = _itp_residue_charges(path)
    assert charges[min(charges)] == pytest.approx(0.0, abs=1e-4)
    assert charges[max(charges)] == pytest.approx(0.0, abs=1e-4)

    section = ""
    atom_labels = {}
    bonds = set()
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        fields = line.split()
        if section == "atoms" and len(fields) >= 5:
            atom_labels[(fields[3], fields[4])] = int(fields[0])
        elif section == "bonds" and len(fields) >= 2:
            bonds.add(tuple(sorted((int(fields[0]), int(fields[1])))))
    assert tuple(sorted((atom_labels[("ACE", "C")], atom_labels[("ALA", "N")]))) in bonds
    assert tuple(sorted((atom_labels[("GLY", "C")], atom_labels[("NME", "N")]))) in bonds

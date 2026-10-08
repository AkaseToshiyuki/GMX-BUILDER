"""Independent geometry, file-format and native-parameter repair contracts."""

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError, ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader
from gmxbuilder.io.mdp import derive_velocity_seed
from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol, write_mdp_files
from gmxbuilder.modules.forcefield.hdb import HDBHydrogenAdder, _compute_h_positions
from gmxbuilder.modules.forcefield.protein_templates import native_atom_types
from gmxbuilder.modules.modifications.protonation import assign_protonation


@pytest.mark.parametrize("method,cosine", [(None, -1 / 3), (3, -0.5), (6, -1 / 3)])
def test_two_hydrogens_obey_their_chemical_geometry(method, cosine):
    positions = np.array(
        _compute_h_positions(np.zeros(3), [np.array([0.1, 0, 0])], 2, method=method)
    )
    lengths = np.linalg.norm(positions, axis=1)
    assert lengths == pytest.approx([0.109, 0.109])
    assert np.dot(positions[0], positions[1]) / np.prod(lengths) == pytest.approx(cosine)


def test_partial_hydrogens_keep_their_named_positions(tmp_path):
    hdb = tmp_path / "test.hdb"
    hdb.write_text("LIG 1\n2 6 H C A B\n")
    adder = HDBHydrogenAdder(hdb)
    names, coordinates, residues, ids, chains = adder.add_hydrogens(
        ["C", "A", "B"],
        np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
        ["LIG"] * 3,
        [1] * 3,
        ["A"] * 3,
    )
    retained = [index for index, name in enumerate(names) if name != "H2"]
    rebuilt, xyz, *_ = adder.add_hydrogens(
        [names[i] for i in retained],
        coordinates[retained],
        [residues[i] for i in retained],
        [ids[i] for i in retained],
        [chains[i] for i in retained],
    )
    assert xyz[rebuilt.index("H2")] == pytest.approx(coordinates[names.index("H2")])
    assert not np.allclose(xyz[rebuilt.index("H1")], xyz[rebuilt.index("H2")])


def test_unknown_hdb_method_is_not_silently_tetrahedral():
    with pytest.raises(ValueError, match="Unsupported HDB"):
        _compute_h_positions(np.zeros(3), [], 2, method=99)


def test_gro_rejects_ambiguous_free_format(tmp_path):
    path = tmp_path / "bad.gro"
    path.write_text("bad\n1\n1 ALA N 1 0.100 0.200 0.300\n1 1 1\n")
    with pytest.raises(ParseError, match="Malformed atom"):
        GROReader().read(path)


@pytest.mark.parametrize("precision", [3, 5])
def test_gro_variable_precision_and_adjacent_negative_fields(tmp_path, precision):
    path = tmp_path / "valid.gro"
    values = [-123.12345, -234.23456, 345.34567]
    line = f"{1:5d}{'ALA':<5}{'CA':>5}{1:5d}"
    line += "".join(f"{value:{precision + 5}.{precision}f}" for value in values)
    path.write_text(f"valid\n1\n{line}\n1000 1000 1000\n")
    assert GROReader().read(path).coordinates[0] == pytest.approx(values, abs=10**-precision)


def test_normalized_web_seed_is_replaced_by_actual_build_seed(tmp_path):
    raw = normalize_protocol({}, has_membrane=True)
    seed = derive_velocity_seed(7)
    config = normalize_protocol(raw, has_membrane=True, velocity_seed=seed)
    assert config["velocity_seed"] == seed != raw["velocity_seed"]
    stages = write_mdp_files(tmp_path, config)
    velocity_stages = [(tmp_path / name).read_text() for _, name in stages]
    velocity_stages = [text for text in velocity_stages if "gen-vel = yes" in text]
    assert velocity_stages
    assert all(f"gen-seed = {seed}" in text for text in velocity_stages)


@pytest.mark.parametrize("ph,name,charge", [(2, "HISH", 1), (8, "HISE", 0)])
def test_opls_histidine_has_native_protonation_names(ph, name, charge):
    result = assign_protonation("HIS", ph, force_field="oplsaa")
    assert result["assigned_name"] == name
    assert result["charge"] == charge
    assert result["is_titratable"]


def test_rtp_parameters_replace_element_guessing():
    system = System(
        structure=Structure(
            coordinates=np.zeros((2, 3)),
            box_vectors=np.eye(3),
            atom_names=["N", "CA"],
            resnames=["ALA", "ALA"],
            resids=[1, 1],
            elements=["N", "C"],
        )
    )
    records = native_atom_types(system, "amber14sb")
    # Published ff14SB ALA atom parameters, distinct from element-level defaults.
    assert records[0].name == "N"
    assert records[0].charge == pytest.approx(-0.4157)
    assert records[1].name == "CX"
    assert records[1].charge == pytest.approx(0.0337)
    assert records[1].mass == pytest.approx(12.01)
    assert records[1].sigma == pytest.approx(0.339966950842)


def test_known_forcefield_never_guesses_missing_residue_charge():
    system = System(
        structure=Structure(coordinates=np.zeros((0, 3)), box_vectors=np.eye(3)),
        metadata={"force_field": "amber14sb"},
    )
    assert system.residue_formal_charge("SEP") == -2
    with pytest.raises(ValueError, match="No amber14sb residue charge"):
        system.residue_formal_charge("NOT_A_RESIDUE")


@pytest.mark.parametrize("element", ["Na", "Ca", "Se", "Fe"])
def test_mol2_two_letter_elements_are_never_shortened(tmp_path, element):
    from gmxbuilder.modules.forcefield.cgenff_import import _parse_mol2

    path = tmp_path / "ligand.mol2"
    path.write_text(f"@<TRIPOS>ATOM\n1 X 0 0 0 {element} 1 LIG 0\n@<TRIPOS>BOND\n")
    with pytest.raises(ModuleConfigError, match="Unsupported MOL2"):
        _parse_mol2(path)

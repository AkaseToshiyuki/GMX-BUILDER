"""Independent geometric counterexamples for identity and periodic imaging."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.geometry.molecular_identity import validate_stereochemistry, whole_molecule
from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability, lipid21_lipids
from gmxbuilder.modules.membrane.lipids import LipidRegistry


def embedded(smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=721) == 0
    return (
        tuple(atom.GetSymbol() for atom in molecule.GetAtoms()),
        tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()),
        molecule.GetConformer().GetPositions() / 10,
    )


def test_stereo_accepts_proper_rotation_and_rejects_mirror():
    smiles = "C[C@H](O)C(=O)O"
    elements, bonds, xyz = embedded(smiles)
    assert validate_stereochemistry(smiles, elements, bonds, xyz)["passed"]
    assert validate_stereochemistry(smiles, elements, bonds, xyz @ np.diag([-1, -1, 1]))["passed"]
    with pytest.raises(ValueError, match="stereochemistry mismatch"):
        validate_stereochemistry(smiles, elements, bonds, xyz @ np.diag([-1, 1, 1]))


def test_double_bond_geometry_rejects_opposite_isomer():
    elements, bonds, xyz = embedded("CC/C=C/CC")
    assert validate_stereochemistry("CC/C=C/CC", elements, bonds, xyz)["passed"]
    with pytest.raises(ValueError, match="E/Z configuration differs"):
        validate_stereochemistry("CC/C=C\\CC", elements, bonds, xyz)


def test_bondwise_unwrap_preserves_all_internal_distances_in_triclinic_box():
    box = np.array([[3.0, 0, 0], [0.4, 3.0, 0], [0.2, 0.3, 3.0]])
    xyz = np.array([[2.9, 2.9, 2.9], [3.03, 3.02, 2.95], [3.14, 3.1, 3.08]])
    wrapped = (xyz @ np.linalg.inv(box) % 1) @ box
    restored = whole_molecule(wrapped, [(0, 1), (1, 2)], box)
    assert np.allclose(restored - restored[0], xyz - xyz[0])


def test_disconnected_and_invalid_bond_graphs_are_not_accepted():
    xyz = np.zeros((3, 3))
    with pytest.raises(ValueError, match="disconnected"):
        whole_molecule(xyz, [(0, 1)], np.eye(3))
    with pytest.raises(ValueError, match="Invalid molecular bond"):
        whole_molecule(xyz, [(0, 4)], np.eye(3))


def test_lipid21_accepts_only_bundled_identities_matching_current_model():
    for name in lipid21_lipids():
        accepted, reason = lipid21_capability(name)
        assert accepted, (name, reason)


def test_same_tail_summary_does_not_make_trans_popc_an_exact_lipid21_match():
    lipid = LipidRegistry.get("POPC")
    molecule = Chem.MolFromSmiles(lipid.smiles)
    for bond in molecule.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOZ:
            bond.SetStereo(Chem.BondStereo.STEREOE)
    changed = replace(lipid, smiles=Chem.MolToSmiles(molecule))
    with LipidRegistry.task_scope({"POPC": changed}):
        accepted, reason = lipid21_capability("POPC")
        assert not accepted
        assert "identity" in reason
        from gmxbuilder.modules.forcefield.lipid21_backend import load_lipid21_geometry

        with pytest.raises(ValueError, match="E/Z"):
            load_lipid21_geometry("POPC")

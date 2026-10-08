"""Independent geometric invariants for bootstrap packing and seed recovery."""

import itertools

import numpy as np
import pytest

from gmxbuilder.geometry import rdkit_lipid as geometry
from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from tests.prerequisites import requires_forcefield


def local_geometry(xyz, names, rtp):
    """Measure every covalent bond and angle, including hydrogens at C3S."""
    adjacency = geometry._adjacency_from_bonds(names, rtp["bonds"])
    lengths, cosines = [], []
    for centre, neighbours in enumerate(adjacency):
        for other in neighbours:
            lengths.append(np.linalg.norm(xyz[other] - xyz[centre]))
        for first, second in itertools.combinations(neighbours, 2):
            a, b = xyz[first] - xyz[centre], xyz[second] - xyz[centre]
            cosines.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return np.array(lengths), np.array(cosines)


@requires_forcefield("charmm36m")
@pytest.mark.parametrize("lipid_name", ["PSM", "BSM", "CER18", "POPC", "PPCPL"])
@pytest.mark.parametrize("seed", range(5))
def test_tail_alignment_preserves_all_bond_lengths_angles_and_identity(lipid_name, seed):
    lipid = LipidRegistry.get(lipid_name)
    _, rtp = lipid_rtp_template(lipid_name, "charmm36m")
    mol, names = geometry._molecule_from_rtp(rtp)
    assert geometry._seed_explicit_stereochemistry(mol, lipid.smiles, seed)
    conformer = mol.GetConformer()
    geometry._sample_lipid_tail_torsions(mol, conformer, names, rtp, seed)
    before = geometry._orient_for_membrane(conformer.GetPositions() / 10, names)
    after = geometry._align_tail_subtrees(before.copy(), names, rtp, smiles=lipid.smiles)
    expected, observed = local_geometry(before, names, rtp), local_geometry(after, names, rtp)
    for first, second in zip(expected, observed, strict=True):
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-10)
    geometry._validate_rtp_bootstrap(lipid.smiles, names, rtp, after, check_overlap=False)


@requires_forcefield("charmm36m")
@pytest.mark.parametrize("seed", range(5))
def test_psm_each_seed_passes_without_relying_on_retry(seed):
    lipid = LipidRegistry.get("PSM")
    _, rtp = lipid_rtp_template("PSM", "charmm36m")
    xyz, names = geometry._build_rtp_seed(lipid.smiles, rtp, seed)
    geometry._validate_rtp_bootstrap(lipid.smiles, names, rtp, xyz)


def test_bad_generated_seed_retries_and_records_selected_seed(monkeypatch, caplog):
    attempted = []
    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.lipid_policy.lipid_rtp_template", lambda *a: ("X", {})
    )

    def build(smiles, rtp, seed):
        attempted.append(seed)
        if seed != 1:
            raise geometry._BootstrapGeometryError("invalid candidate")
        return np.array([[1.0, 2.0, 3.0]]), ("C1",)

    monkeypatch.setattr(geometry, "_build_rtp_seed", build)
    result = geometry._build_cached.__wrapped__("X", "C", "charmm36m", 2, 0)
    assert attempted == [2, 0, 1]
    assert result[1] == ("C1",)
    assert "using validated seed 1" in caplog.text


def test_all_bad_seeds_fail_with_each_attempt_recorded(monkeypatch):
    attempted = []
    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.lipid_policy.lipid_rtp_template", lambda *a: ("X", {})
    )

    def build(smiles, rtp, seed):
        attempted.append(seed)
        raise geometry._BootstrapGeometryError("invalid candidate")

    monkeypatch.setattr(geometry, "_build_rtp_seed", build)
    with pytest.raises(
        geometry._BootstrapGeometryError, match="No valid bootstrap geometry"
    ) as exc:
        geometry._build_cached.__wrapped__("X", "C", "charmm36m", 2, 0)
    assert attempted == [2, 0, 1, 3, 4]
    assert all(f"seed {seed}:" in str(exc.value) for seed in range(5))


@pytest.mark.parametrize("smiles", ["CC/C=C/CC", "C1CCCCC1"])
def test_only_non_ring_single_bonds_are_permitted(smiles):
    from rdkit import Chem

    from gmxbuilder.geometry.tail_torsions import _single_carbon_bonds

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    names = [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in molecule.GetAtoms()]
    rtp = {
        "bonds": [
            (names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]) for b in molecule.GetBonds()
        ]
    }
    permitted = _single_carbon_bonds(smiles, names, rtp)
    for bond in molecule.GetBonds():
        pair = frozenset((names[bond.GetBeginAtomIdx()], names[bond.GetEndAtomIdx()]))
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            assert pair not in permitted
    if "1" in smiles:
        assert not permitted
    else:
        assert permitted  # The check must not disable every torsion.

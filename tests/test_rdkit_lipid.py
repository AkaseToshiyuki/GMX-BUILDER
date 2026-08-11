import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry
from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_name
from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
from gmxbuilder.modules.membrane.builder import _select_spread_positions
from gmxbuilder.modules.membrane.lipid_orientation import (
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry


@pytest.mark.parametrize("lipid_name", ["MGDG", "DGDG"])
def test_galactolipid_registry_identity_matches_curated_structure(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    molecule = Chem.MolFromSmiles(lipid.smiles)

    assert molecule is not None
    assert rdMolDescriptors.CalcMolFormula(molecule) == lipid.formula
    assert Descriptors.MolWt(molecule) == pytest.approx(lipid.mass, abs=0.01)


def test_gm1_registry_is_deprotonated_d18_1_18_0_identity():
    lipid = LipidRegistry.get("GM1")
    molecule = Chem.MolFromSmiles(lipid.smiles)

    assert molecule is not None
    assert rdMolDescriptors.CalcMolFormula(molecule) == "C73H130N3O31-"
    assert Chem.GetFormalCharge(molecule) == lipid.charge == -1
    assert Descriptors.MolWt(molecule) == pytest.approx(lipid.mass, abs=0.01)

    coords, names = build_rdkit_lipid_geometry(
        "GM1", lipid.smiles, force_field="charmm36m", seed=7,
    )
    assert coords.shape == (237, 3)
    assert len(names) == len(set(names)) == 237
    assert np.isfinite(coords).all()


@pytest.mark.parametrize("lipid_name", ["POPC", "POPE", "POPG", "CHOL", "ERG"])
def test_rdkit_lipid_matches_charmm_rtp(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(lipid_name, lipid.smiles, seed=0)
    rtp_name = lipid_rtp_name(lipid_name, "charmm36m")
    template = load_force_field_rtp("charmm36m").get_residue(rtp_name)
    assert coords.shape == (len(names), 3)
    assert np.isfinite(coords).all()
    assert set(names) == {atom[0] for atom in template["atoms"]}


@pytest.mark.parametrize(
    "lipid_name,template_name",
    [
        ("BSM", "LSM"),
        ("CER16", "CER160"),
        ("CER18", "CER180"),
        ("CER24", "CER240"),
        ("DPEPE", "DYPE"),
        ("PUPC", "PDOPC"),
        ("TMCL", "TMCL2"),
        ("TOCL", "TOCL2"),
    ],
)
def test_charmm_identity_alias_builds_exact_rtp_geometry(lipid_name, template_name):
    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name, lipid.smiles, force_field="charmm36m", seed=0,
        net_charge=lipid.charge,
    )
    template = load_force_field_rtp("charmm36m").get_residue(template_name)
    assert coords.shape == (len(names), 3)
    assert np.isfinite(coords).all()
    assert tuple(names) == tuple(atom[0] for atom in template["atoms"])


@pytest.mark.parametrize(
    "lipid_name",
    ["PAPC", "PAPI", "SOPI", "DLIPS", "LPE16", "DSM", "PPCPL", "PPEPL"],
)
def test_modular_charmm_lipid_builds_exact_generated_geometry(lipid_name):
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name, lipid.smiles, force_field="charmm36m", seed=0,
        net_charge=lipid.charge,
    )
    _template_name, template = lipid_rtp_template(lipid_name, "charmm36m")
    assert np.isfinite(coords).all()
    assert tuple(names) == tuple(atom[0] for atom in template["atoms"])


def test_rdkit_builds_geometry_without_rtp(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    from gmxbuilder.geometry import rdkit_lipid
    rdkit_lipid._build_cached.cache_clear()
    lipid = LipidRegistry.get("CAMP")
    coords, names = build_rdkit_lipid_geometry("CAMP", lipid.smiles, seed=0)
    assert coords.shape == (len(names), 3)
    assert np.isfinite(coords).all()
    assert len(names) > 1


@pytest.mark.parametrize("lipid_name", ["20AHC", "22RHC", "24SHC", "25OHC", "27OHC"])
def test_oxysterol_orientation_uses_ring_hydroxyl_head(monkeypatch, tmp_path, lipid_name):
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid

    lipid = LipidRegistry.get(lipid_name)
    template = prepare_gaff_lipid(lipid_name, lipid.smiles, lipid.charge)
    profile = infer_lipid_orientation(template.coordinates, template.atom_names)
    oriented = orient_lipid_to_outward_normal(
        template.coordinates, template.atom_names, upper=True,
    )
    projection, cosine = outward_orientation(
        infer_lipid_orientation(oriented, template.atom_names), upper=True,
    )

    if lipid_name in {"22RHC", "27OHC"}:
        assert len(profile.polar_indices) == 1
    assert projection >= 0.10
    assert cosine >= 0.10


def test_spread_selection_avoids_adjacent_dense_grid_points():
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    points = np.column_stack((x.ravel(), y.ravel())).astype(float)
    chosen = _select_spread_positions(points, 8, np.random.default_rng(7))
    selected = points[chosen]
    distances = np.linalg.norm(selected[:, None] - selected[None, :], axis=2)
    distances[distances == 0.0] = np.inf
    assert distances.min() >= 2.0


def test_spread_selection_treats_opposite_periodic_faces_as_neighbours():
    points = np.asarray([
        [-2.95, 0.0], [2.95, 0.0], [0.0, 0.0],
        [0.0, 2.0], [0.0, -2.0],
    ])
    chosen = _select_spread_positions(
        points, 3, np.random.default_rng(3), box_xy=6.0
    )
    selected = points[chosen]
    periodic_distances = []
    for index, first in enumerate(selected):
        for second in selected[index + 1:]:
            delta = first - second
            delta -= 6.0 * np.round(delta / 6.0)
            periodic_distances.append(np.linalg.norm(delta))

    assert min(periodic_distances) > 1.0


def test_gaff_tail_alignment_never_introduces_intramolecular_overlap():
    lipid = LipidRegistry.get("DPPS")
    coords, names = build_rdkit_lipid_geometry(
        "DPPS", lipid.smiles, force_field="amber14sb", seed=0,
        net_charge=lipid.charge,
    )

    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert len(names) == len(coords)
    assert float(distances.min()) >= 0.05


@pytest.mark.parametrize("lipid_name", ["POPC", "DPPC", "DOPC", "POPE", "POPG"])
def test_phospholipid_conformations_are_not_all_trans_rods(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    spans = []
    for seed in range(5):
        coords, names = build_rdkit_lipid_geometry(
            lipid_name, lipid.smiles, force_field="charmm36m", seed=seed
        )
        name_index = {name: index for index, name in enumerate(names)}
        assert "P" in name_index
        spans.append(float(np.ptp(coords[:, 2])))

    # A fluid-bilayer starting ensemble must be visibly shorter than the
    # former all-trans templates (POPC was 2.70 nm; DOPC was 3.19 nm).
    assert np.mean(spans) < 2.45
    assert np.std(spans) > 0.02


@pytest.mark.parametrize("lipid_name", ["BSM", "NSM", "PSM", "SSM"])
def test_charmm_sphingomyelin_tails_are_extended_and_inward(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name, lipid.smiles, force_field="charmm36m", seed=0,
    )
    index = {name: number for number, name in enumerate(names)}
    f_terminal = max(
        (name for name in names if name.startswith("C") and name.endswith("F")),
        key=lambda name: int(name[1:-1]),
    )
    s_terminal = max(
        (name for name in names if name.startswith("C") and name.endswith("S")),
        key=lambda name: int(name[1:-1]),
    )

    assert np.linalg.norm(coords[index["C1F"]] - coords[index[f_terminal]]) > 1.0
    assert np.linalg.norm(coords[index["C3S"]] - coords[index[s_terminal]]) > 1.0
    assert coords[index[f_terminal], 2] < coords[index["C1F"], 2]
    assert coords[index[s_terminal], 2] < coords[index["C3S"], 2]

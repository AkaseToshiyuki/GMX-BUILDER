from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from gmxbuilder.geometry.rdkit_lipid import (
    _seed_explicit_stereochemistry,
    build_rdkit_lipid_geometry,
)
from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_name
from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
from gmxbuilder.modules.membrane.builder import _select_spread_positions
from gmxbuilder.modules.membrane.lipid_orientation import (
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from tests.prerequisites import requires_forcefield, requires_gaff_runtime


def test_explicit_smiles_stereochemistry_seeds_rtp_ordered_coordinates():
    smiles = "C[C@H](O)C(=O)O"
    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    target = Chem.Mol(reference)
    for atom in target.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    target.RemoveAllConformers()

    assert _seed_explicit_stereochemistry(target, smiles, 17) is True
    Chem.AssignAtomChiralTagsFromStructure(target, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(target, cleanIt=True, force=True)
    Chem.AssignStereochemistry(reference, cleanIt=True, force=True)

    expected = [
        atom.GetProp("_CIPCode") for atom in reference.GetAtoms() if atom.HasProp("_CIPCode")
    ]
    observed = [atom.GetProp("_CIPCode") for atom in target.GetAtoms() if atom.HasProp("_CIPCode")]
    assert observed == expected == ["S"]


@pytest.mark.parametrize("lipid_name", ["MGDG", "DGDG"])
def test_galactolipid_registry_identity_matches_curated_structure(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    molecule = Chem.MolFromSmiles(lipid.smiles)

    assert molecule is not None
    assert rdMolDescriptors.CalcMolFormula(molecule) == lipid.formula
    assert Descriptors.MolWt(molecule) == pytest.approx(lipid.mass, abs=0.01)


@pytest.mark.slow
def test_gm1_registry_is_deprotonated_d18_1_18_0_identity():
    lipid = LipidRegistry.get("GM1")
    molecule = Chem.MolFromSmiles(lipid.smiles)

    assert molecule is not None
    assert rdMolDescriptors.CalcMolFormula(molecule) == "C73H130N3O31-"
    assert Chem.GetFormalCharge(molecule) == lipid.charge == -1
    assert Descriptors.MolWt(molecule) == pytest.approx(lipid.mass, abs=0.01)

    coords, names = build_rdkit_lipid_geometry(
        "GM1",
        lipid.smiles,
        force_field="charmm36m",
        seed=7,
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
@pytest.mark.slow
def test_charmm_identity_alias_builds_exact_rtp_geometry(lipid_name, template_name):
    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name,
        lipid.smiles,
        force_field="charmm36m",
        seed=0,
        net_charge=lipid.charge,
    )
    template = load_force_field_rtp("charmm36m").get_residue(template_name)
    assert coords.shape == (len(names), 3)
    assert np.isfinite(coords).all()
    assert tuple(names) == tuple(atom[0] for atom in template["atoms"])


@pytest.mark.parametrize("lipid_name", ["PAPI", "SOPI"])
def test_corrected_phosphoinositide_isomer_is_preserved(lipid_name):
    """Correct native templates preserve the declared graph and stereochemistry."""
    from gmxbuilder.geometry.molecular_identity import validate_stereochemistry
    from gmxbuilder.modules.membrane.lipid_graph import ordered_graph

    lipid = LipidRegistry.get(lipid_name)
    coordinates, names = build_rdkit_lipid_geometry(
        lipid_name,
        lipid.smiles,
        force_field="charmm36m",
        seed=0,
        net_charge=lipid.charge,
    )
    elements, bonds = ordered_graph(
        lipid_name, "charmm36m", "charmm36m", lipid.smiles, tuple(names)
    )
    assert validate_stereochemistry(lipid.smiles, elements, bonds, coordinates)["passed"]


@pytest.mark.parametrize(
    "lipid_name",
    ["PAPC", "DLIPS", "LPE16", "DSM", "PPCPL", "PPEPL"],
)
@pytest.mark.slow
def test_modular_charmm_lipid_builds_exact_generated_geometry(lipid_name):
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name,
        lipid.smiles,
        force_field="charmm36m",
        seed=0,
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
@requires_gaff_runtime
@pytest.mark.slow
def test_oxysterol_orientation_uses_ring_hydroxyl_head(monkeypatch, tmp_path, lipid_name):
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid

    lipid = LipidRegistry.get(lipid_name)
    template = prepare_gaff_lipid(lipid_name, lipid.smiles, lipid.charge)
    profile = infer_lipid_orientation(template.coordinates, template.atom_names)
    oriented = orient_lipid_to_outward_normal(
        template.coordinates,
        template.atom_names,
        upper=True,
    )
    projection, cosine = outward_orientation(
        infer_lipid_orientation(oriented, template.atom_names),
        upper=True,
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
    points = np.asarray(
        [
            [-2.95, 0.0],
            [2.95, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    chosen = _select_spread_positions(points, 3, np.random.default_rng(3), box_xy=6.0)
    selected = points[chosen]
    periodic_distances = []
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            delta = first - second
            delta -= 6.0 * np.round(delta / 6.0)
            periodic_distances.append(np.linalg.norm(delta))

    assert min(periodic_distances) > 1.0


@requires_gaff_runtime
def test_gaff_tail_alignment_never_introduces_intramolecular_overlap():
    lipid = LipidRegistry.get("DPPS")
    coords, names = build_rdkit_lipid_geometry(
        "DPPS",
        lipid.smiles,
        force_field="amber14sb",
        seed=0,
        net_charge=lipid.charge,
    )

    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert len(names) == len(coords)
    assert float(distances.min()) >= 0.05


def test_rtp_tail_alignment_falls_back_when_it_creates_overlap(monkeypatch):
    import gmxbuilder.geometry.rdkit_lipid as module

    module._build_cached.cache_clear()
    original = module._align_tail_subtrees

    def collapsed(coords, names, rtp, **kwargs):
        result = original(coords, names, rtp, **kwargs)
        result[1] = result[0]
        return result

    monkeypatch.setattr(module, "_align_tail_subtrees", collapsed)
    lipid = LipidRegistry.get("POPC")
    coords, names = module.build_rdkit_lipid_geometry(
        "POPC",
        lipid.smiles,
        force_field="charmm36m",
        seed=928,
    )

    assert not module._has_intramolecular_overlap(coords)
    assert len(coords) == len(names)
    module._build_cached.cache_clear()


@pytest.mark.parametrize("lipid_name", ["POPC", "DPPC", "DOPC", "POPE", "POPG"])
@pytest.mark.slow
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

    # A fluid-bilayer starting ensemble must look like the bilayer, not like
    # the former all-trans templates (POPC was 2.70 nm; DOPC was 3.19 nm). The
    # bound is measured, not assumed: the shipped library holds 50 conformers
    # per lipid taken from a simulated bilayer, and their z-spans mean 2.56 nm
    # for POPC, 2.59 for DPPC, 2.57 for DOPC, 2.54 for POPE and 2.41 for POPG.
    # A flat 2.45 nm stood here before the lipids declared their
    # stereochemistry, and it was tighter than the simulation it was meant to
    # describe -- it would have rejected the library's own frames for three of
    # these five.
    simulated_mean = {"POPC": 2.563, "DPPC": 2.590, "DOPC": 2.573, "POPE": 2.540, "POPG": 2.406}[
        lipid_name
    ]
    assert np.mean(spans) < simulated_mean + 0.10
    assert np.std(spans) > 0.02


@pytest.mark.parametrize("lipid_name", ["BSM", "NSM", "PSM", "SSM"])
def test_charmm_sphingomyelin_tails_are_extended_and_inward(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    coords, names = build_rdkit_lipid_geometry(
        lipid_name,
        lipid.smiles,
        force_field="charmm36m",
        seed=0,
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


# ---------------------------------------------------------------------------
# Headgroup classification for uploaded lipids


def test_uploaded_cholesterol_is_recognised_as_a_sterol():
    """The upload path's single most likely molecule, and it used to be broken.

    Category decides whether the equilibration builder pre-equilibrates a lipid
    at 40 mol% in a POPC host. Sterols need that host because they cannot form
    a neat bilayer at all. Classification by SMILES substring never matched the
    sterol nucleus -- RDKit canonicalisation rewrites ring closures and inserts
    stereo markers -- so cholesterol fell through to the PC default, lost its
    host, and was built as a pure cholesterol bilayer that cannot exist.
    """
    from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid

    reference = LipidRegistry.get("CHOL")
    properties = parse_custom_lipid(reference.smiles, "CHL2")
    assert properties["category"] == "ST"
    assert properties["formula"] == "C27H46O"


@pytest.mark.parametrize(
    "name",
    ["CHOL", "ERG", "SITO", "STIG", "CAMP", "CER16", "LPC16", "GM1", "PAPI", "POP2"],
)
def test_lipids_that_need_a_popc_host_keep_it(name):
    """Every class the equilibration builder hosts must survive classification."""
    from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid

    hosted = {"ST", "LPC", "DG", "CER", "GM1", "PIP"}
    reference = LipidRegistry.get(name)
    assert reference.category in hosted, "fixture no longer describes a hosted class"
    assert parse_custom_lipid(reference.smiles, "LIPX")["category"] in hosted


def test_classification_recovers_the_registry_it_is_judged_against():
    """Feed every built-in lipid its own SMILES; the class must come back.

    The substring matcher recovered 36% and cost nineteen lipids their host.
    LYSPG is the one accepted miss: its class in the registry is lysyl-PG, a
    diacyl lipid, while the same LPG label elsewhere means a lyso lipid, so no
    structural rule can satisfy both readings.
    """
    from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid

    hosted = {"ST", "LPC", "DG", "CER", "GM1", "PIP"}
    recovered, missed, lost_host = 0, [], []
    for name in LipidRegistry.list_builtin():
        reference = LipidRegistry.get(name)
        if not reference.smiles:
            continue
        found = parse_custom_lipid(reference.smiles, "LIPX")["category"]
        if found == reference.category:
            recovered += 1
            continue
        missed.append(name)
        if reference.category in hosted and found not in hosted:
            lost_host.append(name)

    assert not lost_host, f"these lipids would be built without a host: {lost_host}"
    assert missed == ["LYSPG"], f"unexpected misclassification: {missed}"
    assert recovered >= 0.95 * (recovered + len(missed))


def test_estimated_mass_and_formula_come_from_the_structure():
    """The old element counter guessed implicit hydrogens as valence // 2."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from gmxbuilder.modules.membrane.lipids import LipidRegistry, parse_custom_lipid

    for name in ("POPC", "DPPC", "CHOL"):
        reference = LipidRegistry.get(name)
        properties = parse_custom_lipid(reference.smiles, "LIPX")
        exact = Descriptors.MolWt(Chem.MolFromSmiles(reference.smiles))
        assert properties["mass"] == pytest.approx(exact, rel=1e-3)


# ---------------------------------------------------------------------------
# The registry against the force field's own lipid definitions


@requires_forcefield("charmm36m")
@pytest.mark.parametrize("force_field", ["charmm36m", "charmm36"])
def test_corrected_registry_native_mappings_have_no_graph_disagreements(force_field):
    """Correct aliases match the registry without changing molecular identities."""
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_has_rtp
    from gmxbuilder.modules.forcefield.native_lipids import registry_disagreements
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    assert registry_disagreements(force_field) == []
    for name in ("POP2", "PAPI", "SAPI", "SOP2"):
        assert lipid_has_rtp(name, force_field)
        assert resolve_protocol(name, force_field + "-lipid")


def test_every_registry_smiles_matches_its_own_declared_metadata():
    """Formula, mass and charge are declared separately; they must agree.

    This is the check for the lipids no force field defines, where the declared
    formula is the only authority available.
    """
    import re

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    def elements(formula):
        body = re.sub(r"[+-]\d*$", "", formula)
        return {
            match.group(1): int(match.group(2) or 1)
            for match in re.finditer(r"([A-Z][a-z]?)(\d*)", body)
            if match.group(1)
        }

    wrong = []
    for name in LipidRegistry.list_builtin():
        lipid = LipidRegistry.get(name)
        if not lipid.smiles:
            continue
        molecule = Chem.MolFromSmiles(lipid.smiles)
        assert molecule is not None, f"{name} has an unparseable SMILES"
        if elements(rdMolDescriptors.CalcMolFormula(molecule)) != elements(lipid.formula):
            wrong.append(f"{name} formula {lipid.formula}")
        elif abs(Descriptors.MolWt(molecule) - lipid.mass) / max(lipid.mass, 1) > 0.005:
            wrong.append(f"{name} mass {lipid.mass}")
        elif Chem.GetFormalCharge(molecule) != lipid.charge:
            wrong.append(f"{name} charge {lipid.charge}")
    assert wrong == []


def test_no_two_lipids_share_one_structure():
    """Ergosterol and stigmasterol shipped the same SMILES; they are not equal."""
    from collections import defaultdict

    from rdkit import Chem

    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    seen = defaultdict(list)
    for name in LipidRegistry.list_builtin():
        lipid = LipidRegistry.get(name)
        if lipid.smiles:
            seen[Chem.MolToSmiles(Chem.MolFromSmiles(lipid.smiles))].append(name)
    assert [names for names in seen.values() if len(names) > 1] == []


@requires_forcefield("charmm36m")
def test_native_lipid_matching_separates_lipids_a_fingerprint_cannot():
    """DPPC and SMPC have one formula, one atom count and one WL fingerprint.

    16:0/16:0 against 18:0/14:0: the same thirty-two chain carbons split two
    ways. Every interior methylene looks alike four bonds out and both have two
    chain ends and two esters, so the descriptor multisets are equal. Only an
    exact isomorphism test tells them apart, and without one the upload guard
    would refuse a legitimate lipid as a duplicate of a different one.
    """
    from gmxbuilder.modules.forcefield.native_lipids import (
        _lipid21_graph,
        find_native_lipid,
        graph_key,
        same_molecule,
    )

    root = Path(__file__).resolve().parents[1] / "src" / "gmxbuilder" / "data" / "lipid21" / "itp"
    dppc = _lipid21_graph(root / "DPPC.itp")
    smpc = _lipid21_graph(root / "SMPC.itp")
    assert dppc is not None and smpc is not None
    assert graph_key(dppc) == graph_key(smpc), "fixture no longer exercises the collision"
    assert not same_molecule(dppc, smpc)

    matched = {
        entry.residue for entry in find_native_lipid(LipidRegistry.get("DPPC").smiles, "charmm36m")
    }
    assert matched == {"DPPC"}


@requires_forcefield("charmm36m")
def test_uploading_a_lipid_the_force_field_defines_is_refused():
    """Official parameters must win over anything generated for an upload."""
    import tempfile

    from gmxbuilder.modules.membrane.lipids import canonical_lipid_identity
    from gmxbuilder.web.custom_lipids import CustomLipidStore

    store = CustomLipidStore(Path(tempfile.mkdtemp()))
    identity = canonical_lipid_identity(LipidRegistry.get("CHOL").smiles)
    with pytest.raises(ValueError, match="shares element-labelled connectivity"):
        store.save_submission(
            {"name": "XCHL", "canonical_smiles": identity["canonical_smiles"]}, "charmm36m"
        )

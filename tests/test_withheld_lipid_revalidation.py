"""Regressions for the concrete reasons that blocked V4 candidates."""

import pytest
from rdkit import Chem

from gmxbuilder.geometry.molecular_identity import itp_graph, validate_stereochemistry
from gmxbuilder.geometry.stereochemistry import stereo_reference
from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_itp_path, load_lipid21_geometry
from gmxbuilder.modules.membrane.lipids import LipidRegistry


def unassigned_carbons(mol):
    return [
        i
        for i, code in Chem.FindMolChiralCenters(
            stereo_reference(mol), includeUnassigned=True, useLegacyImplementation=False
        )
        if code == "?" and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
    ]


@pytest.mark.parametrize("name", ["TMCL", "TOCL"])
def test_symmetric_cardiolipin_has_no_missing_carbon_configuration(name):
    mol = Chem.MolFromSmiles(LipidRegistry.get(name).smiles)
    before = Chem.MolToSmiles(mol)
    projected = stereo_reference(mol)
    assert unassigned_carbons(mol) == []
    assert Chem.GetFormalCharge(projected) == Chem.GetFormalCharge(mol) == -2
    assert Chem.MolToSmiles(mol) == before
    assert sorted(code for _, code in Chem.FindMolChiralCenters(projected)) == ["R", "R"]
    # Isotopically distinguish one terminal methyl: the arms are no longer
    # identical, and the central glycerol really DOES need specification.
    next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetDegree() == 1).SetIsotope(13)
    assert len(unassigned_carbons(mol)) == 1


def test_resonance_projection_does_not_hide_a_real_unspecified_carbon():
    assert len(unassigned_carbons(Chem.MolFromSmiles("CC(O)C(=O)O"))) == 1


@pytest.mark.parametrize("name", ["DAPG", "DMPG", "DOPG", "DPPG", "PAPG", "POPG", "SOPG"])
def test_pg_uses_official_pgs_coordinates_and_parameters(name):
    from gmxbuilder.modules.forcefield.lipid21_backend import _templates, lipid21_sequence

    assert lipid21_sequence(name)[1] == "PGS"
    assert _templates()[name]["source_modules"][1] == "PGS"
    coords, names = load_lipid21_geometry(name)
    order, elements, bonds = itp_graph(lipid21_itp_path(name))
    assert list(order) == names
    assert validate_stereochemistry(LipidRegistry.get(name).smiles, elements, bonds, coords)[
        "passed"
    ]


@pytest.mark.parametrize(
    "lipid,template,proton",
    [
        ("POP2", "POPI24", "HP42"),
        ("SAPI", "SAPI24", "HP42"),
        ("PAPI", "PAPI", "HP42"),
        ("SOP2", "SOP2", "HP32"),
    ],
)
def test_pip_templates_preserve_declared_proton_location(lipid, template, proton):
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    actual, record = lipid_rtp_template(lipid, "charmm36m")
    assert actual == template
    assert proton in {a[0] for a in record["atoms"]}
    assert sum(a[2] for a in record["atoms"]) == pytest.approx(-4)


def test_policy_compatibility_never_reuses_changed_native_parameters():
    from pathlib import Path

    from gmxbuilder.modules.membrane import parameter_provenance as p

    relative = "modules/forcefield/lipid_policy.py"
    source = Path(p.__file__).parents[2] / relative
    assert p.file_hash(source) == p._PIP_TEMPLATE_POLICY_HASH
    assert (
        p._implementation_hash(source, relative, unchanged_pip_template=False)
        == p._PIP_TEMPLATE_POLICY_HASH
    )
    assert p._implementation_hash(source, relative) != p._PIP_TEMPLATE_POLICY_HASH

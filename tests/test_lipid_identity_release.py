"""Independent named-identity references, separate from parameter/MD acceptance."""

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from gmxbuilder.modules.membrane.chain_identity import chain_identity
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from tests.test_lipid_stereochemistry import _neutral

REFERENCES = json.loads(
    (Path(__file__).parent / "fixtures/lipid_identity_references.json").read_text()
)


@pytest.mark.parametrize("reference", REFERENCES, ids=lambda x: x["name"])
def test_corrected_identity_matches_independent_reference(reference):
    lipid = LipidRegistry.get(reference["name"])
    assert _neutral(lipid.smiles) == _neutral(reference["reference_smiles"])
    molecule = Chem.MolFromSmiles(lipid.smiles)
    assert Chem.GetFormalCharge(molecule) == lipid.charge
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    import re

    assert re.sub(r"[+-]\d*$", "", formula) == lipid.formula


@pytest.mark.parametrize("name", ["DAPG", "DLIPG", "DMPG", "DOPG", "DPPG", "PAPG", "POPG", "SOPG"])
def test_pg_uses_natural_rs_not_previous_rr(name):
    mol = Chem.MolFromSmiles(LipidRegistry.get(name).smiles)
    assert sorted(code for _, code in Chem.FindMolChiralCenters(mol)) == ["R", "S"]


@pytest.mark.parametrize("name", ["POPG", "DOPG", "DLIPG", "DAPG"])
def test_pg_mass_matches_formula_atomic_weights(name):
    lipid = LipidRegistry.get(name)
    assert lipid.mass == pytest.approx(
        Descriptors.MolWt(Chem.MolFromSmiles(lipid.smiles)), abs=0.002
    )


def test_lysine_is_headgroup_not_third_acyl_chain():
    assert sorted(
        c["length"] for c in chain_identity(LipidRegistry.get("LYSPG").smiles)["chains"]
    ) == [16, 18]


@pytest.mark.parametrize("name", ["POP2", "PAPI", "SAPI", "SOP2", "POPG", "LYSPG", "TMCL", "TOCL"])
def test_recertified_native_models_are_queue_candidates(name):
    assert resolve_protocol(name, "charmm36m-lipid")["lipid"] == name


def test_construction_acceptance_alone_cannot_claim_equilibrium():
    from gmxbuilder.modules.membrane.v4_evidence import validation_scope

    assert validation_scope({"status": "ready", "area_converged": True})["kind"] == "initialization"

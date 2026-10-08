"""Chain labels distinguish positional and geometric isomers without guessing."""

import pytest

from gmxbuilder.modules.membrane.chain_identity import chain_identity
from gmxbuilder.modules.membrane.lipids import LipidRegistry


@pytest.mark.parametrize(
    "smiles,label",
    [
        ("CCCCCCCC/C=C\\CCCCCCCC(=O)O", "18:1(9Z)"),
        ("CCCCCCCC/C=C/CCCCCCCC(=O)O", "18:1(9E)"),
        ("CCCCCC/C=C\\CCCCCCCCCC(=O)O", "18:1(11Z)"),
        ("CCCCCCCCC=CCCCCCCCC(=O)O", "18:1(9?)"),
    ],
)
def test_position_and_geometry_come_from_the_full_graph(smiles, label):
    assert chain_identity(smiles)["chains"][0]["label"] == label


@pytest.mark.parametrize(
    "name,expected",
    [
        ("POPC", ["16:0", "18:1(9Z)"]),
        ("DPePE", ["16:1(9Z)", "16:1(9Z)"]),
        ("DEPC", ["22:1(13Z)", "22:1(13Z)"]),
        ("PAPC", ["16:0", "20:4(5Z,8Z,11Z,14Z)"]),
        ("TMCL", ["14:0"] * 4),
    ],
)
def test_known_chain_identities(name, expected):
    result = chain_identity(LipidRegistry.get(name).smiles)
    assert sorted(c["label"] for c in result["chains"]) == sorted(expected)


def test_sphingoid_backbone_is_not_lost_as_a_zero_length_tail():
    chains = chain_identity(LipidRegistry.get("CER18").smiles)["chains"]
    assert {(c["linkage"], c["label"]) for c in chains} == {
        ("amide", "18:0"),
        ("sphingoid", "18:1(4E)"),
    }


def test_vinyl_ether_is_not_reported_as_an_ester():
    chains = chain_identity(LipidRegistry.get("PPEPL").smiles)["chains"]
    assert any(c["linkage"] == "vinyl-ether" for c in chains)
    assert len(chains) == 2


def test_invalid_or_cyclic_structure_does_not_invent_chain_positions():
    assert chain_identity("")["chains"] == []
    assert chain_identity(LipidRegistry.get("CHOL").smiles)["chains"] == []


def test_callers_cannot_mutate_cached_chain_annotations():
    smiles = LipidRegistry.get("POPC").smiles
    chain_identity(smiles)["chains"].clear()
    assert len(chain_identity(smiles)["chains"]) == 2

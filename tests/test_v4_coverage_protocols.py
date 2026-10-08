"""Independent reference values and identity invariants for expanded V4 coverage."""

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from gmxbuilder.geometry.molecular_identity import validate_stereochemistry
from gmxbuilder.modules.forcefield.gaff_lipid_identity import write_lipid_sdf
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol


@pytest.mark.parametrize("family", ["charmm36-lipid", "charmm36m-lipid"])
@pytest.mark.parametrize(
    "name,temperature",
    [
        ("DAPC", 308.15),
        ("DMPG", 308.15),
        ("DOPG", 308.15),
        ("DOPS", 308.15),
        ("DPPG", 329.15),
        ("DSPC", 338.0),
        ("POPA", 316.15),
        ("POPG", 308.15),
    ],
)
def test_recertified_charmm_pairs_retain_thermal_protocol(name, temperature, family):
    # Experimental Tm references are not mislabelled as simulation temperatures.
    protocol = resolve_protocol(name, family)
    assert protocol["temperature_K"] == pytest.approx(temperature)
    assert protocol["environment"] == "pure"
    assert protocol["composition"] == {name: 100}
    if name not in {"DMPG", "POPG", "DSPC"}:
        assert "reference_temperature_K" not in protocol
        assert protocol["evidence_type"] == "experimental_transition"


def test_gaff_temperature_is_not_claimed_as_lipid21_reference():
    protocol = resolve_protocol("DPPC", "amber-gaff2")
    assert protocol["bilayer_transition_K"] == 314.15  # Experimental 41 C.
    assert protocol["temperature_K"] == pytest.approx(329.15)
    assert "reference_temperature_K" not in protocol
    assert resolve_protocol("DPPC", "amber-lipid21")["reference_temperature_K"] == 323.0


def test_unknown_pure_phase_uses_explicit_conformer_pilot():
    protocol = resolve_protocol("DAPE", "charmm36m-lipid")
    assert protocol["composition"] == {"POPC": 90, "DAPE": 10}
    assert protocol["temperature_K"] == 315.15
    assert protocol["lipids_per_leaflet"] == 200
    assert protocol["pure_phase_suitability"] == "unresolved"
    assert protocol["intended_use"] == "initialization_conformers_only"
    assert "bilayer_transition_K" not in protocol
    assert "reference_temperature_K" not in protocol


@pytest.mark.parametrize(
    "family", ["charmm36m-lipid", "charmm36-lipid", "amber-lipid21", "amber-gaff2"]
)
def test_dilipe_known_nonlamellar_boundary_uses_host_not_pure_tm_rule(family):
    protocol = resolve_protocol("DLIPE", family)
    assert protocol["hexagonal_transition_K"] == 258.15
    assert protocol["temperature_K"] == 315.15
    assert protocol["composition"] == {"POPC": 90, "DLIPE": 10}
    assert protocol["environment"] == "host"


def test_ergosterol_matches_independent_pubchem_reference():
    # CID 444679, independently retrieved 2026-09-11. Different atom traversal
    # prevents a text-equality test from simply mirroring the registry edit.
    reference = (
        "C[C@H](/C=C/[C@H](C)C(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3C2=CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
    )
    assert Chem.MolToSmiles(
        Chem.MolFromSmiles(LipidRegistry.get("ERG").smiles)
    ) == Chem.MolToSmiles(Chem.MolFromSmiles(reference))


def test_acpype_input_retains_specified_stereo_and_mirror_is_rejected(tmp_path):
    smiles = "C[C@H](O)/C=C/C"
    path = tmp_path / "input.sdf"
    write_lipid_sdf(smiles, path)
    molecule = Chem.SDMolSupplier(str(path), removeHs=False)[0]
    elements = tuple(a.GetSymbol() for a in molecule.GetAtoms())
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())
    coordinates = np.asarray(molecule.GetConformer().GetPositions()) / 10
    assert validate_stereochemistry(smiles, elements, bonds, coordinates)["passed"]
    coordinates[:, 0] *= -1
    with pytest.raises(ValueError, match="stereochemistry mismatch"):
        validate_stereochemistry(smiles, elements, bonds, coordinates)


def test_protocol_catalog_is_complete_and_unique():
    path = Path(__file__).parents[1] / "src/gmxbuilder/data/v4_protocols.json"
    records = json.loads(path.read_text())["records"]
    families = {"charmm36-lipid", "charmm36m-lipid", "amber-lipid21", "amber-gaff2"}
    assert len(records) == len(LipidRegistry.list_builtin()) * len(families)
    assert {(r["lipid"], r["family"]) for r in records} == {
        (name, family) for name in LipidRegistry.list_builtin() for family in families
    }


def test_cache_reader_rejects_mirror_despite_matching_atom_names_and_charge(tmp_path):
    from gmxbuilder.core.structure import Structure
    from gmxbuilder.io.gro import GROWriter
    from gmxbuilder.modules.forcefield.gaff_backend import _load_cached

    smiles = "C[C@H](O)F"
    write_lipid_sdf(smiles, tmp_path / "input.sdf")
    molecule = Chem.SDMolSupplier(str(tmp_path / "input.sdf"), removeHs=False)[0]
    names = tuple(f"{a.GetSymbol()}{a.GetIdx()}" for a in molecule.GetAtoms())
    atoms = [
        f"{i + 1} t 1 LIP {names[i]} 1 0 {atom.GetMass():.4f}"
        for i, atom in enumerate(molecule.GetAtoms())
    ]
    bonds = [f"{b.GetBeginAtomIdx() + 1} {b.GetEndAtomIdx() + 1} 1" for b in molecule.GetBonds()]
    (tmp_path / "lipid.itp").write_text(
        "[ atoms ]\n" + "\n".join(atoms) + "\n[ bonds ]\n" + "\n".join(bonds)
    )
    (tmp_path / "atomtypes.itp").write_text("[ atomtypes ]\n")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            dict(
                name="LIP",
                smiles=smiles,
                atom_names=names,
                num_atoms=len(names),
                net_charge=0,
                charge_method="bcc",
            )
        )
    )
    structure = Structure(
        coordinates=np.asarray(molecule.GetConformer().GetPositions()) / 10,
        box_vectors=np.eye(3) * 5,
        atom_names=names,
        resnames=["LIP"] * len(names),
        resids=[1] * len(names),
    )
    GROWriter.write(structure, tmp_path / "lipid.gro")
    assert _load_cached(tmp_path, expected_smiles=smiles) is not None
    structure.coordinates[:, 0] *= -1
    GROWriter.write(structure, tmp_path / "lipid.gro")
    assert _load_cached(tmp_path) is not None  # The old structural checks still pass.
    assert _load_cached(tmp_path, expected_smiles=smiles) is None


def test_failed_library_retry_is_scoped_and_does_not_unlock_public_support(
    unpopulated_default_lipid_library,
):
    from gmxbuilder.modules.forcefield.lipid_policy import (
        gaff_lipid_capability,
        rebuilding_library_entry,
    )

    assert not gaff_lipid_capability("CHOL")[0]
    with rebuilding_library_entry():
        assert not gaff_lipid_capability("CHOL")[0]
    with rebuilding_library_entry(retry_failed_validation=True):
        assert gaff_lipid_capability("CHOL")[0]
        with rebuilding_library_entry():
            assert gaff_lipid_capability("CHOL")[0]
    assert not gaff_lipid_capability("CHOL")[0]


def test_gm1_matches_independent_pubchem_stereochemistry():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    # PubChem CID 9963963, neutral GM1a(d18:1/18:0), retrieved 2026-09-11.
    # Compare after neutralisation: the registry deliberately stores sialate.
    reference = (
        "CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O[C@H]2[C@"
        "@H]([C@H]([C@H]([C@H](O2)CO)O[C@H]3[C@@H]([C@H]([C@H]([C@H](O3)CO)O)O[C@H]4[C@@H"
        "]([C@H]([C@H]([C@H](O4)CO)O)O)O)NC(=O)C)O[C@@]5(C[C@@H]([C@H]([C@@H](O5)[C@@H](["
        "C@@H](CO)O)O)NC(=O)C)O)C(=O)O)O)O)O)[C@@H](/C=C/CCCCCCCCCCCCC)O"
    )
    molecule = Chem.MolFromSmiles(LipidRegistry.get("GM1").smiles)
    assert Chem.GetFormalCharge(molecule) == -1
    assert Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(molecule)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(reference)
    )


@pytest.mark.parametrize("name", ["POP3", "SOP3"])
def test_pip3_exact_graph_survives_phosphate_symmetry_and_rejects_wrong_hydrogens(name):
    from gmxbuilder.geometry.molecular_identity import reference_mappings
    from gmxbuilder.geometry.rdkit_lipid import _molecule_from_rtp
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template
    from tests.prerequisites import forcefield_parameters_available

    if not forcefield_parameters_available("charmm36m"):
        pytest.skip("CHARMM36m parameters are not installed")
    _, template = lipid_rtp_template(name, "charmm36m")
    molecule, names = _molecule_from_rtp(template)
    elements = tuple(a.GetSymbol() for a in molecule.GetAtoms())
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())
    smiles = LipidRegistry.get(name).smiles
    assert reference_mappings(smiles, elements, bonds)
    # Move the native P5 proton onto a non-phosphate carbon: same elements,
    # same atom count and net charge, but a different bonded-H identity.
    oxygen, hydrogen = names.index("OP52"), names.index("HP52")
    bad = tuple(
        (names.index("C2"), hydrogen) if {i, j} == {oxygen, hydrogen} else (i, j) for i, j in bonds
    )
    with pytest.raises(ValueError, match="cannot be mapped"):
        reference_mappings(smiles, elements, bad)


@pytest.mark.parametrize("distortion", ["scale", "mirror"])
def test_optional_gaff_packing_cannot_corrupt_valid_coordinates(tmp_path, monkeypatch, distortion):
    from types import SimpleNamespace

    from gmxbuilder.geometry import rdkit_lipid

    smiles = "C[C@H](O)F"
    write_lipid_sdf(smiles, tmp_path / "input.sdf")
    molecule = Chem.SDMolSupplier(str(tmp_path / "input.sdf"), removeHs=False)[0]
    names = tuple(f"{a.GetSymbol()}{a.GetIdx()}" for a in molecule.GetAtoms())
    xyz = np.asarray(molecule.GetConformer().GetPositions()) / 10
    atoms = [
        f"{i + 1} t 1 LIP {names[i]} 1 0 {a.GetMass():.4f}"
        for i, a in enumerate(molecule.GetAtoms())
    ]
    bonds = [f"{b.GetBeginAtomIdx() + 1} {b.GetEndAtomIdx() + 1} 1" for b in molecule.GetBonds()]
    path = tmp_path / "input.itp"
    path.write_text("[ atoms ]\n" + "\n".join(atoms) + "\n[ bonds ]\n" + "\n".join(bonds))

    def corrupt(coordinates, names, smiles):
        if distortion == "scale":
            return coordinates * 1.2  # Same CIP, wrong covalent geometry.
        coordinates[:, 0] *= -1
        return coordinates

    monkeypatch.setattr(rdkit_lipid, "_align_gaff_tail_subtrees", corrupt)
    output = rdkit_lipid._validated_gaff_geometry(
        SimpleNamespace(itp_path=path, atom_names=names, coordinates=xyz), smiles
    )
    expected = rdkit_lipid._orient_for_membrane(xyz.copy(), list(names))
    expected -= expected.mean(axis=0)
    np.testing.assert_allclose(output, expected, atol=1e-12)

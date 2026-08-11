"""Regression tests for structure processing component ownership."""

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.modules.modifications.patches import (
    effective_patch_charge_shift,
    list_patches_for_residue,
)
from gmxbuilder.pipeline.step_executor import StepRunner


def test_hydrogens_are_assigned_to_their_parent_protein_component(tmp_path, monkeypatch):
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 5.0,
        atom_names=["CA", "CA"],
        resnames=["ALA", "GLY"],
        resids=[1, 2],
        chain_ids=["A", "B"],
        elements=["C", "C"],
    )
    system = System(
        structure=structure,
        components=[
            Component("PROTEIN_A", ComponentKind.PROTEIN, np.array([0])),
            Component("PROTEIN_B", ComponentKind.PROTEIN, np.array([1])),
        ],
        metadata={"force_field": "charmm36"},
    )

    class FakeHydrogenAdder:
        def __init__(self, _path):
            pass

        def add_hydrogens(self, names, coordinates, resnames, resids, chain_ids):
            return (
                names + ["HA", "HA"],
                np.vstack([coordinates, [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]),
                resnames + ["ALA", "GLY"],
                resids + [1, 2],
                chain_ids + ["A", "B"],
            )

    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: tmp_path
    )
    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.hdb.HDBHydrogenAdder", FakeHydrogenAdder
    )

    result = StructureProcessor().run(
        system, {"skip_protonation": True, "prepare_standard_termini": False}
    )

    assert result.success
    assert result.system.components[0].atom_indices.tolist() == [0, 1]
    assert result.system.components[1].atom_indices.tolist() == [2, 3]
    assert sorted(
        idx for component in result.system.components for idx in component.atom_indices
    ) == [0, 1, 2, 3]


def _residue_system(resname, atom_names, elements):
    structure = Structure(
        coordinates=np.arange(len(atom_names) * 3, dtype=float).reshape(-1, 3) / 10,
        box_vectors=np.eye(3) * 5.0,
        atom_names=list(atom_names),
        resnames=[resname] * len(atom_names),
        resids=[1] * len(atom_names),
        chain_ids=["A"] * len(atom_names),
        elements=list(elements),
    )
    return System(
        structure=structure,
        components=[
            Component("PROTEIN_A", ComponentKind.PROTEIN, np.arange(len(atom_names)))
        ],
        metadata={"force_field": "charmm36"},
    )


def _disulfide_system(force_field="amber14sb", cross_chain=False):
    atom_names = ["N", "CA", "C", "O", "CB", "SG"] * 2
    first = np.array([
        [0.00, 0.00, 0.00], [0.145, 0.00, 0.00], [0.245, 0.10, 0.00],
        [0.225, 0.22, 0.00], [0.145, -0.12, 0.00], [0.20, -0.25, 0.00],
    ])
    second = np.array([
        [0.37, 0.08, 0.00], [0.47, 0.16, 0.00], [0.59, 0.10, 0.00],
        [0.69, 0.16, 0.00], [0.47, -0.02, 0.00], [0.403, -0.25, 0.00],
    ])
    structure = Structure(
        coordinates=np.vstack([first, second]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=atom_names,
        resnames=["CYS"] * 12,
        resids=[1] * 6 + [2] * 6,
        chain_ids=["A"] * 6 + (["B"] * 6 if cross_chain else ["A"] * 6),
        elements=["N", "C", "C", "O", "C", "S"] * 2,
    )
    return System(
        structure=structure,
        components=[
            Component("PROTEIN", ComponentKind.PROTEIN, np.arange(12))
        ],
        metadata={"force_field": force_field},
    )
def test_amber_forcefield_translates_standard_ile_and_charmm_histidine_names(monkeypatch):
    system = _residue_system("ILE", ["N", "CA", "CB", "CG1", "CG2", "CD1", "C", "O"], ["N", "C", "C", "C", "C", "C", "C", "O"])
    system.metadata["force_field"] = "amber99sb-ildn"
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    result = StructureProcessor().run(
        system, {"skip_protonation": True, "prepare_standard_termini": False}
    )

    assert "CD" in result.system.structure.atom_names
    assert "CD1" not in result.system.structure.atom_names

    histidine = _residue_system("HSE", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    histidine.metadata["force_field"] = "amber99sb-ildn"
    result = StructureProcessor().run(
        histidine, {"skip_protonation": True, "prepare_standard_termini": False}
    )
    assert set(result.system.structure.resnames) == {"HIE"}


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_charmm_forcefields_translate_pdb_ile_cd1_from_actual_rtp(monkeypatch, force_field):
    system = _residue_system(
        "ILE", ["N", "CA", "CB", "CG1", "CG2", "CD1", "C", "O"],
        ["N", "C", "C", "C", "C", "C", "C", "O"],
    )
    system.metadata["force_field"] = force_field
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    result = StructureProcessor().run(
        system, {"skip_protonation": True, "prepare_standard_termini": False}
    )

    assert "CD" in result.system.structure.atom_names
    assert "CD1" not in result.system.structure.atom_names


def test_unassigned_histidine_fails_instead_of_guessing_tautomer(monkeypatch):
    system = _residue_system("HIS", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    system.metadata["force_field"] = "amber99sb-ildn"
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    with pytest.raises(ModuleConfigError, match="Unassigned HIS protonation state"):
        StructureProcessor().run(
            system, {"skip_protonation": True, "prepare_standard_termini": False}
        )


def test_enabled_protonation_rejects_incomplete_assignments(monkeypatch):
    system = _residue_system("HIS", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    with pytest.raises(ModuleConfigError, match="Incomplete protonation assignments.*A:1 HIS"):
        StructureProcessor().run(
            system,
            {
                "pH": 7.0,
                "protonation": [],
                "skip_protonation": False,
                "prepare_standard_termini": False,
            },
        )


def test_enabled_protonation_accepts_complete_histidine_assignment(monkeypatch):
    system = _residue_system("HIS", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    result = StructureProcessor().run(
        system,
        {
            "pH": 7.0,
            "protonation": [{"index": 0, "assigned_name": "HSE"}],
            "skip_protonation": False,
            "prepare_standard_termini": False,
        },
    )

    assert result.success
    assert set(result.system.structure.resnames) == {"HSE"}


def test_confirmed_forcefield_rejects_incomplete_heavy_atom_residue(monkeypatch):
    system = _residue_system("ARG", ["N", "CA", "C", "O", "CB"], ["N", "C", "C", "O", "C"])
    system.metadata.update(
        {"force_field": "amber99sb-ildn", "requested_force_field": "amber99sb-ildn"}
    )
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    with pytest.raises(ModuleConfigError, match="Incomplete protein heavy atoms"):
        StructureProcessor().run(
            system, {"skip_protonation": True, "prepare_standard_termini": False}
        )


@pytest.mark.parametrize(
    "source,patch_id,product,carbonyl,oxygen,old_atom,new_atom",
    [
        ("ASN", "DEA_ASN", "ASP", "CG", "OD1", "ND2", "OD2"),
        ("GLN", "DEG_GLN", "GLU", "CD", "OE1", "NE2", "OE2"),
    ],
)
def test_deamidation_changes_atom_identity_and_residue(
    monkeypatch, source, patch_id, product, carbonyl, oxygen, old_atom, new_atom
):
    atom_names = ["N", "CA", "C", "O", "CB", carbonyl, oxygen, old_atom]
    system = _residue_system(
        source, atom_names, ["N", "C", "C", "O", "C", "C", "O", "N"]
    )
    original_coordinates = system.structure.coordinates.copy()
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    result = StructureProcessor().run(
        system,
        {
            "skip_protonation": True,
            "prepare_standard_termini": False,
            "modifications": [{"index": 0, "patch_id": patch_id}],
        },
    )

    assert result.success
    assert set(result.system.structure.resnames) == {product}
    assert old_atom not in result.system.structure.atom_names
    changed_index = result.system.structure.atom_names.index(new_atom)
    assert result.system.structure.elements[changed_index] == "O"
    np.testing.assert_array_equal(
        result.system.structure.coordinates, original_coordinates
    )
    assert result.system.metadata["n_modifications"] == 1


def test_unsupported_patch_fails_before_mutating_structure(monkeypatch):
    system = _residue_system(
        "SER", ["N", "CA", "C", "O", "CB", "OG"], ["N", "C", "C", "O", "C", "O"]
    )
    original = system.structure.copy()
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    with pytest.raises(ModuleConfigError, match="PHOS_SER is unavailable"):
        StructureProcessor().run(
            system,
            {
                "skip_protonation": True,
                "modifications": [{"index": 0, "patch_id": "PHOS_SER"}],
            },
        )

    assert system.structure.atom_names == original.atom_names
    assert system.structure.resnames == original.resnames
    np.testing.assert_array_equal(system.structure.coordinates, original.coordinates)


def test_charmm36m_phosphoserine_builds_native_heavy_atoms(monkeypatch):
    system = _residue_system(
        "SER", ["N", "CA", "C", "O", "CB", "OG"],
        ["N", "C", "C", "O", "C", "O"],
    )
    system.metadata["force_field"] = "charmm36m"
    monkeypatch.setattr(
        "gmxbuilder.modules.modifications.processor._find_hdb", lambda _name: None
    )

    result = StructureProcessor().run(
        system,
        {
            "skip_protonation": True,
            "prepare_standard_termini": False,
            "modifications": [{"index": 0, "patch_id": "PHOS_SER"}],
        },
    )

    assert set(result.system.structure.resnames) == {"SEP"}
    assert {"P", "O1P", "O2P", "O3P"}.issubset(result.system.structure.atom_names)
    assert np.isfinite(result.system.structure.coordinates).all()


def test_unparameterized_terminus_cap_fails_before_mutating_structure():
    system = _residue_system("ALA", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    with pytest.raises(ModuleConfigError, match="FOR cap is unavailable"):
        StructureProcessor().run(
            system,
            {
                "skip_protonation": True,
                "termini": {"A": {"nter": "FOR", "cter": ""}},
            },
        )
    assert set(system.structure.resnames) == {"ALA"}


def test_patch_catalog_is_forcefield_specific():
    asn = {item["id"]: item for item in list_patches_for_residue("ASN", "charmm36")}
    ser = {item["id"]: item for item in list_patches_for_residue("SER", "charmm36")}
    charmm36m_ser = {
        item["id"]: item for item in list_patches_for_residue("SER", "charmm36m")
    }
    assert asn["DEA_ASN"]["supported"] is True
    assert ser["PHOS_SER"]["supported"] is False
    assert ser["PHOS_SER"]["support_reason"]
    assert charmm36m_ser["PHOS_SER"]["supported"] is True
    assert charmm36m_ser["PHOS_SER"]["charge_shift"] == -1
    amber_ser = {
        item["id"]: item for item in list_patches_for_residue("SER", "amber14sb")
    }
    assert amber_ser["PHOS_SER"]["supported"] is True
    assert amber_ser["PHOS_SER"]["charge_shift"] == -2
    assert amber_ser["PHOS1_SER"]["supported"] is True
    assert amber_ser["PHOS1_SER"]["charge_shift"] == -1
    assert effective_patch_charge_shift("PHOS_SER", "amber14sb") == -2


def test_system_formal_charge_uses_forcefield_specific_modified_template():
    amber = _residue_system("SEP", ["P"], ["P"])
    amber.metadata["force_field"] = "amber14sb"
    charmm = _residue_system("SEP", ["P"], ["P"])
    charmm.metadata["force_field"] = "charmm36m"
    assert amber.residue_formal_charge("SEP") == -2
    assert charmm.residue_formal_charge("SEP") == -1


def test_charmm_cysteine_oxidation_labels_match_native_templates():
    patches = {
        item["id"]: item
        for item in list_patches_for_residue("CYS", "charmm36m")
    }
    assert patches["CSO_CYS"]["supported"] is True
    assert "protonated sulfenic acid" in patches["CSO_CYS"]["description"]
    assert patches["CSO_CYS"]["formula_addition"] == "O"
    assert patches["CSO_CYS"]["charge_shift"] == 0
    assert patches["CSX_CYS"]["supported"] is True
    assert "deprotonated sulfenic acid" in patches["CSX_CYS"]["description"]
    assert patches["CSX_CYS"]["formula_addition"] == "O"
    assert patches["CSX_CYS"]["charge_shift"] == -1
    assert patches["CSD_CYS"]["supported"] is False


def test_new_charmm_ptm_catalog_uses_unambiguous_native_chemistry():
    lysine = {
        item["id"]: item
        for item in list_patches_for_residue("LYS", "charmm36m")
    }
    assert lysine["KME_LYS"]["product_name"] == "MLZ"
    assert lysine["KME2_LYS"]["product_name"] == "MLY"
    assert lysine["KME3_LYS"]["product_name"] == "M3L"
    assert lysine["CARBOXY_LYS"]["product_name"] == "KCX"
    assert lysine["CARBOXY_LYS"]["charge_shift"] == -2
    assert lysine["CBM_LYS"]["supported"] is False
    assert "homocitrulline" in lysine["CBM_LYS"]["support_reason"]
    assert lysine["MAL_LYS"]["supported"] is False
    assert "dimethyllysine" in lysine["MAL_LYS"]["support_reason"]

    arginine = {
        item["id"]: item
        for item in list_patches_for_residue("ARG", "charmm36m")
    }
    assert arginine["RME2_ARG"]["product_name"] == "2MR"
    assert arginine["RME2A_ARG"]["product_name"] == "DA2"
    assert arginine["RME_ARG"]["supported"] is False

    cysteine = {
        item["id"]: item
        for item in list_patches_for_residue("CYS", "charmm36m")
    }
    assert cysteine["CSN_CYS"]["product_name"] == "SNC"
    assert cysteine["SMC_CYS"]["supported"] is True
    assert cysteine["OCS_CYS"]["charge_shift"] == -1


def test_ambiguous_methionine_sulfoxide_remains_fail_closed():
    from gmxbuilder.modules.modifications.patches import patch_capability

    supported, reason = patch_capability("MSO_MET", "charmm36m")
    assert supported is False
    assert "MSO_R_MET" in reason


@pytest.mark.parametrize(
    "patch_id,force_field",
    [
        ("MSO_R_MET", "charmm36m"),
        ("HYP_PRO", "charmm36m"),
        ("HYP_PRO", "amber14sb"),
        ("HYP_PRO", "amber99sb"),
        ("HYP_PRO", "amber99sb-ildn"),
        ("HYL_LYS", "charmm36m"),
    ],
)
def test_explicit_stereochemistry_ptms_are_enabled(patch_id, force_field):
    from gmxbuilder.modules.modifications.patches import patch_capability

    assert patch_capability(patch_id, force_field) == (True, "")


def test_disulfide_pair_builds_cyx_and_explicit_sg_bond():
    result = StructureProcessor().run(
        _disulfide_system(),
        {
            "skip_protonation": True,
            "prepare_standard_termini": False,
            "crosslinks": [
                {"type": "disulfide", "first_index": 0, "second_index": 1}
            ],
        },
    )
    assert result.success
    assert set(result.system.structure.resnames) == {"CYX"}
    assert "HG" not in result.system.structure.atom_names
    sulphurs = [
        index for index, name in enumerate(result.system.structure.atom_names)
        if name == "SG"
    ]
    assert len(sulphurs) == 2
    assert any(
        {bond.i, bond.j} == set(sulphurs) for bond in result.system.topology.bonds
    )
    assert result.system.metadata["crosslinks"][0]["status"] == "passed"


def test_disulfide_rejects_distant_or_unsupported_pair_before_mutation():
    distant = _disulfide_system()
    distant.structure.coordinates[11] += np.array([1.0, 0.0, 0.0])
    original = distant.structure.copy()
    config = {
        "skip_protonation": True,
        "prepare_standard_termini": False,
        "crosslinks": [
            {"type": "disulfide", "first_index": 0, "second_index": 1}
        ],
    }
    with pytest.raises(ModuleConfigError, match="will not drag distant side chains"):
        StructureProcessor().run(distant, config)
    assert distant.structure.resnames == original.resnames

    unsupported = _disulfide_system("charmm36m")
    with pytest.raises(ModuleConfigError, match="unavailable for charmm36m"):
        StructureProcessor().run(unsupported, config)


def test_noncontiguous_repeated_residue_identifier_fails_closed():
    system = _disulfide_system()
    # A:1 is split by A:2 in atom order.  Grouping solely by residue number
    # would silently merge two atom blocks and make frontend indices ambiguous.
    order = list(range(6)) + list(range(6, 12))
    order.insert(7, order.pop(1))
    system.structure.coordinates = system.structure.coordinates[order]
    for attribute in (
        "atom_names", "resnames", "resids", "chain_ids", "elements",
        "occupancies", "tempfactors", "segids",
    ):
        values = getattr(system.structure, attribute)
        setattr(system.structure, attribute, [values[index] for index in order])

    with pytest.raises(ModuleConfigError, match="Non-contiguous repeated residue identifier"):
        StructureProcessor().run(
            system,
            {"skip_protonation": True, "prepare_standard_termini": False},
        )


@pytest.mark.parametrize(
    "force_field,patch_id",
    [
        ("amber14sb", "PALM_CYS"),
        ("amber14sb", "FAR_CYS"),
        ("amber14sb", "GCS_SER"),
        ("charmm36m", "GCT_THR"),
        ("charmm36m", "MYRI_GLY"),
        ("charmm36", "CYX_CYS"),
    ],
)
def test_complex_unvalidated_modifications_remain_explicitly_unavailable(
    force_field, patch_id
):
    from gmxbuilder.modules.modifications.patches import patch_capability

    supported, reason = patch_capability(patch_id, force_field)
    assert supported is False
    assert reason
    assert any(word in reason.lower() for word in ("requires", "validated", "unavailable"))


@pytest.mark.parametrize(
    "patch_id,force_field,reason_fragment",
    [
        ("PCA_GLN", "charmm36m", "different atom inventory"),
        ("MYRI_GLY", "charmm36m", "free myristic acid"),
        ("GPL_GLY", "charmm36m", "chemically unrelated"),
        ("RME_ARG", "charmm36m", "monomethylarginine"),
        ("CSD_CYS", "charmm36m", "atom-complete topology"),
    ],
)
def test_name_collisions_and_missing_assets_report_the_scientific_boundary(
    patch_id, force_field, reason_fragment
):
    from gmxbuilder.modules.modifications.patches import patch_capability

    supported, reason = patch_capability(patch_id, force_field)
    assert supported is False
    assert reason_fragment in reason


def test_step_runner_returns_config_error_instead_of_http_500(tmp_path):
    system = _residue_system("ALA", ["N", "CA", "C", "O"], ["N", "C", "C", "O"])
    runner = StepRunner(tmp_path, pipeline_type="membrane-bilayer")
    system.save_checkpoint(runner.step_dir("forcefield"))

    result = runner.run_step("structure", {
        "skip_protonation": True,
        "prepare_standard_termini": False,
        "termini": {"A": {"nter": "FOR", "cter": ""}},
    })

    assert result["status"] == "error"
    assert "FOR cap is unavailable" in result["error"]

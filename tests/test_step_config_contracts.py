import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from gmxbuilder.modules.membrane.builder import (
    MembraneBuilder,
    _reconcile_lipid_selection,
)
from gmxbuilder.modules.membrane.orient_module import OrientModule
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.modules.solvation.solvate import SolvationBuilder


def _one_unknown_molecule():
    structure = Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C1"],
        resnames=["LIG"],
        resids=[1],
        chain_ids=["L"],
        elements=["C"],
    )
    system = System(structure=structure)
    system.add_component(Component(
        name="UNKNOWN", kind=ComponentKind.UNKNOWN, atom_indices=np.array([0])
    ))
    return system


@pytest.mark.parametrize(
    "module,config",
    [
        (ForceFieldSelector(), {"name": "charmm36m", "unused_option": "oplsaa"}),
        (StructureProcessor(), {"pH": 7.0, "unused_option": True}),
        (OrientModule(), {"method": "manual", "unused_option": True}),
        (MembraneBuilder(), {"lipid_type": "POPC", "unused_option": True}),
        (SolvationBuilder(), {"water_model": "tip3p", "unused_option": True}),
    ],
)
def test_steps_reject_unconsumed_config_values(module, config):
    with pytest.raises(ModuleConfigError, match="Unsupported"):
        module.validate_config(config)


def test_auto_orientation_rejects_manual_overrides():
    with pytest.raises(ModuleConfigError, match="manual-orientation"):
        OrientModule().validate_config({"method": "ppm", "z_offset": 1.0})


@pytest.mark.parametrize("value", [0.49, 3.01, float("nan"), "wide"])
def test_orientation_rejects_invalid_hydrophobic_half_thickness(value):
    with pytest.raises(ModuleConfigError, match="half_thickness"):
        OrientModule().validate_config({"method": "ppm", "half_thickness": value})


@pytest.mark.parametrize("key,value", [
    ("tilt", float("nan")),
    ("z_offset", "not-a-number"),
    ("phi", float("inf")),
])
def test_manual_orientation_rejects_nonfinite_values(key, value):
    with pytest.raises(ModuleConfigError, match="finite number"):
        OrientModule().validate_config({"method": "manual", key: value})


@pytest.mark.parametrize("updates", [
    {"lipid_composition": {
        "upper": [{"name": "POPC", "ratio": 100}],
        "lower": [],
    }},
    {"lipid_type": "POPC", "box_padding": 2.0, "pad": 3.0},
    {"lipid_type": "POPC", "bilayer_size": [8.0, float("nan")]},
    {"lipid_type": "POPC", "orient_method": "unknown"},
    {"lipid_type": "POPC", "embed_method": "unknown"},
    {"lipid_type": "POPC", "seed": 1.25},
])
def test_membrane_rejects_ambiguous_or_nonfinite_geometry_config(updates):
    with pytest.raises(ModuleConfigError):
        MembraneBuilder().validate_config(updates)


def test_membrane_omitted_lower_leaflet_remains_explicit_symmetric_mode():
    assert MembraneBuilder().validate_config({
        "lipid_composition": {
            "upper": [{"name": "POPC", "ratio": 100}],
        },
        "bilayer_size": [8.0, 9.0],
    })


@pytest.mark.parametrize("config", [
    {"modifications": ["bad"], "skip_protonation": True},
    {"modifications": [{"index": 0, "patch_id": "X", "ignored": True}],
     "skip_protonation": True},
    {"crosslinks": [{"type": "disulfide", "first_index": 0,
                     "second_index": 1, "ignored": True}]},
    {"crosslinks": [{"type": "unknown", "first_index": 0,
                     "second_index": 1}]},
    {"crosslinks": [{"type": "disulfide", "first_index": 0,
                     "second_index": True}]},
    {"termini": {"A": []}, "skip_protonation": True},
    {"termini": {"A": {"nter": "ACE", "ignored": "X"}},
     "skip_protonation": True},
])
def test_structure_rejects_malformed_nested_chemistry_config(config):
    with pytest.raises(ModuleConfigError):
        StructureProcessor().validate_config(config)


def test_structure_accepts_minimal_forcefield_derived_modification_contract():
    config = {
        "modifications": [
            {"index": 50, "patch_id": "PHOS_TYR"},
            {"index": 61, "patch_id": "PHOS_TYR"},
            {"index": 138, "patch_id": "PHOS_TYR"},
            {"index": 149, "patch_id": "PHOS_TYR"},
        ],
        "skip_protonation": True,
    }

    assert StructureProcessor().validate_config(config)


def test_structure_accepts_minimal_disulfide_crosslink_contract():
    assert StructureProcessor().validate_config({
        "crosslinks": [
            {"type": "disulfide", "first_index": 4, "second_index": 17}
        ],
        "skip_protonation": True,
    })


def test_structure_rejects_client_supplied_modification_charge_shift():
    config = {
        "modifications": [
            {"index": 50, "patch_id": "PHOS_TYR", "charge_shift": -1},
        ],
        "skip_protonation": True,
    }

    with pytest.raises(ModuleConfigError, match="charge_shift"):
        StructureProcessor().validate_config(config)


def test_forcefield_rejects_incompatible_unparameterized_input_molecule():
    module = ForceFieldSelector()
    config = {
        "name": "charmm36m", "lipid_names": ["POPC"],
        "lipid_ff": "charmm36m", "ligand_ff": "cgenff",
    }
    module.validate_config(config)
    with pytest.raises(ModuleConfigError, match="incompatible"):
        module.run(_one_unknown_molecule(), config)


def test_membrane_requires_forcefield_reconfirmation_when_family_changes(empty_system):
    empty_system.metadata.update({
        "requested_force_field": "charmm36m",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
        "selected_lipid_names": ["POPC"],
    })
    config = {
        "lipid_composition": {
            "upper": [{"name": "20AHC", "ratio": 100}],
            "lower": [{"name": "20AHC", "ratio": 100}],
        },
        "n_lipids_per_leaflet": 64,
    }
    with pytest.raises(ModuleConfigError, match="differs from the Step 2"):
        MembraneBuilder().run(empty_system, config)


def test_membrane_rejects_changed_unvalidated_amber_gaff2_lipid(empty_system):
    empty_system.metadata.update({
        "force_field": "amber14sb",
        "lipid_ff": "gaff2",
        "selected_lipid_names": ["POPC"],
        "gaff_lipids": ["POPC"],
    })

    with pytest.raises(ModuleConfigError, match="CHARMM36m"):
        _reconcile_lipid_selection(empty_system, ["POPC", "CAMP"])


def test_membrane_revalidates_changed_supported_charmm_mixture(empty_system):
    empty_system.metadata.update({
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
        "selected_lipid_names": ["POPC"],
    })

    _reconcile_lipid_selection(empty_system, ["POPC", "CHOL"])

    assert empty_system.metadata["selected_lipid_names"] == ["CHOL", "POPC"]


def test_membrane_preserves_explicit_coherent_amber_gaff2_backend(empty_system):
    empty_system.metadata.update({
        "force_field": "amber14sb",
        "lipid_ff": "gaff2",
        "selected_lipid_names": ["POPC"],
        "gaff_lipids": ["POPC"],
    })

    message = _reconcile_lipid_selection(empty_system, ["POPC"])

    assert message is None
    assert empty_system.metadata["lipid_ff"] == "gaff2"
    assert empty_system.metadata["gaff_lipids"] == ["POPC"]

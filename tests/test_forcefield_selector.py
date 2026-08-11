"""Regression tests for force-field selection validation."""

import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector


def test_forcefield_selector_uses_default_force_field(empty_system):
    selector = ForceFieldSelector()

    assert selector.validate_config({})
    result = selector.run(empty_system, {})

    assert result.success
    assert result.system.metadata["force_field"] == "amber14sb"
    assert result.system.metadata["force_field_release"] == "GROMACS-2026.3"
    assert result.system.metadata["force_field_family"] == "amber"
    assert result.system.metadata["ff_water_model"] == "tip3p"


@pytest.mark.parametrize("config", [{"name": ""}, {"name": "not-a-force-field"}])
def test_forcefield_selector_rejects_invalid_names(config):
    with pytest.raises(ModuleConfigError):
        ForceFieldSelector().validate_config(config)


@pytest.mark.parametrize("ligand_pH", [0.9, 13.1, "7", True])
def test_forcefield_selector_rejects_invalid_ligand_ph(ligand_pH):
    with pytest.raises(ModuleConfigError, match="ligand_pH"):
        ForceFieldSelector().validate_config({"ligand_pH": ligand_pH})


@pytest.mark.parametrize(
    "config,match",
    [
        ({"name": "charmm36m", "lipid_names": ["POPC"],
          "lipid_ff": "gaff2", "ligand_ff": "none"}, "incompatible"),
        ({"name": "oplsaa", "lipid_names": ["POPC"],
          "lipid_ff": "oplsaa", "ligand_ff": "none"}, "Available alternatives"),
    ],
)
def test_forcefield_selector_rejects_cross_family_combinations(empty_system, config, match):
    with pytest.raises(ModuleConfigError, match=match):
        ForceFieldSelector().run(empty_system, config)


def test_forcefield_selector_accepts_charmm_family_lipids(empty_system):
    result = ForceFieldSelector().run(empty_system, {
        "name": "charmm36m", "lipid_names": ["POPC"],
        "lipid_ff": "charmm36m", "ligand_ff": "none",
    })
    assert result.system.metadata["force_field"] == "charmm36m"
    assert result.system.metadata["lipid_ff"] == "charmm36m"


def test_forcefield_selector_accepts_recommended_amber_lipid21(empty_system):
    result = ForceFieldSelector().run(empty_system, {
        "name": "amber14sb", "lipid_names": ["POPC"],
        "lipid_ff": "lipid21", "ligand_ff": "none",
    })
    assert result.system.metadata["force_field"] == "amber14sb"
    assert result.system.metadata["lipid_ff"] == "lipid21"
    assert result.system.metadata["lipid21_lipids"] == ["POPC"]


def test_missing_charmm_lipid_names_installed_alternative(empty_system, monkeypatch):
    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.compatibility.gaff_available", lambda: True,
    )
    with pytest.raises(ModuleConfigError, match=r"20AHC -> no installed force field"):
        ForceFieldSelector().run(empty_system, {
            "name": "charmm36m", "lipid_names": ["20AHC"],
            "lipid_ff": "charmm36m", "ligand_ff": "none",
        })

"""Validation tests for membrane composition configuration."""

import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.membrane.builder import MembraneBuilder


def test_membrane_config_accepts_valid_asymmetric_composition():
    config = {
        "lipid_composition": {
            "upper": [
                {"name": "POPC", "ratio": 60},
                {"name": "POPE", "ratio": 40},
            ],
            "lower": [
                {"name": "POPC", "ratio": 50},
                {"name": "POPE", "ratio": 50},
            ],
        },
        "n_lipids_per_leaflet": 64,
    }

    assert MembraneBuilder().validate_config(config)


def test_membrane_config_preserves_custom_lipid_support():
    config = {
        "lipid_composition": {
            "upper": [{"name": "CUSTOM", "ratio": 100, "category": "PC"}],
            "lower": None,
        }
    }

    assert MembraneBuilder().validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"lipid_composition": {"upper": []}},
        {"lipid_composition": {"upper": [{"name": "POPC", "ratio": 90}]}},
        {"lipid_composition": {"upper": [{"name": "POPC", "ratio": -100}]}},
        {
            "lipid_composition": {
                "upper": [{"name": "POPC", "ratio": 100}],
                "lower": [{"name": "NOTREAL", "ratio": 100}],
            }
        },
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 63},
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 5001},
        {"lipid_type": "POPC", "bilayer_size": [101.0, 8.0]},
    ],
)
def test_membrane_config_rejects_invalid_compositions(config):
    with pytest.raises(ModuleConfigError):
        MembraneBuilder().validate_config(config)


def test_membrane_config_accepts_the_documented_maximum_leaflet_count():
    assert MembraneBuilder().validate_config({"lipid_type": "POPC", "n_lipids_per_leaflet": 5000})


def test_sterol_composition_warning_does_not_reject_valid_input():
    from gmxbuilder.modules.membrane.composition_warnings import composition_warnings

    assert MembraneBuilder().validate_config({"lipid_type": "CHOL", "n_lipids_per_leaflet": 64})
    warnings = composition_warnings(
        {"upper": [("CHOL", 60), ("CHOL", 40), ("POPC", 0)], "lower": [("POPC", 100)]}
    )
    assert len(warnings) == 1
    assert warnings[0]["affected_leaflets"] == ["upper"]
    assert warnings[0]["blocking"] is False
    assert not composition_warnings({"upper": [("POPC", 100)]})


def test_dope_phase_warning_does_not_transfer_experimental_phase_to_martini():
    from gmxbuilder.modules.membrane.composition_warnings import composition_warnings

    leaflets = {"upper": [("DOPE", 100)], "lower": [("DOPE", 10), ("POPC", 90)]}
    warnings = composition_warnings(leaflets)
    assert len(warnings) == 1
    assert warnings[0]["affected_leaflets"] == ["upper"]
    assert warnings[0]["evidence_refs"]
    assert not warnings[0]["blocking"]
    assert not composition_warnings(leaflets, resolution="martini3")

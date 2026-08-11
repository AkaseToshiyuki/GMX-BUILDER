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
        {"lipid_type": "CHOL", "n_lipids_per_leaflet": 64},
    ],
)
def test_membrane_config_rejects_invalid_compositions(config):
    with pytest.raises(ModuleConfigError):
        MembraneBuilder().validate_config(config)

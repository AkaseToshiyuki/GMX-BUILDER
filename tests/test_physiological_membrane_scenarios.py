"""Hidden physiological compositions used only as regression scenarios.

They are deliberately private test fixtures, not presets returned by the API.
The exact percentages are coverage scenarios rather than claims that one
composition represents every cell, organism, organelle, or condition.
"""

import pytest

from gmxbuilder.modules.membrane.builder import MembraneBuilder


_PHYSIOLOGICAL_SCENARIOS = {
    "mammalian_plasma_asymmetric": {
        "upper": [
            {"name": "POPC", "ratio": 45},
            {"name": "PSM", "ratio": 30},
            {"name": "CHOL", "ratio": 25},
        ],
        "lower": [
            {"name": "POPE", "ratio": 40},
            {"name": "POPS", "ratio": 20},
            {"name": "POPC", "ratio": 15},
            {"name": "CHOL", "ratio": 25},
        ],
    },
    "gram_negative_inner_membrane": {
        "upper": [
            {"name": "POPE", "ratio": 75},
            {"name": "POPG", "ratio": 20},
            {"name": "TOCL", "ratio": 5},
        ],
        "lower": None,
    },
    "mitochondrial_inner_membrane": {
        "upper": [
            {"name": "POPC", "ratio": 35},
            {"name": "POPE", "ratio": 45},
            {"name": "TOCL", "ratio": 20},
        ],
        "lower": None,
    },
}


@pytest.mark.parametrize(
    "composition", _PHYSIOLOGICAL_SCENARIOS.values(), ids=_PHYSIOLOGICAL_SCENARIOS
)
def test_hidden_physiological_compositions_satisfy_step5_contract(composition):
    assert MembraneBuilder().validate_config(
        {
            "lipid_composition": composition,
            "n_lipids_per_leaflet": 64,
        }
    )
    for leaflet in (composition["upper"], composition.get("lower")):
        if leaflet:
            assert sum(entry["ratio"] for entry in leaflet) == 100

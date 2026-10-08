"""The construction area: what it changes, and what it must leave alone.

The whole point of this change is that it is *narrow*. One class of number was
not a measurable quantity and is now a measured one; nothing else moves. A
correction that quietly shifted every membrane would be a different and much
larger claim, so most of these tests exist to pin down what did not change.
"""

from __future__ import annotations

import json

import pytest

from gmxbuilder.modules.membrane.area_model import (
    CHOLESTEROL_PARTIAL_AREA_NM2,
    converged_library_area,
    is_sterol,
    leaflet_area_per_lipid,
    species_area,
    sterol_partial_area,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from tests.prerequisites import requires_lfs_assets

VALIDATION_MIX = [("POPC", 40), ("POPE", 25), ("POPS", 10), ("CHOL", 25)]


# --------------------------------------------------------------------------
# What changes


def test_cholesterol_contributes_its_measured_partial_area():
    area, source = species_area("CHOL")
    assert area == pytest.approx(CHOLESTEROL_PARTIAL_AREA_NM2)
    assert source == "literature-partial-molar-area"
    assert area < LipidRegistry.get("CHOL").area_per_lipid, "the registry value was too large"


def test_every_sterol_is_corrected_not_only_cholesterol():
    """A patch that only fixed CHOL would be a special case, not a correction."""
    sterols = [name for name, t in LipidRegistry._lipids.items() if t.category == "ST"]
    assert len(sterols) >= 10, "the registry should carry a family of sterols"
    for name in sterols:
        area, source = species_area(name)
        assert source == "literature-partial-molar-area"
        assert area < LipidRegistry.get(name).area_per_lipid


def test_the_sterols_keep_their_relative_ordering():
    """Sitosterol is bigger than cholesterol; the correction must not flatten that."""
    assert sterol_partial_area("SITO") > sterol_partial_area("CHOL")
    assert sterol_partial_area("27OHC") > sterol_partial_area("CHOL")
    # Rescaling, not replacement: applying 0.24 to all of them would assert
    # that every sterol is the same size.
    assert len({round(sterol_partial_area(n), 4) for n in ("CHOL", "SITO", "27OHC")}) == 3


def test_the_validation_membrane_moves_towards_what_was_measured():
    """56.3 A^2 was predicted; 46.3 / 49.2 A^2 were measured after 50 ns."""
    area = leaflet_area_per_lipid(VALIDATION_MIX).area_per_lipid_nm2 * 100.0
    assert 51.0 < area < 54.0, "expected roughly 52.8 A^2"
    assert abs(area - 49.2) < abs(56.3 - 49.2), "must be closer to CHARMM's measurement"
    assert abs(area - 46.3) < abs(56.3 - 46.3), "must be closer to Amber's measurement"


# --------------------------------------------------------------------------
# What must not change


@pytest.mark.parametrize("lipid", ["POPC", "POPE", "POPS", "DOPC", "DMPC", "PSM"])
def test_a_non_sterol_keeps_the_area_it_already_had(lipid):
    area, source = species_area(lipid)
    assert source == "registry-experimental"
    assert area == pytest.approx(LipidRegistry.get(lipid).area_per_lipid)


def test_a_sterol_free_membrane_is_sized_exactly_as_before():
    """No user building a plain bilayer should see their box change."""
    mix = [("POPC", 70), ("POPE", 30)]
    expected = 0.7 * LipidRegistry.get("POPC").area_per_lipid + 0.3 * (
        LipidRegistry.get("POPE").area_per_lipid
    )
    assert leaflet_area_per_lipid(mix).area_per_lipid_nm2 == pytest.approx(expected)


def test_ratios_need_not_sum_to_one_hundred():
    a = leaflet_area_per_lipid([("POPC", 1), ("CHOL", 1)])
    b = leaflet_area_per_lipid([("POPC", 50), ("CHOL", 50)])
    assert a.area_per_lipid_nm2 == pytest.approx(b.area_per_lipid_nm2)


def test_a_zero_ratio_lipid_contributes_nothing():
    with_zero = leaflet_area_per_lipid([("POPC", 100), ("CHOL", 0)])
    assert [t.lipid for t in with_zero.terms] == ["POPC"]


def test_an_empty_composition_is_refused():
    with pytest.raises(ValueError, match="positive lipid ratio"):
        leaflet_area_per_lipid([("POPC", 0)])


def test_an_unregistered_lipid_falls_back_rather_than_raising():
    area, source = species_area("ZZZZZ")
    assert source == "default-unknown-lipid"
    assert area > 0


# --------------------------------------------------------------------------
# The library tier, and why it is dormant


@requires_lfs_assets
def test_no_shipped_entry_supplies_a_measured_area_today():
    """V3 areas are one frame of a 1 ns run and must never be used as targets.

    This is the assertion that keeps the change honest: the measured tier is
    written and dormant, and if a future rebuild starts feeding it, this test
    fails and forces the claim to be re-examined rather than absorbed.
    """
    for lipid in ("POPC", "POPE", "DMPC", "CHOL"):
        for force_field, lipid_ff in (("charmm36m", None), ("amber14sb", "lipid21")):
            assert converged_library_area(lipid, force_field, lipid_ff) is None


def test_a_converged_entry_is_used_when_one_exists(tmp_path, monkeypatch):
    from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD

    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "observables": {
                    "autocorrelation_method": AUTOCORRELATION_METHOD,
                    "measures": "pure-bilayer-area-per-lipid",
                    "converged": True,
                    "area_per_lipid_nm2": {"mean": 0.6380},
                }
            }
        )
    )
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY", str(tmp_path))
    monkeypatch.setattr(
        "gmxbuilder.runtime.prebuilt_assets.ensure_prebuilt_assets", lambda *a, **k: None
    )

    assert converged_library_area("POPC", "charmm36m") == pytest.approx(0.6380)
    assert species_area("POPC", "charmm36m") == (
        pytest.approx(0.6380),
        "measured-converged-library",
    )

    path = entry / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["v4_protocol"] = {"temperature_K": 308.15}
    path.write_text(json.dumps(metadata))
    assert converged_library_area("POPC", "charmm36m") is None
    assert converged_library_area("POPC", "charmm36m", temperature_K=315.15) is None
    assert converged_library_area("POPC", "charmm36m", temperature_K=308.15) == pytest.approx(0.638)
    metadata["observables"].pop("autocorrelation_method")
    path.write_text(json.dumps(metadata))
    assert converged_library_area("POPC", "charmm36m", temperature_K=308.15) is None


@pytest.mark.parametrize(
    "observables",
    [
        {
            "measures": "pure-bilayer-area-per-lipid",
            "converged": False,
            "area_per_lipid_nm2": {"mean": 0.6380},
        },
        {
            "measures": "host-mixture-area-per-lipid",
            "converged": True,
            "area_per_lipid_nm2": {"mean": 0.4612},
        },
        {"measures": "pure-bilayer-area-per-lipid", "converged": True},
    ],
    ids=["not-converged", "host-mixture-not-the-lipid's-own", "no-value"],
)
def test_an_entry_that_does_not_qualify_is_refused(observables, tmp_path, monkeypatch):
    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(json.dumps({"observables": observables}))
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY", str(tmp_path))
    monkeypatch.setattr(
        "gmxbuilder.runtime.prebuilt_assets.ensure_prebuilt_assets", lambda *a, **k: None
    )

    assert converged_library_area("POPC", "charmm36m") is None


def test_a_v3_entry_is_not_mistaken_for_a_measurement(tmp_path, monkeypatch):
    """The exact shape of every shipped entry: quality, and no observables."""
    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps({"quality": {"passed": True, "area_per_lipid_nm2": 0.5891}})
    )
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY", str(tmp_path))
    monkeypatch.setattr(
        "gmxbuilder.runtime.prebuilt_assets.ensure_prebuilt_assets", lambda *a, **k: None
    )

    assert converged_library_area("POPC", "charmm36m") is None


# --------------------------------------------------------------------------
# Provenance


def test_the_result_records_where_every_number_came_from():
    """A reviewer must be able to see that a sterol was treated differently."""
    metadata = leaflet_area_per_lipid(VALIDATION_MIX).as_metadata()
    json.dumps(metadata, allow_nan=False)

    sources = {term["lipid"]: term["source"] for term in metadata["terms"]}
    assert sources["CHOL"] == "literature-partial-molar-area"
    assert sources["POPC"] == "registry-experimental"
    assert sum(term["fraction"] for term in metadata["terms"]) == pytest.approx(1.0)


def test_is_sterol_does_not_raise_on_an_unknown_lipid():
    assert is_sterol("CHOL") is True
    assert is_sterol("POPC") is False
    assert is_sterol("ZZZZZ") is False


def test_a_missing_asset_archive_costs_the_measured_tier_not_the_build():
    """Sizing a box must not depend on an optional download being present.

    Constructing the library materialises the prebuilt archive, which fails on
    a source checkout whose Git LFS pointers are unfetched. Propagating that
    took the whole membrane step down with it; the right answer is that no
    measured area is available and the registry stands in.
    """
    from unittest.mock import patch

    with patch(
        "gmxbuilder.runtime.prebuilt_assets.ensure_prebuilt_assets",
        side_effect=RuntimeError("Prebuilt lipid assets are absent."),
    ):
        assert converged_library_area("POPC", "charmm36m") is None
        assert species_area("POPC", "charmm36m") == (
            pytest.approx(LipidRegistry.get("POPC").area_per_lipid),
            "registry-experimental",
        )
        # And the correction this module exists for still applies.
        assert species_area("CHOL", "charmm36m")[1] == "literature-partial-molar-area"

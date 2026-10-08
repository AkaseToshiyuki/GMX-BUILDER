"""Protocol and sampling regression cases independent of simulation output."""

import copy

import numpy as np
import pytest

from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from gmxbuilder.modules.membrane.v4_sampling import (
    assess_series,
    quantitative_area,
    replica_agreement,
    sampling_failures,
)


def noise_series(seed=67, n=1000):
    rng = np.random.default_rng(seed)
    return {
        "area_per_lipid_nm2": 0.65 + rng.normal(0, 0.003, n),
        "box_z_nm": 8 + rng.normal(0, 0.01, n),
        "volume_nm3": 400 + rng.normal(0, 1, n),
        "head_to_head_nm": 3.8 + rng.normal(0, 0.01, n),
    }


def test_reference_temperature_is_preserved_when_floor_applies():
    protocol = resolve_protocol("POPC", "charmm36m-lipid")
    assert protocol["reference_temperature_K"] == 303.15
    assert protocol["temperature_K"] == 308.15
    assert protocol["temperature_adjusted"]
    assert resolve_protocol("DPPC", "charmm36m-lipid")["temperature_K"] == 323.15


def test_host_is_exactly_ten_percent_at_42_celsius():
    protocol = resolve_protocol("CER18", "charmm36m-lipid")
    assert protocol["temperature_K"] == 315.15
    assert protocol["composition"] == {"POPC": 90, "CER18": 10}
    assert protocol["lipids_per_leaflet"] * 0.10 == 20
    assert protocol["host_smiles"]
    # Missing pure-phase evidence now has an explicitly approved host pilot;
    # it must not invent a pure reference or imply a measured guest area.
    pilot = resolve_protocol("BSM", "amber-gaff2")
    assert pilot["composition"] == {"POPC": 90, "BSM": 10}
    assert pilot["temperature_K"] == 315.15
    assert pilot["intended_use"] == "initialization_conformers_only"
    assert "reference_temperature_K" not in pilot


@pytest.mark.parametrize(
    "family", ["charmm36m-lipid", "charmm36-lipid", "amber-lipid21", "amber-gaff2"]
)
def test_dope_phase_suitability_precedes_pure_reference_temperature(family):
    protocol = resolve_protocol("dope", family)
    assert protocol["reference_temperature_K"] == 318.15
    assert protocol["hexagonal_transition_K"] == 283.15
    assert protocol["environment"] == "host"
    assert protocol["temperature_K"] == 315.15
    assert protocol["composition"] == {"POPC": 90, "DOPE": 10}
    assert protocol["reference_environment"] == "pure bilayer"


def test_late_volume_collapse_cannot_hide_behind_flat_area():
    series = noise_series()
    series["volume_nm3"][940:] -= 50
    result = assess_series(np.arange(1000) * 100.0, series)
    assert not result["passed"]
    assert any("volume_nm3" in failure for failure in result["failures"])


def test_stationary_control_and_different_replica_are_distinguished():
    times = np.arange(1000) * 100.0
    first = assess_series(times, noise_series())
    assert not sampling_failures(first)
    other = copy.deepcopy(first)
    other["observables"]["area_per_lipid_nm2"]["mean"] += 0.04
    assert not replica_agreement([first, other])["passed"]
    assert not quantitative_area([first, other])["converged"]


def test_precision_does_not_change_conformer_sampling_acceptance():
    first = assess_series(np.arange(1000) * 100.0, noise_series())
    imprecise = copy.deepcopy(first)
    imprecise["observables"]["area_per_lipid_nm2"]["standard_error"] = 0.03
    assert not sampling_failures(imprecise)
    assert not quantitative_area([imprecise, imprecise])["converged"]
    assert quantitative_area([first, first])["converged"]


def test_a_stored_passed_flag_cannot_override_missing_samples():
    first = assess_series(np.arange(1000) * 100.0, noise_series())
    first["passed"] = True
    first["observables"]["volume_nm3"]["effective_samples"] = 2
    assert sampling_failures(first)


@pytest.mark.parametrize("scope", ["analysis", "stationarity"])
def test_obsolete_correlation_statistics_need_reanalysis_even_with_passed_flag(scope):
    result = assess_series(np.arange(1000) * 100.0, noise_series())
    assert not sampling_failures(result)
    target = result if scope == "analysis" else result["stationarity"]
    target.pop("autocorrelation_method")
    target["passed"] = True
    assert "reanalysis required" in "; ".join(sampling_failures(result))
    assert not quantitative_area([result, result])["converged"]


def test_joint_probability_is_rechecked_from_contrast_evidence():
    result = assess_series(np.arange(1000) * 100.0, noise_series())
    result["stationarity"]["details"]["volume_nm3"]["contrasts"]["reference_shift"][
        "difference"
    ] = 1000
    result["stationarity"]["passed"] = True
    assert any("contrast evidence" in reason for reason in sampling_failures(result))


@pytest.mark.parametrize("start", [750, 940])
def test_completed_level_shift_and_late_collapse_are_detected(start):
    series = noise_series()
    series["volume_nm3"][start:] -= 50
    result = assess_series(np.arange(1000) * 100.0, series)
    assert result["stationarity"]["uncertainty_estimated"]
    assert not result["stationarity"]["passed"]
    assert "volume_nm3" in " ".join(sampling_failures(result))


def test_duplicate_observables_do_not_multiply_the_joint_test():
    from gmxbuilder.modules.membrane.v4_stationarity import joint_stationarity

    rng = np.random.default_rng(67)
    ref, hold = 8 + rng.normal(0, 0.01, (750, 1)), 8 + rng.normal(0, 0.01, (250, 1))
    first = joint_stationarity(ref, hold, ["box_z_nm"])
    duplicate = joint_stationarity(
        np.repeat(ref, 2, axis=1), np.repeat(hold, 2, axis=1), ["box_z_nm", "volume_nm3"]
    )
    assert first["comparisons"] == duplicate["comparisons"]
    assert first["critical_value"] == duplicate["critical_value"]
    assert first["details"]["box_z_nm"] == duplicate["details"]["box_z_nm"]


def test_frozen_terminal_coordinates_cannot_establish_stationarity():
    series = noise_series()
    series["volume_nm3"][750:] = 400.0
    result = assess_series(np.arange(1000) * 100.0, series)
    assert not result["stationarity"]["uncertainty_estimated"]
    assert any("Constant holdout" in reason for reason in sampling_failures(result))


def test_missing_or_irregular_time_series_is_rejected():
    with pytest.raises(ValueError, match="uniform"):
        assess_series(np.arange(1000) ** 2, noise_series())
    series = noise_series()
    series["volume_nm3"][5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        assess_series(np.arange(1000), series)

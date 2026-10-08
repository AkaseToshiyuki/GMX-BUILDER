"""Independent boundaries for practical-equivalence acceptance and comparison."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import lfilter

from gmxbuilder.modules.membrane.v4_comparison import system_composition
from gmxbuilder.modules.membrane.v4_sampling import (
    assess_series,
    replica_agreement,
    sampling_failures,
)
from gmxbuilder.modules.membrane.v4_stationarity import interval_status, joint_stationarity


def data(seed=11):
    rng = np.random.default_rng(seed)
    return {
        k: mean + rng.normal(0, sd, 1000)
        for k, mean, sd in (
            ("area_per_lipid_nm2", 0.65, 0.003),
            ("head_to_head_nm", 4, 0.01),
            ("volume_nm3", 400, 1),
            ("box_z_nm", 8, 0.01),
        )
    }


def analysis():
    return assess_series(np.arange(1000) * 100, data())


def test_exact_equivalence_boundary_does_not_pass():
    assert interval_status(-0.9, 0.9, 1) == "equivalent"
    assert interval_status(-1, 0.9, 1) == "insufficient_evidence"
    assert interval_status(0.9, 1.1, 1) == "insufficient_evidence"
    assert interval_status(1.01, 1.5, 1) == "drift_detected"


def test_wide_intervals_cannot_be_accepted_by_nonsignificance_or_flag():
    rng = np.random.default_rng(0)
    y = lfilter([1], [1, -0.99], rng.normal(size=5000))[4000:]
    result = joint_stationarity((400 + y[:750])[:, None], (400 + y[750:])[:, None], ["volume_nm3"])
    assert result["reference_blocks"] < 8
    assert result["status"] == "insufficient_evidence"
    a = analysis()
    contrast = a["stationarity"]["details"]["volume_nm3"]["contrasts"]["reference_shift"]
    critical = a["stationarity"]["critical_value"]
    contrast.update(
        standard_error=100,
        ci_lower=contrast["difference"] - 100 * critical,
        ci_upper=contrast["difference"] + 100 * critical,
        status="equivalent",
    )
    a["passed"] = a["stationarity"]["passed"] = True
    assert sampling_failures(a)


def test_fewer_than_eight_blocks_is_not_an_automatic_rejection():
    rng = np.random.default_rng(0)
    y = lfilter([1], [1, -0.99], rng.normal(size=5000))[4000:]
    result = joint_stationarity(
        (400 + 0.001 * y[:750])[:, None], (400 + 0.001 * y[750:])[:, None], ["volume_nm3"]
    )
    assert 2 <= result["reference_blocks"] < 8
    assert result["passed"]


@pytest.mark.parametrize("fraction", [0.75, 0.94])
def test_correlated_collapse_never_passes_even_with_enough_effective_samples(fraction):
    values = data()
    rng = np.random.default_rng(42)
    y = lfilter([1], [1, -0.99], rng.normal(size=5000))[4000:]
    values["volume_nm3"] = 400 + y
    values["volume_nm3"][int(fraction * 1000) :] -= 50
    a = assess_series(np.arange(1000) * 100, values)
    assert not a["passed"]
    assert any("volume_nm3" in failure for failure in sampling_failures(a))


def test_unknown_observable_has_no_implicit_margin():
    result = joint_stationarity(np.arange(20)[:, None], np.arange(10)[:, None], ["unreviewed"])
    assert not result["passed"]
    assert "No reviewed" in result["reason"]


def test_hydration_exemption_does_not_hide_membrane_disagreement():
    first, second = analysis(), analysis()
    first["system_composition"] = {
        "molecule_counts": {"POPC": 400, "SOL": 10000, "NA": 80, "CL": 80}
    }
    second["system_composition"] = {
        "molecule_counts": {"POPC": 400, "SOL": 11000, "NA": 89, "CL": 89}
    }
    second["observables"]["volume_nm3"]["mean"] += 40
    second["observables"]["box_z_nm"]["mean"] += 1
    assert replica_agreement([first, second])["passed"]
    second["observables"]["area_per_lipid_nm2"]["mean"] += 0.04
    assert not replica_agreement([first, second])["passed"]
    # Hydration only affects the between-run comparison; internal volume
    # equivalence remains required for each trajectory.
    second["stationarity"]["details"]["volume_nm3"]["contrasts"]["reference_shift"][
        "difference"
    ] += 40
    assert sampling_failures(second)


def test_identical_or_unknown_composition_cannot_exempt_box_disagreement():
    first, second = analysis(), analysis()
    second["observables"]["volume_nm3"]["mean"] += 40
    assert not replica_agreement([first, second])["passed"]
    for a in (first, second):
        a["system_composition"] = {"molecule_counts": {"POPC": 400, "SOL": 10000}}
    assert not replica_agreement([first, second])["passed"]
    second["system_composition"]["molecule_counts"]["UNKN"] = 1
    assert "solute compositions differ" in replica_agreement([first, second])["failures"][0]


def test_nonfinite_uncertainty_cannot_be_hidden_by_hydration():
    first, second = analysis(), analysis()
    second["observables"]["volume_nm3"]["standard_error"] = np.nan
    assert not replica_agreement([first, second])["passed"]


def test_residue_wrap_counts_whole_molecules():
    structure = SimpleNamespace(resids=[99999] * 3 + [0] * 3 + [1], resnames=["SOL"] * 6 + ["SOD"])
    assert system_composition(structure, {})["molecule_counts"] == {"SOL": 2, "SOD": 1}

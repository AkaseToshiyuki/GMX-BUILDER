"""What the area measurement must refuse to call converged.

The V3 library's headline area came from one frame of a 1 ns run, so every
number it published was a sample masquerading as a measurement. The point of
this module is not that it computes a mean -- any two lines do that -- it is
that it declines to publish one when the evidence is not there. These tests
are therefore mostly about rejection.

The series here are synthetic on purpose: a real trajectory cannot be asked
to have a known answer, and every property under test (drift, correlation,
plateau) is one that can be constructed exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmxbuilder.modules.membrane.area_observable import (
    MAX_DRIFT_SIGMA,
    MAX_REPLICA_SPREAD_SIGMA,
    MIN_EFFECTIVE_SAMPLES,
    MIN_RETAINED_FRACTION,
    AreaMeasurement,
    integrated_autocorrelation,
    measure_area_per_lipid,
    pool_replicas,
    summarise,
)

N_LEAFLET = 64


def _series(areas: np.ndarray, frame_ps: float = 10.0):
    """Turn a per-lipid area series into the (t, box_x, box_y) a caller passes."""
    times = np.arange(len(areas), dtype=float) * frame_ps
    edge = np.sqrt(np.asarray(areas, dtype=float) * N_LEAFLET)
    return times, edge, edge


def _plateau(n: int = 5000, mean: float = 0.60, noise: float = 0.005, seed: int = 0):
    rng = np.random.default_rng(seed)
    return mean + rng.normal(0.0, noise, n)


def _correlated(
    n: int, tau_frames: float, mean: float = 0.60, seed: int = 0, spread: float = 0.005
):
    """An AR(1) walk, whose integrated autocorrelation time is known."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-1.0 / tau_frames)
    # Scale the innovations so the stationary spread stays realistic for a
    # bilayer area (~1% of the mean) whatever tau is asked for; otherwise a
    # long tau produces a series that wanders over half its own value.
    noise = rng.normal(0.0, spread * np.sqrt(1.0 - phi**2), n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return mean + out


# --------------------------------------------------------------------------
# The mean, when the evidence supports one


def test_a_flat_series_is_measured_and_accepted():
    measurement = measure_area_per_lipid(*_series(_plateau()), N_LEAFLET)
    assert measurement.converged, measurement.rejection
    assert measurement.mean_nm2 == pytest.approx(0.60, abs=0.001)
    assert measurement.standard_error_nm2 < 0.001
    assert measurement.retained_fraction == 1.0


def test_the_reported_error_shrinks_with_more_independent_data():
    short = measure_area_per_lipid(*_series(_plateau(n=500)), N_LEAFLET)
    long = measure_area_per_lipid(*_series(_plateau(n=50000)), N_LEAFLET)
    assert long.standard_error_nm2 < short.standard_error_nm2 / 5.0


# --------------------------------------------------------------------------
# Rejection: the whole point


def test_a_still_contracting_run_is_refused_however_flat_its_tail_looks():
    """The V3 failure mode, exactly: judge the end and miss the drift."""
    times = np.arange(5000, dtype=float) * 10.0
    contracting = 0.60 + 0.10 * np.exp(-times / 40000.0)
    measurement = measure_area_per_lipid(*_series(contracting), N_LEAFLET)

    assert not measurement.converged
    # Rejected for having too few independent samples: a series that is still
    # relaxing never decorrelates, so the trend shows up as a correlation time
    # comparable to the run. The cure named is the right one either way.
    assert "longer run" in measurement.rejection
    assert measurement.effective_samples < MIN_EFFECTIVE_SAMPLES
    # And it must not quietly report the tail value as the answer.
    assert measurement.mean_nm2 > 0.62


def test_trimming_a_run_to_a_flat_tail_cannot_manufacture_convergence():
    """Without the retained-fraction rule the drift test judges its own window.

    Measured on the four 50 ns validation trajectories, the detector kept
    between 2.7% and 74% of the series and two runs passed the drift test on
    the strength of that trimming while still contracting at over 1% per 10 ns.
    """
    times = np.arange(4000, dtype=float) * 10.0
    # Ninety percent steep drift, ten percent plateau: the tail alone is clean.
    drifting = np.where(times < 36000, 0.75 - 0.15 * times / 36000.0, 0.60)
    measurement = measure_area_per_lipid(*_series(drifting), N_LEAFLET)

    # The detector is not permitted to reach that clean tail at all, ...
    assert measurement.retained_fraction >= MIN_RETAINED_FRACTION
    # ... so the drift it was trying to escape is still in the measurement.
    assert not measurement.converged
    assert measurement.mean_nm2 > 0.60, "the tail value must not be reported as the answer"


def test_a_heavily_correlated_run_is_refused_however_many_frames_it_has():
    """Writing frames often is not sampling; R3's concern, in the area.

    A stationary walk with no drift, so the only gate that can fire is the
    sampling one. The seed is fixed because the autocorrelation estimator has
    real variance on a finite series -- across seeds this same process yields
    effective counts from 10 to 27 -- and this case must isolate one gate.
    """
    measurement = measure_area_per_lipid(
        *_series(_correlated(n=20000, tau_frames=900.0, seed=2)), N_LEAFLET
    )

    assert not measurement.converged
    assert measurement.drift_sigma < MAX_DRIFT_SIGMA, "not a drift failure"
    assert measurement.effective_samples < MIN_EFFECTIVE_SAMPLES
    assert measurement.n_frames > 5000, "the frames are there; the information is not"
    assert "longer run" in measurement.rejection


def test_a_drifting_series_that_survives_trimming_still_fails_on_drift():
    """A gentle whole-run slope keeps every frame, so drift must catch it."""
    times = np.arange(6000, dtype=float) * 10.0
    sloped = 0.60 + 0.05 * (1.0 - times / times[-1])
    measurement = measure_area_per_lipid(*_series(sloped), N_LEAFLET)

    assert measurement.retained_fraction >= MIN_RETAINED_FRACTION
    assert not measurement.converged
    assert measurement.effective_samples < MIN_EFFECTIVE_SAMPLES, (
        "a slope is a correlation; it must not survive as 'enough samples'"
    )


# --------------------------------------------------------------------------
# The statistics themselves


def test_the_autocorrelation_time_recovers_a_known_one():
    tau = integrated_autocorrelation(_correlated(n=200000, tau_frames=50.0))
    # AR(1) has tau_int = (1 + phi) / (1 - phi); for tau_frames = 50 that is
    # about 99 frames. A factor-of-two band is the honest tolerance for a
    # windowed estimator on a finite series.
    assert 50.0 < tau < 200.0


def test_uncorrelated_data_costs_one_frame_per_sample():
    assert integrated_autocorrelation(_plateau(n=20000)) < 2.0


def test_a_constant_series_does_not_divide_by_zero():
    assert integrated_autocorrelation(np.full(1000, 0.6)) == 1.0


@pytest.mark.parametrize("length", [37, 100, 751])
def test_correlation_matches_independent_pair_normalized_finite_sum(length):
    # Slow finite signals expose the extra taper that a long white-noise test misses.
    values = np.cos(np.arange(length) * 0.019) + np.arange(length) * 0.001
    centred = values - values.mean()
    variance = sum(float(value) ** 2 for value in centred) / length
    expected = 1.0
    for lag in range(1, length):
        pairs = [float(a * b) for a, b in zip(centred[:-lag], centred[lag:], strict=True)]
        correlation = sum(pairs) / len(pairs) / variance
        if correlation <= 0:
            break
        expected += 2 * correlation * (1 - lag / length)
    assert integrated_autocorrelation(values) == pytest.approx(max(1, expected), rel=1e-12)


def test_old_area_statistics_cannot_be_relabelled_as_current():
    from gmxbuilder.modules.membrane.area_observable import (
        AUTOCORRELATION_METHOD,
        AreaMeasurement,
        measurement_rejection,
    )

    measured = measure_area_per_lipid(*_series(_plateau()), N_LEAFLET)
    block = measured.as_metadata()
    assert block["autocorrelation_method"] == AUTOCORRELATION_METHOD
    assert measurement_rejection(AreaMeasurement.from_metadata(block)) is None
    block.pop("autocorrelation_method")
    block["schema_version"] = 1
    block["converged"] = True
    legacy = AreaMeasurement.from_metadata(block)
    assert "reanalysis required" in measurement_rejection(legacy)
    assert not pool_replicas([legacy, measured]).converged
    assert legacy.as_metadata()["autocorrelation_method"] != AUTOCORRELATION_METHOD


# --------------------------------------------------------------------------
# Contract with the callers


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lipids_per_leaflet": 0},
        {"lipids_per_leaflet": -4},
    ],
)
def test_a_nonsensical_leaflet_count_is_refused(kwargs):
    times, x, y = _series(_plateau(n=100))
    with pytest.raises(ValueError):
        measure_area_per_lipid(times, x, y, **kwargs)


def test_mismatched_series_lengths_are_refused():
    times, x, y = _series(_plateau(n=100))
    with pytest.raises(ValueError, match="equal length"):
        measure_area_per_lipid(times[:-1], x, y, N_LEAFLET)


def test_the_metadata_block_carries_the_criteria_it_was_judged_against():
    """A reviewer must be able to see the thresholds, not just the verdict."""
    block = measure_area_per_lipid(*_series(_plateau()), N_LEAFLET).as_metadata()

    assert block["area_per_lipid_nm2"]["mean"] == pytest.approx(0.60, abs=0.001)
    assert block["converged"] is True
    assert block["rejection"] is None
    assert block["criteria"]["max_drift_sigma"] == MAX_DRIFT_SIGMA
    assert block["criteria"]["min_retained_fraction"] == MIN_RETAINED_FRACTION
    assert set(block["criteria"]) == {
        "max_drift_sigma",
        "min_effective_samples",
        "min_blocks",
        "min_retained_fraction",
    }


def test_a_rejection_says_which_gate_failed_and_by_how_much():
    times = np.arange(5000, dtype=float) * 10.0
    measurement = measure_area_per_lipid(
        *_series(0.60 + 0.10 * np.exp(-times / 40000.0)), N_LEAFLET
    )
    text = summarise(measurement)
    assert "NOT converged" in text
    assert "%" in text and "nm^2" in text


def test_the_summary_of_an_accepted_measurement_says_so():
    assert "-- converged" in summarise(measure_area_per_lipid(*_series(_plateau()), N_LEAFLET))


def test_the_measurement_is_immutable():
    measurement = measure_area_per_lipid(*_series(_plateau(n=200)), N_LEAFLET)
    assert isinstance(measurement, AreaMeasurement)
    with pytest.raises(AttributeError):
        measurement.mean_nm2 = 1.0


def test_the_metadata_is_strict_json_even_when_nothing_could_be_measured():
    """``Infinity`` is a Python token, not JSON; a reviewer's parser rejects it.

    Reproduced by a real test-mode build, which runs 0.5 ps and so cannot
    measure drift at all: the block went out carrying ``Infinity`` and the
    file stopped being readable by anything but Python.
    """
    block = measure_area_per_lipid(
        np.array([0.0, 0.5]), np.array([6.4, 6.41]), np.array([6.4, 6.41]), N_LEAFLET
    ).as_metadata()

    json.dumps(block, allow_nan=False)
    assert block["area_per_lipid_nm2"]["relative_half_drift"] is None
    assert block["converged"] is False
    assert "too few to measure" in block["rejection"]


def test_a_run_with_too_few_frames_says_so_rather_than_blaming_the_bilayer():
    measurement = measure_area_per_lipid(
        np.array([0.0, 0.5]), np.array([6.4, 6.41]), np.array([6.4, 6.41]), N_LEAFLET
    )
    assert "too short to publish an area from" in measurement.rejection
    assert "still relaxing" not in measurement.rejection


def test_a_drift_gate_must_be_reachable_at_some_run_length():
    """A fixed fraction is not a threshold; it can sit below the noise floor.

    The drift statistic's own uncertainty is set by the observable's spread and
    how often it decorrelates. On a real 100 ns POPC bilayer -- spread 2.3% of
    the mean, correlation time 6 ns, about four independent samples per quarter
    -- that floor is 1.56%, above the 1% the gate used to demand. No run of any
    length could pass, and the result would have been read as "100 ns is not
    enough" when the run was fine and the test was not.

    The fixture reproduces that measurement's statistics: spread 2.4% of the
    mean, correlation time 600 frames, about twenty independent samples. It
    lands at 1.9% drift and 1.6 sigma, against the real run's 2.18% and 1.4.
    """
    measurement = measure_area_per_lipid(
        *_series(_correlated(n=10000, tau_frames=600.0, seed=9, spread=0.0144)), N_LEAFLET
    )

    assert measurement.relative_half_drift > 0.01, "the fixed 1% gate would have rejected this"
    assert measurement.drift_sigma < MAX_DRIFT_SIGMA, (
        "but the difference is within its own uncertainty, so it is not a drift"
    )
    assert measurement.converged, measurement.rejection


# --------------------------------------------------------------------------
# Pooling replicas: the check V3 could not make


def _measured(mean: float, n: int = 20000, tau: float = 200.0, seed: int = 0):
    return measure_area_per_lipid(
        *_series(_correlated(n=n, tau_frames=tau, mean=mean, seed=seed)), N_LEAFLET
    )


def test_two_agreeing_replicas_pool_to_a_tighter_answer():
    pooled = pool_replicas([_measured(0.600, seed=1), _measured(0.600, seed=2)])

    assert pooled.converged, pooled.rejection
    assert pooled.mean_nm2 == pytest.approx(0.600, abs=0.002)
    assert pooled.effective_samples > max(m.effective_samples for m in pooled.replicas)


def test_replicas_that_disagree_are_refused_however_tidy_each_one_looks():
    """Each run can be internally converged and the pair still be wrong.

    Two seeds of one system must give one area. Disagreement beyond their
    combined error means the entry is not measured -- and V3, with a single
    trajectory and no error bar, had no way to notice.
    """
    pooled = pool_replicas([_measured(0.600, seed=1), _measured(0.660, seed=2)])

    assert all(m.converged for m in pooled.replicas), "each replica is fine on its own"
    assert not pooled.converged
    assert "seeds have not converged to the same bilayer" in pooled.rejection


def test_a_single_replica_is_refused_by_name():
    pooled = pool_replicas([_measured(0.600, seed=1)])
    assert not pooled.converged
    assert "V3 failure this replaces" in pooled.rejection


def test_an_unconverged_replica_disqualifies_the_pool():
    times = np.arange(5000, dtype=float) * 10.0
    drifting = measure_area_per_lipid(*_series(0.60 + 0.10 * np.exp(-times / 40000.0)), N_LEAFLET)
    pooled = pool_replicas([_measured(0.600, seed=1), drifting])

    assert not pooled.converged
    assert "1 of 2 replicas" in pooled.rejection


def test_the_pooled_mean_weights_by_independent_samples_not_frames():
    """A replica that ran longer without decorrelating has not earned more say."""
    many_frames = _measured(0.700, n=40000, tau=4000.0, seed=1)  # few independent
    fewer_frames = _measured(0.600, n=20000, tau=100.0, seed=2)  # many independent

    pooled = pool_replicas([many_frames, fewer_frames])
    assert many_frames.effective_samples < fewer_frames.effective_samples
    assert abs(pooled.mean_nm2 - 0.600) < abs(pooled.mean_nm2 - 0.700)


def test_the_pooled_metadata_is_strict_json_and_keeps_every_replica():
    pooled = pool_replicas([_measured(0.600, seed=1), _measured(0.600, seed=2)])
    block = pooled.as_metadata()

    json.dumps(block, allow_nan=False)
    assert block["pooled_replicas"] == 2
    assert len(block["replicas"]) == 2
    assert block["measures"] == "pure-bilayer-area-per-lipid"
    assert block["criteria"]["max_replica_spread_sigma"] == MAX_REPLICA_SPREAD_SIGMA


def test_pooling_nothing_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError, match="at least one replica"):
        pool_replicas([])


def test_the_published_error_is_never_narrower_than_the_replicas_disagree():
    """Two seeds scattering more than each run claims is information, not noise.

    A single run's block average cannot see what its own correlation-time
    estimate got wrong; the scatter between independent seeds can. Reporting
    the narrower of the two would publish a precision the pair has not earned.
    """
    pooled = pool_replicas([_measured(0.600, seed=1), _measured(0.612, seed=2)])
    half_spread = abs(pooled.replicas[0].mean_nm2 - pooled.replicas[1].mean_nm2) / 2.0

    assert pooled.standard_error_nm2 >= half_spread * 0.99


def test_the_agreement_gate_does_not_reject_pairs_that_agree():
    """A gate is chosen from its false-rejection rate, not from a round number.

    Over 120 pairs drawn from one distribution -- agreeing by construction --
    the rate is 5.0% at 3 sigma, 0.8% at 4, 0.0% at 5, because the per-replica
    error rests on an estimated correlation time that is sometimes estimated
    small. At 3 sigma about fourteen of the library's 277 entries would have
    been sent back for a rerun they did not need.
    """
    rejected = 0
    for pair in range(30):
        pooled = pool_replicas(
            [_measured(0.600, seed=2 * pair), _measured(0.600, seed=2 * pair + 1)]
        )
        rejected += not pooled.converged
    assert rejected <= 2, f"{rejected}/30 agreeing pairs rejected"

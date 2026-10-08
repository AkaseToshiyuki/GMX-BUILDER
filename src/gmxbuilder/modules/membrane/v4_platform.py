"""Deterministic membrane-platform screening for initialization conformers.

This is an engineering tolerance on observed APL/DHH changes, not a confidence
test or a claim of equilibrium. Autocorrelation and source-sampling requirements
are enforced separately. Raw primary series let every consumer recompute the
same decision rather than trusting a persisted ``passed`` flag.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD

PRIMARY_OBSERVABLES = ("area_per_lipid_nm2", "head_to_head_nm")
METHOD = "apl-dhh-construction-platform-1"
RELATIVE_TOLERANCE = 0.02  # Reviewed observed-change tolerance, not a confidence bound.
MAXIMUM_NS = 200.0  # Compute budget per replica, not an equilibrium timescale.


def construction_policy():
    return {
        "method": METHOD,
        "scope": "initialization_conformers_only",
        "observables": list(PRIMARY_OBSERVABLES),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "window": "existing-common-post-burn-in-window",
        "checks": [
            "quarter_mean_range",
            "terminal_half_shift",
            "linear_change",
            "terminal_block_shift",
        ],
        "minimum_effective_samples": 10,
        "maximum_npt_ns": MAXIMUM_NS,
        "maximum_primary_replica_spread_se": 4.0,
        "equilibrium_certification": False,
    }


def platform_metrics(values):
    """Measure shape on the entire window, including an incomplete final part."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 10 or not np.isfinite(values).all():
        raise ValueError("Platform requires at least ten finite primary observations")
    if np.any(values <= 0):
        raise ValueError("APL and DHH must be positive")
    mean = float(values.mean())
    # Compare the two halves of the final quarter, not the two halves of the full window.
    terminal = values[3 * len(values) // 4 :]
    # Each terminal half needs two points; the relative floor rejects frozen series.
    if len(terminal) < 4 or min(values.var(), terminal.var()) <= (mean * 1e-12) ** 2:
        raise ValueError("Frozen or too short primary series cannot establish a platform")
    quarters = [float(part.mean()) for part in np.array_split(values, 4)]
    half = len(terminal) // 2
    # Normalizing time to a unit span makes the slope an end-to-end change,
    # independent of trajectory duration and physical time units.
    time = np.linspace(-0.5, 0.5, len(values))
    change = float(np.dot(time, values - mean) / np.dot(time, time))
    from gmxbuilder.modules.membrane.area_observable import integrated_autocorrelation

    correlation = integrated_autocorrelation(values)
    terminal_size = max(1, min(len(values), int(np.ceil(correlation))))
    return {
        "mean": mean,
        "quarter_means": quarters,
        "quarter_mean_range": (max(quarters) - min(quarters)) / mean,
        "terminal_half_shift": float(terminal[half:].mean() - terminal[:half].mean()) / mean,
        "linear_change": change / mean,
        "terminal_block_shift": float(values[-terminal_size:].mean() - mean) / mean,
        "correlation_frames": float(correlation),
        "terminal_block_frames": terminal_size,
    }


def platform_evidence(times, series, start):
    return {
        "policy": construction_policy(),
        "time_ps": np.asarray(times[start:], dtype=float).tolist(),
        "series": {
            key: np.asarray(series[key][start:], dtype=float).tolist()
            for key in PRIMARY_OBSERVABLES
        },
    }


def assess_platform(analysis):
    """Raise for invalid evidence; return deficits for an observed moving membrane."""
    evidence = analysis["construction_platform"]
    if evidence["policy"] != construction_policy():
        raise ValueError("Missing or incompatible construction platform policy")
    times = np.asarray(evidence["time_ps"], dtype=float)
    if times.ndim != 1 or len(times) < 10 or not np.isfinite(times).all():
        raise ValueError("Invalid platform times")
    if not np.allclose([times[0], times[-1]], analysis["analysis_window_ps"], atol=0.01, rtol=0):
        raise ValueError("Platform and sampling windows differ")
    # 100 ps is the structural sampling cadence; 0.01 ps tolerates timestamp roundoff.
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, 100.0, atol=0.01, rtol=0):
        raise ValueError("Platform requires the recorded 100 ps coordinate sampling")
    if set(evidence["series"]) != set(PRIMARY_OBSERVABLES):
        raise ValueError("Platform primary observables differ")
    from gmxbuilder.modules.membrane.v4_measurement import INCOMPLETE, validate_measurement_coverage

    validate_measurement_coverage(analysis)
    details, failures = {}, [INCOMPLETE] if analysis.get("measurement_coverage") else []
    for key in PRIMARY_OBSERVABLES:
        values = np.asarray(evidence["series"][key], dtype=float)
        if values.shape != times.shape:
            raise ValueError("Platform series and times differ in length")
        detail = platform_metrics(values)
        measured = analysis["observables"][key]
        if measured["n_frames"] != len(values) or not np.isclose(
            detail["mean"], measured["mean"], rtol=1e-10, atol=0
        ):
            raise ValueError("Platform does not match recorded sampling statistics")
        recorded_correlation = detail["correlation_frames"]
        if analysis.get("autocorrelation_method") != AUTOCORRELATION_METHOD:
            if (
                analysis.get("autocorrelation_method") is not None
                or analysis.get("schema") != "v4-common-window-3"
            ):
                raise ValueError("Unknown platform statistics version; reanalysis required")
            # Check the integrity of the one known historical representation.
            # Its raw series is reusable, but its twice-tapered uncertainty is
            # never used for current platform metrics or construction sampling.
            centred = values - values.mean()
            products = np.correlate(centred, centred, mode="full")[len(values) - 1 :]
            positive_end = next(
                (lag for lag in range(1, len(values)) if products[lag] <= 0), len(values)
            )
            lags = np.arange(1, positive_end)
            recorded_correlation = max(
                1.0,
                float(1 + 2 * np.sum((1 - lags / len(values)) * products[lags] / products[0])),
            )
        if not np.allclose(
            [measured["autocorrelation_ps"], measured["effective_samples"]],
            [recorded_correlation * 100.0, len(values) / recorded_correlation],
            rtol=1e-10,
            atol=0,
        ):
            raise ValueError("Primary sampling evidence differs from its coordinate series")
        for check in construction_policy()["checks"]:
            if abs(detail[check]) > RELATIVE_TOLERANCE:
                failures.append(f"{key}/{check}: {detail[check]:.4%} exceeds 2% platform tolerance")
        details[key] = detail
    return {"passed": not failures, "failures": failures, "details": details}

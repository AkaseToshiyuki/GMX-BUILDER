"""Common-window diagnostics for V4; conformer and area acceptance are separate.

Burn-in window selection maximizes autocorrelation-corrected samples (Chodera,
2016, doi:10.1021/acs.jctc.5b00784). The last quarter is withheld from window
selection and checked for late drift. These diagnostics do not prove a phase
or exclude an unobserved slower process.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from gmxbuilder.modules.membrane.area_observable import (
    AUTOCORRELATION_METHOD,
    integrated_autocorrelation,
)
from gmxbuilder.modules.membrane.v4_stationarity import joint_stationarity, stationarity_failures

MIN_EFFECTIVE_SAMPLES = 10  # Maintainer's minimum independent time samples per observable.
# Preserve the existing burn-in search resolution. This numerical safeguard
# against fitting an FFT correlation to a tiny tail is NOT an acceptance gate.
WINDOW_SEARCH_MIN_FRAMES = 32


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sampling_failures(analysis, *, include_stationarity=True):
    """Recheck stored evidence instead of trusting a serialized passed flag."""
    try:
        if analysis.get("autocorrelation_method") != AUTOCORRELATION_METHOD:
            return ["Obsolete autocorrelation statistics; reanalysis required"]
        from gmxbuilder.modules.membrane.v4_measurement import (
            INCOMPLETE,
            validate_measurement_coverage,
        )

        validate_measurement_coverage(analysis)
        window = analysis["analysis_window_ps"]
        if len(window) != 2 or not 0 <= window[0] < window[1]:
            return ["Invalid common analysis window"]
        fields = analysis["observables"]
        if not {"area_per_lipid_nm2", "box_z_nm", "volume_nm3", "head_to_head_nm"} <= fields.keys():
            return ["Missing required slow observables"]
        failures = (
            stationarity_failures(analysis.get("stationarity", {}), fields)
            if include_stationarity
            else []
        )
        if analysis.get("measurement_coverage") is not None:
            failures.append(INCOMPLETE)
        for key, block in fields.items():
            numbers = [
                block[k]
                for k in (
                    "mean",
                    "standard_error",
                    "effective_samples",
                    "autocorrelation_ps",
                )
            ]
            if not all(v is not None and np.isfinite(v) for v in numbers):
                failures.append(f"{key}: missing or non-finite statistics")
                continue
            if block["effective_samples"] < MIN_EFFECTIVE_SAMPLES or block["n_blocks"] < 2:
                failures.append(f"{key}: insufficient independent sampling")
            if block["standard_error"] <= 0 or block["autocorrelation_ps"] <= 0:
                failures.append(f"{key}: invalid uncertainty or correlation time")
        return failures
    except (KeyError, TypeError, ValueError):
        return ["Malformed trajectory diagnostics"]


def quantitative_area(replicas):
    """Equal-replica mean with Welch block uncertainty and between-run scatter."""
    agreement = replica_agreement(replicas)
    blocks = [r["observables"]["area_per_lipid_nm2"] for r in replicas]
    means = np.array([v["mean"] for v in blocks])
    variances = np.array([v["standard_error"] ** 2 for v in blocks])
    counts = np.array([v["n_blocks"] for v in blocks])
    n = len(blocks)
    mean = float(means.mean())
    se = float(np.sqrt(variances.sum()) / n)
    denominator = np.sum(variances**2 / np.maximum(counts - 1, 1))
    df = float(variances.sum() ** 2 / denominator) if denominator > 0 else 1.0
    between = float(means.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("inf")
    # Use between-replica scatter if larger; its uncertainty has only n - 1 degrees of freedom.
    if between > se:
        se, df = between, float(n - 1)
    # 0.975 is the upper quantile of a two-sided 95% Student interval.
    half = float(student_t.ppf(0.975, df) * se)
    relative = half / mean if mean > 0 else float("inf")
    failures = [reason for r in replicas for reason in sampling_failures(r)]
    failures.extend(agreement["failures"])
    if not np.isfinite(relative) or relative > 0.02:
        failures.append("APL 95% CI relative half-width exceeds 2%")
    return json_safe(
        {
            "schema_version": 2,
            "autocorrelation_method": AUTOCORRELATION_METHOD,
            "area_per_lipid_nm2": {
                "mean": mean,
                "standard_error": se,
                "relative_ci95_half_width": relative,
                "ci95_half_width": half,
                "degrees_of_freedom": df,
                "effective_samples": sum(v["effective_samples"] for v in blocks),
            },
            "converged": not failures,
            "rejection": "; ".join(failures) if failures else None,
            "criteria": {"max_relative_ci95_half_width": 0.02, "max_replica_spread_sigma": 4},
            "replica_agreement": agreement,
        }
    )


def _window_start(values):
    # Fit burn-in using only the first half; reserve later data for drift diagnostics.
    training = values[: len(values) // 2]
    last = len(training) - WINDOW_SEARCH_MIN_FRAMES
    if last <= 0:
        return 0
    best, score = 0, -1.0
    # Limit the search to 80 candidate starts to bound FFT cost, not acceptance.
    for index in np.unique(np.linspace(0, last, 80).astype(int)):
        tail = training[index:]
        effective = len(tail) / integrated_autocorrelation(tail)
        if effective > score:
            best, score = int(index), effective
    return best


def _statistics(values, frame_ps):
    tau = integrated_autocorrelation(values)
    size = max(1, int(np.ceil(tau)))
    count = len(values) // size
    mean = float(np.mean(values))
    error = float("inf")
    if count >= 2:
        # Include the terminal samples; dropping an incomplete last block can
        # discard precisely the late collapse this diagnostic must detect.
        means = np.array([chunk.mean() for chunk in np.array_split(values, count)])
        error = float(means.std(ddof=1) / np.sqrt(count))
    half_width = float(student_t.ppf(0.975, count - 1) * error) if count >= 2 else float("inf")
    reasons = []
    if len(values) / tau < MIN_EFFECTIVE_SAMPLES:
        reasons.append("fewer than ten effective time samples")
    if count < 2:
        reasons.append("fewer than two blocks cannot estimate a standard error")
    return {
        "mean": mean,
        "standard_error": error,
        "n_frames": len(values),
        "n_blocks": count,
        "effective_samples": float(len(values) / tau),
        "autocorrelation_ps": float(tau * frame_ps),
        "relative_ci95_half_width": half_width / max(abs(mean), 1e-12),
        "failures": reasons,
    }


def assess_series(times_ps, series: dict[str, np.ndarray]) -> dict:
    times = np.asarray(times_ps, dtype=float)
    if times.ndim != 1 or len(times) < MIN_EFFECTIVE_SAMPLES or not np.isfinite(times).all():
        raise ValueError("Too few finite trajectory times for V4 stationarity analysis")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, np.median(steps), rtol=1e-4, atol=0.01):
        raise ValueError("V4 analysis requires a strictly increasing, uniform time series")
    values = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    if not values or any(
        v.shape != times.shape or not np.isfinite(v).all() for v in values.values()
    ):
        raise ValueError("Missing, non-finite or unequal observable series")
    # The latest selected start gives every observable the same retained window.
    start = max(_window_start(v) for v in values.values())
    holdout_start = 3 * len(times) // 4
    measured = {k: _statistics(v[start:], float(np.median(steps))) for k, v in values.items()}
    names = sorted(values)
    # Estimate reference noise before the final-quarter holdout. This diagnostic
    # does not establish an error-rate guarantee across repeated queue checks.
    reference_start = start
    training = np.column_stack([values[k][reference_start:holdout_start] for k in names])
    holdout = np.column_stack([values[k][holdout_start:] for k in names])
    stationarity = joint_stationarity(training, holdout, names)
    from gmxbuilder.modules.membrane.v4_platform import platform_evidence

    failures = [f"{k}: {reason}" for k, v in measured.items() for reason in v["failures"]]
    failures.extend(stationarity_failures(stationarity, names))
    return {
        "schema": "v4-common-window-4",
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "stationarity": stationarity,
        "construction_platform": platform_evidence(times, values, start),
        "analysis_window_ps": [float(times[start]), float(times[-1])],
        "reference_window_ps": [float(times[reference_start]), float(times[holdout_start - 1])],
        "holdout_window_ps": [float(times[holdout_start]), float(times[-1])],
        "retained_fraction": (len(times) - start) / len(times),
        "minimum_effective_samples": min(v["effective_samples"] for v in measured.values()),
        "sample_spacing_ps": max(v["autocorrelation_ps"] for v in measured.values()),
        "observables": measured,
        "passed": not failures,
        "failures": failures,
        "sampling_status": "sufficient"
        if all(v["effective_samples"] >= MIN_EFFECTIVE_SAMPLES for v in measured.values())
        else "insufficient",
        "limiting_observable": min(measured, key=lambda k: measured[k]["effective_samples"]),
    }


def replica_agreement(replicas: list[dict]) -> dict:
    """Compare membrane observables and composition-compatible absolute box sizes."""
    from gmxbuilder.modules.membrane.v4_comparison import comparison_policy

    if len(replicas) < 2:
        return {"passed": False, "failures": ["At least two independent replicas required"]}
    keys = set(replicas[0]["observables"])
    if any(set(r["observables"]) != keys for r in replicas):
        return {"passed": False, "failures": ["Replica observables differ"]}
    policy = comparison_policy(replicas)
    spreads, failures = {}, list(policy["failures"])
    for key in sorted(keys):
        blocks = [r["observables"][key] for r in replicas]
        maximum = 0.0
        for i, a in enumerate(blocks):
            for b in blocks[i + 1 :]:
                if (
                    not all(
                        np.isfinite(item[field])
                        for item in (a, b)
                        for field in ("mean", "standard_error")
                    )
                    or min(a["standard_error"], b["standard_error"]) <= 0
                ):
                    failures.append(f"{key}: invalid replica uncertainty")
                    maximum = float("inf")
                    continue
                denominator = np.hypot(a["standard_error"], b["standard_error"])
                spread = (
                    abs(a["mean"] - b["mean"]) / denominator if denominator > 0 else float("inf")
                )
                maximum = max(maximum, float(spread))
        spreads[key] = maximum
        if key not in policy["excluded"] and (not np.isfinite(maximum) or maximum > 4):
            failures.append(f"{key}: replica spread {maximum:.3g} exceeds 4 combined SE")
    return {
        "passed": not failures,
        "spread_sigma": spreads,
        "failures": failures,
        "comparison_policy": policy,
    }

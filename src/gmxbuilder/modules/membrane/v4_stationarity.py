"""Reference-based late-drift diagnostics with finite-block uncertainty.

Estimate noise from a reference window that excludes the terminal holdout.
Fewer than two reference blocks cannot estimate variance; otherwise Student
intervals use the observed degrees of freedom. Require intervals inside the
declared equivalence margins, with Bonferroni adjustment across contrasts.

The nominal 5% family level assumes valid fixed-look intervals. Short-trajectory
uncertainty and repeated-look error control are not established. See Grossfield
et al. (2018), doi:10.33011/livecoms.1.1.5067, for sampling uncertainty limits.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from gmxbuilder.modules.membrane.area_observable import (
    AUTOCORRELATION_METHOD,
    integrated_autocorrelation,
)

METHOD = "late-window-equivalence-1"
RELATIVE_MARGIN = 0.02  # Reviewed relative margin for positive size observables.
ORDER_MARGIN = 0.05  # Absolute molecular-axis P2 margin; P2 may be near zero.
FAMILY_ALPHA = 0.05  # Nominal fixed-look family level, not a sequential error budget.
DIAGNOSTICS = ("reference_shift", "terminal_half_shift")


def _unique_groups(reference, holdout, names):
    """An exact duplicate measurement must not multiply the hypothesis family."""
    groups = []
    for index, name in enumerate(names):
        for representative, aliases in groups:
            if np.array_equal(reference[:, index], reference[:, representative]) and np.array_equal(
                holdout[:, index], holdout[:, representative]
            ):
                aliases.append(name)
                break
        else:
            groups.append((index, [name]))
    return groups


def _reference_uncertainty(reference):
    inefficiency = np.array([integrated_autocorrelation(column) for column in reference.T])
    # Twice the largest correlation length is the heuristic span for block means;
    # it does not guarantee that those means are independent.
    block_frames = max(1, int(np.ceil(2 * inefficiency.max())))
    count = len(reference) // block_frames
    evidence = {"block_frames": block_frames, "reference_blocks": count}
    if count < 2:
        return None, evidence
    chunks = np.array_split(reference, count)
    block_means = np.array([chunk.mean(axis=0) for chunk in chunks])
    block_variance = block_means.var(axis=0, ddof=1) * (len(reference) / count)
    correlation_variance = reference.var(axis=0, ddof=1) * inefficiency
    # Finite blocks lose correlations at their edges. Do not report a smaller
    # noise scale than the autocorrelation estimate, particularly for mixed
    # fast/slow fluctuations. Taking the larger estimate is conservative; it
    # does not make either estimator exact for a short trajectory.
    variance = np.maximum(block_variance, correlation_variance)
    evidence.update(
        degrees_of_freedom=count - 1,
        block_variance=block_variance.tolist(),
        correlation_variance=correlation_variance.tolist(),
        long_run_variance=variance.tolist(),
    )
    return variance, evidence


def equivalence_margin(name, mean):
    if name in {"area_per_lipid_nm2", "box_z_nm", "volume_nm3", "head_to_head_nm"}:
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("Positive physical reference mean required")
        return RELATIVE_MARGIN * mean
    if name.endswith("_axis_P2"):
        return ORDER_MARGIN
    raise ValueError(f"No reviewed equivalence margin for {name}")


def analysis_policy():
    return {
        "relative_margin": RELATIVE_MARGIN,
        "axis_P2_absolute_margin": ORDER_MARGIN,
        "family_alpha": FAMILY_ALPHA,
        "contrasts": list(DIAGNOSTICS),
        "reference": "post-burn-in through first 75% of trajectory",
        "holdout": "last 25%; compare its mean to reference and its two half means",
    }


def joint_stationarity(reference: np.ndarray, holdout: np.ndarray, names: list[str]) -> dict:
    """Require simultaneous intervals INSIDE margins, never merely p > alpha.

    The fixed contrasts describe late-window means, not instantaneous values or
    every possible change point. All-frame molecular/geometry checks remain
    separate. A short terminal excursion can be diluted in a time average.
    """
    result = {
        "method": METHOD,
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "family_alpha": FAMILY_ALPHA,
        "policy": analysis_policy(),
        "scope": "simultaneous late-window mean contrasts in one replica assessment",
        "observables": names,
        "diagnostics": list(DIAGNOSTICS),
        "reference_frames": len(reference),
        "holdout_frames": len(holdout),
        "status": "insufficient_evidence",
        "uncertainty_estimated": False,
        "calibration_status": "not-established",
        "passed": False,
    }
    if (
        reference.ndim != 2
        or holdout.ndim != 2
        or not names
        or reference.shape[1] != len(names)
        or holdout.shape[1] != len(names)
        or len(set(names)) != len(names)
        or len(reference) < 2
        or len(holdout) < 4
        or not np.isfinite(reference).all()
        or not np.isfinite(holdout).all()
    ):
        return {**result, "reason": "Missing or non-finite reference/holdout observations"}
    for label, values in (("reference", reference), ("holdout", holdout)):
        scale = np.maximum(np.abs(values.mean(axis=0)), 1.0)
        if np.any(values.var(axis=0) <= (scale * 1e-12) ** 2):
            return {
                **result,
                "reason": f"Constant {label} observables prevent uncertainty estimation",
            }
    try:
        margins = {
            name: equivalence_margin(name, float(reference[:, i].mean()))
            for i, name in enumerate(names)
        }
    except ValueError as exc:
        return {**result, "reason": str(exc)}
    groups = _unique_groups(reference, holdout, names)
    indices = [index for index, _ in groups]
    reference, holdout = reference[:, indices], holdout[:, indices]
    variance, evidence = _reference_uncertainty(reference)
    result.update(evidence, observable_groups=[aliases for _, aliases in groups])
    if variance is None:
        return {
            **result,
            "reason": "Reference variance is unidentifiable from fewer than two blocks",
        }
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        return {**result, "reason": "Invalid reference uncertainty"}
    df = evidence["degrees_of_freedom"]
    n, half = len(holdout), len(holdout) // 2
    differences = {
        "reference_shift": holdout.mean(0) - reference.mean(0),
        "terminal_half_shift": holdout[half:].mean(0) - holdout[:half].mean(0),
    }
    weights = {
        "reference_shift": 1 / n + 1 / len(reference),
        "terminal_half_shift": 1 / half + 1 / (n - half),
    }
    # A 95% simultaneous two-sided confidence family is more conservative than
    # the 90% interval of a conventional single 5%-level TOST. Bonferroni does
    # not assume independent observables. Short references widen the intervals.
    comparisons = len(groups) * len(DIAGNOSTICS)
    critical = float(student_t.ppf(1 - FAMILY_ALPHA / (2 * comparisons), df))
    details, states = {}, []
    for i, (_, aliases) in enumerate(groups):
        for name in aliases:
            contrasts = {}
            for diagnostic in DIAGNOSTICS:
                difference = float(differences[diagnostic][i])
                error = float(np.sqrt(variance[i] * weights[diagnostic]))
                lower, upper = difference - critical * error, difference + critical * error
                margin = margins[name]
                state = interval_status(lower, upper, margin)
                states.append(state)
                contrasts[diagnostic] = {
                    "difference": difference,
                    "standard_error": error,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "margin": margin,
                    "status": state,
                }
            details[name] = {
                "reference_mean": float(reference[:, i].mean()),
                "contrasts": contrasts,
            }
    state = combined_status(states)
    return {
        **result,
        "uncertainty_estimated": True,
        "status": state,
        "passed": state == "equivalent",
        "comparisons": comparisons,
        "critical_value": critical,
        "details": details,
        "reason": None
        if state == "equivalent"
        else (
            "Late-window drift exceeds equivalence margin"
            if state == "drift_detected"
            else "Late-window confidence intervals do not establish equivalence"
        ),
    }


def interval_status(lower, upper, margin):
    if lower > -margin and upper < margin:
        return "equivalent"
    if lower > margin or upper < -margin:
        return "drift_detected"
    return "insufficient_evidence"


def combined_status(states):
    if "drift_detected" in states:
        return "drift_detected"
    return (
        "equivalent" if all(state == "equivalent" for state in states) else "insufficient_evidence"
    )


def stationarity_failures(result: dict, names) -> list[str]:
    """Rebuild every interval and margin; serialized pass flags cannot override it."""
    try:
        if (
            result["method"] != METHOD
            or result.get("autocorrelation_method") != AUTOCORRELATION_METHOD
            or set(result["observables"]) != set(names)
            or result["policy"] != analysis_policy()
        ):
            return ["Missing or incompatible equivalence evidence; reanalysis required"]
        # Old records used "calibrated" to mean only "variance computed".
        # Reading that legacy key must never be presented as external validation.
        if not result.get("uncertainty_estimated", result.get("calibrated", False)):
            return [
                "Stationarity evidence insufficient: " + (result.get("reason") or "unknown reason")
            ]
        df = result["degrees_of_freedom"]
        groups = result["observable_groups"]
        aliases = [name for group in groups for name in group]
        count = len(DIAGNOSTICS) * len(groups)
        if (
            df != result["reference_blocks"] - 1
            or df < 1
            or result["family_alpha"] != FAMILY_ALPHA
            or result["comparisons"] != count
            or len(aliases) != len(set(aliases))
            or set(aliases) != set(names)
            or set(result["details"]) != set(names)
        ):
            return ["Invalid equivalence uncertainty evidence"]
        reference_n, holdout_n = result["reference_frames"], result["holdout_frames"]
        block_frames = result["block_frames"]
        if (
            type(block_frames) is not int
            or block_frames < 1
            or type(reference_n) is not int
            or type(holdout_n) is not int
            or reference_n < 2
            or holdout_n < 4
            or reference_n // block_frames != result["reference_blocks"]
        ):
            return ["Invalid reference block partition"]
        variance = np.asarray(result["long_run_variance"], dtype=float)
        block_variance = np.asarray(result["block_variance"], dtype=float)
        correlation_variance = np.asarray(result["correlation_variance"], dtype=float)
        if (
            any(v.shape != (len(groups),) for v in (variance, block_variance, correlation_variance))
            or not all(
                np.isfinite(v).all() and np.all(v > 0)
                for v in (variance, block_variance, correlation_variance)
            )
            or not np.allclose(
                variance, np.maximum(block_variance, correlation_variance), rtol=1e-12, atol=0
            )
        ):
            return ["Invalid reference variance evidence"]
        group_index = {name: i for i, group in enumerate(groups) for name in group}
        half = holdout_n // 2
        weights = {
            "reference_shift": 1 / holdout_n + 1 / reference_n,
            "terminal_half_shift": 1 / half + 1 / (holdout_n - half),
        }
        critical = float(student_t.ppf(1 - FAMILY_ALPHA / (2 * count), df))
        if not np.isclose(result["critical_value"], critical, rtol=1e-12):
            return ["Invalid simultaneous confidence critical value"]
        failures = []
        for name, detail in result["details"].items():
            margin = equivalence_margin(name, detail["reference_mean"])
            if set(detail["contrasts"]) != set(DIAGNOSTICS):
                return ["Incomplete equivalence contrasts"]
            for diagnostic, contrast in detail["contrasts"].items():
                difference, error = contrast["difference"], contrast["standard_error"]
                if not np.isfinite(difference) or not np.isfinite(error) or error <= 0:
                    return ["Invalid equivalence contrast uncertainty"]
                expected_error = np.sqrt(variance[group_index[name]] * weights[diagnostic])
                if not np.isclose(error, expected_error, rtol=1e-12, atol=0):
                    return ["Equivalence uncertainty does not match reference evidence"]
                lower, upper = difference - critical * error, difference + critical * error
                if not np.allclose(
                    [contrast["ci_lower"], contrast["ci_upper"], contrast["margin"]],
                    [lower, upper, margin],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    return ["Equivalence interval does not match contrast evidence"]
                state = interval_status(lower, upper, margin)
                if state != "equivalent":
                    failures.append(
                        f"{name}/{diagnostic}: {state}; CI [{lower:.5g}, {upper:.5g}] "
                        f"must fit within +/-{margin:.5g}"
                    )
        return failures
    except (KeyError, TypeError, ValueError):
        return ["Malformed equivalence evidence"]

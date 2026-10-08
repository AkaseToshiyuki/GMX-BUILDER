"""Practical, correlation-aware admission for relaxed single-lipid conformers.

These engineering margins concern initialization coordinates. Nominal intervals
are conditional diagnostics, not simultaneous, sequential or equilibrium claims.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.stats import t as student_t

from gmxbuilder.modules.membrane.area_observable import (
    AUTOCORRELATION_METHOD,
    integrated_autocorrelation,
)

METHOD = "single-lipid-relaxation-2"
MAXIMUM_NS = 200.0


def construction_policy():
    return {
        "method": METHOD,
        "scope": "relaxed_single_lipid_initialization_conformers",
        "window": "fixed-final-half-of-NPT",
        "shape_relative_margin": 0.05,
        "shape_absolute_floor_nm": 0.02,
        "bond_mean_margin_nm": 0.002,
        "angle_mean_margin_degrees": 2.0,
        "torsion_population_margin": 0.10,
        "hydration_relative_margin": 0.10,
        "hydration_absolute_floor": 0.10,
        "hydration_cutoff_nm": 0.35,
        "minimum_effective_samples": 10,
        "minimum_segment_blocks": 2,
        "minimum_extraction_spacing_ps": 1000.0,
        "hard_interval_observables": ["bond", "angle", "hydration"],
        "shape_screen": "observed-median-change; interval-is-diagnostic",
        "torsion_populations": "diagnostic; equilibrium-weights-not-required",
        "nominal_interval": 0.95,
        "maximum_npt_ns": MAXIMUM_NS,
        "equilibrium_certification": False,
        "sequential_error_control": False,
    }


def evidence(times, series, definition):
    times = np.asarray(times, dtype=float)
    start = len(times) // 2
    return {
        "policy": construction_policy(),
        "definition": copy.deepcopy(definition),
        "trajectory_start_ps": float(times[0]),
        "total_frames": len(times),
        "time_ps": times[start:].tolist(),
        "series": {
            key: np.asarray(value, dtype=float)[start:].tolist()
            for key, value in sorted(series.items())
        },
    }


def _read(evidence):
    from gmxbuilder.modules.membrane.local_conformations import (
        population_descriptors,
        validate_definition,
    )

    if evidence["policy"] != construction_policy():
        raise ValueError("Missing or incompatible local-conformer policy")
    definition = evidence["definition"]
    validate_definition(definition)
    times = np.asarray(evidence["time_ps"], dtype=float)
    total = evidence["total_frames"]
    if type(total) is not int or total < 20 or times.shape != (total - total // 2,):
        raise ValueError("Local relaxation requires the complete final half of the trajectory")
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 100, rtol=0, atol=0.01):
        raise ValueError("Invalid local-conformation sampling times")
    origin = float(evidence["trajectory_start_ps"])
    if (
        not math.isfinite(origin)
        or abs(origin) > 0.01
        or abs(times[0] - (origin + total // 2 * 100)) > 0.01
    ):
        raise ValueError("Local-conformation evidence does not cover a complete NPT trajectory")
    expected = population_descriptors(
        np.zeros((2, len(definition["atom_names"]), 3)),
        definition,
        np.array([True, False]),
        np.zeros(2),
    )
    series = {key: np.asarray(value, dtype=float) for key, value in evidence["series"].items()}
    if set(series) != set(expected):
        raise ValueError("Missing or changed local-conformation descriptors")
    if any(
        value.shape != times.shape or not np.isfinite(value).all() or np.any(value < 0)
        for value in series.values()
    ):
        raise ValueError("Malformed local-conformation series")
    for key, value in series.items():
        if key.startswith("bond:") and np.any((value < 0.05) | (value > 0.35)):
            raise ValueError("Implausible mean heavy-atom covalent distance")
        if key.startswith("angle:") and np.any((value < 30) | (value > 180)):
            raise ValueError("Implausible mean heavy-atom bond angle")
        if key.startswith("torsion:") and np.any(value > 1):
            raise ValueError("Invalid torsion population")
        if ":trans:" in key:
            total_population = (
                value
                + series[key.replace(":trans:", ":minus:")]
                + series[key.replace(":trans:", ":plus:")]
            )
            if not np.allclose(total_population, 1, rtol=0, atol=1e-10):
                raise ValueError("Torsion populations do not sum to one")
        if key.endswith(":q50") and (
            np.any(series[key[:-3] + "q10"] > value) or np.any(value > series[key[:-3] + "q90"])
        ):
            raise ValueError("Shape quantiles are not ordered")
    if all(np.ptp(value) < 1e-10 for key, value in series.items() if key.startswith("shape:")):
        raise ValueError("Frozen molecular shapes cannot establish relaxation")
    return times, series


def _statistics(value):
    constant = bool(np.ptp(value) < 1e-12)
    correlation = 1.0 if constant else max(1.0, float(integrated_autocorrelation(value)))
    return {
        "mean": float(value.mean()),
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "correlation_frames": correlation,
        "effective_samples": float(len(value) / correlation),
        "constant_observed": constant,
    }


def _blocks(values, size):
    count = len(values) // size
    if count < construction_policy()["minimum_segment_blocks"]:
        return None
    if count == len(values):
        return values
    # Exactly np.array_split means without allocating one array per tiny block.
    lengths = np.full(count, len(values) // count, dtype=int)
    lengths[: len(values) % count] += 1
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    return np.add.reduceat(values, starts) / lengths


def _margin(key, mean):
    policy = construction_policy()
    if key.startswith("bond:"):
        return policy["bond_mean_margin_nm"]
    if key.startswith("angle:"):
        return policy["angle_mean_margin_degrees"]
    if key.startswith("shape:"):
        return max(policy["shape_absolute_floor_nm"], abs(mean) * policy["shape_relative_margin"])
    if key.startswith("torsion:"):
        return policy["torsion_population_margin"]
    if key.startswith("hydration:"):
        return max(
            policy["hydration_absolute_floor"], abs(mean) * policy["hydration_relative_margin"]
        )
    raise ValueError("Unknown local-conformation descriptor")


def _hard_interval(key):
    return key.split(":", 1)[0] in construction_policy()["hard_interval_observables"]


def _typical_shape(key):
    return key.startswith("shape:") and key.endswith(":q50")


def _observed_shape(key, first, second, diagnostic):
    delta = float(np.mean(second) - np.mean(first))
    margin = _margin(key, float((np.mean(first) + np.mean(second)) / 2))
    return {
        "passed": bool(abs(delta) <= margin),
        "difference": delta,
        "margin": margin,
        "reason": "typical molecular shape within practical margin"
        if abs(delta) <= margin
        else "typical molecular shape changes beyond practical margin",
        "interval_diagnostic": diagnostic,
    }


def compare(key, first, second, first_size, second_size):
    a, b = _blocks(first, first_size), _blocks(second, second_size)
    if a is None or b is None:
        return {"passed": False, "reason": "insufficient correlation-length blocks"}
    delta = float(b.mean() - a.mean())
    variance_a, variance_b = float(a.var(ddof=1) / len(a)), float(b.var(ddof=1) / len(b))
    variance = variance_a + variance_b
    denominator = variance_a**2 / (len(a) - 1) + variance_b**2 / (len(b) - 1)
    df = variance**2 / denominator if denominator > 0 else math.inf
    half = float(student_t.ppf(0.975, df) * math.sqrt(variance)) if variance > 0 else 0.0
    margin = _margin(key, float((a.mean() + b.mean()) / 2))
    upper = abs(delta) + half
    return {
        "passed": bool(upper <= margin),
        "difference": delta,
        "ci95_half_width": half,
        "margin": margin,
        "upper_absolute_difference": upper,
        "blocks": [len(a), len(b)],
        "reason": "within practical margin"
        if upper <= margin
        else "difference interval exceeds practical margin",
    }


def assess_local(evidence):
    times, series = _read(evidence)
    result = {
        "policy": construction_policy(),
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "passed": False,
        "failures": [],
        "warnings": [],
        "statistics": {},
        "comparisons": {},
        "analysis_window_ps": [float(times[0]), float(times[-1])],
    }
    for key, values in sorted(series.items()):
        stats = _statistics(values)
        result["statistics"][key] = stats
        hard = _hard_interval(key)
        if stats["effective_samples"] < construction_policy()["minimum_effective_samples"]:
            result["failures" if hard else "warnings"].append(
                f"{key}: fewer than ten effective time samples"
            )
        size = max(1, int(np.ceil(stats["correlation_frames"])))
        middle, quarter = len(values) // 2, 3 * len(values) // 4
        checks = {
            "early_late": compare(key, values[:middle], values[middle:], size, size),
            "terminal": compare(key, values[middle:quarter], values[quarter:], size, size),
        }
        if _typical_shape(key):
            checks = {
                "early_late": _observed_shape(
                    key, values[:middle], values[middle:], checks["early_late"]
                ),
                "terminal": _observed_shape(
                    key, values[middle:quarter], values[quarter:], checks["terminal"]
                ),
            }
        result["comparisons"][key] = checks
        for label, check in checks.items():
            if not check["passed"]:
                result["failures" if hard or _typical_shape(key) else "warnings"].append(
                    f"{key}/{label}: {check['reason']}"
                )
    primary = [v for key, v in result["statistics"].items() if _hard_interval(key)]
    result["sample_spacing_ps"] = max(
        construction_policy()["minimum_extraction_spacing_ps"],
        max(v["correlation_frames"] for v in primary) * 100,
    )
    result["minimum_effective_samples"] = min(v["effective_samples"] for v in primary)
    result["passed"] = not result["failures"]
    return result


def compare_local_replicas(evidences):
    if len(evidences) < 2:
        raise ValueError("At least two independent local-conformation samples required")
    if any(item["definition"] != evidences[0]["definition"] for item in evidences[1:]):
        raise ValueError("Replica local-conformation definitions differ")
    series = [_read(item)[1] for item in evidences]
    checks, failures, warnings = {}, [], []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            pair = checks[f"{i + 1}-{j + 1}"] = {}
            for key in sorted(series[i]):
                a, b = series[i][key], series[j][key]
                sizes = [max(1, int(np.ceil(_statistics(v)["correlation_frames"]))) for v in (a, b)]
                check = compare(key, a, b, *sizes)
                if _typical_shape(key):
                    check = _observed_shape(key, a, b, check)
                pair[key] = check
                if not check["passed"]:
                    (failures if _hard_interval(key) or _typical_shape(key) else warnings).append(
                        f"{key}: replicas {i + 1}/{j + 1} {check['reason']}"
                    )
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "comparisons": checks,
    }

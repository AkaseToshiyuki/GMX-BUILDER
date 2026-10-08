#!/usr/bin/env python3
"""Independent V4 acceptance controls; no dynamics or library writes.

AR(1) and a sum of fast/slow AR(1) processes have a known stationary generating
law. Eight observables include cross-correlations. Fixed-look false alarms,
indeterminate evidence, sample deficits and deliberately shifted controls are
reported separately. Multiple looks quantify, but do not claim to control,
repeated-testing errors. Seeds used for development must not be reused as an
independent validation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gmxbuilder.modules.membrane.v4_sampling import assess_series  # noqa: E402
from gmxbuilder.modules.membrane.v4_stationarity import METHOD  # noqa: E402

OBSERVABLES = (
    ("area_per_lipid_nm2", 0.65, 0.003),
    ("box_z_nm", 8.0, 0.01),
    ("volume_nm3", 400.0, 1.0),
    ("head_to_head_nm", 3.8, 0.01),
    ("POPC_upper_axis_P2", 0.7, 0.02),
    ("POPC_lower_axis_P2", 0.7, 0.02),
    ("CER18_upper_axis_P2", 0.6, 0.02),
    ("CER18_lower_axis_P2", 0.6, 0.02),
)
MODELS = {"white": 0.0, "ar08": 0.8, "ar095": 0.95, "multiscale": 0.8}


def stationary_control(seed: int, rho: float, frames: int, *, multiscale=False):
    rng = np.random.default_rng(seed)
    # The slowest mode (.99) loses exp(-40) of its initial value in this burn-in.
    innovations = rng.normal(size=(frames + 4000, len(OBSERVABLES)))
    values = lfilter([np.sqrt(1 - rho * rho)], [1, -rho], innovations, axis=0)
    if multiscale:
        slow = lfilter(
            [np.sqrt(1 - 0.99**2)], [1, -0.99], rng.normal(size=innovations.shape), axis=0
        )
        values = (values + slow) / np.sqrt(2)
    values = values[-frames:]
    values[:, 1] = 0.8 * values[:, 0] + 0.6 * values[:, 1]
    values[:, 2] = -0.8 * values[:, 0] + 0.6 * values[:, 2]
    return {
        name: mean + deviation * values[:, i]
        for i, (name, mean, deviation) in enumerate(OBSERVABLES)
    }


def assess_controls(model, frames, seeds, scenario, looks):
    details = []
    for seed in seeds:
        series = stationary_control(seed, MODELS[model], frames, multiscale=model == "multiscale")
        volume = series["volume_nm3"]
        if scenario == "level_shift":
            volume[3 * frames // 4 :] -= 50
        elif scenario == "late_collapse":
            volume[int(0.94 * frames) :] -= 50
        elif scenario == "late_ramp":
            start = 3 * frames // 4
            volume[start:] -= np.linspace(0, 50, frames - start)
        elif scenario == "boundary_shift":
            volume[3 * frames // 4 :] += 8.08
        elif scenario == "benign_shift":
            volume[3 * frames // 4 :] += 4
        elif scenario == "frozen":
            volume[3 * frames // 4 :] = 400
        snapshots = []
        for end in looks:
            analysis = assess_series(
                np.arange(end) * 100.0, {key: value[:end] for key, value in series.items()}
            )
            test = analysis["stationarity"]
            snapshots.append(
                {
                    "frames": end,
                    "status": test["status"],
                    "computable": test["uncertainty_estimated"],
                    "passed": analysis["passed"],
                    "minimum_effective_samples": analysis["minimum_effective_samples"],
                }
            )
        details.append({"seed": seed, "looks": snapshots})
    last = [record["looks"][-1] for record in details]
    tested = sum(row["computable"] for row in last)
    rejected = sum(row["status"] == "drift_detected" for row in last)
    ci = binomtest(rejected, tested).proportion_ci() if tested else None
    return {
        "total": len(details),
        "tested": tested,
        "drift_detected": rejected,
        "uncomputable": len(details) - tested,
        "insufficient_evidence": sum(row["status"] == "insufficient_evidence" for row in last),
        "equivalent": sum(row["status"] == "equivalent" for row in last),
        "insufficient_effective_samples": sum(
            row["minimum_effective_samples"] < 10 for row in last
        ),
        "sampling_passed": sum(row["passed"] for row in last),
        "rejection_ci95": [float(ci.low), float(ci.high)] if ci else None,
        "ever_drift_detected": sum(
            any(row["status"] == "drift_detected" for row in record["looks"]) for record in details
        ),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--first-seed", type=int, default=900000)
    parser.add_argument("--controls", type=int, default=200)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "stationary",
            "level_shift",
            "late_collapse",
            "late_ramp",
            "frozen",
            "boundary_shift",
            "benign_shift",
        ],
        choices=[
            "stationary",
            "level_shift",
            "late_collapse",
            "late_ramp",
            "frozen",
            "boundary_shift",
            "benign_shift",
        ],
    )
    parser.add_argument("--looks", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    looks = args.looks or [args.frames]
    if (
        args.frames < 10
        or args.controls < 1
        or looks != sorted(set(looks))
        or looks[0] < 10
        or looks[-1] != args.frames
    ):
        parser.error(
            "require frames >= 10, positive controls, and increasing looks ending at frames"
        )
    results = {}
    for scenario in args.scenarios:
        for model in args.models:
            result = assess_controls(
                model,
                args.frames,
                range(args.first_seed, args.first_seed + args.controls),
                scenario,
                looks,
            )
            results[f"{scenario}/{model}"] = result
            print(scenario, model, {k: v for k, v in result.items() if k != "details"}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "method": METHOD,
                "configuration": {**vars(args), "output": str(args.output)},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

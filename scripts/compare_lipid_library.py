#!/usr/bin/env python3
"""Decide whether the V4 area protocol is better than V3, from evidence.

V3 published ``box_x * box_y / 64`` of the final frame of a 1 ns NPT run. V4
block-averages the converged part of a long run and states an uncertainty and
a convergence verdict. "Better" is not a matter of preference between those,
and it is not settled by V4 being more elaborate -- so this script asks the
two questions that can actually be answered with data:

1. **Is the V3 number reproducible?** Run the same system twice with different
   velocity seeds. Physics says both must give the same area. If the two V3
   values differ by far more than the two V4 values do, V3 was reporting the
   seed, not the bilayer.

2. **How long does the run have to be?** Truncate one long trajectory at
   increasing lengths and ask V4's criteria about each. The shortest length
   that passes is the answer to "how long should the library run for", which
   V3 fixed at 1 ns by assumption and never checked.

Both questions are answered from the *same* trajectories, and V3's answer is
extracted from them too -- the box at 1 ns of the very run V4 measures. Same
construction, same seed, same integrator, so nothing differs but the analysis.

    python scripts/compare_lipid_library.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is an output directory of the protocol runner, holding
``work/npt.edr``. Directories whose names differ only in a trailing replica
tag are paired automatically. Exit status is 0 when V4 is supported by the
evidence, 1 when it is not.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Truncations to test, in nanoseconds. 1 ns is V3's fixed length and is here
#: so its verdict appears in the same table as the rest.
TRUNCATIONS_NS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0)

LIPIDS_PER_LEAFLET = 64


def _series(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from gmxbuilder.modules.membrane.area_observable import box_series_from_edr
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    edr = next(
        (p for p in (run / "work" / "npt.edr", run / "work" / "npt_cpu.edr") if p.is_file()),
        None,
    )
    if edr is None:
        raise SystemExit(f"{run} has no retained NPT energy file")
    gmx = find_gromacs_executable()
    if not gmx:
        raise SystemExit("no GROMACS executable found")
    return box_series_from_edr(gmx, edr, run / "work")


def _truncation_table(times, box_x, box_y, lipids_per_leaflet=LIPIDS_PER_LEAFLET) -> list[dict]:
    from gmxbuilder.modules.membrane.area_observable import measure_area_per_lipid

    rows = []
    simulated_ns = float(times[-1] / 1000.0)
    for length in TRUNCATIONS_NS:
        # A truncation longer than the run is the whole run again. Listing it
        # would repeat one row under several headings and read as though the
        # answer had stopped changing with length, which is the opposite of
        # what the table is for.
        if length > simulated_ns:
            break
        keep = times <= length * 1000.0
        if keep.sum() < 4:
            continue
        measurement = measure_area_per_lipid(
            times[keep], box_x[keep], box_y[keep], lipids_per_leaflet
        )
        rows.append(
            {
                "ns": length,
                "mean": measurement.mean_nm2,
                "error": measurement.standard_error_nm2,
                "converged": measurement.converged,
                "rejection": measurement.rejection,
                "drift": measurement.relative_half_drift,
                "n_eff": measurement.effective_samples,
            }
        )
    return rows


def _v3_value(times, box_x, box_y, lipids_per_leaflet=LIPIDS_PER_LEAFLET) -> float | None:
    """What V3 would have published: the box at the end of its 1 ns run."""
    index = int(np.searchsorted(times, 1000.0))
    if index >= len(times):
        return None
    return float(box_x[index] * box_y[index] / lipids_per_leaflet)


def _pair_key(name: str) -> str:
    return re.sub(r"_[a-z]$", "", name)


def _leaflet_count(run: Path) -> int:
    import json

    for path in (run / "v4-protocol.json", run / "work/v4-protocol.json", run / "metadata.json"):
        if path.is_file():
            record = json.loads(path.read_text())
            protocol = record.get("v4_protocol", record)
            count = protocol.get("lipids_per_leaflet")
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                return count
    raise ValueError(f"{run}: no explicit lipids_per_leaflet; cannot infer an area denominator")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    arguments = parser.parse_args()

    print("Area diagnostics only; pair threshold is 3 sigma, not the production admission gate.")
    results: dict[str, dict] = {}
    for run in arguments.runs:
        count = _leaflet_count(run)
        times, box_x, box_y = _series(run)
        results[run.name] = {
            "path": run,
            "simulated_ns": float(times[-1] / 1000.0),
            "lipids_per_leaflet": count,
            "v3": _v3_value(times, box_x, box_y, count),
            "truncations": _truncation_table(times, box_x, box_y, count),
        }

    # ---- question 2, per run: how long does the protocol need to be? ----
    print("=" * 78)
    print("How long must the NPT run be? (V3 fixed this at 1 ns without checking)")
    print("=" * 78)
    shortest_passing: list[float] = []
    for name, entry in sorted(results.items()):
        print(f"\n{name}  ({entry['simulated_ns']:.0f} ns simulated)")
        print(f"  {'length':>8s} {'area/lipid':>18s} {'drift':>8s} {'n_eff':>7s}  verdict")
        first_pass = None
        for row in entry["truncations"]:
            drift = "  n/a" if not np.isfinite(row["drift"]) else f"{row['drift'] * 100:5.2f}%"
            verdict = "converged" if row["converged"] else (row["rejection"] or "")[:40]
            if row["converged"] and first_pass is None:
                first_pass = row["ns"]
            print(
                f"  {row['ns']:6.0f}ns {row['mean']:11.4f} +- {row['error']:.4f}"
                f" {drift:>8s} {row['n_eff']:7.1f}  {verdict}"
            )
        entry["first_pass_ns"] = first_pass
        if first_pass is not None:
            shortest_passing.append(first_pass)
        print(f"  -> shortest converged run: {first_pass if first_pass else 'none of these'}")

    # ---- question 1, per pair: is the number reproducible? ----
    pairs: dict[str, list[str]] = defaultdict(list)
    for name in results:
        pairs[_pair_key(name)].append(name)

    print("\n" + "=" * 78)
    print("Is the published number reproducible across velocity seeds?")
    print("=" * 78)
    verdicts: list[bool] = []
    for key, names in sorted(pairs.items()):
        if len(names) < 2:
            print(f"\n{key}: only one replica, reproducibility not testable")
            continue
        first, second = (results[n] for n in sorted(names)[:2])

        v3_spread = abs(first["v3"] - second["v3"]) / np.mean([first["v3"], second["v3"]])

        def converged_value(entry):
            rows = [r for r in entry["truncations"] if r["converged"]]
            return rows[-1] if rows else None

        a, b = converged_value(first), converged_value(second)
        print(f"\n{key}")
        print(
            f"  V3 (1 ns final frame):  {first['v3']:.4f}  vs  {second['v3']:.4f} nm^2"
            f"   -> {v3_spread * 100:.2f}% apart"
        )
        if a is None or b is None:
            print("  V4: at least one replica never converged; cannot compare")
            verdicts.append(False)
            continue
        v4_spread = abs(a["mean"] - b["mean"]) / np.mean([a["mean"], b["mean"]])
        combined = float(np.hypot(a["error"], b["error"]))
        sigma = abs(a["mean"] - b["mean"]) / combined if combined > 0 else np.inf
        print(
            f"  V4 (block average):     {a['mean']:.4f} +- {a['error']:.4f}  vs  "
            f"{b['mean']:.4f} +- {b['error']:.4f} nm^2"
        )
        print(f"                          -> {v4_spread * 100:.2f}% apart, {sigma:.1f} sigma")

        better = v4_spread < v3_spread
        agrees = sigma <= 3.0
        print(
            f"  reproducibility: V4 {'tighter' if better else 'NOT tighter'} than V3; "
            f"replicas {'agree' if agrees else 'DISAGREE'} within stated error"
        )
        verdicts.append(bool(better and agrees))

    print("\n" + "=" * 78)
    if shortest_passing:
        print(
            f"R2 answer: the NPT stage needs at least {max(shortest_passing):.0f} ns "
            f"for these systems (V3 used 1 ns)."
        )
    else:
        print("R2 answer: none of the truncations converged; run longer before concluding.")

    if not verdicts:
        print("VERDICT: no replica pairs supplied; V4 is not yet demonstrated better than V3.")
        return 1
    if all(verdicts):
        print("VERDICT: V4 is supported -- it reproduces across seeds where V3 does not,")
        print("         and it states the uncertainty that makes that checkable.")
        return 0
    print("VERDICT: the evidence does not support retiring V3. Do not delete it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

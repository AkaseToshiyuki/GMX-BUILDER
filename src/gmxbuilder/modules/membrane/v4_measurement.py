"""Explicit coverage for unavailable early membrane measurements; no imputation."""

from __future__ import annotations

import numpy as np

METHOD = "early-membrane-measurement-1"
INCOMPLETE = "Early membrane thickness measurements unavailable; bulk certification incomplete"


def assess_measured_series(times, series, errors):
    """Describe a contiguous measured suffix without changing the local window."""
    from gmxbuilder.modules.membrane.v4_sampling import assess_series

    if not errors:
        return assess_series(times, series)
    times = np.asarray(times, dtype=float)
    missing = [item["frame_index"] for item in errors]
    start = missing[-1] + 1
    if start > len(times) // 2:
        raise ValueError("Unmeasurable membrane thickness reaches the fixed local analysis window")
    for key, values in series.items():
        finite = np.isfinite(values)
        expected = np.ones(len(times), dtype=bool)
        if key == "head_to_head_nm":
            expected[missing] = False
        if not np.array_equal(finite, expected):
            raise ValueError("Non-finite trajectory values outside recorded thickness failures")
    result = assess_series(times[start:], {key: values[start:] for key, values in series.items()})
    result["retained_fraction"] = float(np.mean(times >= result["analysis_window_ps"][0]))
    result["measurement_coverage"] = {
        "method": METHOD,
        "source_time_ps": times.tolist(),
        "unmeasurable_frames": errors,
        "diagnostic_input_window_ps": [float(times[start]), float(times[-1])],
        "complete": False,
        "imputed_frames": 0,
    }
    validate_measurement_coverage(result)
    result["passed"] = False
    result["failures"].append(INCOMPLETE)
    return result


def validate_measurement_coverage(analysis):
    """Check recorded missing measurements rather than trust a coverage flag."""
    coverage = analysis.get("measurement_coverage")
    if coverage is None:
        return
    if (
        coverage.get("method") != METHOD
        or coverage.get("complete") is not False
        or coverage.get("imputed_frames") != 0
    ):
        raise ValueError("Invalid membrane measurement coverage policy")
    times = np.asarray(coverage["source_time_ps"], dtype=float)
    if (
        times.ndim != 1
        or len(times) < 20
        or not np.isfinite(times).all()
        or abs(times[0]) > 0.01
        or not np.allclose(np.diff(times), 100, rtol=0, atol=0.01)
    ):
        raise ValueError("Measurement coverage must retain the complete NPT time axis")
    missing = []
    for item in coverage["unmeasurable_frames"]:
        index = item["frame_index"]
        separation, height = float(item["core_separation_nm"]), float(item["box_z_nm"])
        if (
            type(index) is not int
            or not 0 <= index < len(times) // 2
            or abs(float(item["time_ps"]) - times[index]) > 0.01
            or item["kind"] != "ambiguous_periodic_leaflet_cores"
            or item["maximum_core_separation_fraction"] != 0.40
            or not np.isfinite([separation, height]).all()
            or height <= 0
            or not 0.40 * height < abs(separation) <= 0.50 * height
            or not item.get("reason")
        ):
            raise ValueError("Invalid unmeasurable-frame evidence")
        missing.append(index)
    if not missing or sorted(set(missing)) != missing:
        raise ValueError("Missing or duplicated unmeasurable-frame indices")
    window = coverage["diagnostic_input_window_ps"]
    if not np.allclose(window, [times[missing[-1] + 1], times[-1]], rtol=0, atol=0.01):
        raise ValueError("Bulk diagnostic input window differs from measured suffix")
    if (
        not window[0] <= analysis["analysis_window_ps"][0] < window[1]
        or abs(analysis["analysis_window_ps"][1] - window[1]) > 0.01
    ):
        raise ValueError("Bulk diagnostic window is outside measured coverage")
    local = analysis.get("local_conformations")
    if local is not None and (
        local["total_frames"] != len(times)
        or not np.allclose(local["time_ps"], times[len(times) // 2 :], rtol=0, atol=0.01)
    ):
        raise ValueError("Missing bulk measurements changed the fixed local analysis window")

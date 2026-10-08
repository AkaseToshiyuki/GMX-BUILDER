"""Validate V4 coordinate provenance independently of queue scheduling."""

from __future__ import annotations

import math


def assembled_evidence_valid(metadata: dict) -> bool:
    """Production readers must not mistake a per-replica candidate for a library."""
    from gmxbuilder.modules.membrane.v4_construction import assembled_replicas, qualify_construction
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    try:
        protocol = resolve_protocol(metadata["lipid_name"], metadata["parameter_family"])
        return qualify_construction(assembled_replicas(metadata), protocol)["accepted"]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def replica_evidence_failures(metadata: dict, protocol: dict) -> list[str]:
    """Return sampling deficits; raise for missing/inconsistent provenance.

    A valid but undersampled trajectory can be extended. A mismatched identity,
    composition or source record cannot be repaired by simply running longer.
    """
    failures = []
    analysis = metadata["trajectory_analysis"]
    from gmxbuilder.modules.membrane.v4_measurement import validate_measurement_coverage

    validate_measurement_coverage(analysis)
    from gmxbuilder.modules.membrane.v4_atom_selection import validate_selections

    validate_selections(analysis, protocol)
    expected_counts = {
        name: int(protocol["lipids_per_leaflet"] * ratio / 100)
        for name, ratio in protocol["composition"].items()
    }
    if analysis.get("actual_leaflet_counts") != {
        "upper": expected_counts,
        "lower": expected_counts,
    }:
        raise ValueError("Trajectory leaflet counts differ from the protocol")
    required_order = {
        f"{name}_{leaflet}_axis_P2" for name in expected_counts for leaflet in ("upper", "lower")
    }
    if not required_order <= analysis["observables"].keys():
        raise ValueError("Missing species/leaflet molecular-axis order diagnostics")
    provenance = metadata["conformer_provenance"]
    if len(provenance) != metadata["n_conformations"] or len(
        {item["file"] for item in provenance}
    ) != len(provenance):
        raise ValueError("Conformer provenance is incomplete or duplicated")
    if metadata.get("conformer_validation", {}).get("identity_passed") is not True:
        raise ValueError("Missing conformer identity validation")
    if metadata.get("conformer_validation", {}).get("intrinsic_geometry_passed") is not True:
        raise ValueError("Missing conformer intrinsic-geometry validation")
    accepted_times = sorted({float(item["time_ps"]) for item in provenance})
    from gmxbuilder.modules.membrane.local_relaxation import assess_local

    local = assess_local(analysis["local_conformations"])
    start, end = local["analysis_window_ps"]
    if not accepted_times or accepted_times[0] < start or accepted_times[-1] > end:
        raise ValueError("Conformer sources lie outside the analysis window")
    if len(accepted_times) < 10:
        failures.append("fewer than ten retained sampling times")
    spacing = max(100.0, local["sample_spacing_ps"])
    if any(b - a + 0.01 < spacing for a, b in zip(accepted_times, accepted_times[1:])):
        raise ValueError("Conformer times do not respect autocorrelation spacing")
    end_time = float(analysis["trajectory_end_ps"])
    if not math.isfinite(end_time) or abs(end_time - metadata["npt_ps"]) > 0.01:
        raise ValueError("Trajectory endpoint differs from recorded NPT duration")
    if any(not math.isfinite(t) for t in accepted_times):
        raise ValueError("Non-finite conformer source time")
    return failures


def validation_scope(metadata: dict) -> dict:
    """Describe evidence on an already admitted entry without promoting a flag."""
    from gmxbuilder.modules.membrane.v4_construction import assembled_replicas
    from gmxbuilder.modules.membrane.v4_platform import assess_platform
    from gmxbuilder.modules.membrane.v4_sampling import (
        quantitative_area,
        replica_agreement,
        sampling_failures,
    )

    scope = {
        "kind": "initialization",
        "label": "Initialization conformers",
        "description": (
            "Validated for initial construction; bulk membrane equilibrium is not certified. "
            "Equilibrate the constructed system before production MD."
        ),
    }
    try:
        analyses = [m["trajectory_analysis"] for m in assembled_replicas(metadata)]
        if (
            len(analyses) >= 2
            and all(not sampling_failures(a) for a in analyses)
            and all(not assess_platform(a)["failures"] for a in analyses)
            and not replica_agreement(analyses)["failures"]
            and quantitative_area(analyses)["converged"]
        ):
            return {
                "kind": "pre_equilibrated",
                "label": "Pre-equilibrated conformers",
                "description": (
                    "Replica sampling, bulk stationarity and area convergence criteria passed "
                    "under the recorded library conditions. Equilibrate each newly constructed "
                    "system before production MD."
                ),
            }
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        pass
    return scope

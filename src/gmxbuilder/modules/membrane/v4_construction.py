"""One construction-use decision shared by scheduler, assembler and reader."""

from __future__ import annotations

import math

from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD
from gmxbuilder.modules.membrane.local_relaxation import (
    MAXIMUM_NS,
    assess_local,
    compare_local_replicas,
    construction_policy,
)
from gmxbuilder.modules.membrane.v4_platform import assess_platform

# The assembler stores only replica-specific fields here. Molecular identity is
# shared by the assembled entry; analyses and source records already have their
# own indexed storage, avoiding another full copy of large candidate metadata.
REPLICA_FIELDS = (
    "status",
    "test_mode",
    "quality",
    "npt_ps",
    "n_conformations",
    "conformer_validation",
)
IDENTITY_FIELDS = (
    "lipid_name",
    "parameter_family",
    "canonical_smiles",
    "force_field",
    "lipid_ff",
    "topology_sha256",
    "parameter_fingerprint",
    "atom_names",
    "temperature_K",
    "equilibration_host",
    "v4_protocol",
)


def _validate_structure(metadata, protocol):
    """Keep the existing final-frame physical bounds, not just its pass flag."""
    from gmxbuilder.modules.membrane.lipid_ensemble import MIN_ORIENTED_FRACTION
    from gmxbuilder.modules.membrane.lipid_orientation import MAX_TAIL_CORE_GAP_NM

    quality = metadata["quality"]
    from gmxbuilder.modules.membrane.initial_water import initial_water_evidence_valid

    if not initial_water_evidence_valid(quality.get("initial_water_exclusion")):
        raise ValueError("Initial full-membrane water exclusion is missing or invalid; rebuild")
    orientation, core = quality["orientation"], quality["hydrophobic_core"]
    apl, gap, fraction = (
        float(quality["apl_ratio"]),
        float(core["tail_core_gap_nm"]),
        float(orientation["correct_fraction"]),
    )
    if not all(math.isfinite(v) for v in (apl, gap, fraction)) or not (
        # Broad final-frame construction bounds; this is not the 2% platform screen.
        0.75 <= apl <= 1.35
        and gap <= MAX_TAIL_CORE_GAP_NM
        and MIN_ORIENTED_FRACTION <= fraction <= 1.0
        and orientation["passed"] is True
        and core["passed"] is True
        and min(orientation["upper_lipids"], orientation["lower_lipids"]) >= 20
    ):
        raise ValueError("Final-frame membrane geometry is invalid")
    if protocol["environment"] == "pure":
        # Only a pure membrane is compared with the lipid's reference thickness.
        ratio = float(quality["dhh_ratio"])
        if not math.isfinite(ratio) or not 0.70 <= ratio <= 1.30:
            raise ValueError("Pure-bilayer final thickness is outside structural limits")


def _validate_statistics(analysis):
    """Separate malformed sampling evidence from valid but undersampled runs."""
    required = {"area_per_lipid_nm2", "head_to_head_nm", "volume_nm3", "box_z_nm"}
    fields = analysis["observables"]
    if not required <= fields.keys():
        raise ValueError("Missing required trajectory observables")
    for block in fields.values():
        for field in ("mean", "effective_samples", "autocorrelation_ps"):
            if not math.isfinite(float(block[field])):
                raise ValueError("Non-finite sampling evidence")
        if block["effective_samples"] <= 0 or block["autocorrelation_ps"] <= 0:
            raise ValueError("Invalid correlation or effective sample count")
        if (
            type(block["n_frames"]) is not int
            or block["n_frames"] < 1
            or type(block["n_blocks"]) is not int
            or block["n_blocks"] < 0
        ):
            raise ValueError("Invalid sampling frame/block count")
        if block["n_blocks"] >= 2 and (
            not math.isfinite(float(block["standard_error"])) or block["standard_error"] <= 0
        ):
            raise ValueError("Invalid sampling standard error")


def qualify_construction(metadatas, protocol):
    """Invalid evidence fails; valid sampling/platform deficits may be extended.

    Statistical certification stays in the legacy diagnostic/area path. It
    must neither be overwritten as passed nor impersonate this narrower scope.
    """
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature
    from gmxbuilder.modules.membrane.v4_comparison import comparison_policy
    from gmxbuilder.modules.membrane.v4_evidence import replica_evidence_failures
    from gmxbuilder.modules.membrane.v4_sampling import replica_agreement, sampling_failures

    result = {
        "policy": construction_policy(),
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "status": "failed",
        "accepted": False,
        "failures": [],
        "warnings": [],
        "platforms": [],
    }
    try:
        if len(metadatas) < 2 or protocol.get("construction_policy") != construction_policy():
            raise ValueError("Two replicas and the reviewed construction policy are required")
        for field in IDENTITY_FIELDS:
            if field not in metadatas[0] or any(
                m.get(field) != metadatas[0][field] for m in metadatas
            ):
                raise ValueError(f"Replica identity differs or is missing: {field}")
        first = metadatas[0]
        if (
            first["v4_protocol"] != protocol
            or first["lipid_name"] != protocol["lipid"]
            or first["parameter_family"] != protocol["family"]
            or first["canonical_smiles"] != protocol["smiles"]
            or first["temperature_K"] != protocol["temperature_K"]
        ):
            raise ValueError("Replica identity or conditions differ from reviewed protocol")
        if not first["atom_names"] or first["topology_sha256"] != topology_signature(
            first["atom_names"], first["force_field"], first["lipid_ff"]
        ):
            raise ValueError("Invalid coordinate atom-order signature")
        from gmxbuilder.modules.membrane.v4_atom_selection import selections_by_lipid

        selections = [selections_by_lipid(m["trajectory_analysis"], protocol) for m in metadatas]
        if any(value != selections[0] for value in selections[1:]):
            raise ValueError("Replica chemical observable definitions differ")
        reused_sources = None
        if any("reused_from" in m for m in metadatas):
            from gmxbuilder.modules.membrane.v4_reuse import validate_reused_metadata

            reused_sources = [validate_reused_metadata(m) for m in metadatas]
            source_decision = qualify_construction(reused_sources, reused_sources[0]["v4_protocol"])
            if not source_decision["accepted"]:
                raise ValueError("Reused source ensemble is not accepted")
        lengths, analyses = [], []
        for index, metadata in enumerate(metadatas, 1):
            from gmxbuilder.modules.membrane.parameter_provenance import parameters_current

            if not parameters_current(metadata):
                raise ValueError(f"replica {index}: parameter fingerprint is missing or stale")
            if (
                metadata["status"] != "ready"
                or metadata["test_mode"]
                or metadata["quality"].get("passed") is not True
            ):
                raise ValueError(f"replica {index}: no structurally valid production candidate")
            _validate_structure(metadata, protocol)
            ns = float(metadata["npt_ps"]) / 1000
            if not math.isfinite(ns) or not 0 < ns <= MAXIMUM_NS:
                raise ValueError("Replica duration is outside the 200 ns production budget")
            lengths.append(ns)
            analysis = metadata["trajectory_analysis"]
            # These functions rebuild their decisions from stored numeric
            # evidence. A ready flag is never a construction admission token.
            source_deficits = replica_evidence_failures(metadata, protocol)
            platform = assess_platform(analysis)
            result["platforms"].append(platform)
            local = assess_local(analysis["local_conformations"])
            definition = analysis["local_conformations"]["definition"]
            from gmxbuilder.modules.membrane.local_conformations import descriptor_definition

            selection = next(
                item for item in analysis["atom_selections"] if item["lipid"] == protocol["lipid"]
            )
            atom_names = selection["atom_names"]
            expected_definition = descriptor_definition(
                protocol["smiles"],
                tuple(atom_names),
                tuple(selection["elements"]),
                tuple(map(tuple, selection["bonds"])),
                tuple(atom_names.index(name) for name in selection["polar_atom_names"]),
                tuple(atom_names.index(name) for name in selection["tail_atom_names"]),
                atom_names.index(selection["anchor_atom_name"]),
            )
            if (
                definition != expected_definition
                or abs(local["analysis_window_ps"][-1] - ns * 1000) > 0.01
            ):
                raise ValueError(
                    "Local relaxation differs from the target chemistry or trajectory endpoint"
                )
            result.setdefault("local_relaxation", []).append(local)
            result["warnings"].extend(
                f"replica {index} conformational distribution: {reason}"
                for reason in local["warnings"]
            )
            deficits = source_deficits + local["failures"]
            result["failures"].extend(f"replica {index}: {reason}" for reason in deficits)
            result["warnings"].extend(
                f"replica {index} certification: {reason}" for reason in sampling_failures(analysis)
            )
            result["warnings"].extend(
                f"replica {index} bulk platform: {reason}" for reason in platform["failures"]
            )
            analyses.append(analysis)
        comparison = comparison_policy(analyses)
        if comparison["failures"]:
            raise ValueError("; ".join(comparison["failures"]))
        agreement = replica_agreement(analyses)
        local_agreement = compare_local_replicas([a["local_conformations"] for a in analyses])
        result["local_replica_agreement"] = local_agreement
        result["failures"].extend(local_agreement["failures"])
        result["warnings"].extend(local_agreement["warnings"])
        result["warnings"].extend(agreement["failures"])
        result["replica_agreement"] = agreement
        result["ns_per_replica"] = lengths
        result["accepted"] = not result["failures"]
        result["status"] = (
            "accepted"
            if result["accepted"]
            else "capped_unaccepted"
            if min(lengths) >= MAXIMUM_NS
            else "extendable"
        )
        # Old bulk-platform exceptions do not authorize missing or failed local
        # molecular-relaxation evidence under the new policy.
        result["automatic_status"] = result["status"]
        result["automatic_accepted"] = result["accepted"]
        from gmxbuilder.modules.membrane.v4_approval import reviewed_local_sampling_approval

        approval = reviewed_local_sampling_approval(metadatas, result)
        if approval is not None:
            result["manual_approval"] = approval
            result["accepted"] = True
            result["status"] = "manually_accepted"
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
        OverflowError,
        OSError,
    ) as exc:
        result["failures"].append(f"Invalid construction evidence: {exc}")
        result["status"], result["accepted"] = "failed", False
    return result


def assembled_replicas(metadata):
    """Reconstruct the same per-replica inputs from an assembled artifact."""
    records, analyses = metadata["replica_records"], metadata["replica_analyses"]
    if len(records) != len(analyses) or len(records) < 2:
        raise ValueError("Incomplete assembled replica records")
    names = [record["replica"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate replica identifiers")
    provenance = metadata["conformer_provenance"]
    if any(item["replica"] not in names for item in provenance):
        raise ValueError("Unknown conformer source replica")
    if len(provenance) != metadata["n_conformations"] or len(
        {item["file"] for item in provenance}
    ) != len(provenance):
        raise ValueError("Incomplete assembled conformer provenance")
    common = {key: metadata[key] for key in IDENTITY_FIELDS}
    return [
        {
            **common,
            **{key: record[key] for key in REPLICA_FIELDS},
            "trajectory_analysis": analysis,
            **({"reused_from": record["reused_from"]} if "reused_from" in record else {}),
            "conformer_provenance": [
                item for item in provenance if item["replica"] == record["replica"]
            ],
        }
        for record, analysis in zip(records, analyses, strict=True)
    ]

"""Maintainer-reviewed exceptions for exact, otherwise valid V4 ensembles.

Approval lives with reviewed application data, not in a candidate-supplied
boolean. It cannot travel to another trajectory, condition or failed check.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def evidence_digest(metadata):
    """Hash evidence shared by replica candidates and their assembled records.

    Assembly renumbers filenames and adds a replica label. Neither changes a
    conformer's physical source; all other provenance and analysis fields stay
    in the digest. Coordinate file hashes are recorded separately at publication.
    """
    from gmxbuilder.modules.membrane.v4_construction import IDENTITY_FIELDS, REPLICA_FIELDS

    evidence = {key: metadata[key] for key in (*IDENTITY_FIELDS, *REPLICA_FIELDS)}
    evidence["trajectory_analysis"] = metadata["trajectory_analysis"]
    evidence["conformer_provenance"] = [
        {key: value for key, value in source.items() if key not in {"file", "replica"}}
        for source in metadata["conformer_provenance"]
    ]
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _approval_records():
    path = Path(__file__).parents[2] / "data" / "v4_manual_approvals.json"
    payload = json.loads(path.read_text())
    if payload["schema"] != "v4-maintainer-approvals-1":
        raise ValueError("Unknown V4 approval registry schema")
    return payload["approvals"]


def reviewed_approval(metadatas, result):
    """Only an exact, nonempty set of reviewed platform deficits is waivable."""
    if result["accepted"] or result["status"] == "failed":
        return None
    # This route cannot waive sample counts, replica disagreement or any other
    # gate even if someone mistakenly includes it in an approval record.
    pattern = (
        r"replica \d+: (area_per_lipid_nm2|head_to_head_nm)/"
        r"(quarter_mean_range|terminal_half_shift|linear_change|terminal_block_shift): "
    )
    if not result["failures"] or not all(re.match(pattern, f) for f in result["failures"]):
        return None
    approvals = _approval_records()
    digests = [evidence_digest(metadata) for metadata in metadatas]
    for approval in approvals:
        if (
            approval["replica_evidence_sha256"] == digests
            and approval["waived_failures"] == result["failures"]
            and approval["waived_failures"]
            and approval["lipid"] == metadatas[0]["lipid_name"]
            and approval["family"] == metadatas[0]["parameter_family"]
        ):
            return approval
    return None


def reviewed_local_sampling_approval(metadatas, result):
    """Match an explicit, evidence-bound hydration sampling exception.

    Historical bulk-platform approvals do not activate this route. Invalid
    structure/identity, other local deficits and quantitative area are never
    waived. The caller keeps the automatic verdict and failures unchanged.
    """
    if result.get("accepted") or result.get("status") != "capped_unaccepted":
        return None
    failures = result.get("failures", [])
    pattern = (
        r"replica [12]: hydration:contacts:(upper|lower): fewer than ten effective time samples"
    )
    if not failures or not all(re.fullmatch(pattern, failure) for failure in failures):
        return None
    digests = [evidence_digest(metadata) for metadata in metadatas]
    for approval in _approval_records():
        if (
            approval.get("scope") == "local-hydration-effective-samples-1"
            and approval.get("replica_evidence_sha256") == digests
            and approval.get("waived_failures") == failures
            and approval.get("lipid") == metadatas[0]["lipid_name"]
            and approval.get("family") == metadatas[0]["parameter_family"]
            and approval.get("approved_by")
            and approval.get("authorization")
            and approval.get("equilibrium_certified") is False
            and approval.get("quantitative_area_override") is False
        ):
            return approval
    return None

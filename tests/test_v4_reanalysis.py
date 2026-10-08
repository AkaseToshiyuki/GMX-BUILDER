"""Analysis upgrades preserve MD identity and retain the previous evidence."""

import copy
import hashlib
import json

import pytest

from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from gmxbuilder.modules.membrane.v4_reanalysis import (
    LEGACY_METHOD,
    analysis_upgrade_allowed,
    archive_analysis,
)


def signed(protocol):
    protocol.pop("sha256", None)
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return protocol


def pair():
    current = resolve_protocol("CER18", "charmm36m-lipid")
    old = copy.deepcopy(current)
    old["stationarity_method"] = LEGACY_METHOD
    old.pop("stationarity_policy")
    return signed(old), current


def test_only_an_explicit_known_analysis_change_is_reusable():
    old, current = pair()
    assert analysis_upgrade_allowed(old, current)
    assert analysis_upgrade_allowed(current, current)
    old["stationarity_method"] = "unreviewed"
    assert not analysis_upgrade_allowed(signed(old), current)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_K", 310),
        ("lipids_per_leaflet", 100),
        ("smiles", "CCO"),
        ("composition", {"CER18": 100}),
    ],
)
def test_analysis_upgrade_cannot_change_md_conditions(field, value):
    old, current = pair()
    old[field] = value
    assert not analysis_upgrade_allowed(signed(old), current)


def test_bad_hash_or_policy_is_not_an_analysis_upgrade():
    old, current = pair()
    old["sha256"] = "bad"
    assert not analysis_upgrade_allowed(old, current)
    old, current = pair()
    current["stationarity_policy"]["relative_margin"] = 0.2
    assert not analysis_upgrade_allowed(old, signed(current))


def test_bulk_to_local_policy_is_an_exact_analysis_only_upgrade():
    from gmxbuilder.modules.membrane.v4_platform import construction_policy as old_policy

    current = resolve_protocol("POPC", "amber-lipid21")
    previous = copy.deepcopy(current)
    previous["construction_policy"] = old_policy()
    assert analysis_upgrade_allowed(signed(previous), current)
    previous["temperature_K"] += 1
    assert not analysis_upgrade_allowed(signed(previous), current)


def test_previous_assessment_is_archived_without_relabelling(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    old, _ = pair()
    payload = json.dumps({"v4_protocol": old, "npt_ps": 100000}).encode()
    (output / "metadata.json").write_bytes(payload)
    (work / "whole.gro").write_bytes(b"previous geometry")
    (work / "trajectory_geometry.npz").write_bytes(b"per-frame geometry evidence")
    (work / "trajectory_unmeasurable_frames.json").write_text('{"unmeasurable_frames": []}')
    record = archive_analysis(work, output)[0]
    from pathlib import Path

    archive = Path(record["archive"])
    assert (archive / "metadata.json").read_bytes() == payload
    assert (archive / "whole.gro").read_bytes() == b"previous geometry"
    assert (archive / "trajectory_geometry.npz").read_bytes() == b"per-frame geometry evidence"
    assert (archive / "trajectory_unmeasurable_frames.json").read_text() == (
        '{"unmeasurable_frames": []}'
    )
    (work / "whole.gro").write_bytes(b"new geometry")
    assert archive_analysis(work, output)[0] == record
    assert (archive / "whole.gro").read_bytes() == b"previous geometry"

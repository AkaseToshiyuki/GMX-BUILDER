"""Old bulk-platform approvals cannot bypass local molecular relaxation."""

import copy
import json

import pytest

from gmxbuilder.modules.membrane import v4_approval
from gmxbuilder.modules.membrane.v4_construction import qualify_construction
from tests.test_v4_construction import pair  # noqa: F401


@pytest.fixture
def denied(request, monkeypatch):
    records, protocol = request.getfixturevalue("pair")
    series = records[1]["trajectory_analysis"]["local_conformations"]["series"]
    for key, values in series.items():
        if key.startswith("shape:"):
            series[key] = [v * 1.15 for v in values]
    automatic = qualify_construction(records, protocol)
    assert automatic["status"] == "capped_unaccepted"
    record = {
        "id": "old-review",
        "lipid": "POPC",
        "family": "charmm36m-lipid",
        "replica_evidence_sha256": [v4_approval.evidence_digest(m) for m in records],
        "waived_failures": automatic["failures"],
    }
    monkeypatch.setattr(v4_approval, "_approval_records", lambda: [record])
    return records, protocol, automatic


def test_registry_cannot_waive_local_deficits(denied):
    records, protocol, automatic = denied
    result = qualify_construction(records, protocol)
    assert not result["accepted"] and not result["automatic_accepted"]
    assert result["failures"] == automatic["failures"]
    assert "manual_approval" not in result


def test_candidate_supplied_approval_is_not_authority(denied):
    records, protocol, _ = denied
    records[0]["manual_approval"] = {"approved": True}
    assert not qualify_construction(records, protocol)["accepted"]


@pytest.mark.parametrize("change", ["geometry", "sampling", "trajectory", "identity"])
def test_evidence_digest_tracks_physical_and_diagnostic_mutations(request, change):
    records, _ = request.getfixturevalue("pair")
    before = v4_approval.evidence_digest(records[0])
    modified = copy.deepcopy(records[0])
    if change == "geometry":
        modified["quality"]["hydrophobic_core"]["tail_core_gap_nm"] = 0.9
    elif change == "sampling":
        modified["conformer_provenance"][0]["time_ps"] += 1
    elif change == "trajectory":
        modified["trajectory_analysis"]["stationarity"]["calibration_status"] = "changed"
    else:
        modified["canonical_smiles"] = "CCO"
    assert v4_approval.evidence_digest(modified) != before


def test_failed_local_assembly_preserves_sources_and_does_not_publish(denied, tmp_path):
    import build_v4_library as queue
    import numpy as np
    from assemble_v4_library import assemble_entry

    records, protocol, _ = denied
    roots, original_bytes = [], []
    for replica, metadata in enumerate(records, 1):
        root = tmp_path / f"replica-{replica}"
        roots.append(root)
        directory = root / "charmm36m-lipid/POPC"
        directory.mkdir(parents=True)
        payload = json.dumps(metadata)
        original_bytes.append(payload)
        (directory / "metadata.json").write_text(payload)
        for item in metadata["conformer_provenance"]:
            np.savez(
                directory / item["file"],
                coords=np.zeros((len(metadata["atom_names"]), 3)),
                atom_names=metadata["atom_names"],
            )
    entry = {"lipid": "POPC", "family": "charmm36m-lipid", "v4_protocol": protocol}
    assert not queue.entry_assessment(tmp_path, entry, 2)["queue_complete"]
    result = assemble_entry(tmp_path, roots, entry["family"], entry["lipid"], tmp_path / "library")
    assert result.get("skipped")
    assert not (tmp_path / "library/charmm36m-lipid/POPC").exists()
    assert [
        (root / "charmm36m-lipid/POPC/metadata.json").read_text() for root in roots
    ] == original_bytes


@pytest.fixture
def approved_hydration(request, monkeypatch):
    import numpy as np

    records, protocol = request.getfixturevalue("pair")
    series = records[0]["trajectory_analysis"]["local_conformations"]["series"]
    series["hydration:contacts:upper"] = (
        2 + 0.001 * np.sin(np.linspace(0, 5.6 * np.pi, len(series["hydration:contacts:upper"])))
    ).tolist()
    from gmxbuilder.modules.membrane.local_relaxation import assess_local

    local = assess_local(records[0]["trajectory_analysis"]["local_conformations"])
    spacing = np.ceil(local["sample_spacing_ps"] / 100) * 100
    for index, item in enumerate(records[0]["conformer_provenance"]):
        item["time_ps"] = local["analysis_window_ps"][0] + (index // 2) * spacing
        item["molecule_index"] = index % 2
    automatic = qualify_construction(records, protocol)
    assert automatic["status"] == "capped_unaccepted", automatic["failures"]
    assert automatic["failures"] == [
        "replica 1: hydration:contacts:upper: fewer than ten effective time samples"
    ]
    approval = {
        "id": "exact-hydration-review",
        "scope": "local-hydration-effective-samples-1",
        "lipid": "POPC",
        "family": "charmm36m-lipid",
        "approved_by": "Maintainer",
        "authorization": "Explicit approval of this ensemble and sampling deficit",
        "replica_evidence_sha256": [v4_approval.evidence_digest(m) for m in records],
        "waived_failures": automatic["failures"],
        "equilibrium_certified": False,
        "quantitative_area_override": False,
    }
    monkeypatch.setattr(v4_approval, "_approval_records", lambda: [approval])
    return records, protocol, automatic, approval


def test_explicit_hydration_approval_preserves_automatic_failure(approved_hydration):
    records, protocol, automatic, approval = approved_hydration
    result = qualify_construction(records, protocol)
    assert result["accepted"] and result["status"] == "manually_accepted"
    assert result["automatic_status"] == "capped_unaccepted"
    assert result["automatic_accepted"] is False
    assert result["failures"] == automatic["failures"]
    assert result["manual_approval"] == approval


@pytest.mark.parametrize("change", ["evidence", "scope", "authority", "area", "extra_failure"])
def test_hydration_approval_is_narrow_and_evidence_bound(approved_hydration, change):
    records, protocol, automatic, approval = approved_hydration
    if change == "evidence":
        records[0]["trajectory_analysis"]["local_conformations"]["series"][
            "hydration:contacts:upper"
        ][0] += 0.00001
    elif change == "scope":
        approval.pop("scope")
    elif change == "authority":
        approval.pop("authorization")
    elif change == "area":
        approval["quantitative_area_override"] = True
    else:
        automatic["failures"] += ["replica 2: angle:test: insufficient sampling"]
        assert v4_approval.reviewed_local_sampling_approval(records, automatic) is None
        return
    assert not qualify_construction(records, protocol)["accepted"]


def test_hydration_approval_cannot_override_invalid_structure(approved_hydration):
    records, protocol, _, approval = approved_hydration
    records[0]["quality"]["hydrophobic_core"]["tail_core_gap_nm"] = 10
    approval["replica_evidence_sha256"] = [v4_approval.evidence_digest(m) for m in records]
    result = qualify_construction(records, protocol)
    assert result["status"] == "failed" and not result["accepted"]


def test_approved_hydration_assembly_and_reader(approved_hydration, tmp_path):
    import build_v4_library as queue
    import numpy as np
    from assemble_v4_library import assemble_entry

    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary

    records, protocol, automatic, _ = approved_hydration
    roots, originals = [], []
    for replica, metadata in enumerate(records, 1):
        root = tmp_path / f"replica-{replica}"
        roots.append(root)
        directory = root / "charmm36m-lipid/POPC"
        directory.mkdir(parents=True)
        payload = json.dumps(metadata)
        originals.append(payload)
        (directory / "metadata.json").write_text(payload)
        for source in metadata["conformer_provenance"]:
            np.savez(
                directory / source["file"],
                coords=np.zeros((len(metadata["atom_names"]), 3)),
                atom_names=metadata["atom_names"],
            )
    entry = {"family": "charmm36m-lipid", "lipid": "POPC", "v4_protocol": protocol}
    assert queue.entry_assessment(tmp_path, entry, 2)["queue_complete"]
    result = assemble_entry(tmp_path, roots, entry["family"], entry["lipid"], tmp_path / "library")
    assert result["conformers"] == 40
    published = json.loads((tmp_path / "library/charmm36m-lipid/POPC/metadata.json").read_text())
    verdict = published["construction_acceptance"]
    assert verdict["accepted"] and not verdict["automatic_accepted"]
    assert verdict["failures"] == automatic["failures"]
    assert EquilibratedLipidLibrary(roots=[tmp_path / "library"]).has(
        "POPC", "charmm36m", "charmm36m"
    )
    assert [
        (root / "charmm36m-lipid/POPC/metadata.json").read_text() for root in roots
    ] == originals

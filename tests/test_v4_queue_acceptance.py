"""Queue acceptance against recorded failures, without running dynamics."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.dry_initial_fixture import dry_initial_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_v4_library as queue  # noqa: E402
from v4_queue_state import queue_ownership, source_manifest, verify_source  # noqa: E402


def test_priority_lipids_cross_families_and_keep_reuse_source_first():
    entries = [
        {"family": family, "lipid": lipid}
        for family in ("amber-gaff2", "charmm36-lipid", "amber-lipid21", "charmm36m-lipid")
        for lipid in ("ZZZ", "POPE", "AAA", "POPC")
    ]
    ordered = queue.order_pending(entries, " popc, POPE,POPC ")
    assert [(e["lipid"], e["family"]) for e in ordered[:8]] == [
        (lipid, family)
        for lipid in ("POPC", "POPE")
        for family in ("amber-lipid21", "charmm36m-lipid", "charmm36-lipid", "amber-gaff2")
    ]
    assert [(e["lipid"], e["family"]) for e in ordered[8:]] == [
        (lipid, family)
        for lipid in ("AAA", "ZZZ")
        for family in ("amber-lipid21", "charmm36m-lipid", "charmm36-lipid", "amber-gaff2")
    ]
    assert len(ordered) == len(entries)


def test_default_queue_order_is_unchanged_and_priority_typos_fail():
    entries = [
        {"family": "charmm36-lipid", "lipid": "POPC"},
        {"family": "charmm36m-lipid", "lipid": "POPC"},
        {"family": "charmm36m-lipid", "lipid": "CHOL"},
    ]
    assert queue.order_pending(entries) == [entries[2], entries[1], entries[0]]
    with pytest.raises(ValueError, match="POCP"):
        queue.order_pending(entries, "POCP")


@pytest.fixture
def entry():
    return {
        "family": "charmm36m-lipid",
        "lipid": "BSM",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
    }


def write_pair(root, entry, *, samples=(40, 40), means=(0.5, 0.501), errors=(0.002, 0.002)):
    from gmxbuilder.modules.membrane.area_observable import AreaMeasurement
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature

    directories = []
    for replica in (1, 2):
        directory = root / f"replica-{replica}" / entry["family"] / entry["lipid"]
        directory.mkdir(parents=True)
        m = AreaMeasurement(
            mean_nm2=means[replica - 1],
            standard_error_nm2=errors[replica - 1],
            n_frames=90000,
            n_blocks=40,
            analysis_window_ps=(10000, 100000),
            equilibration_ps=10000,
            retained_fraction=0.9,
            relative_half_drift=0.001,
            drift_sigma=0.5,
            autocorrelation_ps=1000,
            effective_samples=samples[replica - 1],
            converged=True,
            rejection=None,
        )
        metadata = {
            "status": "ready",
            "quality": {"passed": True, "initial_water_exclusion": dry_initial_fixture()},
            "test_mode": False,
            "lipid_name": entry["lipid"],
            "parameter_family": entry["family"],
            "canonical_smiles": "CCO",
            "force_field": entry["force_field"],
            "lipid_ff": entry["lipid_ff"],
            "topology_sha256": topology_signature(
                ["C1", "O1"], entry["force_field"], entry["lipid_ff"]
            ),
            "temperature_K": 310,
            "atom_names": ["C1", "O1"],
            "npt_ps": 100000,
            "n_conformations": 1,
            "observables": m.as_metadata(),
        }
        from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

        metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        np.savez(directory / "conf_0000.npz", coords=np.zeros((2, 3)), atom_names=["C1", "O1"])
        directories.append(directory)
    return directories


def test_bsm_completes_the_ten_sample_queue_despite_stricter_area_diagnostic(tmp_path, entry):
    write_pair(tmp_path, entry, samples=(19.98152215699331, 19.38302180153519))
    result = queue.entry_assessment(tmp_path, entry, 2)
    assert result["built"]
    assert result["queue_complete"]
    assert not result["area_converged"]
    assert "sample size" in result["area_rejection"]


def test_camp_disagreement_is_reported_without_extending_the_queue(tmp_path, entry):
    write_pair(
        tmp_path,
        entry,
        samples=(30.58564, 27.13331),
        means=(0.43867920358873386, 0.4450159150015824),
        errors=(0.0007214286835023289, 0.0006952385435092129),
    )
    result = queue.entry_assessment(tmp_path, entry, 2)
    assert result["built"] and result["queue_complete"]
    assert not result["area_converged"]
    assert result["spread_sigma"] == pytest.approx(6.324655530717114)
    assert "disagree" in result["area_rejection"]


def test_matching_converged_pair_is_done(tmp_path, entry):
    write_pair(tmp_path, entry)
    assert queue.entry_done(tmp_path, entry, 2)


@pytest.mark.parametrize("samples,done", [(9.999, False), (10.0, True), (19.0, True)])
def test_ten_is_the_queue_threshold_not_a_sample_truncation(tmp_path, entry, samples, done):
    write_pair(tmp_path, entry, samples=(samples, samples))
    assert queue.entry_done(tmp_path, entry, 2) is done
    assert queue.effective_samples(tmp_path, entry, 2) == samples


def test_launch_failure_waits_for_an_already_started_replica():
    reaped = []

    class Child:
        def wait(self):
            reaped.append(True)
            return 0

    def fail():
        raise OSError("cannot launch second replica")

    with pytest.raises(OSError, match="cannot launch"):
        queue.wait_for_replicas([(1, Child), (2, fail)])
    assert reaped == [True]


@pytest.mark.parametrize(
    "change",
    [
        {"test_mode": True},
        {"quality": {"passed": False}},
        {"temperature_K": 300},
        {"canonical_smiles": "CCC"},
        {"topology_sha256": None},
        {"topology_sha256": "stale-signature"},
        {"atom_names": ["O1", "C1"]},
    ],
)
def test_bad_or_mismatched_replica_cannot_be_accepted(tmp_path, entry, change):
    directories = write_pair(tmp_path, entry)
    p = directories[1] / "metadata.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), **change}))
    assert not queue.entry_done(tmp_path, entry, 2)


def test_resume_keeps_valid_100ns_result_instead_of_rebuilding_50ns(tmp_path, entry):
    directories = write_pair(tmp_path, entry, samples=(19, 19))
    assert queue.replica_action(directories[0], tmp_path / "work", 50) == ("present", 100)
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        queue.replica_action(partial, tmp_path / "work", 50)
    assert partial.is_dir()


def test_missing_checkpoint_is_not_resumable(tmp_path):
    for name in ("npt.tpr", "npt.edr", "topol.top"):
        (tmp_path / name).write_text("retained")
    assert not queue.continuation_available(tmp_path)
    (tmp_path / "npt.cpt").write_text("checkpoint")
    assert queue.continuation_available(tmp_path)


def test_source_change_prevents_new_replica(tmp_path, monkeypatch):
    monkeypatch.setattr("v4_queue_state.subprocess.check_output", lambda *a, **k: "commit")
    source = tmp_path / "src"
    source.mkdir()
    path = source / "model.py"
    path.write_text("first")
    original = source_manifest(tmp_path)
    verify_source(tmp_path, original)
    path.write_text("second")
    with pytest.raises(RuntimeError, match="changed"):
        verify_source(tmp_path, original)


def test_second_queue_cannot_own_the_same_library(tmp_path, monkeypatch):
    monkeypatch.setattr("v4_queue_state.source_manifest", lambda root: {"commit": "test"})
    with queue_ownership(tmp_path, tmp_path):
        with pytest.raises(RuntimeError, match="Another V4 queue"):
            with queue_ownership(tmp_path, tmp_path):
                pytest.fail("duplicate queue acquired the lock")
    with queue_ownership(tmp_path, tmp_path):
        pass


@pytest.mark.parametrize("existing", ["replica", "work"])
def test_reuse_never_replaces_existing_replica_or_simulation(tmp_path, entry, existing):
    write_pair(tmp_path, entry)
    destination = {**entry, "family": "charmm36-lipid", "force_field": "charmm36"}
    location = (
        tmp_path / "replica-1" / destination["family"] / entry["lipid"]
        if existing == "replica"
        else tmp_path / "work" / destination["family"] / entry["lipid"]
    )
    location.mkdir(parents=True)
    sentinel = location / "retained.data"
    sentinel.write_bytes(b"irreplaceable previous simulation")
    assert not queue.publish_reuse(tmp_path, destination, entry, 2)
    assert sentinel.read_bytes() == b"irreplaceable previous simulation"


def test_reuse_is_accepted_by_the_library_reader_under_the_destination_namespace(
    tmp_path, entry, monkeypatch
):
    from gmxbuilder.modules.membrane.equilibrated_library import (
        ACCEPTED_METHOD,
        MIN_CONFORMERS,
        SCHEMA_VERSION,
        EquilibratedLipidLibrary,
        topology_signature,
    )

    # Synthetic custom identity isolates the namespace/reader contract;
    # force-field equivalence and conformer physics have separate real tests.
    entry = {**entry, "lipid": "TSTREAD"}
    origins = write_pair(tmp_path, entry)
    for origin in origins:
        path = origin / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata.update(
            schema_version=SCHEMA_VERSION,
            coordinate_handedness="preserved",
            leaflet_transform="proper_rotation",
            method=ACCEPTED_METHOD,
            force_field="charmm36m",
            lipid_ff="charmm36m",
            n_conformations=MIN_CONFORMERS,
        )
        metadata["quality"]["orientation"] = {"passed": True, "n_lipids_checked": MIN_CONFORMERS}
        metadata["topology_sha256"] = topology_signature(
            metadata["atom_names"], "charmm36m", "charmm36m"
        )
        path.write_text(json.dumps(metadata))
        for i in range(1, MIN_CONFORMERS):
            (origin / f"conf_{i:04d}.npz").write_bytes((origin / "conf_0000.npz").read_bytes())
    monkeypatch.setattr(queue, "_reuse_reason", lambda lipid: "equivalence verified separately")
    target = {
        **entry,
        "family": "charmm36-lipid",
        "force_field": "charmm36",
        "lipid_ff": "charmm36",
    }
    assert queue.publish_reuse(tmp_path, target, entry, 2)
    for replica in (1, 2):
        root = tmp_path / f"replica-{replica}"
        reader = EquilibratedLipidLibrary(roots=[root, root])
        source = reader.inspect(entry["lipid"], "charmm36m", "charmm36m")
        reused = reader.inspect(entry["lipid"], "charmm36", "charmm36")
        assert source is not None and reused is not None
        assert reused.metadata["topology_sha256"] != source.metadata["topology_sha256"]
        assert (
            reused.metadata["reused_from"]["source_topology_sha256"]
            == source.metadata["topology_sha256"]
        )


@pytest.fixture(autouse=True)
def _storage_parameter_sources(parameter_sources):
    return parameter_sources

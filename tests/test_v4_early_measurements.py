"""Independent controls for missing early measurements and completed-run recovery."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from mdtraj.formats import XTCTrajectoryFile

from gmxbuilder.io.gro import GROWriter
from gmxbuilder.modules.membrane import lipid_equilibration as equil
from gmxbuilder.modules.membrane.v4_measurement import (
    INCOMPLETE,
    assess_measured_series,
    validate_measurement_coverage,
)
from gmxbuilder.modules.membrane.v4_sampling import quantitative_area, sampling_failures
from gmxbuilder.modules.membrane.v4_trajectory import TrajectorySampler
from tests.test_construction_recovery import bare_builder
from tests.test_v4_protocol_sampling import noise_series


def missing_example(indices=(0, 1, 6)):
    series = noise_series(n=301)
    errors = []
    for index in indices:
        series["head_to_head_nm"][index] = np.nan
        errors.append(
            dict(
                frame_index=index,
                time_ps=index * 100.0,
                kind="ambiguous_periodic_leaflet_cores",
                core_separation_nm=3.15,
                box_z_nm=7.76,
                maximum_core_separation_fraction=0.40,
                reason="explicit ambiguous geometry",
            )
        )
    return np.arange(301) * 100.0, series, errors


def test_missing_prefix_is_recorded_and_cannot_certify_bulk_equilibrium():
    times, series, errors = missing_example()
    result = assess_measured_series(times, series, errors)
    coverage = result["measurement_coverage"]
    assert coverage["source_time_ps"] == times.tolist()
    assert coverage["unmeasurable_frames"] == errors
    assert coverage["diagnostic_input_window_ps"] == [700, 30000]
    assert coverage["imputed_frames"] == 0
    assert not result["passed"]
    assert INCOMPLETE in sampling_failures(result)
    assert not quantitative_area([result, result])["converged"]
    # A complete local half has the original 151 observations, not half the suffix.
    result["local_conformations"] = {"total_frames": 301, "time_ps": times[150:].tolist()}
    validate_measurement_coverage(result)
    result["local_conformations"]["time_ps"] = times[154:].tolist()
    with pytest.raises(ValueError):
        validate_measurement_coverage(result)


@pytest.mark.parametrize("index", [150, 300])
def test_unmeasurable_local_window_is_not_rescued_by_moving_its_start(index):
    with pytest.raises(ValueError, match="fixed local analysis window"):
        assess_measured_series(*missing_example((index,)))


def test_other_nonfinite_values_are_not_hidden_by_dropping_the_prefix():
    times, series, errors = missing_example()
    series["area_per_lipid_nm2"][0] = np.nan
    with pytest.raises(ValueError, match="outside recorded"):
        assess_measured_series(times, series, errors)


@pytest.mark.parametrize("mutation", ["flag", "threshold", "time", "window", "geometry"])
def test_modified_missing_frame_records_fail_closed(mutation):
    result = assess_measured_series(*missing_example())
    bad = copy.deepcopy(result)
    coverage = bad["measurement_coverage"]
    frame = coverage["unmeasurable_frames"][0]
    if mutation == "flag":
        coverage["complete"] = True
    elif mutation == "threshold":
        frame["maximum_core_separation_fraction"] = 0.5
    elif mutation == "time":
        frame["time_ps"] = 10000
    elif mutation == "window":
        coverage["diagnostic_input_window_ps"][0] = 900
    else:
        frame["core_separation_nm"] = 1
    with pytest.raises(ValueError):
        validate_measurement_coverage(bad)


def ambiguous_frame():
    sampler = TrajectorySampler.__new__(TrajectorySampler)
    sampler.protocol = {"lipid": "POPS"}
    sampler.n_leaflet = 1
    sampler.groups = [
        dict(name="POPS", polar=[0], tails=[1], anchor=0, upper=np.array([True, False]))
    ]
    xyz = np.zeros((2, 2, 3))
    # Independent hand-calculated case: periodic core separation 1.8 / box 4 = 0.45.
    xyz[:, :, 2] = [[0.5, -1.0], [3.7, 5.2]]
    return sampler, [xyz]


def test_ambiguous_frame_keeps_orientation_but_never_has_fake_thickness():
    sampler, blocks = ambiguous_frame()
    with pytest.raises(equil.BilayerThicknessError):
        sampler.metrics(np.eye(3) * 4, blocks)
    metrics, passed, detail = sampler.metrics(np.eye(3) * 4, blocks, allow_unmeasurable=True)
    assert not passed
    assert metrics["head_to_head_nm"] is None
    assert detail["core_gap_nm"] is None
    assert detail["oriented_fraction"] == 1
    assert detail["measurement_error"]["core_separation_nm"] == pytest.approx(1.8)
    # Translating and independently wrapping whole molecules cannot remove ambiguity.
    blocks[0][:, :, 2] += np.array([5.3, -6.7])[:, None]
    _, passed, detail = sampler.metrics(np.eye(3) * 4, blocks, allow_unmeasurable=True)
    assert not passed
    assert detail["measurement_error"]["core_separation_nm"] == pytest.approx(1.8)


def test_unexpected_runtime_failure_still_stops_analysis(monkeypatch):
    sampler, blocks = ambiguous_frame()

    def fail(*args):
        raise RuntimeError("unrelated coordinate failure")

    monkeypatch.setattr(equil, "head_to_head_distance", fail)
    with pytest.raises(RuntimeError, match="unrelated"):
        sampler.metrics(np.eye(3) * 4, blocks, allow_unmeasurable=True)


@pytest.mark.parametrize("defect", [None, "time", "energy", "gro"])
def test_completed_recovery_never_runs_dynamics(tmp_path, monkeypatch, defect):
    from gmxbuilder.core.structure import Structure
    from gmxbuilder.modules.membrane import v4_tpr

    builder = bare_builder(monkeypatch)
    record = dict(
        lipid_name="POPS",
        force_field="charmm36m",
        lipid_ff="charmm36m",
        replica_seed=12,
        npt_steps=15000000,
    )
    monkeypatch.setattr(equil, "validate_prepared_work", lambda *a, **kw: record)
    monkeypatch.setattr(
        builder,
        "_entry_context",
        lambda *a, **kw: SimpleNamespace(
            force_field="charmm36m", output_dir=tmp_path / "library" / "POPS"
        ),
    )
    monkeypatch.setattr(
        builder, "_energy_end_time", lambda *a: 29000 if defect == "energy" else 30000
    )
    monkeypatch.setattr(builder, "_mdrun", lambda *a, **kw: pytest.fail("MD launched"))
    monkeypatch.setattr(builder, "_mdrun_continue", lambda *a, **kw: pytest.fail("MD resumed"))
    monkeypatch.setattr(v4_tpr, "verify_tpr", lambda *a: None)
    published = []
    monkeypatch.setattr(builder, "_publish_prepared", lambda *a, **kw: published.append(True))
    for name in ("production-started.json", "npt.tpr", "npt.cpt", "npt.edr"):
        (tmp_path / name).write_text("original")
    xyz = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=np.float32)
    box = np.eye(3, dtype=np.float32)[None] * 8
    with XTCTrajectoryFile(str(tmp_path / "npt.xtc"), "w") as f:
        f.write(
            xyz, time=np.array([29000 if defect == "time" else 30000], dtype=np.float32), box=box
        )
    final = xyz[0].copy()
    if defect == "gro":
        final[0, 0] += 0.1
    GROWriter.write(
        Structure(
            coordinates=final,
            box_vectors=box[0],
            atom_names=["C", "C"],
            resnames=["POPS", "POPS"],
            resids=[1, 2],
        ),
        tmp_path / "npt.gro",
    )
    if defect:
        with pytest.raises(RuntimeError, match="endpoint|final frame"):
            builder.recover_prepared(tmp_path)
        assert not published
    else:
        builder.recover_prepared(tmp_path)
        assert published == [True]

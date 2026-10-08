"""Independently generated acceptance/rejection controls for the local policy."""

import copy

import numpy as np
import pytest

from gmxbuilder.modules.membrane.local_relaxation import assess_local, compare_local_replicas
from tests.local_relaxation_fixture import local_fixture


def test_stationary_molecular_populations_pass_independent_runs():
    pair = [local_fixture(seed=seed) for seed in (101, 102)]
    assert all(assess_local(item)["passed"] for item in pair)
    assert compare_local_replicas(pair)["passed"]


@pytest.mark.parametrize("kind", ["drift", "late_jump", "hydration", "bond", "angle"])
def test_known_unrelaxed_controls_are_rejected(kind):
    item = local_fixture()
    series = item["series"]
    prefix = kind if kind in {"hydration", "bond", "angle"} else "shape"
    key = next(
        k
        for k in series
        if k.startswith(prefix + ":") and (prefix != "shape" or k.endswith(":q50"))
    )
    magnitude = {"hydration": 0.8, "bond": 0.008, "angle": 8}.get(kind, 0.2)
    shift = np.linspace(0, magnitude, len(series[key]))
    if kind == "late_jump":
        shift[:] = 0
        shift[3 * len(shift) // 4 :] = magnitude
    series[key] = (np.array(series[key]) + shift).tolist()
    if prefix == "shape":
        upper = key[:-3] + "q90"
        series[upper] = (np.array(series[upper]) + shift).tolist()
    result = assess_local(item)
    assert not result["passed"]
    assert any(key in reason for reason in result["failures"])


def test_rotamer_weight_drift_is_reported_without_equilibrium_certification():
    item = local_fixture()
    series = item["series"]
    key = next(k for k in series if ":minus:" in k)
    shift = np.linspace(0, 0.3, len(series[key]))
    series[key] = (np.array(series[key]) + shift).tolist()
    other = key.replace(":minus:", ":trans:")
    series[other] = (np.array(series[other]) - shift).tolist()
    result = assess_local(item)
    assert result["passed"]
    assert any(key in reason for reason in result["warnings"])
    assert result["policy"]["equilibrium_certification"] is False


def test_different_stable_conformational_basins_do_not_pass_replica_comparison():
    first, second = local_fixture(seed=10), local_fixture(seed=11)
    for key, values in second["series"].items():
        if key.startswith("shape:"):
            second["series"][key] = (np.array(values) * 1.12).tolist()
    assert assess_local(first)["passed"] and assess_local(second)["passed"]
    assert not compare_local_replicas([first, second])["passed"]


def test_slow_small_amplitude_motion_is_not_mistaken_for_independent_samples():
    item = local_fixture()
    key = next(k for k in item["series"] if k.startswith("bond:"))
    values = np.asarray(item["series"][key])
    item["series"][key] = (0.153 + 0.0002 * np.sin(np.linspace(0, 3 * np.pi, len(values)))).tolist()
    result = assess_local(item)
    assert not result["passed"]
    assert any("ten effective" in reason for reason in result["failures"])


def test_vectorized_blocks_match_independent_array_split_reference():
    from gmxbuilder.modules.membrane.local_relaxation import _blocks

    rng = np.random.default_rng(711)
    for size in (1, 2, 3, 7, 15):
        values = rng.normal(size=73)
        expected = [chunk.mean() for chunk in np.array_split(values, len(values) // size)]
        np.testing.assert_allclose(_blocks(values, size), expected, atol=1e-14)


def test_constant_unvisited_torsion_sector_is_explicit_and_not_a_frozen_trajectory():
    item = local_fixture()
    for key in item["series"]:
        if key.startswith("torsion:"):
            item["series"][key] = [float(":trans:" in key)] * len(item["time_ps"])
    result = assess_local(item)
    assert result["passed"]
    assert any(value["constant_observed"] for value in result["statistics"].values())
    for key in item["series"]:
        item["series"][key] = [item["series"][key][0]] * len(item["time_ps"])
    with pytest.raises(ValueError, match="Frozen"):
        assess_local(item)


@pytest.mark.parametrize("defect", ["missing", "time", "policy", "nonfinite", "population"])
def test_incomplete_or_mutated_evidence_fails_closed(defect):
    item = copy.deepcopy(local_fixture())
    if defect == "missing":
        item["series"].pop(next(iter(item["series"])))
    elif defect == "time":
        item["time_ps"][0] += 100
    elif defect == "policy":
        item["policy"]["shape_relative_margin"] = 0.9
    elif defect == "population":
        key = next(k for k in item["series"] if ":trans:" in k)
        item["series"][key][0] += 0.1
    else:
        item["series"][next(iter(item["series"]))][0] = float("nan")
    with pytest.raises(ValueError):
        assess_local(item)

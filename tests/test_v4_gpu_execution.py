"""V4 must obey explicit GPU execution policy on both fresh and resumed MD."""

import pytest

from gmxbuilder.modules.membrane.lipid_equilibration import (
    LipidEquilibrationBuilder,
    lipid_gpu_device,
)


@pytest.fixture(autouse=True)
def isolate_preprocessing_policy(monkeypatch):
    # These tests exercise GPU dispatch; charge-policy admission has its own tests.
    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.grompp_policy.ensure_dynamics_policy", lambda *a: None
    )


def builder():
    instance = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    instance.gmx = "gmx"
    instance.threads = 16
    instance.require_gpu_update = True
    return instance


@pytest.mark.parametrize("resume", [False, True])
def test_v4_executes_single_rank_with_sixteen_threads_and_gpu_update(tmp_path, monkeypatch, resume):
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY_GPU", "1")
    instance = builder()
    calls = []
    instance._run = lambda args, *_args, **_kwargs: calls.append(args)
    instance._energy_end_time = lambda *_args: 1000.0
    with lipid_gpu_device(1):
        if resume:
            instance._mdrun_continue("npt", tmp_path, until_ps=1000, timeout=10)
        else:
            instance._mdrun("npt", tmp_path, timeout=10)
    assert len(calls) == 1
    command = calls[0]
    for key, value in {"-ntmpi": "1", "-ntomp": "16", "-update": "gpu", "-gpu_id": "1"}.items():
        assert command[command.index(key) + 1] == value


def test_failed_gpu_run_is_not_accepted_from_stale_coordinates_or_retried_on_cpu(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY_GPU", "1")
    (tmp_path / "npt.gro").write_text("stale" * 100)
    instance = builder()
    calls = []

    def fail(args, *_args, **_kwargs):
        calls.append(args)
        raise RuntimeError("GPU update unavailable")

    instance._run = fail
    with pytest.raises(RuntimeError, match="GPU update unavailable"):
        instance._mdrun("npt", tmp_path, timeout=10)
    assert len(calls) == 1


@pytest.mark.parametrize("resume", [False, True])
def test_gpu_policy_conflict_fails_before_starting(tmp_path, monkeypatch, resume):
    monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY_GPU", "0")
    instance = builder()
    with pytest.raises(RuntimeError, match="GPU execution is disabled"):
        if resume:
            instance._mdrun_continue("npt", tmp_path, until_ps=1000, timeout=10)
        else:
            instance._mdrun("npt", tmp_path, timeout=10)

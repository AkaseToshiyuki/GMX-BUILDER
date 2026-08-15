import signal
import subprocess

import pytest

from gmxbuilder.modules.forcefield import gaff_backend


class _TimedOutProcess:
    pid = 4321
    returncode = -signal.SIGTERM

    def __init__(self):
        self.calls = 0

    def communicate(self, *, timeout):
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired(["acpype"], timeout)
        return "partial stdout", "partial stderr"


def test_external_timeout_terminates_the_complete_process_group(monkeypatch, tmp_path):
    process = _TimedOutProcess()
    popen_options = {}
    signals = []

    def fake_popen(args, **kwargs):
        popen_options.update(kwargs)
        return process

    monkeypatch.setattr(gaff_backend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gaff_backend.os, "name", "posix")
    monkeypatch.setattr(
        gaff_backend.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="timed out after 2 seconds"):
        gaff_backend._run_external(
            ["acpype", "-i", "molecule.mol2"],
            cwd=tmp_path,
            env={},
            timeout=2,
        )

    assert popen_options["start_new_session"] is True
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.calls == 2


def test_gaff_thread_limit_is_scoped_to_external_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_ENV", str(tmp_path / "gaff"))
    monkeypatch.setenv("GMXBUILDER_GAFF_THREADS", "24")
    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "24")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)

    child_env = gaff_backend._gaff_tool_environment()

    assert child_env["OMP_NUM_THREADS"] == "24"
    assert child_env["OMP_THREAD_LIMIT"] == "24"
    assert "OMP_NUM_THREADS" not in gaff_backend.os.environ


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_gaff_thread_limit_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("GMXBUILDER_GAFF_THREADS", value)
    with pytest.raises(ValueError, match="positive integer"):
        gaff_backend._gaff_tool_environment()


def test_smiles_parameterization_retries_stochastic_coordinate_clashes(monkeypatch, tmp_path):
    results = iter(
        [
            subprocess.CompletedProcess(
                ["acpype"],
                1,
                "",
                "Atoms TOO close\nCoordinates issues with your system",
            ),
            subprocess.CompletedProcess(["acpype"], 0, "ok", ""),
        ]
    )
    calls = []

    def fake_run(args, *, cwd, env, timeout):
        calls.append(cwd)
        return next(results)

    monkeypatch.setattr(gaff_backend, "_run_external", fake_run)
    result, generated_work = gaff_backend._run_smiles_acpype(
        ["acpype", "-i", "SMILES"],
        work=tmp_path,
        env={},
        timeout=30,
    )

    assert result.returncode == 0
    assert generated_work == tmp_path / "acpype_attempt_2"
    assert calls == [tmp_path / "acpype_attempt_1", tmp_path / "acpype_attempt_2"]

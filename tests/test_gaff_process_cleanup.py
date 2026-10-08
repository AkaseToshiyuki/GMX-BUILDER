import json
import signal
import subprocess

import numpy as np
import pytest

from gmxbuilder.modules.forcefield import gaff_backend


def test_coordinate_cache_signature_is_linear_versioned_and_translation_invariant():
    names = ("C1", "N2", "O3")
    elements = ("C", "N", "O")
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [0.21, 0.08, 0.0]],
        dtype=float,
    )

    signature = gaff_backend._coordinate_identity_signature(names, elements, coordinates, 7.0)
    translated = gaff_backend._coordinate_identity_signature(
        names, elements, coordinates + np.asarray([100.0, -20.0, 3.0]), 7.0
    )
    changed = coordinates.copy()
    changed[2, 1] += 0.001

    assert translated == signature
    assert gaff_backend._coordinate_identity_signature(names, elements, changed, 7.0) != signature
    assert (
        gaff_backend._coordinate_identity_signature(
            tuple(reversed(names)),
            tuple(reversed(elements)),
            coordinates[::-1],
            7.0,
        )
        != signature
    )
    payload = json.loads(signature)
    assert payload["schema"].startswith("gaff-coordinate-v4-")
    assert payload["atom_count"] == 3
    assert set(payload) == {"atom_count", "protonation_pH", "schema", "sha256"}


def test_gaff_rejects_oversized_ligand_before_external_tool_discovery(monkeypatch):
    def unexpected_tool_check():
        raise AssertionError("tool discovery must not run for an oversized ligand")

    monkeypatch.setattr(gaff_backend, "gaff_available", unexpected_tool_check)
    indices = tuple(range(gaff_backend.MAX_GAFF_LIGAND_ATOMS + 1))
    with pytest.raises(ValueError, match="2048 atoms"):
        gaff_backend.prepare_gaff_molecule("LIG", object(), indices, 0)


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


@pytest.mark.parametrize("available_cpus", [2, 32])
def test_gaff_thread_limit_is_scoped_to_external_tools(monkeypatch, tmp_path, available_cpus):
    monkeypatch.setattr(gaff_backend.os, "cpu_count", lambda: available_cpus)
    monkeypatch.setenv("GMXBUILDER_GAFF_ENV", str(tmp_path / "gaff"))
    monkeypatch.setenv("GMXBUILDER_GAFF_THREADS", "24")
    monkeypatch.setenv("GMXBUILDER_TASK_THREADS", "24")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)

    child_env = gaff_backend._gaff_tool_environment()

    assert child_env["OMP_NUM_THREADS"] == str(min(24, available_cpus))
    assert child_env["OMP_THREAD_LIMIT"] == str(min(24, available_cpus))
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

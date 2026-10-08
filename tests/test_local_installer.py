"""Static safety and resource-contract tests for the local installer."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_parallel_package_downloads_keep_lock_order_and_validate_hashes(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import io
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    spec = importlib.util.spec_from_file_location(
        "gaff_installer", ROOT / "scripts/install_gaff_runtime.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    payloads = {"https://test.invalid/" + str(i): bytes([i]) * 300 for i in range(4)}
    records = [
        {"url": url, "sha256": hashlib.sha256(body).hexdigest()} for url, body in payloads.items()
    ]
    monkeypatch.setattr(
        installer.urllib.request, "urlopen", lambda url, **kwargs: io.BytesIO(payloads[url])
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        urls = list(pool.map(lambda row: installer._fetch_archive(row, tmp_path), records))
    assert urls == [
        (tmp_path / str(i)).as_uri() + "#" + row["sha256"] for i, row in enumerate(records)
    ]
    with pytest.raises(RuntimeError, match="checksum"):
        installer._fetch_archive({**records[0], "sha256": "0" * 64}, tmp_path)
    assert (tmp_path / "0").read_bytes() == payloads[records[0]["url"]]
    assert not list(tmp_path.glob("*.download-*"))


def test_local_installer_exposes_documented_unattended_defaults():
    script = (ROOT / "install-local.sh").read_text()

    assert "DEFAULT_HOST=127.0.0.1" in script
    assert "GMXBUILDER_DEPLOYMENT_MODE" in script
    assert "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT" in script
    assert "trusted-lan" in script
    assert "DEFAULT_PORT=7788" in script
    assert "INTERACTIVE=0" in script
    assert "--interactive" in script
    assert "--allow-unsafe-deployment" in script
    assert "AVAILABLE_CORES" in script
    assert "DEFAULT_CPU_CORES=$((AVAILABLE_CORES / 2))" in script
    assert "choose_default_slots" in script
    assert "CPU_CORES % QUEUE_SLOTS" in script
    assert "TASK_THREADS=$((CPU_CORES / QUEUE_SLOTS))" in script
    assert "--gmx-bin" in script
    assert "GROMACS 2026.0 or newer" in script
    assert "export GMX_BIN" in script
    assert '"$PYTHON_BIN" -m venv "$BOOTSTRAP_DIR"' in script
    assert "--require-hashes" in script
    assert "uv-bootstrap.txt" in script
    assert '"$UV_BIN" sync' in script
    assert '"$PYTHON_BIN" "$ROOT_DIR/scripts/fetch_prebuilt_assets.py"' in script
    assert '"$VENV_DIR/bin/gmxbuilder" prebuilt-assets install' in script
    assert "git lfs pull" not in script
    assert "systemctl --user enable --now gmxbuilder.service" in script
    assert "lipid-library queue" not in script
    assert "lipid_equilibrated_v4/library" in script
    assert "disable --now gmxbuilder-lipid-library-watchdog.timer" in script
    assert "enable --now gmxbuilder-lipid-library-watchdog.timer" not in script
    assert "gmxbuilder-lipid-library-watchdog.timer" in script
    assert "output/lipid-library" not in script
    assert "GMXBUILDER_TASK_DIR" in script


def test_local_installer_help_is_non_mutating_and_documents_overrides():
    result = subprocess.run(
        ["bash", str(ROOT / "install-local.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "installation is unattended" in result.stdout
    assert "--bind-host" in result.stdout
    assert "--allow-unsafe-deployment" in result.stdout
    assert "--cpu-cores" in result.stdout
    assert "--queue-slots" in result.stdout
    assert "--gmx-bin" in result.stdout


def test_external_asset_manifest_has_direct_verified_https_downloads():
    manifest = json.loads((ROOT / "scripts/external_assets.json").read_text())
    assert manifest["schema_version"] == 1
    assert {asset["target"] for asset in manifest["assets"]} == {
        "charmm36",
        "charmm36m",
        "amber14sb_ol24",
    }
    for asset in manifest["assets"]:
        assert asset["source_url"].startswith("https://")
        assert asset["url"].startswith("https://")
        assert len(asset["sha256"]) == 64
        assert asset["required_files"]


def test_an_asset_that_merges_names_a_base_this_repository_actually_ships():
    """``merge_base`` is resolved at install time on the user's machine.

    A typo there fails during installation, on a fresh clone, with the network
    already used -- so it is checked here instead, where the answer is known.
    """
    manifest = json.loads((ROOT / "scripts/external_assets.json").read_text())
    forcefields = ROOT / "src" / "gmxbuilder" / "data" / "forcefields"
    merging = [asset for asset in manifest["assets"] if asset.get("merge_base")]
    assert merging, "no merging asset; this test would prove nothing"
    for asset in merging:
        base = forcefields / str(asset["merge_base"])
        assert base.is_dir(), f"{asset['name']} merges onto a missing {asset['merge_base']}"
        assert (base / "aminoacids.rtp").is_file(), (
            f"{asset['merge_base']} has no residue database to protect"
        )


def test_gromacs_bootstrap_is_official_pinned_and_noninteractive():
    helper = (ROOT / "scripts" / "install_gromacs.py").read_text()
    installer = (ROOT / "install-local.sh").read_text()

    assert "https://ftp.gromacs.org/gromacs/" in helper
    assert 'VERSION = "2026.3"' in helper
    assert len(helper.split('SOURCE_SHA256 = "', 1)[1].split('"', 1)[0]) == 64
    assert "GMX_BUILD_OWN_FFTW=ON" in helper
    assert "GMX_THREAD_MPI=ON" in helper
    assert "GMX_GPU={'CUDA' if cuda else 'OFF'}" in helper
    assert "scripts/install_gromacs.py" in installer
    assert "GMXBUILDER_GROMACS_FORCE_CPU" in installer
    assert "scripts/install_gaff_runtime.py" in installer
    assert "GMXBUILDER_GAFF_ENV" in installer

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_gromacs.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--target-root" in result.stdout
    assert "--force-cpu" in result.stdout


def test_gaff_bootstrap_is_pinned_verified_and_noninteractive():
    helper = (ROOT / "scripts" / "install_gaff_runtime.py").read_text()

    assert 'MICROMAMBA_VERSION = "2.8.1-0"' in helper
    assert "github.com/mamba-org/micromamba-releases/releases/download/" in helper
    assert '"ambertools=24.8"' in helper
    assert '"acpype=2023.10.27"' in helper
    assert '"openbabel=3.1.1"' in helper
    assert "--offline" in helper
    assert "gaff-linux-64.lock.json" in helper
    assert "checksum verification failed" in helper

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_gaff_runtime.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prefix" in result.stdout
    assert "--runtime-root" in result.stdout


def test_installer_refuses_an_unsafe_bind_before_doing_any_work():
    """The opt-in gate must run before the runtime bootstrap, not after it.

    A guard that fires only after GROMACS has been built does not guard
    anything, so this asserts the real exit behaviour rather than the source
    text. The timeout is the actual assertion for that ordering: if the gate
    ever moves back below the bootstrap, this fails instead of starting an
    hour-long build.
    """
    for host in ("0.0.0.0", "::", "192.0.2.10"):
        completed = subprocess.run(
            [str(ROOT / "install-local.sh"), "--bind-host", host],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
        )
        assert completed.returncode != 0, host
        combined = completed.stdout + completed.stderr
        assert "--allow-unsafe-deployment" in combined, host
        # Nothing may be installed or built before the address is accepted.
        assert "Installing the required GROMACS runtime" not in combined, host


def test_installer_rejects_a_malformed_unsafe_deployment_flag():
    completed = subprocess.run(
        [str(ROOT / "install-local.sh"), "--bind-host", "0.0.0.0"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env={**os.environ, "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT": "maybe"},
    )
    assert completed.returncode != 0
    assert "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT must be" in (completed.stdout + completed.stderr)

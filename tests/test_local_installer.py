"""Static safety and resource-contract tests for the local installer."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def test_local_installer_exposes_documented_unattended_defaults():
    script = (ROOT / "install-local.sh").read_text()

    assert "DEFAULT_HOST=127.0.0.1" in script
    assert "GMXBUILDER_DEPLOYMENT_MODE" in script
    assert "trusted-lan" in script
    assert "DEFAULT_PORT=7788" in script
    assert "INTERACTIVE=0" in script
    assert "--interactive" in script
    assert "AVAILABLE_CORES" in script
    assert "DEFAULT_CPU_CORES=$((AVAILABLE_CORES / 2))" in script
    assert "choose_default_slots" in script
    assert "CPU_CORES % QUEUE_SLOTS" in script
    assert "TASK_THREADS=$((CPU_CORES / QUEUE_SLOTS))" in script
    assert "--gmx-bin" in script
    assert "GROMACS 2026.0 or newer" in script
    assert "export GMX_BIN" in script
    assert '"$PYTHON_BIN" -m venv "$BOOTSTRAP_DIR"' in script
    assert '"uv==0.11.22"' in script
    assert '"$UV_BIN" sync' in script
    assert '"$PYTHON_BIN" "$ROOT_DIR/scripts/fetch_prebuilt_assets.py"' in script
    assert '"$VENV_DIR/bin/gmxbuilder" prebuilt-assets install' in script
    assert "git lfs pull" not in script
    assert "systemctl --user enable --now gmxbuilder.service" in script
    assert "lipid-library status" in script
    assert "--json-output" in script
    assert "PENDING" in script
    assert "StandardOutput=journal" in script
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
    assert "--cpu-cores" in result.stdout
    assert "--queue-slots" in result.stdout
    assert "--gmx-bin" in result.stdout


def test_external_asset_manifest_has_direct_verified_https_downloads():
    manifest = json.loads((ROOT / "scripts/external_assets.json").read_text())
    assert manifest["schema_version"] == 1
    assert {asset["target"] for asset in manifest["assets"]} == {
        "charmm36",
        "charmm36m",
    }
    for asset in manifest["assets"]:
        assert asset["source_url"].startswith("https://")
        assert asset["url"].startswith("https://")
        assert len(asset["sha256"]) == 64
        assert asset["required_files"]


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
    assert '"ambertools=26.0"' in helper
    assert '"acpype=2023.10.27"' in helper
    assert '"openbabel=3.1.1"' in helper
    assert "--strict-channel-priority" in helper
    assert "checksum verification failed" in helper

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_gaff_runtime.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prefix" in result.stdout
    assert "--runtime-root" in result.stdout

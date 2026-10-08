"""The local deployment must be verifiable, not merely assumed.

The service runs an editable install, so source changes need no
reinstallation -- but a running process keeps the modules it imported at
start-up. Nothing detected the gap, and the deployment ended up fourteen
releases behind the working tree while appearing healthy.

These tests cover the script's decisions rather than its effects: restarting a
real service is not something a test suite should do.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_local.sh"


def _run(*arguments: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        env={**os.environ, **(env or {})},
    )


def test_the_script_is_executable_and_valid_shell():
    assert os.access(SCRIPT, os.X_OK)
    parsed = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr


def test_check_mode_reports_an_unreachable_service_in_words():
    """A bare exit status is not a usable report.

    An earlier version let curl's status escape through pipefail and set -e,
    so the operator saw exit 7 and no explanation.
    """
    # --allow-dirty keeps this independent of the working tree's state: the
    # uncommitted-changes gate runs first, so without it this test would pass
    # on a clean checkout and fail while anyone is mid-change.
    result = _run(
        "--check",
        "--allow-dirty",
        env={"GMXBUILDER_HEALTH_URL": "http://127.0.0.1:59999/health"},
    )
    assert result.returncode == 1
    assert "not answering" in result.stderr


def test_check_mode_never_restarts_anything():
    source = SCRIPT.read_text(encoding="utf-8")
    guarded = re.search(r"if \(\( ! check_only \)\); then(.*?)\nfi\n", source, re.S)
    assert guarded, "the restart block is no longer guarded by check_only"
    assert "systemctl --user restart" in guarded.group(1)
    # Exactly one restart, and it is inside the guard.
    assert source.count("systemctl --user restart") == 1


def test_the_script_refuses_to_interrupt_running_builds():
    """A restart drops in-flight work, and a build can be an hour of GROMACS."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "builds_active" in source
    assert "builds_queued" in source
    assert "Refusing to restart" in source


def test_a_version_mismatch_is_an_error_not_a_note():
    """The exit status has to carry the verdict so a release step can gate on it."""
    source = SCRIPT.read_text(encoding="utf-8")
    mismatch = source[source.index('if [[ "$actual" != "$expected" ]]') :]
    assert "exit 1" in mismatch.split("fi")[0]


def test_the_version_is_read_from_the_same_file_the_package_uses():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "src/gmxbuilder/__version__.py" in source
    assert (ROOT / "src" / "gmxbuilder" / "__version__.py").is_file()


def test_an_uncommitted_working_tree_blocks_deployment():
    """Matching version strings do not prove the deployment matches the repo.

    The editable install means the service runs the working tree, so an
    uncommitted change is served while both sides report the same number --
    the same shape of silent divergence this script exists to catch.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'git -C "$REPO_ROOT" status --porcelain' in source
    assert "allow_dirty" in source

    guard = source[source.index('if [[ -n "$dirty" ]]') :]
    assert "exit 1" in guard.split("fi\n")[0]


def test_unknown_options_are_refused():
    """Silently ignoring a typo would mean a flag quietly does nothing."""
    result = _run("--nonsense")
    assert result.returncode == 2
    assert "Unknown option" in result.stderr


def test_allow_dirty_is_reported_not_hidden():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Warning: deploying an uncommitted working tree" in source
    # The success line must say which of the two states it is reporting.
    assert "from an uncommitted working tree" in source
    assert "matching the repository" in source


def test_the_installer_carries_extension_settings_into_the_runner():
    """A private module must survive a reinstall.

    Without this the settings would have to be re-added to the generated
    launcher by hand after every run of install-local.sh, which is precisely
    the kind of step that gets forgotten.
    """
    installer = (ROOT / "install-local.sh").read_text(encoding="utf-8")
    assert 'EXTENSIONS="${GMXBUILDER_EXTENSIONS:-}"' in installer
    assert "printf 'export GMXBUILDER_EXTENSIONS=%q" in installer
    assert "printf 'export PYTHONPATH=%q" in installer
    # Written only when configured: a public install must emit neither.
    block = installer[installer.index("RUNNER=") : installer.index('} > "$RUNNER"')]
    conditional = block[block.index('if [[ -n "$EXTENSIONS" ]]') :]
    assert "GMXBUILDER_EXTENSIONS" in conditional.split("fi")[0]


def test_deploy_reports_a_launcher_that_predates_the_installer():
    """Restarting picks up source changes but never installer changes.

    The unit runs a generated launcher, so a setting the installer learned to
    write after the last install is simply absent, and a restart walks straight
    past it.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "predates the current installer" in source
    assert "run-local.sh" in source


def test_the_drift_scan_ignores_settings_that_are_not_always_written():
    """A check that fires on every healthy install is one people learn to ignore.

    The scan is scoped to the run-local.sh block so the lipid-queue runner's
    variables are not counted, and to two-space indentation so the conditional
    extension block is excluded.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert '/^RUNNER=/,/^} > "\\$RUNNER"/p' in source
    assert "s/^  printf 'export" in source

    installer = (ROOT / "install-local.sh").read_text(encoding="utf-8")
    block = installer[installer.index("RUNNER=") : installer.index('} > "$RUNNER"')]
    scanned = {
        line.split("export ")[1].split("=")[0]
        for line in block.splitlines()
        if line.startswith("  printf 'export ")
    }
    # V4 readiness is now shared with Web, so its library path is mandatory.
    assert "GMXBUILDER_LIPID_LIBRARY" in scanned
    # Worker-only and optional extension settings stay outside the Web scan.
    assert "GMXBUILDER_GAFF_CACHE" not in scanned
    assert "GMXBUILDER_EXTENSIONS" not in scanned
    assert "GMXBUILDER_TASK_DIR" in scanned

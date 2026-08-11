"""Static safety and resource-contract tests for the local installer."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_local_installer_exposes_documented_interactive_defaults():
    script = (ROOT / "install-local.sh").read_text()

    assert "DEFAULT_HOST=127.0.0.1" in script
    assert "GMXBUILDER_DEPLOYMENT_MODE" in script
    assert "trusted-lan" in script
    assert "DEFAULT_PORT=7788" in script
    assert "AVAILABLE_CORES" in script
    assert "DEFAULT_CPU_CORES=$((AVAILABLE_CORES / 2))" in script
    assert "choose_default_slots" in script
    assert "CPU_CORES % QUEUE_SLOTS" in script
    assert "TASK_THREADS=$((CPU_CORES / QUEUE_SLOTS))" in script
    assert '"$PYTHON_BIN" -m venv' in script
    assert '"$VENV_DIR/bin/python" -m pip install -e' in script
    assert '"$VENV_DIR/bin/gmxbuilder" prebuilt-assets install' in script
    assert "systemctl --user enable --now gmxbuilder.service" in script
    assert "GMXBUILDER_TASK_DIR" in script

"""Feedback while a Check runs.

Every step whose Check writes a checkpoint shows a progress bar under its own
Check button: a percentage from the server, the phase underneath, a running
clock, and a green "Complete" with the total time when it finishes.

The bars are per-button and built on demand. A single shared element was tried
first and cannot work -- the button that starts a Check lives in a wizard panel
that is hidden and shown as the user moves between steps -- so what is checked
here is that every Check path creates its own bar and closes it out on every
exit, including failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gmxbuilder.web import server
from gmxbuilder.web.server import app
from gmxbuilder.web.task_manager import TaskManager

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "gmxbuilder" / "web" / "static"
TEMPLATE = ROOT / "src" / "gmxbuilder" / "web" / "templates" / "index.html"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
IONS_JS = (STATIC / "ions.js").read_text(encoding="utf-8")
VERIFY_JS = (STATIC / "app_parts" / "system_verification.js").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Every Check that writes files reports progress


# The Check buttons that run a pipeline step on the server. Each one writes a
# checkpoint, so each one must show a bar.
CHECK_BUTTONS = {
    "input-check-btn": APP_JS,
    "forcefield-check-btn": APP_JS,
    "structure-check-btn": APP_JS,
    "orient-check-btn": APP_JS,
    "check-composition-btn": VERIFY_JS,
    "solv-check-btn": APP_JS,
    "ion-check-btn": IONS_JS,
    "cg-model-check": APP_JS,
    "cg-mapping-check": APP_JS,
    "cg-orientation-check": APP_JS,
    "cg-environment-check": APP_JS,
    "cg-solvation-check": APP_JS,
    "cg-system-check": APP_JS,
}


@pytest.mark.parametrize("button_id", sorted(CHECK_BUTTONS))
def test_every_check_button_exists_in_the_page(button_id):
    assert f'id="{button_id}"' in TEMPLATE.read_text(encoding="utf-8")


@pytest.mark.parametrize("button_id", sorted(CHECK_BUTTONS))
def test_every_check_that_writes_a_checkpoint_shows_a_bar(button_id):
    """The generic path covers most steps; three panels handle their own."""
    source = CHECK_BUTTONS[button_id]
    if source is APP_JS and button_id not in ("orient-check-btn",):
        # Routed through _doCheckStep, which starts the bar for whatever
        # button id it was given.
        assert "startStepProgress(stepName, btnId, statusElId)" in APP_JS
        assert f"'{button_id}'" in APP_JS or f'"{button_id}"' in APP_JS
    else:
        started = re.search(r"startStepProgress\([^)]*" + re.escape(button_id), source)
        assert started, f"{button_id} starts no progress bar"


@pytest.mark.parametrize("source", [APP_JS, IONS_JS, VERIFY_JS], ids=["app", "ions", "verify"])
def test_no_check_path_can_leave_a_bar_polling_forever(source):
    """Each handler closes its bar in `finally`, not only on the happy path."""
    for match in re.finditer(r"(?<!function )startStepProgress\(", source):
        tail = source[match.start() : match.start() + 9000]
        finallys = [m.start() for m in re.finditer(r"\}\s*finally\s*\{", tail)]
        assert finallys, "a Check starts a bar with no finally block after it"
        block = tail[finallys[0] :]
        assert any(
            name in block[:800] for name in ("finishStepProgress(", "finishUnfinishedStepProgress(")
        ), "the finally block never closes the bar"


def test_finishing_twice_is_harmless():
    """The finally net runs after the branch that already reported the result."""
    body = APP_JS[APP_JS.index("function finishStepProgress(") :][:400]
    assert "handle.done" in body


# --------------------------------------------------------------------------
# What the bar shows


def test_the_bar_carries_a_percentage_a_phase_and_a_clock():
    builder = APP_JS[APP_JS.index("function _stepProgressElement(") :]
    builder = builder[: builder.index("\nfunction ")]
    for part in (
        "step-progress-bar",
        "step-progress-percent",
        "step-progress-phase",
        "step-progress-elapsed",
    ):
        assert part in builder, part


def test_the_bar_reports_itself_to_assistive_technology():
    builder = APP_JS[APP_JS.index("function _stepProgressElement(") :]
    builder = builder[: builder.index("\nfunction ")]
    assert 'role="progressbar"' in builder
    assert 'aria-valuemin="0"' in builder and 'aria-valuemax="100"' in builder
    assert "aria-live" in builder
    assert "aria-valuenow" in APP_JS[APP_JS.index("function _paintStepProgress(") :][:600]


def test_a_finished_check_says_complete_with_its_total_time():
    body = APP_JS[APP_JS.index("function finishStepProgress(") :]
    body = body[: body.index("\n/**")]
    assert "'Complete'" in body
    assert "s total" in body


def test_running_is_blue_and_animated_and_finished_is_green():
    running = CSS[CSS.index('.step-progress[data-state="running"]') :]
    running = running[: running.index('.step-progress[data-state="done"]')]
    assert "animation: step-progress-roll" in running
    assert "@keyframes step-progress-roll" in CSS
    # Blue while running comes from the shared primary token.
    assert "background: var(--primary)" in CSS

    done = CSS[CSS.index('.step-progress[data-state="done"]') :]
    done = done[: done.index('.step-progress[data-state="error"]')]
    assert "background: var(--success)" in done, "the fill must turn green"
    assert "border-color: #86efac" in done, "the frame must turn green too"


def test_the_animation_yields_to_a_reduced_motion_preference():
    assert "prefers-reduced-motion" in CSS
    reduced = CSS[CSS.index("@media (prefers-reduced-motion: reduce)") :][:200]
    assert "animation: none" in reduced


# --------------------------------------------------------------------------
# When a bar must not be on screen


def test_no_bar_exists_before_a_check_is_clicked():
    """The template ships no bar; the first Check on a step creates it."""
    assert "step-progress" not in TEMPLATE.read_text(encoding="utf-8")
    assert "document.createElement('div')" in APP_JS


def test_invalidating_a_step_takes_its_completed_bar_with_it():
    """A green "Complete" claims a checkpoint that still exists."""
    loop = APP_JS[APP_JS.index("for (var i = idx + 1; i < state.wizardSteps.length; i++)") :][:300]
    assert "_checkedSteps.delete(state.wizardSteps[i]);" in loop
    assert "clearStepProgress(state.wizardSteps[i]);" in loop
    assert "clearStepProgress(step);" in IONS_JS, "server-side invalidation clears bars too"
    assert "clearStepProgress();" in APP_JS, "returning to task selection clears every bar"


def test_the_percentage_never_runs_backwards():
    body = APP_JS[APP_JS.index("function startStepProgress(") :]
    body = body[: body.index("\n/**")]
    assert "reported > handle.fraction" in body


# --------------------------------------------------------------------------
# The server side of the same feature


def test_a_step_reports_progress_through_its_phases():
    """The runner must call back as it advances, not only at the end."""
    import inspect

    from gmxbuilder.pipeline.step_executor import StepRunner

    signature = inspect.signature(StepRunner.run_step)
    assert "on_progress" in signature.parameters

    source = inspect.getsource(StepRunner.run_step)
    # One report per documented phase, plus start and completion.
    assert source.count("report(") >= 7
    assert 'report(1.0, "Complete")' in source


def test_a_failing_progress_callback_cannot_break_the_step():
    """Reporting is decoration; it must never cost a build."""
    import inspect

    from gmxbuilder.pipeline.step_executor import StepRunner

    source = inspect.getsource(StepRunner.run_step)
    guarded = source[source.index("def report(") : source.index("# ---- 1.")]
    assert "except Exception" in guarded


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_RATE_LIMIT_DB", str(tmp_path / "rate.sqlite3"))
    monkeypatch.setattr(server, "task_manager", TaskManager(tmp_path / "tasks"))
    with TestClient(app) as started:
        yield started


def test_progress_for_an_unknown_task_is_a_404(client):
    assert client.get("/api/step/" + "a" * 32 + "/progress").status_code == 404


def test_progress_endpoint_reports_nothing_running_rather_than_failing(client, monkeypatch):
    """Between polls a step can finish; that is a normal answer, not an error."""
    task = client.post("/api/tasks", json={"task_type": "pure-membrane"}).json()
    response = client.get(f"/api/step/{task['task_id']}/progress")
    assert response.status_code == 200
    assert response.json() == {"running": False}


def test_a_published_progress_entry_is_reported_with_elapsed_time(client):
    import time

    task = client.post("/api/tasks", json={"task_type": "pure-membrane"}).json()
    task_id = task["task_id"]
    with server._step_progress_lock:
        server._step_progress[task_id] = {
            "step": "membrane",
            "fraction": 0.42,
            "phase": "Running the scientific module",
            "started_at": time.time() - 3.0,
            "updated_at": time.time(),
        }
    try:
        body = client.get(f"/api/step/{task_id}/progress").json()
        assert body["running"] is True
        assert body["step"] == "membrane"
        assert body["fraction"] == 0.42
        assert body["phase"] == "Running the scientific module"
        assert body["elapsed_s"] >= 3.0
    finally:
        with server._step_progress_lock:
            server._step_progress.pop(task_id, None)

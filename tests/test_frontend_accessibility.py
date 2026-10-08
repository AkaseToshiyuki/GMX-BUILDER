"""Source-level accessibility and CSP contracts for the web client."""

import re
from pathlib import Path

from tests.frontend_bundle import frontend_source

ROOT = Path(__file__).parents[1]
STYLE = ROOT / "src/gmxbuilder/web/static/style.css"
TEMPLATE = ROOT / "src/gmxbuilder/web/templates/index.html"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _app_bundle() -> str:
    return frontend_source()


def test_frontend_has_no_inline_event_handlers() -> None:
    combined = _source(TEMPLATE) + _app_bundle()
    inline_event = re.compile(
        r"\bon(?:click|change|dblclick|keydown|keyup|input|focus|blur|submit)\s*=",
        re.IGNORECASE,
    )
    assert inline_event.search(combined) is None


# Stage toggles, label association and native card controls used to be checked
# by searching for their spelling here. They are now asserted against the
# running page in tests/test_browser_interaction.py, which can see whether the
# control actually does anything -- something no source search can establish.
#
# Keyboard handling inside the residue listbox keeps a source check for now:
# reaching that widget means driving several workflow steps, so a behavioural
# version would be an end-to-end test of something else.
def test_residue_listbox_handles_arrow_keys() -> None:
    app = _app_bundle()

    assert "option.closest('[role=\"listbox\"]')" in app
    assert "event.key === 'ArrowDown'" in app


def test_modals_have_dialog_semantics_focus_management_and_restoration() -> None:
    template = _source(TEMPLATE)
    app = _app_bundle()

    assert 'id="custom-lipid-modal" role="dialog"' in template
    assert 'aria-modal="true" aria-labelledby="custom-lipid-title"' in template
    assert "function modalFocusableElements(modal)" in app
    assert "if (event.key === 'Escape')" in app
    assert "if (event.key !== 'Tab') return" in app
    assert "modal._returnFocus = returnFocus || document.activeElement" in app
    assert "returnFocus.focus()" in app


def test_navigation_and_membrane_budget_contracts() -> None:
    app = _app_bundle()
    template = _source(TEMPLATE)
    render_nav = app.split("function renderStepNav()", 1)[1].split(
        "// ===================================================================", 1
    )[0]
    delegated_navigation = app.split("// Button Wiring", 1)[1].split(
        "// ==================================================================="
        "\n// Options Loading",
        1,
    )[0]

    assert render_nav.count("goToWizardStep(idx)") == 1
    assert "#step-nav .step[data-step-module]" not in delegated_navigation
    assert 'id="n-lipids-per-leaflet" value="150" step="10" min="64" max="5000"' in template
    assert "nLipids > 5000" in app
    assert "n_lipids_per_leaflet: Math.max" not in app


def test_mobile_layout_constrains_content_to_the_viewport() -> None:
    """Layout rules that a headless desktop browser would not exercise.

    The colour assertion that used to sit here was replaced by a computed
    contrast check in tests/test_browser_interaction.py: asserting that a hex
    string appears in the stylesheet says nothing about what a reader sees,
    since the rule may be overridden and the element may sit on a different
    background.
    """
    style = _source(STYLE)
    app = _app_bundle()

    assert "max-width: calc(100vw - 16px)" in style
    assert "width: min(560px, calc(100vw - 32px))" in style
    assert ".custom-lipid-structure canvas" in style
    assert "viewportWidth - margin * 2" in app

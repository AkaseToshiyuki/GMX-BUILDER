"""Behaviour of the browser client, asserted by driving it.

The client has long been checked by searching its source text for expected
strings. That catches nothing: a source search cannot tell whether clicking a
control does anything, and it fails whenever an equivalent refactor changes the
spelling. Round 9 built the harness -- a live server and a headless browser --
so the checks that are really about *behaviour* can be made against behaviour.

What stays a source check, deliberately: the absence of inline event handlers.
That is a property of the file, required by the Content-Security-Policy, and a
regex over the source is the right tool for it.
"""

from __future__ import annotations

import pytest

from tests.prerequisites import requires_lfs_assets

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# --------------------------------------------------------------------------
# Stage cards: the collapse control


def _reveal_stage_panel(page):
    """Show the simulation-parameter panel, which the landing page hides.

    The stage cards live inside a container the workflow only reveals several
    steps in. Driving the whole workflow to reach them would make this a
    slow end-to-end test of something else, so the panel is unhidden directly:
    what is under test is the toggle, not what decides to show the panel.
    """
    from selenium.webdriver.common.by import By

    # Two layers hide it: the step panel, which the wizard swaps, and the
    # coarse-grained variant inside it. Both are unhidden explicitly rather
    # than by walking ancestors, so a change in structure fails loudly here
    # instead of silently leaving the panel invisible.
    revealed = page.execute_script(
        """
        const panel = document.getElementById('panel-simparams');
        const inner = document.getElementById('cg-simparams');
        if (!panel || !inner) { return false; }
        panel.classList.add('active');
        panel.classList.remove('hidden');
        panel.style.display = 'block';
        inner.classList.remove('hidden');
        return true;
        """
    )
    assert revealed, "the simulation-parameter panel is no longer where this test expects"
    toggles = [
        element
        for element in page.find_elements(By.CSS_SELECTOR, "#cg-simparams .sim-stage-toggle")
        if element.is_displayed()
    ]
    assert toggles, "the stage panel was revealed but no toggle became visible"
    return toggles[0]


def test_activating_a_stage_toggle_changes_its_expanded_state(page):
    """The source check could only see that the attribute is written somewhere."""
    toggle = _reveal_stage_panel(page)
    before = toggle.get_attribute("aria-expanded")
    assert before in {"true", "false"}

    toggle.click()
    after = toggle.get_attribute("aria-expanded")
    assert after != before, "aria-expanded did not change when the toggle was activated"

    toggle.click()
    assert toggle.get_attribute("aria-expanded") == before


def test_a_stage_toggle_controls_the_element_it_names(page):
    """aria-controls must point at a real element whose visibility follows."""
    from selenium.webdriver.common.by import By

    toggle = _reveal_stage_panel(page)
    controlled_id = toggle.get_attribute("aria-controls")
    assert controlled_id

    body = page.find_element(By.ID, controlled_id)
    expanded_before = toggle.get_attribute("aria-expanded") == "true"
    assert body.is_displayed() == expanded_before

    toggle.click()
    assert body.is_displayed() == (not expanded_before)


def test_stage_toggles_are_reachable_by_keyboard(page):
    """A div with a click handler looks identical in source and is not focusable."""
    toggle = _reveal_stage_panel(page)
    assert toggle.tag_name.lower() == "button"

    page.execute_script("arguments[0].focus();", toggle)
    focused = page.execute_script("return document.activeElement === arguments[0];", toggle)
    assert focused, "the toggle could not take keyboard focus"


# --------------------------------------------------------------------------
# Labels: association, not spelling


def test_every_labelled_control_resolves_to_an_existing_input(page):
    """A label whose `for` names nothing is invisible to a source search."""
    orphans = page.execute_script(
        """
        const missing = [];
        document.querySelectorAll('label[for]').forEach(function (label) {
          const target = document.getElementById(label.htmlFor);
          if (!target) { missing.push(label.htmlFor); }
        });
        return missing;
        """
    )
    assert orphans == [], f"labels point at ids that do not exist: {orphans}"


def test_form_controls_on_the_first_screen_have_an_accessible_name(page):
    """Every visible control must be identifiable to someone not seeing it."""
    unnamed = page.execute_script(
        """
        const unnamed = [];
        document.querySelectorAll('input, select, textarea').forEach(function (control) {
          if (control.type === 'hidden' || control.offsetParent === null) { return; }
          const labelled = document.querySelector('label[for="' + control.id + '"]');
          const named = control.getAttribute('aria-label')
            || control.getAttribute('aria-labelledby')
            || control.closest('label')
            || labelled;
          if (!named) {
            unnamed.push(control.id || control.name || control.outerHTML.slice(0, 60));
          }
        });
        return unnamed;
        """
    )
    assert unnamed == [], f"controls without an accessible name: {unnamed}"


# --------------------------------------------------------------------------
# Duplicate ids: getElementById silently returns the first


def test_the_page_has_no_duplicate_element_ids(page):
    """Duplicates make getElementById return whichever came first, silently."""
    duplicates = page.execute_script(
        """
        const seen = {};
        const duplicates = [];
        document.querySelectorAll('[id]').forEach(function (element) {
          if (seen[element.id]) { duplicates.push(element.id); }
          seen[element.id] = true;
        });
        return duplicates;
        """
    )
    assert duplicates == [], f"duplicate ids: {duplicates}"


# --------------------------------------------------------------------------
# Modals: hidden ones must be inert


def test_hidden_modals_are_not_reachable_by_keyboard(page):
    """A modal left in the tab order traps focus in an invisible dialog."""
    reachable = page.execute_script(
        """
        const offenders = [];
        document.querySelectorAll('.modal-overlay.hidden').forEach(function (modal) {
          const focusable = modal.querySelectorAll(
            'a[href], button, input, select, textarea, [tabindex]'
          );
          focusable.forEach(function (element) {
            if (element.offsetParent !== null) { offenders.push(modal.id); }
          });
        });
        return offenders;
        """
    )
    assert reachable == [], f"hidden modals expose focusable controls: {set(reachable)}"


def test_modals_declare_dialog_semantics(page):
    incomplete = page.execute_script(
        """
        const bad = [];
        document.querySelectorAll('.modal-overlay').forEach(function (modal) {
          if (modal.getAttribute('role') !== 'dialog'
              || modal.getAttribute('aria-modal') !== 'true'
              || !modal.getAttribute('aria-labelledby')) {
            bad.push(modal.id || '(unnamed)');
          }
        });
        return bad;
        """
    )
    assert incomplete == [], f"modals missing dialog semantics: {incomplete}"


def test_each_modal_label_target_exists(page):
    """aria-labelledby pointing at a missing id leaves the dialog unnamed."""
    dangling = page.execute_script(
        """
        const bad = [];
        document.querySelectorAll('.modal-overlay[aria-labelledby]').forEach(function (modal) {
          if (!document.getElementById(modal.getAttribute('aria-labelledby'))) {
            bad.push(modal.id || '(unnamed)');
          }
        });
        return bad;
        """
    )
    assert dangling == [], f"modals label themselves with missing ids: {dangling}"


# --------------------------------------------------------------------------
# Workflow cards


def test_workflow_cards_are_buttons_and_announce_their_selection(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    # The grid is populated by script after load, so wait rather than race it.
    WebDriverWait(page, 30).until(
        lambda driver: driver.find_elements(By.CSS_SELECTOR, "#task-grid button")
    )
    cards = page.find_elements(By.CSS_SELECTOR, "#task-grid button")
    assert cards, "no workflow cards were rendered"
    for card in cards:
        assert card.get_attribute("aria-pressed") in {"true", "false"}


# --------------------------------------------------------------------------
# Contrast: computed from what the browser actually paints


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255.0
        channels.append(
            channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground, background) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _parse_rgb(value: str) -> tuple[float, float, float] | None:
    import re

    numbers = re.findall(r"[\d.]+", value or "")
    if len(numbers) < 3:
        return None
    return tuple(float(number) for number in numbers[:3])


def test_visible_text_meets_the_contrast_floor(page):
    """Computed from the painted colours, not from a colour written in the CSS.

    The previous check asserted that a particular hex string appeared in the
    stylesheet, which says nothing about what a reader sees: the rule may be
    overridden, the element may sit on a different background, and any
    equivalent colour spelling breaks the assertion while changing nothing.
    """
    samples = page.execute_script(
        """
        const out = [];
        const nodes = document.querySelectorAll('p, h1, h2, h3, label, button, .hint');
        nodes.forEach(function (node) {
          if (node.offsetParent === null) { return; }
          if (!node.textContent.trim()) { return; }
          const style = getComputedStyle(node);
          if (parseFloat(style.opacity) < 0.99) { return; }
          let ancestor = node;
          let background = 'rgba(0, 0, 0, 0)';
          while (ancestor) {
            const value = getComputedStyle(ancestor).backgroundColor;
            if (value && !value.startsWith('rgba(0, 0, 0, 0')) { background = value; break; }
            ancestor = ancestor.parentElement;
          }
          out.push({
            text: node.textContent.trim().slice(0, 40),
            size: parseFloat(style.fontSize),
            weight: style.fontWeight,
            color: style.color,
            background: background === 'rgba(0, 0, 0, 0)' ? 'rgb(255, 255, 255)' : background,
          });
        });
        return out;
        """
    )
    assert samples, "no visible text was found to measure"

    failures = []
    for sample in samples:
        foreground = _parse_rgb(sample["color"])
        background = _parse_rgb(sample["background"])
        if foreground is None or background is None:
            continue
        ratio = _contrast_ratio(foreground, background)
        large = sample["size"] >= 24 or (
            sample["size"] >= 18.66 and str(sample["weight"]) in {"700", "bold", "800", "900"}
        )
        floor = 3.0 if large else 4.5
        if ratio < floor:
            failures.append(f"{sample['text']!r}: {ratio:.2f} < {floor}")

    assert failures == [], "text below the WCAG AA contrast floor:\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# Choosing a workflow enters the wizard


def _await_task_grid(page, task_id):
    """Wait for the landing grid, which is populated from /api/task-types."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions
    from selenium.webdriver.support.ui import WebDriverWait

    return WebDriverWait(page, 20).until(
        expected_conditions.visibility_of_element_located(
            (By.CSS_SELECTOR, f'.task-card[data-task-id="{task_id}"]')
        )
    )


def _choose_workflow(page, task_id):
    """Click a workflow card and wait for the wizard to open on step 1."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions
    from selenium.webdriver.support.ui import WebDriverWait

    card = _await_task_grid(page, task_id)
    page.execute_script("arguments[0].click()", card)
    WebDriverWait(page, 20).until(
        expected_conditions.invisibility_of_element_located((By.ID, "task-grid"))
    )


def test_choosing_a_workflow_replaces_the_task_grid_with_step_one(page):
    """The wizard replaces the landing page; it does not append to it.

    A stray close tag once let #task-grid escape the panel that hides it, so
    every workflow opened with all ten task cards still stacked above step 1.
    Nothing in the JavaScript was wrong, which is why only driving the page
    finds it.
    """
    from selenium.webdriver.common.by import By

    _await_task_grid(page, "pure-membrane")
    assert page.find_element(By.ID, "task-grid").is_displayed()

    _choose_workflow(page, "pure-membrane")

    assert not page.find_element(By.ID, "task-grid").is_displayed()
    task_panel = page.find_element(By.ID, "panel-task-type")
    assert "active" not in task_panel.get_attribute("class").split()
    assert not task_panel.is_displayed()

    open_panels = page.execute_script(
        "return Array.from(document.querySelectorAll('.panel'))"
        "  .filter(p => p.offsetParent !== null).map(p => p.id);"
    )
    assert open_panels == ["panel-forcefield"], open_panels


# --------------------------------------------------------------------------
# Check progress


_PROGRESS_STATE = """
  var el = document.querySelector('#step-progress-' + arguments[0]);
  if (!el) return null;
  var bar = el.querySelector('.step-progress-bar');
  return {
    state: el.getAttribute('data-state'),
    percent: el.querySelector('.step-progress-percent').textContent,
    phase: el.querySelector('.step-progress-phase').textContent,
    elapsed: el.querySelector('.step-progress-elapsed').textContent,
    barColour: getComputedStyle(bar).backgroundColor,
    animation: getComputedStyle(bar).animationName,
    frameColour: getComputedStyle(el).borderTopColor,
    visible: el.offsetParent !== null,
    belowButton: !!(document.getElementById(arguments[0]).compareDocumentPosition(el) & 4),
    inSamePanel: document.getElementById(arguments[0]).closest('.panel').contains(el),
  };
"""


def test_no_progress_bar_exists_until_a_check_is_clicked(page):
    from selenium.webdriver.common.by import By

    _choose_workflow(page, "pure-membrane")
    assert page.find_elements(By.CSS_SELECTOR, ".step-progress") == []


def test_a_running_check_shows_a_blue_animated_bar_under_its_button(page):
    """Drive the live start function: a fast step would race the assertion."""
    _choose_workflow(page, "pure-membrane")
    page.execute_script(
        "window.__probe = startStepProgress('forcefield', 'forcefield-check-btn',"
        " 'forcefield-check-status');"
    )
    try:
        state = page.execute_script(_PROGRESS_STATE, "forcefield-check-btn")
        assert state is not None, "no bar was created"
        assert state["state"] == "running"
        assert state["visible"] and state["belowButton"] and state["inSamePanel"]
        assert state["barColour"] == "rgb(37, 99, 235)", state["barColour"]
        assert state["animation"] == "step-progress-roll"
        assert state["phase"].startswith("Starting")
        assert state["elapsed"].endswith("s")
    finally:
        page.execute_script("finishStepProgress(window.__probe, true);")


@requires_lfs_assets
def test_a_finished_check_turns_green_and_reports_its_total_time(page):
    """A real Check, end to end: click, wait, read what the user would see.

    Every step enters the custom-lipid scope, which materialises the prebuilt
    lipid library, so this needs the Git LFS assets that CI does not fetch.
    The bar itself is checked without them by the two tests above.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions
    from selenium.webdriver.support.ui import WebDriverWait

    # Entering the panel starts compatibility loading asynchronously. A scripted
    # click on its disabled button is a no-op, even though the panel is visible.
    # Hold the real response to reproduce that ordering deterministically.
    page.execute_script("""
      window.__realCompatibilityFetch = window.fetch;
      window.__heldCompatibility = [];
      window.fetch = (url, ...args) => String(url).includes('/forcefield-compatibility/')
        ? new Promise(resolve => __heldCompatibility.push(
            () => resolve(__realCompatibilityFetch(url, ...args))))
        : __realCompatibilityFetch(url, ...args);
    """)
    _choose_workflow(page, "pure-membrane")
    WebDriverWait(page, 20).until(
        lambda driver: driver.execute_script("return __heldCompatibility.length > 0")
    )
    assert not page.find_element(By.ID, "forcefield-check-btn").is_enabled()
    page.execute_script("document.getElementById('forcefield-check-btn').click()")
    assert page.execute_script(_PROGRESS_STATE, "forcefield-check-btn") is None
    page.execute_script("""
      window.fetch = __realCompatibilityFetch;
      __heldCompatibility.forEach(release => release());
    """)
    WebDriverWait(page, 30).until(
        expected_conditions.element_to_be_clickable((By.ID, "forcefield-check-btn"))
    ).click()

    state = WebDriverWait(page, 60).until(
        lambda driver: (
            (found := driver.execute_script(_PROGRESS_STATE, "forcefield-check-btn"))
            and found["state"] in ("done", "error")
            and found
        )
    )
    assert state["state"] == "done", state
    assert state["percent"] == "100%"
    assert state["phase"] == "Complete"
    assert state["elapsed"].endswith("s total")
    # data-state changes before the 240 ms CSS colour transition finishes.
    state = WebDriverWait(page, 2).until(
        lambda driver: (
            (found := driver.execute_script(_PROGRESS_STATE, "forcefield-check-btn"))
            and found["barColour"] == "rgb(22, 163, 74)"
            and found["frameColour"] == "rgb(134, 239, 172)"
            and found
        )
    )
    assert state["barColour"] == "rgb(22, 163, 74)", state["barColour"]
    assert state["frameColour"] == "rgb(134, 239, 172)", state["frameColour"]
    assert state["animation"] == "none"

    status = page.find_element(By.ID, "forcefield-check-status").text
    assert status.startswith("· protein ") and " / lipid " in status and " / water " in status
    assert "s total" not in status


# --------------------------------------------------------------------------
# Viewer: a visibility change is not a reload


# Everything redrawPDBViewerWithChainFilter calls on a viewer. A real 3Dmol
# viewer needs WebGL, which headless Firefox does not provide here or in CI, so
# the decision under test -- reload, or only restyle -- is driven against a
# stand-in implementing exactly this surface. That the stand-in matches the
# real thing was confirmed by running the same toggles against a live viewer:
# two redraws, zero addModel calls, and the ligand's 41 atoms hidden and shown.
_FAKE_VIEWER = """
window.__fake = {
  adds: 0, removes: 0, zooms: 0, styles: [], __gmxLoadedPdb: arguments[0],
  removeAllModels: function() { this.removes++; },
  addModel: function(pdb) { this.adds++; },
  setStyle: function(sel, style) { this.styles.push([sel, style]); },
  zoomTo: function() { this.zooms++; },
  render: function() {},
  setSlab: function() {},
  selectedAtoms: function() { return []; }
};
window._pdbViewer = window.__fake;
window.state.pdbInfo = {pdb_content: arguments[1], small_molecules: []};
_chainState = {A: {included: true}, B: {included: true}};
_smallMolState = {};
redrawPDBViewerWithChainFilter();
var f = window.__fake;
return {adds: f.adds, removes: f.removes, zooms: f.zooms,
        styleCalls: f.styles.length, tracked: f.__gmxLoadedPdb};
"""

_PDB = "ATOM      1  N   ALA A   1       0.000   0.000   0.000\nEND"


def test_an_unchanged_structure_is_restyled_rather_than_reparsed(page):
    """Ticking a chain changes what is shown, not what is loaded.

    The redraw ran removeAllModels() then addModel() on every tick, re-parsing
    the whole structure and rebuilding its geometry to hide one chain.
    """
    _choose_workflow(page, "membrane-bilayer")
    result = page.execute_script(_FAKE_VIEWER, _PDB, _PDB)
    assert result["adds"] == 0, "the structure was re-parsed for a visibility change"
    assert result["removes"] == 0
    assert result["styleCalls"] > 0, "nothing was restyled either"


def test_an_unchanged_structure_keeps_the_camera(page):
    """Re-zooming would throw away the user's view on every checkbox tick."""
    _choose_workflow(page, "membrane-bilayer")
    assert page.execute_script(_FAKE_VIEWER, _PDB, _PDB)["zooms"] == 0


def test_a_changed_structure_is_loaded_and_framed(page):
    """Reuse is keyed on the content, not assumed."""
    _choose_workflow(page, "membrane-bilayer")
    result = page.execute_script(_FAKE_VIEWER, "A DIFFERENT STRUCTURE", _PDB)
    assert result["adds"] == 1, "a genuinely new structure was not loaded"
    assert result["removes"] == 1
    assert result["zooms"] == 1, "a new structure should be framed"
    assert result["tracked"] == _PDB, "the viewer did not record what it now holds"


def test_the_wizard_state_is_reachable_from_outside_the_script(page):
    """`const` at script scope is invisible to anything driving the page."""
    assert page.execute_script("return !!window.state && Array.isArray(window.state.wizardSteps)")


# --------------------------------------------------------------------------
# The force-field dropdown


def test_a_force_field_that_is_not_installed_is_offered_but_not_selectable(page):
    """Greyed out and labelled, rather than hidden.

    Hiding it makes a capability the project has look like one it lacks, and
    leaves a user who was told to select it with nowhere to look. The list is
    stubbed rather than measured: what is installed on the machine running the
    suite is not the behaviour under test.
    """
    rows = page.execute_script(
        """
        window._forceFieldOptions = [
          {name: 'amber14sb', label: 'AMBER ff14SB', installed: true, water_model: 'tip3p'},
          {name: 'notyet', label: 'Something Unbuilt', installed: false, water_model: 'tip3p'}
        ];
        renderProteinForceFieldOptions();
        var select = document.getElementById('ff-protein');
        return {
          options: Array.from(select.options).map(function(option) {
            return {
              value: option.value,
              label: option.textContent,
              disabled: option.disabled,
              title: option.title
            };
          }),
          selected: select.value
        };
        """
    )
    by_value = {option["value"]: option for option in rows["options"]}
    assert set(by_value) == {"amber14sb", "notyet"}

    unavailable = by_value["notyet"]
    assert unavailable["disabled"] is True
    assert unavailable["label"].endswith("Coming Soon")
    assert "install-local.sh" in unavailable["title"], (
        "the tooltip has to say what would make it available"
    )

    available = by_value["amber14sb"]
    assert available["disabled"] is False
    assert "Coming Soon" not in available["label"]
    assert rows["selected"] == "amber14sb"


def test_the_label_does_not_repeat_what_it_already_says(page):
    """The catalog label carries the release; appending produced "(legacy) — legacy"."""
    label = page.execute_script(
        """
        window._forceFieldOptions = [
          {name: 'old', label: 'AMBER ff99SB (legacy)', installed: true, legacy: true,
           water_model: 'tip3p'}
        ];
        renderProteinForceFieldOptions();
        return document.getElementById('ff-protein').options[0].textContent;
        """
    )
    assert label == "AMBER ff99SB (legacy)"


def test_the_selection_falls_back_to_something_that_can_actually_build(page):
    """A resumed task may name a force field this deployment never installed."""
    selected = page.execute_script(
        """
        window._forceFieldOptions = [
          {name: 'gone', label: 'Not Installed Here', installed: false, water_model: 'tip3p'},
          {name: 'present', label: 'Installed', installed: true, water_model: 'tip3p'}
        ];
        renderProteinForceFieldOptions();
        document.getElementById('ff-protein').value = 'gone';
        renderProteinForceFieldOptions();
        return document.getElementById('ff-protein').value;
        """
    )
    assert selected == "present"

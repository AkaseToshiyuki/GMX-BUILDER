"""Does the page actually come up?

The browser client is checked almost entirely by asserting that strings appear
in its source. Those assertions break on harmless refactors and, more
importantly, pass on a page that is broken: a script that throws while loading
leaves every later script unexecuted, and no source-text search can see that.

The scripts are ordered classic files sharing one global scope, so adding a
part or reordering initialisation is exactly the change that can white-screen
the page while the whole Python suite stays green.

This starts the real application, drives a headless browser at it, and fails on
any uncaught error or console error during load. It boots the server itself
rather than requiring GMXBUILDER_BROWSER_URL, so it can run unattended; it
still skips when a browser is unavailable, which is the case on a machine
without geckodriver.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


EXPECTED_SCRIPTS = [
    "appearance.js",
    "constants.js",
    "http.js",
    "viewer_style.js",
    "checkpoint_viewer.js",
    "final_review.js",
    "ions.js",
    "resource_queue.js",
    "app.js",
    "app_parts/custom_lipids.js",
    "app_parts/structure_processing.js",
    "app_parts/simulation.js",
    "app_parts/system_verification.js",
]


def test_page_scripts_complete_in_order_and_workflows_are_interactive(page):
    """One real load covers script completion, ordering and usable task cards."""
    from selenium.webdriver.common.by import By

    assert page.execute_script("return window.__gmxbuilderLoaded || []") == EXPECTED_SCRIPTS
    assert any(
        card.is_enabled() for card in page.find_elements(By.CSS_SELECTOR, "#task-grid button")
    )
    assert not page.execute_script("return document.getElementById('task-grid').inert")
    assert page.execute_script("return window.__browserErrors") == []

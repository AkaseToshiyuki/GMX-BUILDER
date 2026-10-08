"""Progressive startup and optional-library behavior in a real browser."""

import pytest
from selenium.webdriver.support.ui import WebDriverWait

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_workflow_entry_works_while_parameter_catalog_is_preparing(
    browser, live_server, monkeypatch
):
    from gmxbuilder.web.server_parts import option_catalog

    class Pending:
        def read(self, **kwargs):
            return {
                "status": "checking",
                "lipids": [],
                "force_fields": [],
                "water_models": [],
                "library_version": 4,
                "entries": [],
            }

    monkeypatch.setattr(option_catalog, "get_catalog", lambda: Pending())
    browser.get("about:blank")
    browser.get(live_server)
    wait = WebDriverWait(browser, 10)
    wait.until(
        lambda d: d.execute_script(
            "return !document.getElementById('task-grid').inert && initComputeQueueStatus._done"
        )
    )
    assert not browser.execute_script("return window._optionsReady")
    browser.find_element("css selector", '[data-task-id="membrane-bilayer"]').click()
    wait.until(
        lambda d: d.execute_script(
            "return document.getElementById('panel-input').classList.contains('active')"
        )
    )
    assert browser.find_elements("id", "options-load-status")
    browser.get("about:blank")


def test_homepage_defers_optional_libraries_and_shared_load_is_single(page):
    assert page.execute_script("return typeof $3Dmol") == "undefined"
    assert not page.execute_script(
        "return performance.getEntriesByType('resource')"
        ".some(r=>/3Dmol|smiles-drawer/.test(r.name))"
    )
    result = page.execute_async_script("""
      const done=arguments[0];
      Promise.all([GMXAssets.viewer(),GMXAssets.viewer()]).then(()=>done({
        loaded:typeof $3Dmol==='object',
        requests:performance.getEntriesByType('resource').filter(r=>r.name.includes('3Dmol-min')).length
      })).catch(e=>done({error:String(e)}));
    """)
    assert result == {"loaded": True, "requests": 1}

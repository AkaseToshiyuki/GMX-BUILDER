"""A presentation migration must retain controls, states and a usable rollback."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def _ready(browser):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(browser, 30).until(
        lambda driver: driver.execute_script(
            "return typeof initCustomLipidModal === 'function' && "
            "initCustomLipidModal._done === true"
        )
    )


@pytest.fixture
def themed_page(browser, live_server):
    browser.get(live_server + "/?appearance=dark")
    _ready(browser)
    original_size = browser.get_window_size()
    try:
        yield browser
    finally:
        browser.execute_script("sessionStorage.removeItem('gmxbuilder-appearance')")
        browser.set_window_size(**original_size)


def test_classic_rollback_survives_workflow_navigation_and_reload(themed_page, live_server):
    page = themed_page
    page.get(live_server + "/?appearance=classic")
    _ready(page)
    assert page.execute_script("return gmxViewerBackground()") == "#ffffff"
    assert page.execute_script("return getComputedStyle(document.body).backgroundColor") == (
        "rgb(248, 250, 252)"
    )
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    page.refresh()
    _ready(page)
    assert page.execute_script("return document.documentElement.dataset.appearance") == "classic"
    page.get(live_server + "/?appearance=dark")
    _ready(page)
    assert page.execute_script("return gmxViewerBackground()") == "#ffffff"
    assert page.execute_script("return getComputedStyle(document.body).backgroundColor") == (
        "rgb(19, 24, 31)"
    )


@pytest.mark.parametrize("width", [1600, 500])
def test_every_panel_retains_the_same_visible_controls_and_states(themed_page, width):
    page = themed_page
    page.set_window_size(width, 1050)
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    result = page.execute_script("""
        const panels = [...document.querySelectorAll('.panel')];
        const sheet = document.getElementById('dark-appearance');
        const differences = [];
        const controls = () => [...document.querySelectorAll(
          '.panels button,.panels input,.panels select,.panels textarea,.panels a[href]'
        )].filter(el => el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden')
          .map(el => ({id:el.id,tag:el.tagName,text:el.textContent.trim(),type:el.type,
            value:el.value,checked:el.checked,disabled:el.disabled,
            label:el.getAttribute('aria-label'),href:el.getAttribute('href')}));
        panels.forEach(panel => {
          panels.forEach(p => p.classList.toggle('active', p === panel));
          sheet.media = 'not all';
          const classic = JSON.stringify(controls());
          sheet.media = 'all';
          const dark = JSON.stringify(controls());
          if (classic !== dark) differences.push(panel.id);
        });
        return {count: panels.length, differences};
    """)
    assert result["count"] >= 10, "the audit must visit the complete panel collection"
    assert result["differences"] == []


@pytest.mark.parametrize("width", [1600, 500])
def test_all_panels_fit_the_viewport(themed_page, width):
    page = themed_page
    page.set_window_size(width, 1050)
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    overflow = page.execute_script("""
        const panels = [...document.querySelectorAll('.panel')];
        const bad = [];
        panels.forEach(panel => {
          panels.forEach(p => p.classList.toggle('active', p === panel));
          if (document.documentElement.scrollWidth > innerWidth + 1) bad.push(panel.id);
        });
        return bad;
    """)
    assert overflow == []


def test_inline_queue_and_contact_dialog_keep_their_controls_in_dark_mode(themed_page):
    from selenium.webdriver.common.by import By

    page = themed_page
    page.set_window_size(500, 900)
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script("""
        document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
        document.getElementById('panel-simparams').classList.add('active');
        state.buildRunning=true;
        document.getElementById('progress-section').classList.remove('hidden');
        showComputeQueueStatus({task_id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',status:'queued',
          queue_position:13,queue_length:27,ahead:12,waited_seconds:138,
          estimated_wait_seconds:null,expires_at:Date.now()/1000+3600});
    """)
    assert "27 waiting / 12 ahead" in page.find_element(By.ID, "compute-queue-total").text
    assert page.find_elements(By.ID, "compute-queue-modal") == []
    assert page.find_element(By.ID, "compute-queue-copy").is_enabled()
    page.execute_script("openAccessibleModal(document.getElementById('custom-lipid-modal'))")
    assert page.find_element(By.ID, "custom-lipid-contact-link").is_displayed()
    page.find_element(By.ID, "custom-lipid-cancel-btn").click()
    assert not page.find_element(By.ID, "custom-lipid-modal").is_displayed()

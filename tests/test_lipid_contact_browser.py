"""New lipid requests must open administrator guidance without computation."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


@pytest.mark.parametrize("force_field", ["amber14sb", "charmm36m", "oplsaa"])
def test_contact_is_available_without_changing_force_field_or_submitting(page, force_field):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script("return initCustomLipidModal._done === true")
    )
    page.execute_script(
        """
        const protein = document.getElementById('ff-protein');
        protein.add(new Option(arguments[0], arguments[0]));
        protein.value = arguments[0];
        const lipid = document.getElementById('ff-lipid');
        lipid.add(new Option('Existing backend', 'existing-test-backend'));
        lipid.value = 'existing-test-backend';
        state.taskId = null;
        _lipidPickerData = {lipids: [], categories: {}};
        window.contactTestRequests = [];
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {
          if (options && options.method && options.method !== 'GET') {
            window.contactTestRequests.push(String(url));
            return Promise.reject(new Error('Contact must not submit a request'));
          }
          return originalFetch.apply(this, arguments);
        };
        const anchor = document.createElement('button');
        anchor.id = 'contact-test-anchor';
        anchor.textContent = 'Choose lipid';
        const panel = document.getElementById('panel-membrane');
        panel.classList.add('active');
        panel.classList.remove('hidden');
        panel.prepend(anchor);
        anchor.scrollIntoView();
        anchor.focus();
        openLipidDropdown(anchor);
        """,
        force_field,
    )
    button = page.find_element(By.CSS_SELECTOR, ".lipid-option-custom")
    assert button.is_enabled()
    assert "Contact administrator" in button.text
    button.click()
    modal = page.find_element(By.ID, "custom-lipid-modal")
    assert modal.is_displayed()
    assert "SMILES" in modal.text
    assert "homepage or announcement board" in modal.text
    assert modal.find_elements(By.CSS_SELECTOR, "input, textarea") == []
    link = modal.find_element(By.ID, "custom-lipid-contact-link")
    assert link.get_attribute("href") == page.current_url.split("#")[0]
    assert link.get_attribute("target") == "_blank"
    assert "noopener" in link.get_attribute("rel")
    WebDriverWait(page, 5).until(lambda _: page.switch_to.active_element == link)
    link.send_keys(Keys.ESCAPE)
    assert not modal.is_displayed()
    assert page.switch_to.active_element.get_attribute("id") == "contact-test-anchor"
    assert page.execute_script("return document.getElementById('ff-protein').value") == force_field
    assert (
        page.execute_script("return document.getElementById('ff-lipid').value")
        == "existing-test-backend"
    )
    assert page.execute_script("return window.contactTestRequests") == []


def test_legacy_failure_explains_offline_policy_and_contact_dialog_can_close(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script("return initCustomLipidModal._done === true")
    )
    page.execute_async_script(
        """
        const done = arguments[arguments.length - 1];
        state.taskId = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
        window.contactTestRequests = [];
        window.fetch = async function(url, options) {
          window.contactTestRequests.push({
            url: String(url), method: (options || {}).method || 'GET'
          });
          return {ok: true, json: async () => ({lipids: [{name: 'OLD', state: 'failed'}]})};
        };
        loadTaskCustomLipids().then(done);
        """
    )
    modal = page.find_element(By.ID, "custom-lipid-modal")
    assert modal.is_displayed()
    assert "Online calculation and retry are no longer available" in modal.text
    assert "OLD (failed)" in modal.text
    page.find_element(By.ID, "custom-lipid-cancel-btn").click()
    assert not modal.is_displayed()
    assert page.execute_script("return state.customLipidBusy") is True
    requests = page.execute_script("return window.contactTestRequests")
    assert all(item["method"] == "GET" for item in requests)

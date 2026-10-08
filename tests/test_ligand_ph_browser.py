"""Exercise pH edits, async suggestion races and the actual force-field payload."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_ph_edit_invalidates_old_charge_and_discards_late_responses(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait

    # Function declarations exist before async page initialization has wired
    # the pH handlers. Drive an initialized page, as the real workflow does.
    WebDriverWait(page, 20).until(
        lambda driver: driver.execute_script("return initCustomLipidModal._done === true")
    )
    page.execute_async_script("""
        const done = arguments[0];
        state.taskId = 'ligand-ph-browser-test';
        state.taskType = {pipeline: 'solvator', visible_modules: ['forcefield']};
        document.getElementById('panel-forcefield').classList.add('active');
        document.getElementById('panel-forcefield').style.display = 'block';
        document.getElementById('ff-protein').value = 'amber14sb';
        initSimParams();
        window.phRequests = [];
        const originalFetch = window.fetch;
        window.fetch = async function(url, options) {
          if (String(url).includes('forcefield-compatibility')) {
            return {ok: true, json: async () => ({family: 'amber', ligand_names: ['AMP'],
              lipid_options: [{value: 'none', label: 'None', enabled: true}],
              ligand_options: [{value: 'gaff2', label: 'GAFF2', enabled: true}], ligands: []})};
          }
          if (String(url).includes('ligand-charge-suggestions')) {
            const pH = JSON.parse(options.body).pH;
            return new Promise(resolve => phRequests.push({pH, resolve}));
          }
          return originalFetch(url, options);
        };
        refreshForceFieldCompatibility().then(done);
    """)
    assert page.execute_script("return phRequests[0].pH;") == 7.0
    pH = page.find_element(By.ID, "ff-ligand-ph")
    pH.clear()
    pH.send_keys("13", Keys.TAB)
    WebDriverWait(page, 10).until(
        lambda d: d.execute_script("return phRequests.some(r => r.pH === 13);")
    )
    page.execute_script("""
        // Resolve by requested state, not the number of browser change events.
        phRequests.filter(r => r.pH === 13).forEach(r => r.resolve({ok:true,
          json: async () => ({pH:13, suggestions:{AMP:{status:'ok', net_charge:0,
          pH:13, formula:'C9H13N'}}})}));
        phRequests.filter(r => r.pH === 7).forEach(r => r.resolve({ok:true,
          json: async () => ({pH:7, suggestions:{AMP:{status:'ok', net_charge:1,
          pH:7, formula:'C9H14N+'}}})}));
    """)
    charge = page.find_element(By.CSS_SELECTOR, '[data-ligand-charge="AMP"]')
    WebDriverWait(page, 10).until(lambda d: charge.get_attribute("value") == "0")
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield;")
    assert config["ligand_pH"] == 13.0
    assert config["ligand_charges"] == {"AMP": 0}
    assert float(page.find_element(By.ID, "proc-pH").get_attribute("value")) == 13.0
    pH.clear()
    assert charge.get_attribute("value") == ""
    assert "solution pH" in page.execute_script("""
        try { buildModuleConfig('forcefield'); return 'unexpected success'; }
        catch (e) { return e.message; }
    """)
    charge.send_keys("1")  # An explicit user override survives the next pH edit.
    pH.send_keys("7.4", Keys.TAB)
    WebDriverWait(page, 10).until(
        lambda d: d.execute_script("return phRequests.some(r => r.pH === 7.4);")
    )
    assert charge.get_attribute("value") == "1"
    assert (
        page.execute_script("return buildModuleConfig('forcefield').forcefield.ligand_pH;") == 7.4
    )
    page.execute_script("""
        const input = document.getElementById('proc-pH');
        input.value = '8'; input.dispatchEvent(new Event('input', {bubbles:true}));
    """)
    assert pH.get_attribute("value") == "8"
    assert page.execute_script("return buildModuleConfig('forcefield').forcefield.ligand_pH;") == 8


def test_charmm_records_ph_without_silently_changing_explicit_state(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 10).until(lambda d: d.execute_script("return !!window._forceFieldOptions;"))
    page.execute_script("""
        state.taskId = 'charmm-ph-browser-test';
        state.taskType = {pipeline:'solvator', visible_modules:['forcefield']};
        document.getElementById('ff-protein').value = 'charmm36m';
        document.getElementById('panel-forcefield').style.display = 'block';
        document.getElementById('ff-ligand').innerHTML =
          '<option value="charmm_compat">Local</option>';
        window._ffCompatibility = {ligand_names:['AMP']};
        initSimParams();
        renderLigandChargeInputs();
        setLigandEnvironmentPH(7.4);
    """)
    from selenium.webdriver.support.ui import Select

    Select(
        page.find_element(By.CSS_SELECTOR, '[data-charmm-identity-source="AMP"]')
    ).select_by_value("smiles")
    page.find_element(By.CSS_SELECTOR, '[data-charmm-smiles="AMP"]').send_keys(
        "C[C@H]([NH3+])Cc1ccccc1"
    )
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield;")
    assert config["ligand_pH"] == 7.4
    assert config["charmm_compat_smiles"]["AMP"] == "C[C@H]([NH3+])Cc1ccccc1"
    assert "overrides retain their supplied state" in page.execute_script(
        "return document.getElementById('ff-ligand-charges').textContent;"
    )

"""Exercise the actual CHARMM selector, local inputs, upload toggle and payload."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_local_default_inputs_research_toggle_and_upload_preservation(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait

    page.execute_async_script("""
        const done = arguments[arguments.length - 1];
        window.identityErrors = [];
        window.addEventListener('error', event => identityErrors.push(event.message));
        state.taskId = 'local-compat-browser-test';
        state.taskType = {pipeline: 'solvator', visible_modules: ['forcefield']};
        const panel = document.getElementById('panel-forcefield');
        panel.classList.add('active'); panel.style.display = 'block';
        document.getElementById('ff-protein').value = 'charmm36m';
        initSimParams();
        const originalFetch = window.fetch;
        window.fetch = async function(url, options) {
          if (String(url).includes('forcefield-compatibility')) {
            return {ok: true, json: async () => ({family: 'charmm', ligand_names: ['LIG'],
              lipid_options: [{value: 'none', label: 'None', enabled: true}],
              ligand_options: [
                {value: 'charmm_compat', label: 'Local CHARMM-compatible builder', enabled: true},
                {value: 'cgenff', label: 'CGenFF / ParamChem import', enabled: true}],
                ligands: []})};
          }
          return originalFetch(url, options);
        };
        refreshForceFieldCompatibility().then(done);
    """)
    selector = Select(page.find_element(By.ID, "ff-ligand"))
    assert selector.first_selected_option.get_attribute("value") == "charmm_compat"
    assert len(selector.options) == 2
    Select(
        page.find_element(By.CSS_SELECTOR, '[data-charmm-identity-source="LIG"]')
    ).select_by_value("smiles")
    smiles = page.find_element(By.CSS_SELECTOR, '[data-charmm-smiles="LIG"]')
    smiles.send_keys("CCCCO")
    checkbox = page.find_element(By.ID, "ff-charmm-research")
    assert not checkbox.is_selected()
    checkbox.click()
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield;")
    assert config["charmm_compat_smiles"] == {"LIG": "CCCCO"}
    assert config["charmm_compat_allow_research"] is True
    selector.select_by_value("cgenff")
    WebDriverWait(page, 5).until(
        lambda _: page.find_elements(By.CSS_SELECTOR, '[data-cgenff-mol2="LIG"]')
    )
    assert page.find_elements(By.CSS_SELECTOR, '[data-cgenff-mol2="LIG"]'), page.execute_script("""
        return {errors: identityErrors, source: document.getElementById('ff-ligand').value,
          contents: document.getElementById('ff-ligand-charges').innerHTML};
    """)
    assert not page.find_elements(By.ID, "ff-charmm-research")
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield;")
    assert config["charmm_compat_smiles"] == {}
    assert config["charmm_compat_allow_research"] is False
    page.execute_async_script(
        "const done=arguments[0]; refreshForceFieldCompatibility().then(done);"
    )
    assert selector.first_selected_option.get_attribute("value") == "cgenff"
    selector.select_by_value("charmm_compat")
    page.execute_script("_cgenffUploads.LIG = {ready: true, force_field: 'charmm36m'};")
    assert (
        page.execute_script("return buildModuleConfig('forcefield').forcefield.cgenff_parameters;")
        == {}
    )
    assert (
        page.find_element(By.CSS_SELECTOR, '[data-charmm-smiles="LIG"]').get_attribute("value")
        == "CCCCO"
    )
    assert page.find_element(By.ID, "ff-charmm-research").is_selected()


def test_real_mol2_upload_is_available_and_selected_in_payload(page, tmp_path, monkeypatch):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait

    from tests.test_ligand_identity import ethanol_mol2, task

    task_id = task(tmp_path, monkeypatch)
    page.execute_script(
        """
        state.taskId = arguments[0];
        state.taskType = {pipeline:'solvator', visible_modules:['forcefield']};
        document.getElementById('panel-forcefield').style.display = 'block';
        document.getElementById('ff-protein').value = 'charmm36m';
        document.getElementById('ff-ligand').innerHTML =
          '<option value="charmm_compat">Local</option>';
        window._ffCompatibility = {ligand_names:['LIG']};
        initSimParams(); renderLigandChargeInputs();
    """,
        task_id,
    )
    Select(
        page.find_element(By.CSS_SELECTOR, '[data-charmm-identity-source="LIG"]')
    ).select_by_value("mol2")
    file = page.find_element(By.CSS_SELECTOR, '[data-charmm-mol2="LIG"]')
    assert file.is_displayed()

    def select_file(contents):
        # Firefox may run in a separate /tmp namespace. Supply a real browser
        # File and dispatch the same change event as the native file picker.
        page.execute_script(
            """
            const transfer = new DataTransfer();
            transfer.items.add(new File([arguments[1]], 'ligand.mol2', {type:'chemical/x-mol2'}));
            arguments[0].files = transfer.files;
            arguments[0].dispatchEvent(new Event('change', {bubbles:true}));
        """,
            file,
            contents,
        )

    select_file(ethanol_mol2().decode())
    WebDriverWait(page, 15).until(
        lambda _: (
            "supplied MOL2"
            in page.find_element(By.CSS_SELECTOR, '[data-ligand-identity-status="LIG"]').text
        )
    )
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield")
    assert config["charmm_compat_smiles"] == {}
    assert config["charmm_compat_mol2"]["LIG"]["uploaded"] is True
    assert len(config["charmm_compat_mol2"]["LIG"]["sha256"]) == 64
    select_file("not a molecule")
    WebDriverWait(page, 15).until(
        lambda _: (
            "MOL2 must contain"
            in page.find_element(By.CSS_SELECTOR, '[data-ligand-identity-status="LIG"]').text
        )
    )
    assert (
        page.execute_script(
            "return buildModuleConfig('forcefield').forcefield.charmm_compat_mol2.LIG.sha256"
        )
        is None
    )


def test_automatic_default_and_late_identification_cannot_replace_override(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait

    page.execute_script("""
        state.taskId = 'identity-race';
        state.taskType = {pipeline:'solvator', visible_modules:['forcefield']};
        document.getElementById('panel-forcefield').style.display = 'block';
        document.getElementById('ff-protein').value = 'charmm36m';
        document.getElementById('ff-ligand').innerHTML =
          '<option value="charmm_compat">Local</option>';
        window._ffCompatibility = {ligand_names:['LIG']};
        window.identityRequests = [];
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {
          if (String(url).includes('/api/ligand-chemistry/')) {
            return new Promise(resolve => identityRequests.push({
              body:JSON.parse(options.body), resolve}));
          }
          return originalFetch(url, options);
        };
        initSimParams(); renderLigandChargeInputs();
    """)
    mode = Select(page.find_element(By.CSS_SELECTOR, '[data-charmm-identity-source="LIG"]'))
    assert mode.first_selected_option.get_attribute("value") == "auto"
    assert not page.find_element(By.CSS_SELECTOR, '[data-charmm-smiles="LIG"]').is_displayed()
    config = page.execute_script("return buildModuleConfig('forcefield').forcefield;")
    assert config["charmm_compat_smiles"] == {}
    assert config["charmm_compat_mol2"] == {}
    WebDriverWait(page, 10).until(
        lambda _: page.execute_script("return identityRequests.length > 0")
    )
    page.execute_script("setLigandEnvironmentPH(8.0)")
    WebDriverWait(page, 10).until(
        lambda _: page.execute_script("return identityRequests.length >= 2")
    )
    page.execute_script("""
        identityRequests.at(-1).resolve({ok:true,json:async()=>({ligands:{LIG:{
          status:'needs_input',error:'ambiguous at new pH'}}})});
    """)

    def status():
        return page.find_element(By.CSS_SELECTOR, '[data-ligand-identity-status="LIG"]').text

    WebDriverWait(page, 10).until(lambda _: "ambiguous at new pH" in status())
    page.execute_script("""
        identityRequests[0].resolve({ok:true,json:async()=>({ligands:{LIG:{status:'ok',smiles:'OLD',source:'coordinate_perception',net_charge:0}}})});
    """)
    assert "OLD" not in status()
    mode.select_by_value("smiles")
    page.find_element(By.CSS_SELECTOR, '[data-charmm-smiles="LIG"]').send_keys("CCO")
    assert page.execute_script(
        "return buildModuleConfig('forcefield').forcefield.charmm_compat_smiles"
    ) == {"LIG": "CCO"}
    Select(
        page.find_element(By.CSS_SELECTOR, '[data-charmm-identity-source="LIG"]')
    ).select_by_value("auto")
    assert (
        page.execute_script(
            "return buildModuleConfig('forcefield').forcefield.charmm_compat_smiles"
        )
        == {}
    )

"""Browser checks for explicit uploaded-chemistry decisions and stale checks."""

import pytest
from selenium.webdriver.support.ui import WebDriverWait

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def setup_protein(page, sequence):
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script(
        """
      document.getElementById('ff-protein').value='amber14sb';
      state.pdbInfo={sequences:[{chain_id:'A',residues:arguments[0]}]};
      loadProcResidues();
    """,
        [{"resid": i + 1, "resname": name, "is_protein": True} for i, name in enumerate(sequence)],
    )
    page.execute_async_script("reloadModificationCatalog().then(arguments[0])")
    WebDriverWait(page, 10).until(
        lambda d: d.execute_script(
            "return !document.querySelector('.proc-nter-sel option[value=ACE]').disabled"
        )
    )


def test_patch_picker_tracks_current_residue_when_requests_finish_out_of_order(page):
    setup_protein(page, ["PRO", "TYR", "THR", "CYS"])
    page.execute_script(
        r"""
        window.patchReleases = {};
        const originalFetch = window.fetch;
        window.fetch = (url, options) => {
          const match = String(url).match(/\/api\/patches\/(PRO|TYR|THR|CYS)\?/);
          if (!match) return originalFetch(url, options);
          return new Promise(resolve => { window.patchReleases[match[1]] = resolve; });
        };
        """
    )

    def click_residue(index):
        page.execute_script(
            "document.querySelector('.proc-mod-res[data-idx=\"' + arguments[0] + '\"]').click()",
            index,
        )

    def release(residue):
        page.execute_script(
            """
            const residue = arguments[0];
            window.patchReleases[residue](new Response(JSON.stringify([{
              id: residue + '_PATCH', name: residue + ' patch', product_name: residue,
              description: 'browser regression', supported: true, charge_shift: 0
            }]), {status: 200, headers: {'Content-Type': 'application/json'}}));
            """,
            residue,
        )

    def option_text():
        return page.execute_script(
            "return document.getElementById('proc-patch-options').textContent"
        )

    click_residue(0)
    click_residue(1)
    release("TYR")
    WebDriverWait(page, 10).until(lambda _: "TYR patch" in option_text())
    release("PRO")
    page.execute_async_script("setTimeout(arguments[0], 0)")
    assert "PRO patch" not in option_text()
    assert "TYR patch" in option_text()

    click_residue(2)
    assert "TYR patch" not in option_text()
    release("THR")
    WebDriverWait(page, 10).until(lambda _: "THR patch" in option_text())
    click_residue(3)
    assert "THR patch" not in option_text()
    release("CYS")
    WebDriverWait(page, 10).until(lambda _: "CYS patch" in option_text())
    assert "CYS 4" in page.execute_script(
        "return document.getElementById('proc-patch-target').textContent"
    )
    applied = page.execute_script(
        """
        document.querySelector('#proc-patch-options .proc-patch-option').click();
        return serializeStructureModifications().map(mod => ({
          index: mod.index, patch_id: mod.patch_id
        }));
        """
    )
    assert applied == [{"index": 3, "patch_id": "CYS_PATCH"}]


def test_modification_edits_revoke_checks_and_offer_explicit_removal(page):
    setup_protein(page, ["ALA", "SER", "ALA"])
    page.execute_script("""
      setInputModificationReport({recognized:1,records:[{status:'recognized',normalized:true,
        chain:'A',resid:2,original_resname:'SEP',standard_resname:'SER',patch_id:'PHOS_SER'}]});
      _checkedSteps.add('structure');_checkedSteps.add('ions');
      _checkedConfig={structure:{modifications:serializeStructureModifications()}};
      document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-structure'));
    """)
    assert page.execute_script("return serializeStructureModifications()[0].target.resid") == 2
    result = page.execute_script("""
      const box=document.querySelector('#proc-upload-modification-notice input');
      const row=box.closest('label');
      const layout={display:getComputedStyle(row).display,width:getComputedStyle(box).width,
        background:getComputedStyle(row).backgroundColor};
      box.click();
      return {layout,checked:_checkedSteps.has('structure'),ions:_checkedSteps.has('ions'),
        mods:serializeStructureModifications(),
        removals:buildModuleConfig('structure').structure.input_modification_decisions};
    """)
    assert result["layout"] == {
        "display": "flex",
        "width": "18px",
        "background": "rgb(255, 243, 205)",
    }
    assert not result["checked"] and not result["ions"]
    assert result["mods"] == []
    assert result["removals"] == [
        {"action": "remove", "target": {"chain": "A", "resid": 2, "resname": "SER"}}
    ]
    page.execute_script("document.querySelector('#proc-upload-modification-notice input').click()")
    assert page.execute_script("return serializeStructureModifications()[0].target.resid") == 2
    assert (
        page.execute_script(
            "return buildModuleConfig('structure').structure.input_modification_decisions.length"
        )
        == 0
    )


def test_terminal_cap_change_invalidates_structure_check(page):
    setup_protein(page, ["SER", "ALA"])
    result = page.execute_script("""
      _checkedSteps.add('structure');
      const select=document.querySelector('.proc-nter-sel[data-chain="A"]');
      if(!select) throw new Error('Terminal selector missing');
      select.value='ACE';select.dispatchEvent(new Event('change'));
      return {checked:_checkedSteps.has('structure'),
        caps:buildModuleConfig('structure').structure.termini};
    """)
    assert not result["checked"]
    assert result["caps"]["A"]["nter"] == "ACE"


def test_resume_preserves_caps_while_capabilities_are_loading(page):
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    result = page.execute_script("""
      state.pdbInfo={sequences:[{chain_id:'A',residues:[
        {resid:1,resname:'ALA',is_protein:true},{resid:2,resname:'ALA',is_protein:true}]}]};
      loadProcResidues();
      restoreStructureProcessingConfig({termini:{A:{nter:'ACE',cter:'NME'}}});
      return buildModuleConfig('structure').structure.termini;
    """)
    assert result == {"A": {"nter": "ACE", "cter": "NME"}}

"""Browser selection, refresh and request-failure behavior; no molecular jobs."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_v4_list_unlocks_only_accepted_pair_and_blocks_on_refresh_failure(page):
    page.execute_async_script("selectTaskType('pure-membrane').then(arguments[0])")
    result = page.execute_async_script("""
      const done = arguments[0];
      (async () => {
        const original = window.fetch;
        let ready = false, fail = false, posts = 0;
        window.fetch = async (url, options) => {
          if (String(url) === '/api/lipid-library-list') {
            if (fail) throw new Error('offline');
            return {ok:true,json:async()=>({library_version:4,entries:[
              {lipid_name:'POPC',lipid_ff:'charmm36m',ready}
            ]})};
          }
          if (String(url).endsWith('/membrane') && options?.method === 'POST') {
            posts++; throw new Error('must not submit');
          }
          return original(url, options);
        };
        document.getElementById('ff-protein').value = 'charmm36m';
        document.getElementById('ff-lipid').innerHTML =
          '<option value="charmm36m">CHARMM36m</option><option value="charmm36">CHARMM36</option>';
        _mixUpper = [{name:'POPC',ratio:100}]; _asymmetric = false;
        await refreshV4LipidAvailability();
        renderLipidList('popc');
        const blocked = document.querySelector('[data-lipid-name="POPC"]').disabled;
        await checkComposition();
        ready = true;
        await refreshV4LipidAvailability();
        renderLipidList('popc');
        const unlocked = !document.querySelector('[data-lipid-name="POPC"]').disabled;
        const canBuild = !document.getElementById('check-composition-btn').disabled;
        _compositionChecked = true;
        await refreshV4LipidAvailability();
        const keptCheck = _compositionChecked;
        document.getElementById('ff-lipid').value = 'charmm36';
        renderLipidList('popc');
        const otherBlocked = document.querySelector('[data-lipid-name="POPC"]').disabled;
        _compositionChecked = false;
        _checkedSteps.add('membrane');
        fail = true;
        await refreshV4LipidAvailability();
        const failureBlocked = document.getElementById('check-composition-btn').disabled;
        const failureMessage = document.getElementById('v4-lipid-availability').textContent;
        const revokedResume = !_checkedSteps.has('membrane');
        window.fetch = original;
        done({blocked,unlocked,canBuild,keptCheck,otherBlocked,failureBlocked,failureMessage,revokedResume,posts});
      })().catch(error => done({error:String(error),stack:error.stack}));
    """)
    assert "error" not in result, result
    assert result["blocked"] and result["unlocked"] and result["canBuild"]
    assert result["otherBlocked"] and result["failureBlocked"] and result["keptCheck"]
    assert "could not be checked" in result["failureMessage"]
    assert not result["revokedResume"], "Transient polling failure must preserve the checkpoint"
    assert result["posts"] == 0

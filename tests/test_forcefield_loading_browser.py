"""Force-field controls recover from transport faults and reject obsolete results."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]

SETUP = """
  state.taskId='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  state.taskType={id:'solvator',visible_modules:['input','forcefield']};
  state.wizardSteps=['input','forcefield'];state.currentStepIdx=1;
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-forcefield'));
  document.getElementById('ff-protein').value='amber14sb';
  window._ffCompatibility=null;
  window.reportFor=family=>({family,lipid_options:[{value:family==='amber'?'lipid21':'charmm36',
    label:'Lipids',enabled:true}],ligand_options:[{value:family==='amber'?'gaff2':'charmm_compat',
    label:'Ligands',enabled:true}],ligand_names:[],ligands:[]});
  window.jsonResponse=data=>new Response(JSON.stringify(data),
    {headers:{'Content-Type':'application/json'}});
"""


def test_initial_empty_response_recovers_without_switching_force_fields(page):
    result = page.execute_async_script(
        SETUP
        + """
      const done=arguments[0];let calls=0;
      const realFetch=window.fetch;
      window.fetch=(url,...args)=>String(url).includes('/forcefield-compatibility/')
        ? Promise.resolve(++calls===1?new Response(''):jsonResponse(reportFor('amber')))
        : realFetch(url,...args);
      refreshForceFieldCompatibility().then(()=>done({calls,
        lipid:document.getElementById('ff-lipid').value,
        ligand:document.getElementById('ff-ligand').value,
        disabled:document.getElementById('forcefield-check-btn').disabled,
        status:document.getElementById('ff-compatibility-status').textContent}));
    """
    )
    assert result["calls"] == 2, result
    assert result["lipid"] == "lipid21" and result["ligand"] == "gaff2"
    assert result["disabled"] is False
    assert "Compatible family: AMBER" in result["status"]


def test_late_response_cannot_overwrite_new_force_field(page):
    result = page.execute_async_script(
        SETUP
        + """
      const done=arguments[0];let release;
      const realFetch=window.fetch;
      window.fetch=(url,options)=>{
        if(!String(url).includes('/forcefield-compatibility/'))return realFetch(url,options);
        if(JSON.parse(options.body).protein_ff==='amber14sb')return new Promise(r=>release=r);
        return Promise.resolve(jsonResponse(reportFor('charmm')));
      };
      (async()=>{
        const old=refreshForceFieldCompatibility();
        document.getElementById('ff-protein').value='charmm36';
        await refreshForceFieldCompatibility();
        release(jsonResponse(reportFor('amber')));await old;
        done({family:window._ffCompatibility?.family,
          lipid:document.getElementById('ff-lipid').value});
      })().catch(e=>done({error:String(e)}));
    """
    )
    assert result == {"family": "charmm", "lipid": "charmm36"}


def test_persistent_bad_response_has_retry_and_keeps_next_locked(page):
    from selenium.webdriver.support.ui import WebDriverWait

    result = page.execute_async_script(
        SETUP
        + """
      const done=arguments[0];window.compatibilityCalls=0;window.recovered=false;
      _checkedSteps.add('forcefield');
      const realFetch=window.fetch;
      window.fetch=(url,...args)=>String(url).includes('/forcefield-compatibility/')
        ? Promise.resolve((++window.compatibilityCalls,
          window.recovered?jsonResponse(reportFor('amber')):new Response('')))
        : realFetch(url,...args);
      refreshForceFieldCompatibility().then(()=>done({calls:window.compatibilityCalls,
        enabled:!document.getElementById('forcefield-check-btn').disabled,
        next:isCurrentStepFulfilled(),status:document.getElementById('ff-compatibility-status').textContent}));
    """
    )
    assert result["calls"] == 2
    assert not result["enabled"] and not result["next"]
    assert "empty or invalid response" in result["status"]
    assert "Unexpected end" not in result["status"]
    page.execute_script("window.recovered=true")
    page.find_element("id", "ff-compatibility-retry").click()
    WebDriverWait(page, 10).until(
        lambda p: p.execute_script(
            "return !document.getElementById('forcefield-check-btn').disabled"
        )
    )
    assert page.execute_script("return window.compatibilityCalls") == 3
    assert page.execute_script("return document.getElementById('ff-lipid').value") == "lipid21"


@pytest.mark.parametrize("failure", ["empty", "http", "network"])
def test_late_failure_does_not_clear_latest_selection_or_retry_old_request(page, failure):
    result = page.execute_async_script(
        SETUP
        + """
      const failure=arguments[0],done=arguments[1];let calls=0,release,reject;
      const realFetch=window.fetch;
      window.fetch=(url,options)=>{
        if(!String(url).includes('/forcefield-compatibility/'))return realFetch(url,options);
        if(++calls===1)return new Promise((r,j)=>{release=r;reject=j});
        return Promise.resolve(jsonResponse(reportFor('charmm')));
      };
      (async()=>{
        const old=refreshForceFieldCompatibility();
        const pendingDisabled=document.getElementById('forcefield-check-btn').disabled;
        document.getElementById('ff-protein').value='charmm36';
        await refreshForceFieldCompatibility();
        if(failure==='network')reject(new TypeError('Network error'));
        else release(new Response('',{status:failure==='http'?503:200}));
        await old;
        done({calls,pendingDisabled,family:window._ffCompatibility?.family,
          enabled:!document.getElementById('forcefield-check-btn').disabled});
      })().catch(e=>done({error:String(e)}));
    """,
        failure,
    )
    assert result == {"calls": 2, "pendingDisabled": True, "family": "charmm", "enabled": True}


def test_old_task_response_is_ignored(page):
    result = page.execute_async_script(
        SETUP
        + """
      const done=arguments[0];let release;
      window.fetch=()=>new Promise(r=>release=r);
      const old=refreshForceFieldCompatibility();
      state.taskId='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
      document.getElementById('ff-compatibility-status').textContent='New task';
      release(jsonResponse(reportFor('amber')));
      old.then(()=>done({report:window._ffCompatibility,
        status:document.getElementById('ff-compatibility-status').textContent}));
    """
    )
    assert result == {"report": None, "status": "New task"}


@pytest.mark.parametrize("body,status", [("", 200), ("<html>unavailable</html>", 502)])
def test_read_only_catalog_recovers_from_empty_or_gateway_response(page, body, status):
    result = page.execute_async_script(
        """
      const body=arguments[0],status=arguments[1],done=arguments[2];let calls=0;
      const realFetch=window.fetch;
      window.fetch=(url,...args)=>url==='/api/options'
        ? Promise.resolve(++calls===1?new Response(body,{status}):new Response('{"ready":true}'))
        : realFetch(url,...args);
      GMXHttp.json('/api/options').then(data=>done({data,calls})).catch(e=>done({error:String(e),calls}));
    """,
        body,
        status,
    )
    assert result == {"data": {"ready": True}, "calls": 2}


@pytest.mark.parametrize(
    "path,status,body",
    [
        ("/api/step/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/forcefield", 503, ""),
        ("/api/tasks", 200, ""),
        (
            "/api/forcefield-compatibility/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            409,
            '{"error":"Run Check Upload first"}',
        ),
    ],
)
def test_writes_and_domain_failures_are_not_automatically_replayed(page, path, status, body):
    result = page.execute_async_script(
        """
      const path=arguments[0],status=arguments[1],body=arguments[2],done=arguments[3];let calls=0;
      window.fetch=async()=>{calls++;return new Response(body,{status});};
      GMXHttp.json(path,{method:'POST'}).then(()=>done({unexpected:true})).catch(e=>done({calls,error:e.message}));
    """,
        path,
        status,
        body,
    )
    assert result["calls"] == 1
    assert "error" in result
    if status == 409:
        assert result["error"] == "Run Check Upload first"


@pytest.mark.parametrize(
    "loader,endpoint",
    [
        ("reloadModificationCatalog", "patches"),
        ("reloadCrosslinkCapabilities", "crosslink-capabilities"),
        ("reloadTerminalCapabilities", "terminal-capabilities"),
    ],
)
def test_related_catalogs_keep_only_the_latest_force_field(page, loader, endpoint):
    from tests.test_modification_browser import setup_protein

    setup_protein(page, ["CYS", "ALA", "CYS"])
    result = page.execute_async_script(
        SETUP
        + """
      const loader=arguments[0],endpoint=arguments[1],done=arguments[2];let release;
      const realFetch=window.fetch;
      const value=family=>endpoint==='patches'?
        [{id:family,target_residues:['ALA'],supported:family==='charmm'}]:
        endpoint==='crosslink-capabilities'?{disulfide:{supported:family==='charmm'}}:
        {ACE:{supported:family==='charmm'}};
      window.fetch=(url,options)=>{
        if(!String(url).includes('/api/'+endpoint+'?'))return realFetch(url,options);
        if(String(url).includes('amber14sb'))return new Promise(r=>release=r);
        return Promise.resolve(jsonResponse(value('charmm')));
      };
      (async()=>{
        const old=window[loader]();document.getElementById('ff-protein').value='charmm36';
        await window[loader]();release(jsonResponse(value('amber')));await old;
        done(endpoint==='patches'?document.querySelector('.proc-mod-res[data-idx="1"]').classList.contains('modifiable'):
          endpoint==='crosslink-capabilities'?!document.getElementById('proc-disulfide-add').disabled:
          !document.querySelector('.proc-nter-sel option[value=ACE]').disabled);
      })().catch(e=>done({error:String(e)}));
    """,
        loader,
        endpoint,
    )
    assert result is True, result


def test_edit_during_force_field_check_cannot_restore_old_pass(page):
    result = page.execute_async_script(
        SETUP
        + """
      const done=arguments[0];let release;
      const realFetch=window.fetch;
      window.fetch=(url,options)=>String(url).includes('/api/step/')&&String(url).endsWith('/forcefield')
        ? new Promise(r=>release=r):realFetch(url,options);
      const checking=_doCheckStep('forcefield','forcefield-check-status','forcefield-check-btn');
      document.getElementById('ff-protein').value='charmm36';resetForceFieldCheck();
      release(jsonResponse({status:'ok',metrics:{}}));
      checking.then(()=>done({checked:_checkedSteps.has('forcefield'),
        config:_checkedConfig?.forcefield||null,
        error:document.getElementById('step-feedback-forcefield-check-btn').textContent}));
    """
    )
    assert not result["checked"] and result["config"] is None
    assert "settings changed while checking" in result["error"]


@pytest.mark.parametrize("malformed", ["empty", "schema"])
def test_option_failure_is_local_and_can_be_retried(browser, live_server, monkeypatch, malformed):
    from fastapi.responses import Response
    from selenium.webdriver.support.ui import WebDriverWait

    from gmxbuilder.web.server_parts import option_catalog

    original = option_catalog.get_catalog
    calls = []

    class BrokenCatalog:
        def read(self, **kwargs):
            if not kwargs.get("options"):
                return original().read()
            calls.append(1)
            if len(calls) <= 2:
                return Response("") if malformed == "empty" else {}
            return original().read(**kwargs)

    monkeypatch.setattr(option_catalog, "get_catalog", lambda: BrokenCatalog())
    browser.get("about:blank")
    browser.get(live_server)
    wait = WebDriverWait(browser, 30)
    wait.until(lambda d: d.find_elements("css selector", "#options-load-status button"))
    assert len(calls) == 2
    assert not browser.execute_script("return document.getElementById('task-grid').inert")
    browser.find_element("css selector", "#options-load-status button").click()
    wait.until(lambda d: d.execute_script("return window._optionsReady === true"))
    assert not browser.find_elements("id", "options-load-status")
    assert browser.execute_script("return document.getElementById('ff-protein').options.length") > 0


def test_default_composition_survives_pending_availability_without_bypassing_check(page):
    result = page.execute_async_script(
        """
      const done=arguments[0];let release,calls=0;
      const realFetch=window.fetch,realAlert=window.alert;const alerts=[];
      applyV4Availability({status:"checking",library_version:4,entries:[]});
      window.alert=message=>alerts.push(message);
      window.fetch=(url,...args)=>String(url)==='/api/lipid-library-list'
        ? (++calls===1?Promise.resolve(new Response('')):new Promise(r=>release=r))
        :realFetch(url,...args);
      (async()=>{
        const loading=refreshV4LipidAvailability();
        state.taskType={id:'pure-membrane',pipeline:'pure_membrane',default_config:{membrane:{lipid_type:'POPC'}}};
        loadTaskDefaults();
        const pending={alerts:alerts.length,
          disabled:document.getElementById('check-composition-btn').disabled};
        while(!release)await new Promise(r=>setTimeout(r,20));
        release(new Response(JSON.stringify({library_version:4,entries:[]})));await loading;
        const unavailable=document.getElementById('check-composition-btn').disabled;
        selectLipid('POPC');
        window.alert=realAlert;
        done({pending,unavailable,calls,interactiveAlerts:alerts.length,
          message:document.getElementById('v4-lipid-availability').textContent});
      })().catch(e=>done({error:String(e)}));
    """
    )
    assert result["pending"] == {"alerts": 0, "disabled": True}, result
    assert result["unavailable"] and result["interactiveAlerts"] == 1
    assert result["calls"] == 2
    assert "POPC" in result["message"]


def test_verified_availability_stays_enabled_during_an_unchanged_refresh(page):
    result = page.execute_async_script(
        """
      const done=arguments[0];
      (async()=>{
        await refreshV4LipidAvailability();
        state.taskType={id:'pure-membrane',pipeline:'pure_membrane',default_config:{membrane:{lipid_type:'POPC'}}};
        loadTaskDefaults();
        const snapshot={status:'ready',library_version:4,entries:[{lipid_name:'POPC',
          ready:true,lipid_ff:selectedLipidParameterSource(),amber_mixed_ready:true}]};
        applyV4Availability(snapshot);
        const before=document.getElementById('check-composition-btn').disabled;
        let release;const realFetch=window.fetch;
        window.fetch=(url,...args)=>String(url)==='/api/lipid-library-list'
          ?new Promise(r=>release=r):realFetch(url,...args);
        const refreshing=refreshV4LipidAvailability();
        const pending=document.getElementById('check-composition-btn').disabled;
        release(new Response(JSON.stringify(snapshot)));await refreshing;
        window.fetch=realFetch;
        done({before,pending,after:document.getElementById('check-composition-btn').disabled});
      })().catch(e=>done({error:String(e)}));
    """
    )
    assert result == {"before": False, "pending": False, "after": False}

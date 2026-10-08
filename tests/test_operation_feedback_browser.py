"""Visible diagnostics and notice lifetime across independently scheduled tickets."""

import asyncio

import pytest
from fastapi.responses import JSONResponse

pytestmark = [pytest.mark.browser, pytest.mark.slow]
TASK = "b" * 32


def ready_input(page, workflow="solvator"):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda d: d.execute_script("return initComputeQueueStatus._done === true")
    )
    page.execute_async_script("selectTaskType(arguments[0]).then(arguments[1])", workflow)
    page.execute_script(
        """
        state.taskId = arguments[0];
        state.pdbInfo = {small_molecules:[]};
        _chainState = {A:{included:true}};
        document.getElementById('upload-info').classList.remove('hidden');
        document.getElementById('input-check-btn').disabled = false;
        window.__viewerRelease = null;
        window.__realPreviewLoader = window._loadStepViewerPdb;
        window._loadStepViewerPdb = () => new Promise(resolve => {
          window.__viewerRelease = resolve;
        });
        window.redrawPDBViewerWithChainFilter = () => {};
    """,
        TASK,
    )


@pytest.fixture
def staged_check(monkeypatch, live_server):
    from gmxbuilder.web import server

    routes = {r.path: r for r in server.app.routes if hasattr(r, "dependant")}
    control = {"accepted": [], "finished": set(), "delivered": set(), "failure": False}
    ids = {"filter": "c" * 32, "input": "d" * 32}
    names = {v: k for k, v in ids.items()}

    def ticket(name):
        finished = name in control["finished"]
        return {
            "task_id": TASK,
            "operation_id": ids[name],
            "status": "completed" if finished else "queued",
            "response_ready": finished,
            "queue_length": 3,
            "queue_position": 2,
            "ahead": 1,
        }

    def accept(name):
        control["accepted"].append(name)
        return JSONResponse(
            ticket(name), status_code=202, headers={"X-GMXBUILDER-Operation": ids[name]}
        )

    async def filter_input(**kwargs):
        return accept("filter")

    async def step(**kwargs):
        return accept("input")

    async def status(operation_id):
        return ticket(names[operation_id])

    async def result(operation_id):
        name = names[operation_id]
        while name not in control["delivered"]:
            await asyncio.sleep(0.02)
        if name == "input" and control["failure"]:
            return {"status": "error", "error": "Independent chain-break diagnostic"}
        return {"status": "ok", "metrics": {}}

    for path, function in [
        ("/api/filter-pdb/{task_id}", filter_input),
        ("/api/step/{task_id}/{step_name}", step),
        ("/api/operations/{operation_id}", status),
        ("/api/operations/{operation_id}/result", result),
    ]:
        monkeypatch.setattr(routes[path].dependant, "call", function)
    try:
        yield control
    finally:
        control["finished"].update(ids)
        control["delivered"].update(ids)


@pytest.mark.parametrize("failure", [False, True])
def test_notice_survives_ticket_boundaries_until_whole_check_settles(page, staged_check, failure):
    from selenium.webdriver.support.ui import WebDriverWait

    ready_input(page)
    staged_check["failure"] = failure
    page.find_element("id", "input-check-btn").click()
    wait = WebDriverWait(page, 15)
    wait.until(lambda _: "filter" in staged_check["accepted"])
    notice = page.find_element("id", "compute-queue-status")
    assert notice.is_displayed()
    page.execute_script("""
        window.__notice = document.getElementById('compute-queue-status');
        window.__noticeHidden = 0; window.__noticeRemoved = 0; window.__noticeWatch = true;
        window.__observer = new MutationObserver(records => {
          for (const r of records) for (const n of r.removedNodes)
            if (n === window.__notice) window.__noticeRemoved++;
        });
        __observer.observe(__notice.parentElement, {childList:true});
        function sample() {
          if (!__noticeWatch) return;
          if (!__notice.isConnected || getComputedStyle(__notice).display === 'none')
            __noticeHidden++;
          requestAnimationFrame(sample);
        }
        requestAnimationFrame(sample);
    """)
    for name in ["filter", "input"]:
        wait.until(lambda _: name in staged_check["accepted"])
        staged_check["finished"].add(name)
        # Observe the terminal ticket while its final response is still held.
        wait.until(lambda d: "Continuing" in d.find_element("id", "compute-queue-message").text)
        assert notice.is_displayed()
        assert not page.find_element("id", "input-check-btn").is_enabled()
        assert not page.find_element("css selector", "#panel-input .next-btn").is_enabled()
        staged_check["delivered"].add(name)
    if not failure:
        wait.until(lambda d: d.execute_script("return !!window.__viewerRelease || !_stepRunning"))
        assert page.execute_script("return !!window.__viewerRelease"), page.find_element(
            "id", "input-readiness-report"
        ).text
        assert notice.is_displayed()  # Check result arrived; viewer not yet restored.
        assert not page.find_element("css selector", "#panel-input .next-btn").is_enabled()
        page.execute_script("__noticeWatch = false; __viewerRelease(null)")
    wait.until(lambda d: d.execute_script("return !_stepRunning"))
    stats = page.execute_script("""
        __noticeWatch = false; __observer.disconnect();
        return {hidden:__noticeHidden, removed:__noticeRemoved,
          same:__notice === document.getElementById('compute-queue-status')};
    """)
    assert stats["removed"] == 0 and stats["same"] is True
    assert not notice.is_displayed()
    assert page.find_element("css selector", "#panel-input .next-btn").is_enabled() is not failure
    if failure:
        assert (
            page.find_element("tag name", "body").text.count("Independent chain-break diagnostic")
            == 1
        )


def test_input_deduplicates_errors_but_keeps_distinct_warnings_and_retry(page):
    ready_input(page)
    outcome = page.execute_async_script("""
        const done = arguments[0];
        const reason = 'Chain B is broken <img src=x onerror=alert(1)>';
        const second = 'Backbone atom N is missing';
        window.fetch = async url => ({ok:true, json:async () =>
          url.includes('/api/filter-pdb/') ? {status:'ok'} : {
            status:'error', error:reason + '\\n' + second,
            input_validation:{errors:[{message:reason}, {message:second}],
              warnings:['Distinct caution', 'Distinct caution'], repairable_residues:[{}]}
          }});
        _doCheckStep('input','input-check-status','input-check-btn').then(() => done({
          text:document.body.innerText,
          images:document.querySelectorAll('#input-readiness-report img').length,
          next:document.querySelector('#panel-input .next-btn').disabled
        }));
    """)
    assert outcome["text"].count("Chain B is broken") == 1
    assert outcome["text"].count("Backbone atom N is missing") == 1
    assert outcome["text"].count("Distinct caution") == 1
    assert "Resolve the blocking issues" in outcome["text"]
    assert outcome["images"] == 0 and outcome["next"] is True
    result = page.execute_async_script("""
        const done = arguments[0];
        window._loadStepViewerPdb = async () => null;
        window.fetch = async () => ({ok:true,json:async () => ({status:'ok',metrics:{}})});
        _doCheckStep('input','input-check-status','input-check-btn').then(() => done({
          text:document.body.innerText,
          next:document.querySelector('#panel-input .next-btn').disabled,
          notice:getComputedStyle(document.getElementById('compute-queue-status')).display
        }));
    """)
    assert "Chain B is broken" not in result["text"]
    assert "Distinct caution" not in result["text"]
    assert result["next"] is False and result["notice"] == "none"


def test_notice_waits_for_existing_confirmation_gate(page):
    ready_input(page)
    result = page.execute_script("""
        document.getElementById('panel-input').classList.remove('active');
        const panel = document.getElementById('panel-cg_system');
        panel.classList.add('active');
        state.wizardSteps = ['cg_system', 'simparams']; state.currentStepIdx = 0;
        const button = panel.querySelector('button[id*="check"]');
        const handle = startStepProgress('cg_system', button.id, 'unused');
        finishStepProgress(handle, true);
        finishUnfinishedStepProgress(handle);
        const notice = document.getElementById('compute-queue-status');
        const before = getComputedStyle(notice).display !== 'none';
        const blocked = panel.querySelector('.next-btn').disabled;
        // Computational completion no longer doubles as visual confirmation.
        _checkedSteps.add('cg_system'); updateNextButtonState();
        return {before, blocked, after:getComputedStyle(notice).display,
          next:panel.querySelector('.next-btn').disabled};
    """)
    assert result == {"before": False, "blocked": True, "after": "none", "next": False}


def test_build_failure_keeps_log_collapsed_and_one_visible_reason(page):
    ready_input(page)
    result = page.execute_script("""
        document.getElementById('panel-input').classList.remove('active');
        document.getElementById('panel-simparams').classList.add('active');
        const log = document.getElementById('build-log-content');
        log.textContent = 'Distinct build context\\nBuild-specific failure';
        document.getElementById('build-log-panel').classList.remove('hidden');
        document.getElementById('build-log-panel').open = true;
        showBuildFailure('Build-specific failure');
        return {text:document.body.innerText, log:log.textContent,
          open:document.getElementById('build-log-panel').open,
          download:getComputedStyle(document.getElementById('download-link')).display};
    """)
    assert result["text"].count("Build-specific failure") == 1
    assert "Distinct build context" in result["log"]
    assert result["open"] is False and result["download"] == "none"


@pytest.mark.parametrize(
    ("step", "button"),
    [
        ("orient", "orient-check-btn"),
        ("membrane", "check-composition-btn"),
        ("ions", "ion-check-btn"),
    ],
)
def test_dedicated_checks_keep_one_error_and_revoke_a_previous_pass(page, step, button):
    from selenium.webdriver.support.ui import WebDriverWait

    ready_input(page)
    page.execute_async_script(
        """
        const step = arguments[0], button = arguments[1], done = arguments[2];
        document.getElementById('panel-input').classList.remove('active');
        document.getElementById('panel-' + step).classList.add('active');
        state.wizardSteps = [step, 'simparams']; state.currentStepIdx = 0;
        _checkedSteps.add(step); state.completedSteps.add(0);
        window.fetch = async url => ({ok:true,json:async () =>
          url === '/api/lipid-library-list' ? {library_version:4,entries:[
            {lipid_name:'POPC',lipid_ff:'lipid21',ready:true}]} :
          {status:'error',error:'Specific ' + step + ' diagnostic'}});
        (async () => {
          if (step === 'membrane') {
            // Satisfy the independent V4 availability prerequisite before
            // exercising this test's backend-error presentation contract.
            document.getElementById('ff-protein').value = 'amber14sb';
            document.getElementById('ff-lipid').innerHTML =
              '<option value="lipid21">Lipid21</option>';
            await refreshV4LipidAvailability();
          }
          document.getElementById(button).click(); done();
        })().catch(error => done(String(error)));
    """,
        step,
        button,
    )
    WebDriverWait(page, 10).until(
        lambda d: f"Specific {step} diagnostic" in d.find_element("tag name", "body").text
    )
    assert page.find_element("tag name", "body").text.count(f"Specific {step} diagnostic") == 1
    assert not page.find_element("id", "compute-queue-status").is_displayed()
    assert not page.find_element("css selector", f"#panel-{step} .next-btn").is_enabled()
    assert page.find_element("id", button).is_enabled()


def test_late_progress_response_cannot_overwrite_failed_check(page):
    ready_input(page)
    page.execute_script("""
        window.__latePoll = null;
        window.fetch = async () => ({ok:true,json:() => new Promise(r => window.__latePoll = r)});
        window.__handle = startStepProgress('input','input-check-btn','input-check-status');
    """)
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 5).until(lambda d: d.execute_script("return !!window.__latePoll"))
    result = page.execute_async_script("""
        const done = arguments[0];
        finishStepProgress(__handle, false, 'Terminal diagnostic');
        finishUnfinishedStepProgress(__handle);
        __latePoll({running:true,fraction:0.99,phase:'Obsolete phase'});
        setTimeout(() => done(
          document.getElementById('step-progress-input-check-btn').innerText), 50);
    """)
    assert "Check failed" in result and "Obsolete phase" not in result and "99%" not in result


@pytest.mark.parametrize("initial_status", ["queued", "running"])
def test_queued_build_keeps_polling_through_running_failure_and_retry(page, initial_status):
    ready_input(page)
    result = page.execute_async_script(
        """
        const initialStatus = arguments[0], done = arguments[1];
        const pollMs = initialStatus === 'queued' ? 3000 : 2000;
        document.getElementById('panel-input').classList.remove('active');
        document.getElementById('panel-simparams').classList.add('active');
        const timers = [];
        const originalSet = GMXPoll.start, originalClear = GMXPoll.stop;
        GMXPoll.start = (callback, ms) => {
          const timer = {callback, ms, active:true}; timers.push(timer); return timer;
        };
        GMXPoll.stop = timer => {if (timer) timer.active = false;};
        let outcome = {status:'running'};
        window.fetch = async url => ({ok:true,json:async () =>
          url === '/api/build' ? {status:initialStatus,task_id:state.taskId} :
          url.includes('/queue-status') ? outcome : {lines:[],total:0,done:false}});
        (async () => {
          await runBuild();
          let poll = timers.find(t => t.ms === pollMs);
          await poll.callback();
          const activeAfterRunning = poll.active;
          outcome = {status:'failed',error:'Unique finalization failure'};
          await poll.callback();
          const failed = {text:document.body.innerText, stopped:!state.buildRunning};
          timers.length = 0;
          await runBuild();
          poll = timers.find(t => t.ms === pollMs);
          outcome = {status:'completed',
            result:{num_atoms:5,components:[],log:['Preserved build log']}};
          await poll.callback();
          done({activeAfterRunning, failed, text:document.body.innerText,
            stopped:!state.buildRunning,
            log:document.getElementById('build-log-content').textContent,
            open:document.getElementById('build-log-panel').open,
            download:getComputedStyle(document.getElementById('download-link')).display});
        })().catch(e => done({error:String(e)})).finally(() => {
          GMXPoll.start = originalSet; GMXPoll.stop = originalClear;
        });
    """,
        initial_status,
    )
    assert result["activeAfterRunning"] is True, result
    assert result["failed"]["stopped"] is True
    assert result["failed"]["text"].count("Unique finalization failure") == 1
    assert result["stopped"] is True and "Unique finalization failure" not in result["text"]
    assert "Preserved build log" in result["log"] and result["open"] is False
    assert result["download"] != "none"


@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
def test_check_cleanup_stops_polling_and_clock_for_every_outcome(page, outcome):
    ready_input(page)
    observed = page.execute_script(
        """
        const outcome = arguments[0];
        const originalClear = GMXPoll.stop;
        const cleared = [];
        GMXPoll.stop = id => { cleared.push(id); originalClear(id); };
        const handle = startStepProgress('input', 'input-check-btn', 'input-check-status');
        const timers = [handle.poll, handle.clock];
        try {
          if (outcome === 'exception') throw new Error('Unexpected failure');
          finishStepProgress(handle, outcome === 'success', 'Distinct problem');
        } catch (error) {
          // Exercise the finally safety net for a path without a terminal reply.
        } finally {
          finishUnfinishedStepProgress(handle);
        }
        finishUnfinishedStepProgress(handle);
        GMXPoll.stop = originalClear;
        return {timersStopped:timers.every(id => cleared.includes(id)),
          state:handle.element.dataset.state, phase:handle.element.querySelector(
            '.step-progress-phase').textContent, text:document.body.innerText};
    """,
        outcome,
    )
    assert observed["timersStopped"] is True
    assert observed["state"] == ("done" if outcome == "success" else "error")
    assert observed["phase"] == ("Complete" if outcome == "success" else "Check failed")
    if outcome == "failure":
        assert observed["text"].count("Distinct problem") == 1


@pytest.mark.parametrize("workflow", ["martini3-bilayer", "martini3-solvent", "solvator"])
@pytest.mark.parametrize("failure", ["timeout", "http", "render"])
def test_checked_input_unlocks_after_preview_failure_and_can_reload(page, workflow, failure):
    ready_input(page, workflow)
    result = page.execute_async_script(
        """
        const failure = arguments[0], done = arguments[1];
        // Exercise the real timeout loader; only the transport is controlled.
        const sourceLoader = window.__realPreviewLoader;
        window._loadStepViewerPdb = sourceLoader;
        window._VIEWER_REQUEST_TIMEOUT_MS = 25;
        let fail = true;
        window.fetch = (url, opts) => {
          if (url.endsWith('/viewer.pdb')) {
            if (fail && failure === 'timeout') return new Promise((resolve, reject) => {
              opts.signal.addEventListener('abort', () =>
                reject(new DOMException('Aborted','AbortError')));
            });
            return Promise.resolve(new Response('HEADER checked input\\nEND\\n',
              {status:fail && failure === 'http' ? 503 : 200}));
          }
          return Promise.resolve(new Response(JSON.stringify({status:'ok',metrics:{}}),
            {headers:{'Content-Type':'application/json'}}));
        };
        window.redrawPDBViewerWithChainFilter = () => {
          if (fail && failure === 'render') throw new Error('WebGL unavailable');
        };
        (async () => {
          await _doCheckStep('input','input-check-status','input-check-btn');
          const before = {next:document.querySelector('#panel-input .next-btn').disabled,
            checked:_checkedSteps.has('input'), running:_stepRunning,
            notice:document.getElementById('compute-queue-status').classList.contains('hidden'),
            warning:document.getElementById('input-preview-feedback')?.textContent,
            retry:!!document.querySelector('#input-preview-feedback button')};
          fail = false;
          await refreshCheckedInputPreview();
          done({before, warningAfter:!!document.getElementById('input-preview-feedback'),
            nextAfter:document.querySelector('#panel-input .next-btn').disabled});
        })().catch(e => done({error:String(e)}));
        """,
        failure,
    )
    assert "error" not in result, result
    assert result["before"]["checked"] and not result["before"]["running"]
    assert result["before"]["next"] is False and result["before"]["notice"] is True
    assert "Input check passed" in result["before"]["warning"]
    assert result["before"]["retry"] is True
    assert result["warningAfter"] is False and result["nextAfter"] is False


@pytest.mark.parametrize(
    "workflow",
    ["solvator", "membrane-bilayer", "pure-membrane", "martini3-bilayer", "martini3-solvent"],
)
def test_final_build_uses_full_width_shared_progress_and_terminal_styles(page, workflow):
    ready_input(page, workflow)
    result = page.execute_script("""
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-simparams').classList.add('active');
        startBuildProgress();
        const element = _buildProgressHandle.element;
        const section = document.getElementById('progress-section');
        const bar = element.querySelector('[role=progressbar]');
        const running = {state:element.dataset.state,
          width:element.getBoundingClientRect().width,
          available:section.getBoundingClientRect().width,
          percent:bar.getAttribute('aria-valuenow'),
          time:element.querySelector('.step-progress-elapsed').textContent};
        showBuildFailure('Controlled export failure');
        const resultSection = document.getElementById('result-section');
        const failed = {state:element.dataset.state,clock:_buildProgressHandle.clock,
          result:resultSection.dataset.state,
          color:getComputedStyle(document.getElementById('result-heading')).color};
        startBuildProgress();
        _showBuildResult({num_atoms:5,components:[],log:[]});
        return {running,failed,complete:element.dataset.state,result:resultSection.dataset.state,
          color:getComputedStyle(document.getElementById('result-heading')).color,
          percent:bar.getAttribute('aria-valuenow'),clock:_buildProgressHandle.clock,
          legacy:!!document.getElementById('progress-fill')};
    """)
    assert result["running"]["state"] == "running"
    assert abs(result["running"]["width"] - result["running"]["available"]) < 2
    assert result["running"]["percent"] is None
    assert "s" in result["running"]["time"]
    assert result["failed"]["state"] == "error" and result["failed"]["clock"] is None
    assert result["failed"]["result"] == "error" and result["result"] == "done"
    assert result["failed"]["color"] != result["color"]
    assert result["complete"] == "done" and result["percent"] == "100"
    assert result["clock"] is None and result["legacy"] is False


@pytest.mark.parametrize("input_check_required", [False, True])
def test_build_rejection_preserves_recovery_flag_and_revokes_only_stale_passes(
    page, input_check_required
):
    ready_input(page)
    result = page.execute_async_script(
        """
        const required = arguments[0], done = arguments[1];
        const steps = state.wizardSteps;
        state.completedSteps = new Set();
        _checkedSteps.clear();
        steps.forEach((step, i) => { _checkedSteps.add(step); state.completedSteps.add(i); });
        state.currentStepIdx = steps.indexOf('simparams');
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-simparams').classList.add('active');
        window.buildModuleConfig = () => ({});
        const reason = 'Repeat Check Upload <img src=x onerror=alert(1)>';
        window.fetch = async url => String(url) === '/api/build'
          ? {ok:false, json:async()=>({error:reason,input_check_required:required})}
          : {ok:true,json:async()=>({lines:[],done:true})};
        let invalidated = 0;
        window.invalidateFinalReview = () => { invalidated++; };
        runBuild().then(() => {
          const details = document.getElementById('result-details');
          const button = details.querySelector('button');
          const result = {checked:Array.from(_checkedSteps),
            completed:Array.from(state.completedSteps), invalidated,
            text:details.textContent, images:details.querySelectorAll('img').length,
            recovery:!!button, running:state.buildRunning,
            state:document.getElementById('result-section').dataset.state};
          if (button) button.click();
          result.step = steps[state.currentStepIdx];
          result.advance = canGoToStep(1);
          done(result);
        }).catch(error=>done({error:String(error)}));
        """,
        input_check_required,
    )
    assert "error" not in result, result
    assert result["state"] == "error" and result["running"] is False
    assert "Repeat Check Upload" in result["text"] and result["images"] == 0
    assert result["recovery"] is input_check_required
    if input_check_required:
        assert result["checked"] == result["completed"] == []
        assert result["invalidated"] == 1
        assert result["step"] == "input" and result["advance"] is False
    else:
        assert result["checked"] and result["completed"]
        assert result["invalidated"] == 0 and result["step"] == "simparams"


def test_aborted_preview_ticket_can_be_watched_again(page, staged_check):
    ready_input(page, "martini3-bilayer")
    aborted = page.execute_async_script("""
        const done = arguments[0];
        const ticket = {operation_id:'dddddddddddddddddddddddddddddddd',
          task_id:state.taskId,status:'queued'};
        window.__previewReceipt = () => new Response(JSON.stringify(ticket), {
          status:202,headers:{'Content-Type':'application/json',
            'X-GMXBUILDER-Operation':ticket.operation_id}});
        const controller = new AbortController();
        resolveManagedResponse(__previewReceipt(), {signal:controller.signal,
          requestUrl:'/api/step/'+state.taskId+'/input/viewer.pdb'}).then(
            () => done('unexpected success'), error => done(error.name));
        setTimeout(() => controller.abort(), 25);
    """)
    assert aborted == "AbortError"
    staged_check["finished"].add("input")
    staged_check["delivered"].add("input")
    retried = page.execute_async_script("""
        const done = arguments[0];
        resolveManagedResponse(__previewReceipt(), {
          requestUrl:'/api/step/'+state.taskId+'/input/viewer.pdb'}).then(
            response => response.json()).then(done).catch(error => done({error:error.name}));
    """)
    assert retried == {"status": "ok", "metrics": {}}


def test_reconnected_build_reports_observed_time_and_stops_replaced_clock(page):
    ready_input(page)
    result = page.execute_script("""
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-simparams').classList.add('active');
        startBuildProgress(true);
        const oldClock = _buildProgressHandle.clock;
        const clear = window.clearInterval;
        let replaced = false;
        window.clearInterval = id => {if (id === oldClock) replaced = true; clear(id);};
        startBuildProgress(true);
        finishBuildProgress(true);
        window.clearInterval = clear;
        return {replaced,clock:_buildProgressHandle.clock,
          elapsed:_buildProgressHandle.element.querySelector('.step-progress-elapsed').textContent};
    """)
    assert result["replaced"] is True and result["clock"] is None
    assert result["elapsed"].endswith("s since reconnect")

"""Drive real XHR upload controls across an independent 202/result HTTP contract."""

import json
import time
import uuid

import pytest
from fastapi.responses import JSONResponse

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def ready(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 30).until(
        lambda d: d.execute_script(
            "return initComputeQueueStatus._done === true "
            "&& !document.getElementById('task-grid').inert"
        )
    )


@pytest.fixture
def queued_upload(monkeypatch, live_server):
    from gmxbuilder.web import server

    routes = {route.path: route for route in server.app.routes if hasattr(route, "dependant")}
    upload = routes["/api/upload-pdb"].dependant.call
    # Production IDs are unique; reusing one can replay a browser-cached 410
    # from a preceding expired-operation test.
    operation = uuid.uuid4().hex
    control = {"ready": False, "error": None, "uploads": 0}

    async def accept(*args, **kwargs):
        # Real multipart parsing, saved task and structure response. Only the
        # HTTP delivery schedule is controlled independently by this fixture.
        payload = await upload(*args, **kwargs)
        if isinstance(payload, JSONResponse):
            payload = json.loads(payload.body)
        control["payload"] = payload
        control["uploads"] += 1
        return JSONResponse(
            ticket(), status_code=202, headers={"X-GMXBUILDER-Operation": operation}
        )

    def ticket():
        return {
            "operation_id": operation,
            "task_id": control["payload"]["task_id"],
            "status": "queued",
            "queue_length": 1,
            "queue_position": 1,
            "ahead": 0,
            "waited_seconds": 1,
            "expires_at": time.time() + 86400,
            "response_ready": control["ready"],
        }

    async def status(operation_id):
        assert operation_id == operation
        if control.get("status_error"):
            return JSONResponse({"error": control["status_error"]}, status_code=410)
        return ticket()

    async def result(operation_id):
        assert operation_id == operation
        if not control["ready"]:
            return JSONResponse({"status": "queued"}, status_code=202)
        if control["error"]:
            return JSONResponse({"error": control["error"]}, status_code=422)
        return control["payload"]

    for path, function in [
        ("/api/upload-pdb", accept),
        ("/api/operations/{operation_id}", status),
        ("/api/operations/{operation_id}/result", result),
    ]:
        monkeypatch.setattr(routes[path].dependant, "call", function)
    return control


def test_xhr_waits_for_structure_then_refresh_restores_same_task(
    page, queued_upload, small_pdb_file
):
    from selenium.webdriver.support.ui import WebDriverWait

    ready(page)
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    page.execute_script(
        """
        const input = document.getElementById('pdb-file');
        const files = new DataTransfer();
        files.items.add(new File([arguments[0]], 'test.pdb', {type:'chemical/x-pdb'}));
        input.files = files.files;
        input.dispatchEvent(new Event('change', {bubbles:true}));
    """,
        small_pdb_file.read_text(),
    )
    WebDriverWait(page, 30).until(lambda _: "payload" in queued_upload)
    WebDriverWait(page, 10).until(lambda d: d.execute_script("return !!state.taskId"))
    task_id = page.execute_script("return state.taskId")
    assert page.execute_script("return state.pdbInfo") is None
    assert not page.find_element("id", "upload-info").is_displayed()
    assert page.find_element("id", "upload-progress").is_displayed()
    assert not page.find_element("id", "input-check-btn").is_enabled()
    assert page.find_element("id", "compute-queue-copy").is_enabled()
    assert (
        page.execute_script("return !!document.querySelector('.modal-overlay:not(.hidden)')")
        is False
    )
    # A second file selection cannot issue a second request while processing.
    page.execute_script("handleFile(new File(['x'], 'second.pdb'))")
    assert queued_upload["uploads"] == 1
    queued_upload["ready"] = True
    WebDriverWait(page, 30).until(
        lambda d: d.execute_script("return !state.uploadRunning && state.pdbInfo?.num_atoms === 5")
    )
    assert page.find_element("id", "upload-info").is_displayed()
    assert page.execute_script("return document.querySelectorAll('.chain-check input').length") > 0
    page.refresh()
    ready(page)
    WebDriverWait(page, 30).until(
        lambda d: d.execute_script("return state.pdbInfo?.num_atoms === 5")
    )
    assert page.execute_script("return state.taskId") == task_id
    assert page.execute_script("return document.querySelector('.panel.active').id") == "panel-input"
    assert page.current_url.endswith("/BilayerBuilder/Step1")
    assert task_id not in page.current_url


@pytest.mark.parametrize("failure_phase", ["result", "status"])
def test_managed_upload_failure_is_visible_without_a_dialog(
    page, queued_upload, small_pdb_file, failure_phase
):
    from selenium.webdriver.support.ui import WebDriverWait

    ready(page)
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    page.execute_script(
        """
        const input = document.getElementById('pdb-file');
        const files = new DataTransfer();
        files.items.add(new File([arguments[0]], 'test.pdb', {type:'chemical/x-pdb'}));
        input.files = files.files;
        input.dispatchEvent(new Event('change', {bubbles:true}));
    """,
        small_pdb_file.read_text(),
    )
    WebDriverWait(page, 30).until(lambda _: "payload" in queued_upload)
    if failure_phase == "status":
        queued_upload["status_error"] = "Structure contains no usable atoms"
    else:
        queued_upload["error"] = "Structure contains no usable atoms"
    queued_upload["ready"] = True
    WebDriverWait(page, 30).until(
        lambda d: d.execute_script("return state.uploadRunning === false")
    )
    assert "no usable atoms" in page.find_element("id", "validation-errors").text
    assert not page.find_element("id", "compute-queue-status").is_displayed()
    assert page.find_element("tag name", "body").text.count("no usable atoms") == 1
    assert page.find_element("id", "browse-btn").is_enabled()
    assert (
        page.execute_script("return !!document.querySelector('.modal-overlay:not(.hidden)')")
        is False
    )
    assert page.execute_script("return state.pdbInfo") is None


def test_upload_plain_text_server_error_is_actionable(
    page, live_server, small_pdb_file, monkeypatch
):
    from fastapi.responses import PlainTextResponse
    from selenium.webdriver.support.ui import WebDriverWait

    from gmxbuilder.web import server

    route = next(
        route for route in server.app.routes if getattr(route, "path", None) == "/api/upload-pdb"
    )

    async def fail_upload(*args, **kwargs):
        return PlainTextResponse("Internal Server Error", status_code=500)

    monkeypatch.setattr(route.dependant, "call", fail_upload)
    ready(page)
    page.execute_async_script("selectTaskType('membrane-bilayer').then(arguments[0])")
    page.execute_script(
        "handleFile(new File([arguments[0]], 'test.pdb', {type:'chemical/x-pdb'}))",
        small_pdb_file.read_text(),
    )
    WebDriverWait(page, 30).until(
        lambda driver: driver.execute_script("return state.uploadRunning === false")
    )
    message = page.find_element("id", "validation-errors").text
    assert "Upload failed (HTTP 500). Please retry." in message
    assert "Failed to execute 'json'" not in message
    assert page.find_element("id", "browse-btn").is_enabled()


def test_step_url_without_saved_context_returns_to_matching_workflow(page, live_server):
    from selenium.webdriver.support.ui import WebDriverWait

    ready(page)
    page.execute_script("sessionStorage.removeItem('gmxbuilder-current-task')")
    page.get(live_server + "/Solvator/Step5")
    ready(page)
    WebDriverWait(page, 15).until(lambda d: d.current_url.endswith("/Solvator/Step1"))
    assert page.execute_script("return document.querySelector('.panel.active').id") == "panel-input"
    assert page.execute_script("return state.taskType.id") == "solvator"


def test_expired_saved_task_reports_failure_and_resets_step(page, live_server):
    from selenium.webdriver.support.ui import WebDriverWait

    ready(page)
    page.execute_script("""
        sessionStorage.setItem('gmxbuilder-current-task', JSON.stringify({
          task_id:'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', workflow:'solvator', step:4
        }));
    """)
    page.get(live_server + "/Solvator/Step5")
    ready(page)
    WebDriverWait(page, 15).until(lambda d: d.current_url.endswith("/Solvator/Step1"))
    assert page.execute_script("return document.querySelector('.panel.active').id") == "panel-input"
    assert "Could not restore task" in page.find_element("id", "task-resume-error").text
    assert page.execute_script("return state.taskId") is None
    assert page.execute_script("return document.getElementById('panels').inert") is False


def test_exit_during_pending_upload_keeps_saved_task_resumable(
    page, queued_upload, small_pdb_file, live_server
):
    from selenium.webdriver.support.ui import WebDriverWait

    from gmxbuilder.web import server

    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script(
        "handleFile(new File([arguments[0]], 'exit.pdb', {type:'chemical/x-pdb'}))",
        small_pdb_file.read_text(),
    )
    wait = WebDriverWait(page, 30)
    wait.until(lambda _: "payload" in queued_upload)
    wait.until(lambda d: d.execute_script("return !!state.taskId"))
    task_id = page.execute_script("return state.taskId")
    assert page.execute_script("return state.uploadRunning") is True, page.execute_script(
        "return {error:document.getElementById('validation-errors').textContent, "
        "info:state.pdbInfo, task:state.taskId}"
    )
    task_dir = server.task_manager.get_task_dir(task_id)
    saved_state = (task_dir / "state.json").read_bytes()
    saved_pdb = server.task_manager.get_pdb_path(task_id).read_bytes()
    # Record any request from the departing document, including unload callbacks.
    page.execute_script("""
        sessionStorage.setItem('exit-requests', '[]');
        const original = window.fetch;
        window.fetch = (url, options = {}) => {
          const requests = JSON.parse(sessionStorage.getItem('exit-requests'));
          requests.push({url:String(url),method:options.method || 'GET'});
          sessionStorage.setItem('exit-requests', JSON.stringify(requests));
          return original(url, options);
        };
    """)
    assert (
        page.execute_script("return document.getElementById('copy-task-id').nextElementSibling.id")
        == "exit-task"
    )
    page.find_element("id", "exit-task").click()
    wait.until(lambda d: d.current_url.rstrip("/") == live_server.rstrip("/"))
    ready(page)
    assert page.execute_script("return state.taskId") is None
    assert page.execute_script("return sessionStorage.getItem('gmxbuilder-current-task')") is None
    assert page.find_element("id", "panel-task-type").is_displayed()
    assert not page.find_element("id", "header-task-id").is_displayed()
    requests = page.execute_script("return JSON.parse(sessionStorage.getItem('exit-requests'))")
    assert all(row["method"] == "GET" and "cancel" not in row["url"] for row in requests)
    assert (task_dir / "state.json").read_bytes() == saved_state
    assert server.task_manager.get_pdb_path(task_id).read_bytes() == saved_pdb
    # The queued delivery can still complete after the old document has gone.
    queued_upload["ready"] = True
    page.refresh()
    ready(page)
    assert page.execute_script("return state.taskId") is None
    page.find_element("id", "resume-task-id").send_keys(task_id)
    page.find_element("id", "resume-task-btn").click()
    wait.until(lambda d: d.execute_script("return state.pdbInfo?.num_atoms === 5"))
    assert page.execute_script("return state.taskId") == task_id
    wait.until(lambda d: d.current_url.endswith("/Solvator/Step1"))
    assert queued_upload["uploads"] == 1

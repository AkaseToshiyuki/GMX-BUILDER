"""Shared test fixtures, and the --fast option.

Two thirds of this suite's wall time comes from a fifteenth of its tests: the
ones that shell out to GROMACS or AmberTools, embed a 3D conformer with RDKit,
or drive a browser. They earn that time -- they are the only checks that can
see a topology GROMACS will not accept, or a page that white-screens -- but
paying it on every edit is what makes the suite feel too long to run.

``--fast`` deselects them, leaving the pure-Python checks. It is a convenience
for the edit loop, deliberately *not* the default: a run that skips the real
tools must be something you asked for, never something you forgot to turn off.
CI runs the whole suite; local release checks follow the affected-module scope.
"""

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="skip tests marked slow (external tools: GROMACS, AmberTools, RDKit, Firefox)",
    )

    parser.addoption(
        "--construction-only",
        action="store_true",
        default=False,
        help="run construction/preprocessing; omit explicit GROMACS integration probes",
    )


@pytest.fixture
def require_simulation(request):
    """Boundary after construction/grompp, before an optional integration probe.

    Defaults to full verification. Explicit construction-only runs report these
    as skipped (not passed); PDBFixer placement and zero-step energy QA still run.
    """

    def require():
        if request.config.getoption("--construction-only"):
            pytest.skip("construction/grompp passed; integration excluded by --construction-only")

    return require


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--fast"):
        return
    skipped = pytest.mark.skip(reason="deselected by --fast (drives an external tool)")
    deselected = 0
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skipped)
            deselected += 1
    config.stash[_FAST_DESELECTED] = deselected


_FAST_DESELECTED = pytest.StashKey[int]()


@pytest.fixture(scope="session", autouse=True)
def no_implicit_asset_installation():
    """Read installed prerequisites without repairing the user's real cache.

    Installer tests call install_prebuilt_assets with explicit temporary roots.
    Ordinary tests must never bootstrap or replace assets as a lookup side effect.
    """
    import os

    previous = os.environ.get("GMXBUILDER_PREBUILT_AUTO_INSTALL")
    os.environ["GMXBUILDER_PREBUILT_AUTO_INSTALL"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GMXBUILDER_PREBUILT_AUTO_INSTALL", None)
        else:
            os.environ["GMXBUILDER_PREBUILT_AUTO_INSTALL"] = previous


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Never let a fast run be mistaken for a complete one."""
    deselected = config.stash.get(_FAST_DESELECTED, 0)
    if deselected:
        terminalreporter.write_line(
            f"--fast skipped {deselected} tests that drive external tools; "
            "run without --fast before releasing.",
            yellow=True,
        )


@pytest.fixture(scope="session", autouse=True)
def isolated_task_root(tmp_path_factory):
    """Keep task state out of the shared default root.

    ``TaskManager`` defaults to ``/tmp/gmxbuilder_tasks``, so the suite
    otherwise writes real task directories into a location shared with any
    local instance and leaves them behind after the run. Tests that want a
    specific root still replace ``task_manager`` directly.
    """
    import os

    from gmxbuilder.web.task_manager import task_manager

    # The environment variable alone is not enough: TASK_ROOT is read when
    # task_manager is imported, which happens during collection, before any
    # fixture runs. The variable is still set for anything spawned as a
    # subprocess, and the already-constructed singleton is repointed directly.
    previous_environment = os.environ.get("GMXBUILDER_TASK_DIR")
    previous_root = task_manager.root
    root = tmp_path_factory.mktemp("task-root")
    os.environ["GMXBUILDER_TASK_DIR"] = str(root)
    task_manager.root = root
    try:
        yield root
    finally:
        task_manager.root = previous_root
        if previous_environment is None:
            os.environ.pop("GMXBUILDER_TASK_DIR", None)
        else:
            os.environ["GMXBUILDER_TASK_DIR"] = previous_environment


@pytest.fixture(scope="session", autouse=True)
def isolated_rate_limit_database(tmp_path_factory):
    """Give each session its own durable rate-limit database.

    The limiter is a fixed-window counter backed by SQLite, and it defaults to
    a fixed path under the task root that survives between runs. Running the
    suite more than a few times within one window therefore exhausts the
    `heavy` and `finalize` budgets and produces 429/503 responses in tests that
    have nothing to do with rate limiting -- a failure that looks like a
    product regression and reproduces only after repeated local runs.

    A fresh database is necessary and not sufficient. The `heavy` budget is 20
    requests an hour, which is a sensible thing to offer the internet and an
    absurd one to offer a test suite: the suite makes more uploads than that,
    and whether it fails depends on how long it takes. It passed at 95 minutes
    and failed at 46, when caching brought every upload test inside one window
    -- twelve failures with nothing in common except a 429. The budgets are
    therefore raised for the session as well.

    Tests that exercise the limiter deliberately set both the path and the
    budgets themselves with monkeypatch, which overrides this.
    """
    import os

    previous = {
        name: os.environ.get(name)
        for name in (
            "GMXBUILDER_RATE_LIMIT_DB",
            "GMXBUILDER_HEAVY_RATE",
            "GMXBUILDER_FINALIZE_RATE",
        )
    }
    database = tmp_path_factory.mktemp("rate-limits") / "rate-limits.sqlite3"
    os.environ["GMXBUILDER_RATE_LIMIT_DB"] = str(database)
    os.environ["GMXBUILDER_HEAVY_RATE"] = "100000"
    os.environ["GMXBUILDER_FINALIZE_RATE"] = "100000"
    try:
        yield database
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def empty_system():
    """An empty System with a 10 nm cubic box."""
    return System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
    )


@pytest.fixture
def simple_protein_structure():
    """A minimal 3-atom 'protein' structure."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.38, 0.0, 0.0],
            [0.76, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return Structure(
        coordinates=coords,
        box_vectors=np.eye(3) * 5.0,
        atom_names=["N", "CA", "C"],
        resnames=["ALA", "ALA", "ALA"],
        resids=[1, 1, 1],
        chain_ids=["A", "A", "A"],
        elements=["N", "C", "C"],
    )


@pytest.fixture
def small_pdb_file(tmp_path):
    """Create a small PDB file for testing."""
    content = """\
HEADER    TEST PROTEIN
CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.209   2.354   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.752  -1.247  1.00  0.00           C
TER
END
"""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(content)
    return pdb_path


# --------------------------------------------------------------------------
# The live-browser harness
#
# Two modules drive a headless Firefox at a real server. Each used to keep its
# own copy of this setup, so one suite run paid for two Firefox launches to
# look at the same application. They share one browser here instead.
#
# The *browser* is session-scoped; the *server* deliberately is not. The
# application keeps its background tasks in a module-level list, created on the
# event loop that ran its lifespan. A uvicorn instance left running for the
# whole session therefore leaves its tasks in that global, and the next
# ``TestClient(app)`` -- there are dozens, in the web tests -- fails at shutdown
# with "the future belongs to a different loop". Measured: 49 failures and 10
# errors across fifteen files, none of them in a browser test. So the server
# lives and dies with the module that asked for it, which is what keeps that
# global empty for everyone else.
#
# Nothing is imported at module scope. conftest is loaded for every run,
# including runs on machines with no selenium and no geckodriver, and an import
# there would turn a clean skip into a collection error.


def _free_browser_port() -> int:
    import contextlib
    import socket

    with contextlib.closing(socket.socket()) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _require_browser_stack() -> None:
    """Skip -- or fail, when an environment promised a browser -- if absent."""
    import os
    import shutil

    import pytest

    # A check that silently skips is worse than no check, so an environment
    # that promises a browser must fail loudly instead of quietly passing.
    required = os.environ.get("GMXBUILDER_REQUIRE_BROWSER", "").strip() == "1"
    if required:
        pytest.importorskip(
            "selenium",
            reason="GMXBUILDER_REQUIRE_BROWSER=1 but selenium is absent",
        )
        if not shutil.which("geckodriver"):
            pytest.fail("GMXBUILDER_REQUIRE_BROWSER=1 but geckodriver is not installed")
        return
    pytest.importorskip("selenium")
    if not shutil.which("geckodriver"):
        pytest.skip("geckodriver is required for live browser checks")


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Serve the real application on a loopback port, for one module."""
    import os
    import threading
    import time

    _require_browser_stack()

    import uvicorn

    root = tmp_path_factory.mktemp("browser-tasks")
    previous = {
        "GMXBUILDER_TASK_DIR": os.environ.get("GMXBUILDER_TASK_DIR"),
        "GMXBUILDER_RATE_LIMIT_DB": os.environ.get("GMXBUILDER_RATE_LIMIT_DB"),
    }
    os.environ["GMXBUILDER_TASK_DIR"] = str(root)
    os.environ["GMXBUILDER_RATE_LIMIT_DB"] = str(root / "rate.sqlite3")

    from gmxbuilder.web.server import app

    port = _free_browser_port()
    from tests.browser_instrumentation import BrowserInstrumentation

    config = uvicorn.Config(
        BrowserInstrumentation(app), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while time.time() < deadline and not server.started:
        time.sleep(0.1)
    if not server.started:
        server.should_exit = True
        pytest.skip("the application did not start in time")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        # A graceful stop waits for open keep-alive connections, which under
        # load held the suite for the whole timeout. Ask once, then insist.
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="session")
def browser():
    """One headless Firefox, shared by every module that drives the page.

    It takes no server: a driver outlives any one module's server, and the
    page fixture supplies the URL.
    """
    import shutil

    _require_browser_stack()

    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service

    options = Options()
    options.add_argument("-headless")
    driver = webdriver.Firefox(
        options=options,
        service=Service(shutil.which("geckodriver")),
    )
    driver.set_page_load_timeout(60)
    try:
        yield driver
    finally:
        driver.quit()


@pytest.fixture
def page(browser, live_server):
    """The application's landing page, freshly loaded."""
    # Fault-injection tests replace fetch and leave pending promises/timers.
    # Leave that document before reusing the same application URL.
    browser.get("about:blank")
    browser.get(live_server)
    # DOMContentLoaded awaits options before installing event handlers. Selenium's
    # page-load completion alone does not mean those controls are interactive.
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        # A cold real V4 library revalidates numeric evidence (measured at ~36 s
        # with profiling). This is a readiness bound, not a sleep or a bypass.
        WebDriverWait(browser, 60).until(
            lambda driver: driver.execute_script(
                "return initComputeQueueStatus._done === true "
                "&& initCustomLipidModal._done === true "
                "&& !document.getElementById('task-grid').inert && window._optionsReady === true"
            )
        )
    except TimeoutException:
        diagnostic = browser.execute_script(
            "return {errors: window.__browserErrors, "
            "startup: document.getElementById('startup-load-error')?.textContent, "
            "requests: window.__browserRequests, "
            "resources: performance.getEntriesByType('resource').map(r=>({"
            "name:r.name,duration:r.duration,responseStatus:r.responseStatus}))}"
        )
        pytest.fail(f"Application initialization timed out: {diagnostic}")
    return browser


@pytest.fixture
def unpopulated_default_lipid_library(monkeypatch):
    """Policy-only tests specify absent assets without invoking the installer."""
    from gmxbuilder.modules.membrane import equilibrated_library

    library = equilibrated_library.EquilibratedLipidLibrary(roots=[])
    library.require_v4 = True
    monkeypatch.setattr(equilibrated_library, "_library", library)
    return library


@pytest.fixture
def empty_default_lipid_library(monkeypatch):
    """Input-only endpoint tests keep real task scoping without installing lipids."""
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.web import custom_lipids

    monkeypatch.setattr(
        custom_lipids, "EquilibratedLipidLibrary", lambda: EquilibratedLipidLibrary(roots=[])
    )


@pytest.fixture
def parameter_sources(tmp_path, monkeypatch):
    """Synthetic parameter files for storage contracts, with real content hashing.

    Only the provenance reader's source directory is isolated. Parameter admission,
    hashing, mutation detection and the normal topology/geometry code remain real.
    """
    from gmxbuilder.modules.membrane import parameter_provenance as provenance

    root = tmp_path / "synthetic-parameters"
    root.mkdir()
    (root / "forcefield.itp").write_text("; synthetic storage fixture parameters\n")
    monkeypatch.setattr(provenance, "force_field_directory", lambda _ff: root)
    return root

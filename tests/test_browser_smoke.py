"""Optional live-browser regressions for deployment-only UI paths."""

from __future__ import annotations

import os
import shutil
import time

import pytest


@pytest.mark.browser
def test_refresh_and_enter_step_one_accepts_popc_with_lipid21():
    """The default POPC selection must not raise a false compatibility alert."""
    base_url = os.environ.get("GMXBUILDER_BROWSER_URL", "").rstrip("/")
    if not base_url:
        pytest.skip("set GMXBUILDER_BROWSER_URL to run live browser regressions")

    pytest.importorskip("selenium")
    from selenium import webdriver
    from selenium.common.exceptions import NoAlertPresentException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait

    def assert_no_alert(driver, stage: str) -> None:
        try:
            alert = driver.switch_to.alert
        except NoAlertPresentException:
            return
        message = alert.text
        alert.dismiss()
        pytest.fail(f"unexpected alert {stage}: {message}")

    options = Options()
    options.add_argument("-headless")
    driver_path = shutil.which("geckodriver")
    if not driver_path:
        pytest.skip("geckodriver is required for live Firefox regressions")
    driver = webdriver.Firefox(options=options, service=Service(driver_path))
    wait = WebDriverWait(driver, 25)
    try:
        driver.get(f"{base_url}/")
        card_selector = '.task-card[data-task-id="membrane-bilayer"]'
        wait.until(lambda current: current.find_elements(By.CSS_SELECTOR, card_selector))
        time.sleep(1)
        assert_no_alert(driver, "on initial load")

        driver.refresh()
        wait.until(lambda current: current.find_elements(By.CSS_SELECTOR, card_selector))
        time.sleep(1)
        assert_no_alert(driver, "after refresh")

        popc_sources = driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            fetch('/api/options').then(response => response.json()).then(data => {
              const popc = data.lipids.find(item => item.name === 'POPC');
              done(popc ? popc.parameterizations : []);
            }).catch(error => done(['ERROR', String(error)]));
            """
        )
        assert "lipid21" in popc_sources

        driver.find_element(By.CSS_SELECTOR, card_selector).click()
        wait.until(
            lambda current: "active"
            in current.find_element(By.ID, "panel-input").get_attribute("class").split()
        )
        time.sleep(1)
        assert_no_alert(driver, "after entering Step 1")
        assert driver.current_url.rstrip("/").endswith("/BilayerBuilder/Step1")

        driver.refresh()
        wait.until(
            lambda current: "active" in current.find_element(
                By.ID, "panel-task-type"
            ).get_attribute("class").split()
        )
        wait.until(lambda current: current.current_url.rstrip("/") == base_url.rstrip("/"))
        assert driver.current_url.rstrip("/") == base_url.rstrip("/")

        task_id = driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            const pdb = `HEADER    ROUTE TEST
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.209   2.354   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.752   1.247  1.00  0.00           C
TER
END
`;
            const form = new FormData();
            form.append('file', new Blob([pdb], {type: 'chemical/x-pdb'}), 'route.pdb');
            form.append('task_type', 'membrane-bilayer');
            fetch('/api/upload-pdb', {method: 'POST', body: form})
              .then(response => response.json())
              .then(data => done(data.task_id || ['ERROR', data]))
              .catch(error => done(['ERROR', String(error)]));
            """
        )
        assert isinstance(task_id, str) and len(task_id) == 32
        # Legacy task-bearing links are retired and return to the home page.
        driver.get(f"{base_url}/BilayerBuilder/{task_id}/Step1")
        wait.until(
            lambda current: "active" in current.find_element(
                By.ID, "panel-task-type"
            ).get_attribute("class").split()
        )
        wait.until(lambda current: current.current_url.rstrip("/") == base_url.rstrip("/"))
        assert driver.current_url.rstrip("/") == base_url.rstrip("/")
        resume_input = driver.find_element(By.ID, "resume-task-id")
        resume_input.send_keys(task_id)
        driver.find_element(By.ID, "resume-task-btn").click()
        wait.until(
            lambda current: task_id
            in current.find_element(By.ID, "task-id-display").text
        )
        assert driver.current_url.rstrip("/").endswith(
            "/BilayerBuilder/Step1"
        )
        driver.find_element(By.ID, "copy-task-id").click()
        wait.until(
            lambda current: current.find_element(
                By.ID, "copy-task-id-status"
            ).text == "Copied"
        )
        driver.refresh()
        wait.until(lambda current: "active" in current.find_element(
            By.ID, "panel-task-type"
        ).get_attribute("class").split())
        wait.until(lambda current: current.current_url.rstrip("/") == base_url.rstrip("/"))
        assert driver.current_url.rstrip("/") == base_url.rstrip("/")

        hardware = driver.execute_script(
            """
            initSimParams();
            document.getElementById('sim-hw-mode').value = 'external-mpi';
            document.getElementById('sim-hw-cpu').value = '24';
            document.getElementById('sim-hw-mpi').value = '4';
            document.getElementById('sim-hw-use-gpu').checked = true;
            document.getElementById('sim-hw-gpu-count').value = '2';
            document.getElementById('sim-hw-gpu-ids').value = '0,1';
            document.getElementById('sim-hw-gmx').value = 'gmx_mpi';
            document.getElementById('sim-hw-launcher').value = 'srun';
            readStageParams();
            return collectSimulationParams().hardware;
            """
        )
        assert hardware == {
            "mode": "external-mpi",
            "cpu_threads": 24,
            "mpi_ranks": 4,
            "use_gpu": True,
            "gpu_count": 2,
            "gpu_ids": "0,1",
            "gmx_command": "gmx_mpi",
            "mpi_launcher": "srun",
            "pin": "auto",
        }
        simulation_shape = driver.execute_script(
            """
            const config = collectSimulationParams();
            return {
              keys: Object.keys(config).sort(),
              schemaVersion: config.schema_version,
              minimizationOwnsCutoff: Object.prototype.hasOwnProperty.call(config.minimization, 'rlist'),
              equilibrationOwnsCutoff: Object.prototype.hasOwnProperty.call(config.eq_stages[0], 'rlist'),
              productionOwnsPressureGeometry: Object.prototype.hasOwnProperty.call(config.prod_iters[0], 'pcoupl_type'),
              hasGlobalTemperature: Object.prototype.hasOwnProperty.call(config, 'temperature'),
              hasSystemName: Object.prototype.hasOwnProperty.call(config, 'system_name')
            };
            """
        )
        assert simulation_shape == {
            "keys": ["eq_stages", "hardware", "minimization", "prod_iters", "schema_version"],
            "schemaVersion": 2,
            "minimizationOwnsCutoff": True,
            "equilibrationOwnsCutoff": True,
            "productionOwnsPressureGeometry": True,
            "hasGlobalTemperature": False,
            "hasSystemName": False,
        }
        assert driver.find_element(
            By.ID, "sim-hw-omp"
        ).get_attribute("textContent") == "6"
        assert driver.execute_script(
            """
            _DEFAULT_EM.nsteps = 12345;
            initSimParams();
            return _DEFAULT_EM.nsteps;
            """
        ) == 50000
        membrane_protocol = driver.execute_script(
            """
            initSimParams();
            return {
              equilibrationStages: _simStages.length,
              productionSteps: _prodIters[0].nsteps,
              productionSegments: _prodIters[0].repeat,
              temperature: _prodIters[0].temperature,
              commGroups: _prodIters[0].comm_grps,
              rlist: _prodIters[0].rlist,
              switchDistance: _prodIters[0].rvdw_switch,
              dispersionCorrection: _prodIters[0].dispcorr
            };
            """
        )
        assert membrane_protocol == {
            "equilibrationStages": 6,
            "productionSteps": 5_000_000,
            "productionSegments": 5,
            "temperature": 310.15,
            "commGroups": "SOLU_MEMB SOLV",
            "rlist": 1.0,
            "switchDistance": None,
            "dispersionCorrection": "EnerPres",
        }

        driver.get(f"{base_url}/")
        solution_card = '.task-card[data-task-id="solvator"]'
        wait.until(lambda current: current.find_elements(By.CSS_SELECTOR, solution_card))
        driver.find_element(By.CSS_SELECTOR, solution_card).click()
        wait.until(
            lambda current: "active"
            in current.find_element(By.ID, "panel-input").get_attribute("class").split()
        )
        solution_protocol = driver.execute_script(
            """
            initSimParams();
            return {
              emSteps: _DEFAULT_EM.nsteps,
              equilibrationStages: _simStages.length,
              restraintBB: _simStages[0].bb,
              restraintSC: _simStages[0].sc,
              productionSteps: _prodIters[0].nsteps,
              productionSegments: _prodIters[0].repeat,
              commGroups: _prodIters[0].comm_grps
            };
            """
        )
        assert solution_protocol == {
            "emSteps": 5000,
            "equilibrationStages": 1,
            "restraintBB": 400,
            "restraintSC": 40,
            "productionSteps": 500_000,
            "productionSegments": 10,
            "commGroups": "SOLU SOLV",
        }

        charmm_nonbond = driver.execute_script(
            """
            const forceField = document.getElementById('ff-protein');
            forceField.value = 'charmm36m';
            forceField.dispatchEvent(new Event('change', {bubbles: true}));
            return {
              rlist: _prodIters[0].rlist,
              switchDistance: _prodIters[0].rvdw_switch,
              rvdw: _prodIters[0].rvdw,
              rcoulomb: _prodIters[0].rcoulomb,
              dispersionCorrection: _prodIters[0].dispcorr
            };
            """
        )
        assert charmm_nonbond == {
            "rlist": 1.2,
            "switchDistance": 1.0,
            "rvdw": 1.2,
            "rcoulomb": 1.2,
            "dispersionCorrection": "no",
        }

        # Both Martini cards must leave the task chooser. The bilayer card
        # allocates its task before Step 1 because protein-free mode is valid;
        # the solvent card waits for a structure upload.
        for task_type, route, expects_task_id in (
            ("martini3-bilayer", "Martini3BilayerBuilder", True),
            ("martini3-solvent", "Martini3SolventBuilder", False),
        ):
            driver.get(f"{base_url}/")
            selector = f'.task-card[data-task-id="{task_type}"]'
            wait.until(lambda current: current.find_elements(By.CSS_SELECTOR, selector))
            driver.find_element(By.CSS_SELECTOR, selector).click()
            wait.until(
                lambda current: "active" in current.find_element(
                    By.ID, "panel-input"
                ).get_attribute("class").split()
            )
            assert driver.current_url.rstrip("/").endswith(f"/{route}/Step1")
            displayed_task = driver.find_element(By.ID, "task-id-display").text.strip()
            assert (len(displayed_task) == 32) is expects_task_id

        modal_state = driver.execute_script(
            """
            showComputeQueueModal({
              status: 'queued',
              task_id: arguments[0],
              queue_position: 3,
              estimated_wait_seconds: 90,
              estimated_start_at: new Date(Date.now() + 90000).toISOString()
            });
            const modal = document.getElementById('compute-queue-modal');
            const close = document.getElementById('compute-queue-close');
            return {
              visible: !modal.classList.contains('hidden'),
              closeDisabled: close.disabled,
              taskId: document.getElementById('compute-queue-task-id').textContent,
              position: document.getElementById('compute-queue-position').textContent
            };
            """,
            task_id,
        )
        assert modal_state == {
            "visible": True,
            "closeDisabled": True,
            "taskId": task_id,
            "position": "3",
        }
        driver.find_element(By.ID, "compute-queue-saved").click()
        wait.until(
            lambda current: not current.find_element(
                By.ID, "compute-queue-close"
            ).get_attribute("disabled")
        )
        driver.find_element(By.ID, "compute-queue-close").click()
        assert "hidden" in driver.find_element(
            By.ID, "compute-queue-modal"
        ).get_attribute("class").split()
    finally:
        driver.quit()

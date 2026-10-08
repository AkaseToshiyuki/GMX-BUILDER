"""Queue UI communicates real waiting, uncertainty and task expiry."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_inline_queue_exposes_waiting_and_storage_pause(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script(
            "return typeof initComputeQueueStatus === 'function' && "
            "initComputeQueueStatus._done === true"
        )
    )
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script("""
        state.taskId='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
        startStepProgress('input','input-check-btn','input-check-status');
        showComputeQueueStatus({task_id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',status:'queued',
          queue_position:13, queue_length:27, ahead:12, waited_seconds:138,
          estimated_wait_seconds:null, expires_at:Date.now()/1000+3600,
          pause_reason:'Storage is full. New writes and starts are paused.'});
    """)
    assert page.find_element(By.ID, "compute-queue-status").is_displayed()
    assert "27 waiting / 12 ahead" in page.find_element(By.ID, "compute-queue-total").text
    assert "minutes" in page.find_element(By.ID, "compute-queue-waited").text
    assert (
        "Not enough comparable history" in page.find_element(By.ID, "compute-queue-estimate").text
    )
    assert "Storage is full" in page.find_element(By.ID, "compute-queue-resource").text
    assert page.find_element(By.ID, "compute-queue-expires").text != "—"
    assert page.find_elements(By.ID, "compute-queue-saved") == []
    assert page.find_element(By.ID, "compute-queue-status").get_attribute("role") != "dialog"
    assert page.find_element(By.ID, "compute-queue-copy").is_enabled()


def test_completed_operation_does_not_return_to_queued_label(page):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script(
            "return typeof initComputeQueueStatus === 'function' && "
            "initComputeQueueStatus._done === true"
        )
    )
    page.execute_script("""
      showComputeQueueStatus({task_id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',status:'completed'});
    """)
    assert not page.find_element(By.ID, "compute-queue-status").is_displayed()


def test_running_guidance_stays_beside_check_progress_without_moving_focus(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda d: d.execute_script("return initComputeQueueStatus._done === true")
    )
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    result = page.execute_script("""
        const button = document.getElementById('input-check-btn');
        button.focus();
        state.taskId='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
        const handle=startStepProgress('input','input-check-btn','input-check-status');
        const progress=handle.element;
        showComputeQueueStatus({task_id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',status:'running'});
        return {
          adjacent:progress.nextElementSibling.id,
          focus:document.activeElement.id,
          text:document.getElementById('compute-queue-status').textContent
        };
    """)
    assert result["adjacent"] == "compute-queue-status"
    assert result["focus"] == "input-check-btn"
    assert "Copy the Task ID" in result["text"] and "Resume" in result["text"]

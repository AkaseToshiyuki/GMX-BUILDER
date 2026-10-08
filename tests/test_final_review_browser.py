"""Final review stays separate, full-width and closed until the scene is ready."""

import pytest
from selenium.webdriver.support.ui import WebDriverWait

pytestmark = [pytest.mark.browser, pytest.mark.slow]


@pytest.mark.parametrize(
    "workflow",
    ["membrane-bilayer", "pure-membrane", "solvator", "martini3-bilayer", "martini3-solvent"],
)
def test_workflows_have_separate_final_review_and_full_width_checks(page, workflow):
    page.execute_async_script("selectTaskType(arguments[0]).then(arguments[1])", workflow)
    result = page.execute_script("""
      const bad=[];
      for(const panel of document.querySelectorAll('.panel')) {
        document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p===panel));
        for(const btn of panel.querySelectorAll('button[id*="check"], #ion-confirm-system-btn')) {
          if(!btn.getClientRects().length) continue;
          const width=btn.getBoundingClientRect().width;
          const parent=btn.parentElement;
          const css=getComputedStyle(parent);
          const available=parent.clientWidth
            -parseFloat(css.paddingLeft)-parseFloat(css.paddingRight);
          if(width<available-2) bad.push({id:btn.id,width,available});
        }
      }
      return {steps:state.taskType.visible_modules,bad,
        final:document.getElementById('ion-confirm-system-btn').closest('.panel').id,
        ion:document.getElementById('ion-check-btn').closest('.panel').id,
        viewer:document.getElementById('ion-viewer').closest('.panel').id,
        ionText:document.getElementById('panel-ions').textContent};
    """)
    assert "final_review" in result["steps"]
    assert result["final"] == "panel-final_review" and result["ion"] == "panel-ions"
    assert result["viewer"] == "panel-final_review"
    assert "not equilibrium ion sampling" in result["ionText"]
    assert "Experimental: dimensionless Metropolis site optimization" in result["ionText"]
    assert result["bad"] == []


def test_final_review_waits_for_render_and_rejects_stale_task_completion(page):
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script("""
      document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-final_review'));
      state.currentStepIdx=state.wizardSteps.indexOf('final_review');
      window.originalRender=GMXViewer.render;
      GMXViewer.render=()=>new Promise(resolve=>window.finishDisplay=resolve);
      window.reviewDone=false;renderFinalReview().then(()=>window.reviewDone=true);
    """)
    assert page.execute_script("return document.getElementById('ion-confirm-system-btn').disabled")
    page.execute_script("""
      state.taskId='different-task';
      finishDisplay({revision:'old',source_step:'ions',components:[],atom_count:10});
    """)
    WebDriverWait(page, 5).until(lambda d: d.execute_script("return reviewDone"))
    assert page.execute_script(
        "return document.getElementById('ion-confirm-system-btn').disabled "
        "&& !isFinalReviewConfirmed()"
    )
    page.execute_script("""
      window.reviewDone=false;renderFinalReview().then(()=>window.reviewDone=true);
      finishDisplay({revision:'new',source_step:'ions',components:[],atom_count:10});
    """)
    WebDriverWait(page, 5).until(lambda d: d.execute_script("return reviewDone"))
    assert page.execute_script("return !document.getElementById('ion-confirm-system-btn').disabled")
    page.execute_script("""
      document.querySelector('#panel-final_review .next-btn').disabled=false;
      invalidateFinalReview();GMXViewer.render=originalRender;
    """)
    assert page.execute_script(
        "return document.querySelector('#panel-final_review .next-btn').disabled"
    )
    assert page.execute_script("return document.getElementById('ion-confirm-system-btn').disabled")


def test_enabled_categories_and_navigation_have_no_global_queue_notice(page):
    assert (
        page.execute_script(
            "return document.querySelectorAll('#task-grid .card-category-header').length"
        )
        >= 3
    )
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_script(
        "showComputeQueueStatus({status:'running',task_id:state.taskId,queue_position:0})"
    )
    assert page.execute_script(
        "return document.getElementById('compute-queue-status').classList.contains('hidden')"
    )


@pytest.mark.parametrize(
    "slug,workflow",
    [
        ("Martini3BilayerBuilder", "martini3-bilayer"),
        ("Martini3SolventBuilder", "martini3-solvent"),
    ],
)
def test_martini_workflow_route_round_trips_without_task_context(page, live_server, slug, workflow):
    page.get(live_server + "/" + slug + "/Step8")
    WebDriverWait(page, 15).until(
        lambda d: d.execute_script(
            "return typeof state!=='undefined' && state.taskType!==null && state.currentStepIdx===0"
        )
    )
    assert page.execute_script("return state.taskType.id") == workflow
    assert page.current_url.endswith("/" + slug + "/Step1")
    assert "final_review" in page.execute_script("return state.wizardSteps")

"""Stored simulation drafts must remain data at the browser rendering boundary."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


@pytest.mark.parametrize("field", ["bb", "sc", "lipid", "nsteps", "em_nsteps"])
@pytest.mark.parametrize(
    "payload",
    [
        '</span></button><a id="injected-stage" href="https://attacker.invalid" '
        'style="position:fixed;inset:0">Untrusted instructions</a><span>',
        '</span></button><svg id="injected-stage"><text>Untrusted</text></svg><span>',
    ],
)
def test_restored_stage_markup_is_inert_text(page, field, payload):
    result = page.execute_script(
        """
        initSimParams();
        const field = arguments[0], payload = arguments[1];
        const saved = {
          eq_stages: _simStages.map(stage => Object.assign({}, stage)),
          prod_iters: _prodIters.map(stage => Object.assign({}, stage))
        };
        saved.eq_stages[0].enabled = false;
        if (field === 'em_nsteps') saved.minimization = {nsteps: payload};
        else saved.eq_stages[0][field] = payload;
        restoreSimulationParams(saved);
        const injectedInitially = !!document.getElementById('injected-stage');
        renderSimStages();
        const selector = field === 'em_nsteps' ? '[data-stage="em"]' : '[data-stage="eq0"]';
        return {
          injectedInitially,
          injectedAfterRender: !!document.getElementById('injected-stage'),
          summary: document.querySelector(selector + ' .sim-stage-summary').textContent,
          enabled: document.getElementById('eq-enabled-0').checked,
          unchangedValue: field === 'em_nsteps' ? _DEFAULT_EM.nsteps : _simStages[0][field],
          ordinaryStage: document.getElementById('eq-nsteps-1').value
        };
        """,
        field,
        payload,
    )
    assert not result["injectedInitially"]
    assert not result["injectedAfterRender"]
    assert payload in result["summary"]
    assert result["unchangedValue"] == payload
    assert not result["enabled"]
    assert int(result["ordinaryStage"]) > 0


def test_restored_ordinary_and_adjacent_stage_fields(page):
    result = page.execute_script(
        """
        initSimParams();
        const payload = '<a id="injected-stage">untrusted</a>';
        const saved = {
          eq_stages: _simStages.map(stage => Object.assign({}, stage)),
          prod_iters: _prodIters.map(stage => Object.assign({}, stage))
        };
        saved.eq_stages[0].enabled = false;
        saved.eq_stages[0].bb = -10;
        saved.eq_stages[0].dt = 10;
        saved.eq_stages[0].mdp_overrides = {userint1: payload};
        saved.prod_iters[0].repeat = 3;
        restoreSimulationParams(saved, {gmx_command: payload, gpu_ids: payload});
        const normal = {
          em: document.querySelector('[data-stage="em"] .sim-stage-summary').textContent,
          eq: document.querySelector('[data-stage="eq0"] .sim-stage-summary').textContent,
          prod: document.querySelector('[data-stage="prod0"] .sim-stage-summary').textContent,
          gmx: document.getElementById('sim-hw-gmx').value,
          overrides: document.getElementById('eq-mdp-overrides-0').value
        };
        saved.prod_iters[0].nsteps = payload;
        saved.prod_iters[0].repeat = payload;
        restoreSimulationParams(saved);
        return {normal, injected: !!document.getElementById('injected-stage'), payload};
        """
    )
    assert not result["injected"]
    assert "emtol=1000" in result["normal"]["em"]
    assert "SKIPPED" in result["normal"]["eq"]
    assert "BB=-10" in result["normal"]["eq"]
    assert "10.0 fs" in result["normal"]["eq"]
    assert "3 segments" in result["normal"]["prod"]
    assert result["normal"]["gmx"] == result["payload"]
    assert result["payload"] in result["normal"]["overrides"]

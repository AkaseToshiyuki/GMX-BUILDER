/* A confirmation belongs to one rendered final checkpoint, never a navigation index. */
(function () {
  'use strict';
  let ready = null,
    confirmed = null,
    request = 0;
  const button = () => document.getElementById('ion-confirm-system-btn');
  const message = () => document.getElementById('ion-confirm-system-status');
  function invalidate() {
    request++;
    ready = null;
    confirmed = null;
    if (typeof _checkedSteps !== 'undefined') _checkedSteps.delete('final_review');
    if (button()) button().disabled = true;
    if (message())
      message().textContent = 'Load and inspect the current final structure before confirming.';
    if (typeof updateNextButtonState === 'function') updateNextButtonState();
    if (typeof updateStepNavHighlight === 'function') updateStepNavHighlight();
  }
  function source() {
    if (isCoarseGrainedWorkflow()) return 'cg_system';
    return state.taskType?.pipeline === 'pure_membrane' && !pureMembraneIncludesSolvent()
      ? 'membrane'
      : 'ions';
  }
  async function render() {
    invalidate();
    const token = request,
      task = state.taskId;
    const data = await GMXViewer.render('ion-viewer', source());
    if (!data || token !== request || task !== state.taskId) return;
    ready = { ...data, task };
    const summary = document.getElementById('final-component-summary');
    summary.replaceChildren();
    const table = document.createElement('table');
    table.className = 'ion-summary-table';
    const heading = document.createElement('tr');
    ['Component', 'Type', 'Saved atoms', 'Molecules'].forEach((text) => {
      const cell = document.createElement('th');
      cell.textContent = text;
      heading.appendChild(cell);
    });
    table.appendChild(heading);
    for (const component of data.components) {
      const row = document.createElement('tr');
      [
        component.name,
        component.kind,
        component.atoms.toLocaleString(),
        component.lipids ?? component.molecules ?? '—',
      ].forEach((text) => {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.appendChild(cell);
      });
      table.appendChild(row);
    }
    summary.appendChild(table);
    document.getElementById('ion-viewer-label').textContent =
      'Exact checked coordinates. Lipids: orange/purple; water: blue; ions: spheres. Hydrogens are omitted from the display only.';
    message().textContent =
      'Inspect every component and the periodic box, then confirm this structure.';
    button().disabled = false;
  }
  async function confirm() {
    if (
      !ready ||
      ready.task !== state.taskId ||
      !document.getElementById('panel-final_review').classList.contains('active')
    )
      return;
    const selected = ready,
      token = request;
    button().disabled = true;
    try {
      const response = await fetch('/api/task/' + selected.task + '/final-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_step: selected.source_step, revision: selected.revision }),
      });
      const result = await response.json();
      if (!response.ok || !result.confirmed) throw new Error(result.error || 'Confirmation failed');
      // Ignore a response for a task or rendered checkpoint superseded during the request.
      if (token !== request || selected !== ready || selected.task !== state.taskId) return;
      confirmed = { ...result, task: selected.task };
      _checkedSteps.add('final_review');
      const legacy = document.getElementById('cg-confirm-system');
      if (legacy) legacy.checked = true;
      message().textContent = '✓ This checkpoint is confirmed. Continue to Simulation Parameters.';
      updateNextButtonState();
      updateStepNavHighlight();
    } catch (error) {
      if (token === request) {
        invalidate();
        message().textContent = error.message + ' Reload with Retry viewer before confirming.';
      }
    }
  }
  window.invalidateFinalReview = invalidate;
  window.renderFinalReview = render;
  window.isFinalReviewConfirmed = () =>
    Boolean(
      ready &&
        confirmed &&
        confirmed.task === state.taskId &&
        (confirmed.rendered_revision || confirmed.revision) === ready.revision,
    );
  document.addEventListener('DOMContentLoaded', () => {
    button().addEventListener('click', confirm);
    document.getElementById('final-viewer-retry').addEventListener('click', () => {
      GMXViewer.invalidate();
      render();
    });
    document.addEventListener('input', (event) => {
      if (event.target.dataset.viewerControl) return;
      const panel = event.target.closest('.panel');
      if (panel && !['panel-final_review', 'panel-simparams', 'panel-task-type'].includes(panel.id))
        invalidate();
    });
  });
})();

window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push('final_review.js');

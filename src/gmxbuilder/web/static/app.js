/* ============================================================
   GMXBUILDER Web Interface — Application Logic v2
   Dynamic step navigation based on selected task type.
   ============================================================ */

// ----- State -----
const state = {
  taskType: null,         // selected task type detail from /api/task-type/{id}
  wizardSteps: [],        // ordered list of module names to show as steps
  currentStepIdx: 0,      // index into wizardSteps[]
  taskId: null,            // assigned after PDB upload
  completedSteps: new Set(),  // set of step indices that are fulfilled
  pdbInfo: null,
  buildRunning: false,
  customLipidBusy: false,
};
// `const` at script scope is not a property of `window`, which puts the whole
// wizard state out of reach of anything driving the page from outside it --
// a browser test, or the console. Exposed the same way the viewer and the
// chain/molecule selections already are. Mutated in place, never reassigned,
// so the two names never diverge.
window.state = state;
var _cgenffUploads = {};
var _ligandChargeDrafts = {};
var _charmmCompatSmiles = {};
var _charmmCompatResearch = false;
var _charmmIdentitySources = {};
var _charmmMol2Uploads = {};
var _charmmIdentityRequest = 0;
var _charmmIdentityTimer = null;
var _restoredLigandBackend = null;
var _ligandChargeOrigins = {};
var _computedLigandCharges = {};
var _computedLigandChargePH = null;
var _ligandChargeRequest = 0;

// Step metadata — title + icon for each module name
const STEP_META = {
  input:      { title: 'PDB Upload' },
  structure:  { title: 'Structure Processing' },
  membrane:   { title: 'Membrane Builder' },
  orient:     { title: 'Orientation' },
  solvation:  { title: 'Solvent & Box' },
  ions:       { title: 'Ions' },
  final_review: { title: 'Final Structure Review' },
  forcefield: { title: 'Force Field' },
  topology:   { title: 'Topology & Parameters' },
  simparams:  { title: 'Simulation Params' },
  cg_model: { title: 'Martini 3 Model' },
  cg_mapping: { title: 'Protein Mapping' },
  cg_orientation: { title: 'Orientation' },
  cg_environment: { title: 'CG Environment' },
  cg_solvation: { title: 'CG Solvation' },
  cg_system: { title: 'Final CG System' },
};

// Category colors for task cards
const CAT_COLORS = {
  'Membrane':    '#2563eb',
  'Solution':    '#0891b2',
  'Coarse Grained': '#16a34a',
  'Glycan':      '#7c3aed',
  'Ligand':      '#db2777',
  'Nanomaterial':'#ea580c',
};

const WORKFLOW_ROUTES = {
  'membrane-bilayer': 'BilayerBuilder',
  'pure-membrane': 'PureBilayerSystem',
  'solvator': 'Solvator',
  'martini3-bilayer': 'Martini3BilayerBuilder',
  'martini3-solvent': 'Martini3SolventBuilder',
};
const ROUTE_WORKFLOWS = Object.fromEntries(
  Object.entries(WORKFLOW_ROUTES).map(function(entry) { return [entry[1], entry[0]]; })
);
let _restoringRoute = false;
let _resumeError = '';
function showResumeError(message) {
  let notice=document.getElementById('task-resume-error');
  if(!notice) {
    notice=document.createElement('div');notice.id='task-resume-error';
    notice.className='input-check-report error';notice.setAttribute('role','alert');
  }
  const panel=document.querySelector('.panel.active') || document.getElementById('panel-task-type');
  panel.querySelector('h2').after(notice);notice.textContent=message;
}

async function copyTaskIdToClipboard() {
  const taskId = String(state.taskId || '').trim();
  if (!taskId) return;
  const status = document.getElementById('copy-task-id-status');
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(taskId);
    } else {
      const temporary = document.createElement('textarea');
      temporary.value = taskId;
      temporary.setAttribute('readonly', '');
      temporary.style.position = 'fixed';
      temporary.style.opacity = '0';
      document.body.appendChild(temporary);
      temporary.select();
      if (!document.execCommand('copy')) throw new Error('copy command rejected');
      temporary.remove();
    }
    if (status) status.textContent = 'Copied';
  } catch (_error) {
    if (status) status.textContent = 'Copy failed';
  }
  window.setTimeout(function() { if (status) status.textContent = ''; }, 1600);
}

function taskRouteSlug() {
  if (!state.taskType) return '';
  return state.taskType.route_slug || WORKFLOW_ROUTES[state.taskType.id] || '';
}

function syncTaskRoute(stepIdx, forceReplace) {
  const slug = taskRouteSlug();
  if (!slug || stepIdx < 0) return;
  // Keep task identifiers out of URLs; remember only this tab's task context.
  rememberCurrentTask();
  const target = '/' + slug + '/Step' + (stepIdx + 1);
  if (window.location.pathname === target) return;
  if (_restoringRoute || forceReplace) history.replaceState({}, '', target);
  else history.pushState({}, '', target);
}

function parseTaskRoute(pathname) {
  const match = String(pathname || '').match(
    /^\/([A-Za-z0-9]+)\/Step(\d+)\/?$/
  );
  if (!match || !ROUTE_WORKFLOWS[match[1]]) return null;
  return {
    workflow: ROUTE_WORKFLOWS[match[1]],
    stepIdx: Math.max(0, parseInt(match[2], 10) - 1),
  };
}

function rememberCurrentTask() {
  if (!state.taskType || state.currentStepIdx < 0) return;
  try {
    sessionStorage.setItem('gmxbuilder-current-task', JSON.stringify({
      task_id: state.taskId, workflow: state.taskType.id, step: state.currentStepIdx
    }));
  } catch (_error) { /* Manual resume remains available when storage is disabled. */ }
}

function forgetCurrentTask() {
  try { sessionStorage.removeItem('gmxbuilder-current-task'); } catch (_error) {}
}

function exitCurrentTask() {
  forgetCurrentTask();
  // A fresh document releases viewers and polling without cancelling server work.
  window.location.replace('/');
}

function acceptedManagedUpload(ticket) {
  if (!ticket || !/^[a-f0-9]{32}$/.test(ticket.task_id || '')) return;
  state.taskId = ticket.task_id;
  document.getElementById('task-id-display').textContent = ticket.task_id;
  document.getElementById('header-task-id').classList.remove('hidden');
  rememberCurrentTask();
}

async function restoreRouteFromLocation() {
  const route = parseTaskRoute(window.location.pathname);
  if (!route) { goToTaskSelect(); return; }
  var saved = null;
  try { saved = JSON.parse(sessionStorage.getItem('gmxbuilder-current-task')); } catch (_error) {}
  _restoringRoute = true;
  var panels = document.getElementById('panels');
  try {
    if (saved && saved.workflow === route.workflow && /^[a-f0-9]{12}(?:[a-f0-9]{20})?$/.test(saved.task_id || '')) {
      panels.inert = true;
      document.getElementById('panel-task-type').classList.remove('active');
      showComputeQueueStatus({task_id:saved.task_id, status:'running', message:'Restoring the saved task and current operation status…'});
      if (await resumeTask(saved.task_id, route.stepIdx)) return;
      forgetCurrentTask();
    }
    await selectTaskType(route.workflow);
    // A URL without saved task context cannot prove that later steps passed.
    goToWizardStep(0);
    if(_resumeError) showResumeError(_resumeError);
  } finally {
    panels.inert = false;
    _restoringRoute = false;
    if (state.currentStepIdx >= 0) syncTaskRoute(state.currentStepIdx, true);
  }
}

// ----- Init -----
document.addEventListener('DOMContentLoaded', async () => {
  console.log('DOMContentLoaded START');
  const taskGrid = document.getElementById('task-grid');
  taskGrid.inert = true;
  try {
  loadOptions();
  const tasksLoaded = await loadTaskTypes();
  if (!tasksLoaded) {
    const notice = document.createElement('div');
    notice.id = 'startup-load-error';
    notice.setAttribute('role', 'alert');
    notice.textContent = 'Workflow options could not be loaded. Please retry. ';
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.textContent = 'Retry loading page';
    retry.addEventListener('click', () => window.location.reload());
    notice.appendChild(retry);
    taskGrid.before(notice);
    return;
  }
  setupUpload();
  setupRunButton();
  initOrientationStep();
  initStructureProcessing();
  
  initCustomLipidModal();
  initComputeQueueStatus();
  initSimParams();
  initCheckButtons();
  initCoarseGrainedControls();
    // Solvation Check button
  var solvBtn = document.getElementById("solv-check-btn");
  if (solvBtn) {
    solvBtn.addEventListener("click", async function() {
      await _doCheckStep('solvation', 'solv-check-status', 'solv-check-btn');

    });
  }
  // Any Step 6 input change invalidates the saved checkpoint and requires
  // another backend check. Do not replace backend results with a browser estimate.
  ['box-padding', 'overlap-scale'].forEach(function(id) {
    var input = document.getElementById(id);
    if (input) input.addEventListener('input', resetSolvCheck);
  });
  var ffSelect = document.getElementById('ff-protein');
  if (ffSelect) ffSelect.addEventListener('change', function() {
    updateWaterModelOptions(true);
    resetForceFieldCheck();
    refreshForceFieldCompatibility();
    syncMdpNonbondDefaults();
    renderSimStages();
    reloadModificationCatalog();
    reloadCrosslinkCapabilities();
    reloadTerminalCapabilities();
    const dropdown = document.getElementById('lipid-picker-dropdown');
    if (dropdown && !dropdown.classList.contains('hidden')) renderLipidList('');
  });
  var ffWaterSelect = document.getElementById('ff-water-model');
  if (ffWaterSelect) ffWaterSelect.addEventListener('change', function() {
    syncLockedWaterModelDisplay();
    resetForceFieldCheck();
  });
  ['ff-lipid', 'ff-ligand'].forEach(function(id) {
    var select = document.getElementById(id);
    if (select) select.addEventListener('change', function() {
      resetForceFieldCheck();
      renderLigandChargeInputs();
      updateV4CompositionAvailability();
      const dropdown = document.getElementById('lipid-picker-dropdown');
      if (dropdown && !dropdown.classList.contains('hidden')) renderLipidList('');
    });
  });
  var pureSolventToggle = document.getElementById('pure-membrane-include-solvent');
  if (pureSolventToggle) {
    pureSolventToggle.addEventListener('change', function() {
      syncPureMembraneSolvationOption(true);
    });
  }
  // Resume task button
  var resumeBtn = document.getElementById("resume-task-btn");
  if (resumeBtn) {
    resumeBtn.addEventListener("click", function() {
      var tid = document.getElementById("resume-task-id").value.trim();
      if (!tid) { alert("Please enter a task ID."); return; }
      resumeTask(tid);
    });
  }
  var copyTaskIdButton = document.getElementById('copy-task-id');
  if (copyTaskIdButton) copyTaskIdButton.addEventListener('click', copyTaskIdToClipboard);
  document.getElementById('exit-task')?.addEventListener('click', exitCurrentTask);
  await restoreRouteFromLocation();
  window.addEventListener('popstate', function() {
    restoreRouteFromLocation();
  });
  taskGrid.inert = false;
  console.log('DOMContentLoaded END — all init functions called');
  } catch(e) { console.error('DOMContentLoaded error:', e.message, e.stack); }

  // Cleanup build-polling intervals on page unload (prevent stale HTTP requests)
  window.addEventListener('beforeunload', function() {
    if (window._buildPollTimers) {
      window._buildPollTimers.forEach(function(t) { GMXPoll.stop(t); });
      window._buildPollTimers = [];
    }
  });
});

// ===================================================================
// Task Type Loading
// ===================================================================

async function loadTaskTypes() {
  try {
    const data = await GMXHttp.json('/api/task-types', {}, {
      validate: data => Array.isArray(data.task_types)
    });
    renderTaskCards(data.task_types);
    return true;
  } catch (err) {
    console.error('Failed to load task types:', err);
    return false;
  }
}

function renderTaskCards(taskTypes) {
  const grid = document.getElementById('task-grid');
  grid.innerHTML = '';
  const planned = taskTypes.filter(t => !t.enabled);
  const plannedSection = document.createElement('details');
  plannedSection.className = 'planned-workflows';
  const summary = document.createElement('summary');
  summary.textContent = 'Planned workflows (' + planned.length + ')';
  plannedSection.appendChild(summary);
  const plannedGrid = document.createElement('div');
  plannedGrid.className = 'task-grid';
  plannedSection.appendChild(plannedGrid);

  // Group by category
  const grouped = {};
  taskTypes.forEach(t => {
    if (!grouped[t.category]) grouped[t.category] = [];
    grouped[t.category].push(t);
  });

  // Both availability sections keep their own category headings.
  for (const [cat, types] of Object.entries(grouped)) {
    // Category header
    const header = document.createElement('h3');
    header.className = 'card-category-header';
    header.style.color = CAT_COLORS[cat] || '#64748b';
    header.textContent = cat;
    if (types.some(t => t.enabled)) grid.appendChild(header.cloneNode(true));
    if (types.some(t => !t.enabled)) plannedGrid.appendChild(header);

    types.forEach(t => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'task-card' + (t.enabled ? '' : ' disabled');
      card.dataset.taskId = t.id;
      card.disabled = !t.enabled;
      card.setAttribute('aria-pressed', 'false');

      const badge = t.enabled ? '' : '<span class="card-badge">Coming Soon</span>';

      // Safe DOM construction — avoid XSS from API-provided strings
      const iconEl = document.createElement('span'); iconEl.className = 'card-icon'; iconEl.textContent = t.icon;
      const catEl = document.createElement('span'); catEl.className = 'card-category'; catEl.style.color = CAT_COLORS[cat]||'#64748b'; catEl.textContent = t.category;
      const titleEl = document.createElement('span'); titleEl.className = 'card-title'; titleEl.textContent = t.title;
      const descEl = document.createElement('span'); descEl.className = 'card-desc'; descEl.textContent = t.description;
      card.appendChild(iconEl); card.appendChild(catEl); card.appendChild(titleEl); card.appendChild(descEl);
      if (badge) { const badgeEl = document.createElement('span'); badgeEl.className = 'card-badge'; badgeEl.textContent = 'Coming Soon'; card.appendChild(badgeEl); }

      if (t.enabled) {
        card.addEventListener('click', () => selectTaskType(t.id));
      }
      (t.enabled ? grid : plannedGrid).appendChild(card);
    });
  }
  if (planned.length) grid.appendChild(plannedSection);
}

async function selectTaskType(taskId) {
  try {
    // Show selection
    document.querySelectorAll('.task-card').forEach(c => {
      const selected = c.dataset.taskId === taskId;
      c.classList.toggle('selected', selected);
      c.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });

    state.taskType = await GMXHttp.json(`/api/task-type/${taskId}`, {}, {
      validate: data => Array.isArray(data.visible_modules)
    });
    state.wizardSteps = state.taskType.visible_modules || [];
    state.currentStepIdx = 0;
    state.completedSteps = new Set();   // reset locks on new task type
    state.pdbInfo = null;               // reset PDB info
    state.taskId = null;
    _charmmCompatSmiles = {};
    _charmmCompatResearch = false;
    _restoredLigandBackend = null;
    resetLigandPHState();
    window._ffCompatibility = null;
    _ffCompatibilityValid = false;
    var taskIdText = document.getElementById('task-id-display');
    var taskIdBox = document.getElementById('header-task-id');
    if (taskIdText) taskIdText.textContent = '';
    if (taskIdBox) taskIdBox.classList.add('hidden');
    state.customLipidBusy = false;
    _orientedPdbContent = null;
    _membraneCheckpointPdb = null;
    _membraneActualBox = null;
    _membraneActualCounts = null;
    _checkedSteps.clear();
    _checkedConfig = null;
    _compositionChecked = false;
    updateCompositionStatus();
    syncTaskRoute(0);

    // Task type selection itself is always "done"
    state.completedSteps.add(-1);

    // Update header
    document.getElementById('header-task-title').textContent = state.taskType.title;

    // Workflows without an uploaded structure still need a persistent task
    // before their first Check button can create a checkpoint.
    if (state.taskType.requires_input === false) {
      var createResponse = await fetch('/api/tasks', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task_type: taskId}),
      });
      var createResult = await createResponse.json();
      if (!createResponse.ok || !createResult.task_id) {
        throw new Error(createResult.error || 'Could not create task');
      }
      state.taskId = createResult.task_id;
      rememberCurrentTask();
      if (taskIdText) taskIdText.textContent = state.taskId;
      if (taskIdBox) taskIdBox.classList.remove('hidden');
      setTimeout(loadTaskCustomLipids, 0);
      syncTaskRoute(0, true);
    }

    if (state.taskType.requires_input === false && !await ensureOptionsLoaded()) return;
    if (isCoarseGrainedWorkflow()) loadMartiniLipidCapabilities().catch(function() {});
    configureTaskSpecificControls();

    // Adapt UI for liquid builder: no solvation check needed, different labels
    if (taskId === 'liquid-builder') {
      // Compute box dimensions for liquid (pure solvent — no protein)
      var bx = parseFloat(document.getElementById('box-padding')?.value) || 5.0;
      var wm = document.getElementById('ff-water-model')?.value || 'tip3p';
      var boxVol = bx * bx * bx;
      var waterVol = boxVol * 0.96;
      var nWater = Math.round(waterVol / 0.0299);
      _waterVolume = waterVol;
      _waterCount = nWater;
      _solvChecked = true;
      // Update solvation result display
      var solvDims = document.getElementById("solv-box-dims");
      if (solvDims) solvDims.textContent = bx.toFixed(1) + ' x ' + bx.toFixed(1) + ' x ' + bx.toFixed(1) + ' = ' + boxVol.toFixed(0) + ' nm³';
      var solvVol = document.getElementById("solv-box-vol");
      if (solvVol) solvVol.textContent = boxVol.toFixed(0);
      var solvNWat = document.getElementById("solv-n-water");
      if (solvNWat) solvNWat.textContent = nWater.toLocaleString();
      var solvWm = document.getElementById("solv-water-model");
      if (solvWm) solvWm.textContent = wm.toUpperCase();
      var solvResult = document.getElementById("solv-result");
      if (solvResult) solvResult.classList.remove("hidden");
      var wmLabel = document.querySelector('label[for=\"ff-water-model\"]');
      if (wmLabel) wmLabel.textContent = 'Solvent Type:';
      var padLabel = document.querySelector('label[for=\"box-padding\"]');
      if (padLabel) padLabel.textContent = 'Box Size X (nm):';
      var padEl = document.getElementById('box-padding');
      if (padEl) { padEl.placeholder = '5.0'; padEl.value = '5.0'; }
      document.getElementById('solv-check-btn')?.classList.add('hidden');
      document.getElementById('solv-result')?.classList.add('hidden');
      // Populate solvent dropdown with water + organic options
      if (window._allSolvents) {
        populateSelect('ff-water-model', window._allSolvents.map(function(s) {
          var label = s.label + ' (' + s.density.toFixed(2) + ' g/mL)';
          if (s.category === 'organic') label += ' [ITP only]';
          return {value: s.name, label: label};
        }));
      }
    } else {
      // Restore water-model dropdown for non-liquid task types
      var wmLabel2 = document.querySelector('label[for=\"ff-water-model\"]');
      if (wmLabel2) wmLabel2.textContent = 'Water Model:';
      updateWaterModelOptions(true);
      var padLabel2 = document.getElementById('box-padding-label');
      var padHint2 = document.getElementById('box-padding-hint');
      if (state.taskType && state.taskType.pipeline === 'solvator') {
        if (padLabel2) padLabel2.textContent = 'Padding on All Sides (nm)';
        if (padHint2) padHint2.textContent = 'The solute is translated into the box with this clearance on all six faces.';
      } else {
        if (padLabel2) padLabel2.innerHTML = 'Z Padding (nm) <span class="hint">— water thickness above &amp; below membrane</span>';
        if (padHint2) padHint2.textContent = 'Box XY is fixed by Membrane step; this padding only extends the Z direction.';
      }
    }

    // Build step nav
    renderStepNav();

    // Load defaults from task type
    loadTaskDefaults();
    syncPureMembraneSolvationOption(false);
    // Select the scientific protocol only after the exact workflow and force
    // field defaults are known.
    initSimParams();

    // Navigate to first builder step
    goToWizardStep(0);

  } catch (err) {
    console.error('Failed to select task type:', err);
  }
}

function configureTaskSpecificControls() {
  var pipeline = state.taskType ? state.taskType.pipeline : '';
  var coarseGrained = isCoarseGrainedWorkflow();
  var primaryLabel = document.getElementById('ff-primary-label');
  var lipidField = document.getElementById('ff-lipid-field');
  var ligandField = document.getElementById('ff-ligand-field');
  var pureOption = document.getElementById('pure-membrane-solvation-option');
  var cgInputOptions = document.getElementById('cg-input-options');
  var atomisticSimParams = document.getElementById('simparams-stages');
  var cgSimParams = document.getElementById('cg-simparams');
  var simDescription = document.querySelector('#panel-simparams > .section-desc');
  if (primaryLabel) {
    primaryLabel.textContent = pipeline === 'pure_membrane'
      ? 'Force Field Family'
      : 'Protein Force Field';
  }
  if (lipidField) lipidField.classList.toggle('hidden', pipeline === 'solvator');
  if (ligandField) ligandField.classList.toggle('hidden', pipeline === 'pure_membrane');
  if (pureOption) pureOption.classList.toggle('hidden', pipeline !== 'pure_membrane');
  if (cgInputOptions) cgInputOptions.classList.toggle('hidden', !coarseGrained);
  if (atomisticSimParams) atomisticSimParams.classList.toggle('hidden', coarseGrained);
  if (cgSimParams) cgSimParams.classList.toggle('hidden', !coarseGrained);
  if (simDescription) simDescription.textContent = coarseGrained
    ? 'Conservative Martini 3 minimization, optional serial equilibration, and production settings.'
    : 'Six-stage equilibration with decaying restraints, followed by production. Click any stage to expand and tune.';
  var systemName = document.getElementById('system-name');
  if (systemName) {
    if (coarseGrained) systemName.value = 'martini3_system';
    else if (pipeline === 'pure_membrane') systemName.value = 'pure_bilayer';
    else if (pipeline === 'solvator') systemName.value = 'solvator_system';
    else systemName.value = 'membrane_system';
  }
  syncCoarseGrainedInputControls(false);
  updateCgOrientationControlLabels();
  syncCgOrientationMode();
}

function isCoarseGrainedWorkflow() {
  return !!(state.taskType && [
    'martini3-bilayer', 'martini3-solvent'
  ].includes(state.taskType.id));
}

function coarseGrainedEnvironment() {
  if (!state.taskType) return 'bilayer';
  if (state.taskType.id === 'martini3-solvent') return 'solution';
  if (state.taskType.id === 'martini3-bilayer') return 'bilayer';
  return state.taskType.default_config?.input?.environment || 'bilayer';
}

function coarseGrainedIncludesProtein() {
  if (!isCoarseGrainedWorkflow()) return false;
  return document.getElementById('cg-include-protein')?.checked !== false;
}

let _cgOrientMode = 'ppm';
let _cgOrientZOffset = 0.0;
let _cgOrientTilt = 0.0;
let _cgOrientPhi = 0.0;
let _cgOrientedPdbContent = null;
let _cgOrientPreviewRequestId = 0;
let _cgOrientPreviewTimer = null;

function syncCgOrientationMode() {
  document.querySelectorAll('.cg-orient-tab').forEach(function(tab) {
    tab.classList.toggle('active', tab.dataset.method === _cgOrientMode);
  });
  document.getElementById('cg-orient-auto-result')?.classList.toggle(
    'hidden', _cgOrientMode !== 'ppm'
  );
  document.getElementById('cg-orient-manual')?.classList.toggle(
    'hidden', _cgOrientMode !== 'manual'
  );
}

function syncCoarseGrainedInputControls(invalidate) {
  if (!isCoarseGrainedWorkflow()) return;
  var environment = coarseGrainedEnvironment();
  var include = document.getElementById('cg-include-protein');
  if (environment === 'solution' && include) {
    include.checked = true;
    include.disabled = true;
  } else if (include) {
    include.disabled = false;
  }
  var includeProtein = include ? include.checked : true;
  var bilayerControls = document.getElementById('cg-bilayer-environment-controls');
  if (bilayerControls) bilayerControls.classList.toggle('hidden', environment !== 'bilayer');
  var environmentTitle = document.querySelector('#panel-cg_environment > h2');
  var environmentDescription = document.getElementById('cg-environment-description');
  if (environmentTitle) environmentTitle.textContent = environment === 'bilayer'
    ? 'Bilayer Builder' : 'Protein Placement';
  if (environmentDescription) environmentDescription.textContent = environment === 'bilayer'
    ? 'Choose the exact leaflet size and independent upper/lower Martini lipid compositions.'
    : 'Review the mapped protein pose before creating its solvent box.';
  STEP_META.cg_environment.title = environment === 'bilayer' ? 'Bilayer Builder' : 'Protein Placement';
  STEP_META.cg_solvation.title = 'Solvent & Box';
  STEP_META.cg_system.title = 'Ions';
  var upload = document.getElementById('upload-zone');
  var uploadInfo = document.getElementById('upload-info');
  if (upload) upload.classList.toggle('hidden', !includeProtein);
  if (uploadInfo && !includeProtein) uploadInfo.classList.add('hidden');
  var mappingControls = document.getElementById('cg-mapping-controls');
  if (mappingControls) mappingControls.classList.toggle('hidden', !includeProtein);
  var mappingButton = document.getElementById('cg-mapping-check');
  if (mappingButton) mappingButton.textContent = includeProtein ? '✓ Map Protein' : '✓ Skip Protein Mapping';
  document.querySelectorAll('.cg-bilayer-control').forEach(function(element) {
    element.classList.toggle('hidden', environment !== 'bilayer');
  });
  var orientationPanel = document.getElementById('panel-cg_orientation');
  if (orientationPanel) orientationPanel.classList.toggle('cg-not-applicable', !includeProtein);
  var solvent = document.getElementById('cg-include-solvent');
  if (solvent) {
    if (environment === 'solution') solvent.checked = true;
    solvent.disabled = environment === 'solution';
  }
  var paddingLabel = document.getElementById('cg-padding-label');
  var paddingHint = document.getElementById('cg-padding-hint');
  var padding = document.getElementById('cg-padding');
  if (paddingLabel) paddingLabel.textContent = environment === 'bilayer'
    ? 'Z padding above and below the bilayer (nm)'
    : 'Padding on all sides of the protein (nm)';
  if (paddingHint) paddingHint.textContent = environment === 'bilayer'
    ? 'The membrane X/Y area remains fixed by the explicit leaflet size.'
    : 'The box is derived from the mapped protein extent plus this clearance on all six faces.';
  if (padding && !padding.dataset.userEdited) padding.value = environment === 'bilayer' ? '2.0' : '1.5';
  if (invalidate) invalidateCoarseGrainedFrom('input');
}

const _CG_STEP_ORDER = [
  'input', 'cg_model', 'cg_mapping', 'cg_orientation', 'cg_environment', 'cg_solvation', 'cg_system'
];

function invalidateCoarseGrainedFrom(stepName) {
  renderMembraneCompositionWarnings([]);
  if (!isCoarseGrainedWorkflow()) return;
  var start = _CG_STEP_ORDER.indexOf(stepName);
  if (start < 0) return;
  _CG_STEP_ORDER.slice(start).forEach(function(name) {
    _checkedSteps.delete(name);
    if (_checkedConfig) delete _checkedConfig[name];
    var status = document.getElementById(name.replace('_', '-') + '-status');
    if (status) status.textContent = '';
  });
  var confirmation = document.getElementById('cg-confirm-system');
  if (confirmation) { confirmation.checked = false; confirmation.disabled = true; }
  updateNextButtonState();
  updateStepNavHighlight();
}

let _cgLipidCatalog = [];
let _cgMixUpper = [{name: 'POPC', ratio: 100}];
let _cgMixLower = [{name: 'POPC', ratio: 100}];
let _cgAsymmetric = false;
let _cgPickerTarget = null;
let _cgCapabilitiesPromise = null;

function cgCompositionFor(leaflet) {
  return (leaflet === 'upper' ? _cgMixUpper : _cgMixLower).map(function(item) {
    return {name: item.name, ratio: Number(item.ratio)};
  });
}

async function loadMartiniLipidCapabilities() {
  if (_cgLipidCatalog.length) return _cgLipidCatalog;
  if (!_cgCapabilitiesPromise) {
    _cgCapabilitiesPromise = fetch('/api/coarse-grained/capabilities')
      .then(function(response) {
        if (!response.ok) throw new Error('Could not load Martini lipid capabilities');
        return response.json();
      })
      .then(function(data) {
        _cgLipidCatalog = Array.isArray(data.lipids) ? data.lipids : [];
        if (!_cgLipidCatalog.length) throw new Error('No validated Martini lipids are installed');
        renderCgMix('upper');
        renderCgMix('lower');
        updateCgLipidCounts();
        return _cgLipidCatalog;
      })
      .catch(function(error) {
        _cgCapabilitiesPromise = null;
        var status = document.getElementById('cg-environment-status');
        if (status) { status.textContent = '✗ ' + error.message; status.style.color = '#dc2626'; }
        throw error;
      });
  }
  return _cgCapabilitiesPromise;
}

function normalizeCgRatios(mix) {
  var total = mix.reduce(function(sum, item) { return sum + Number(item.ratio || 0); }, 0);
  if (total <= 0) {
    mix.forEach(function(item, index) { item.ratio = index === 0 ? 100 : 0; });
    return;
  }
  var assigned = 0;
  mix.forEach(function(item, index) {
    if (index === mix.length - 1) item.ratio = 100 - assigned;
    else {
      item.ratio = Math.max(0, Math.round(Number(item.ratio || 0) * 100 / total));
      assigned += item.ratio;
    }
  });
}

function renderCgMix(leaflet) {
  var list = document.getElementById('cg-' + leaflet + '-lipid-list');
  if (!list) return;
  var mix = leaflet === 'upper' ? _cgMixUpper : _cgMixLower;
  list.replaceChildren();
  mix.forEach(function(entry, index) {
    var definition = _cgLipidCatalog.find(function(item) { return item.name === entry.name; });
    var row = document.createElement('div');
    row.className = 'lipid-mix-row';
    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'mix-lipid-trigger';
    var name = document.createElement('span');
    name.className = 'mix-lipid-trigger-name';
    name.textContent = entry.name;
    var family = document.createElement('span');
    family.className = 'mix-lipid-trigger-cat';
    family.textContent = definition ? definition.family + ' · APL ' + Number(definition.apl_nm2).toFixed(2) + ' nm²' : '';
    var arrow = document.createElement('span');
    arrow.className = 'mix-lipid-trigger-arrow';
    arrow.innerHTML = '&#9662;';
    trigger.append(name, family, arrow);
    trigger.addEventListener('click', function(event) {
      event.stopPropagation();
      _cgPickerTarget = {leaflet: leaflet, index: index};
      openCgLipidPicker(trigger);
    });
    var ratioWrap = document.createElement('div');
    ratioWrap.className = 'mix-ratio';
    var ratio = document.createElement('input');
    ratio.type = 'number'; ratio.min = '0'; ratio.max = '100'; ratio.step = '1';
    ratio.className = 'mix-ratio-input'; ratio.value = String(entry.ratio);
    ratio.addEventListener('input', function() {
      entry.ratio = Math.max(0, Math.min(100, Number(ratio.value) || 0));
      invalidateCoarseGrainedFrom('cg_environment');
      updateCgLipidCounts();
    });
    ratio.addEventListener('change', function() {
      normalizeCgRatios(mix); renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
    });
    var pct = document.createElement('span'); pct.className = 'mix-pct'; pct.textContent = '%';
    ratioWrap.append(ratio, pct);
    var remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'mix-remove'; remove.textContent = '×';
    remove.disabled = mix.length <= 1;
    remove.addEventListener('click', function() {
      if (mix.length <= 1) return;
      mix.splice(index, 1); normalizeCgRatios(mix);
      if (!_cgAsymmetric && leaflet === 'upper') _cgMixLower = cgCompositionFor('upper');
      renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
      invalidateCoarseGrainedFrom('cg_environment');
    });
    row.append(trigger, ratioWrap, remove);
    list.appendChild(row);
  });
}

function openCgLipidPicker(anchor) {
  var picker = document.getElementById('cg-lipid-picker-dropdown');
  if (!picker) return;
  var rect = anchor.getBoundingClientRect();
  var margin = 8;
  var viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  var dropdownWidth = Math.min(Math.max(rect.width, 500), Math.max(0, viewportWidth - margin * 2));
  var dropdownLeft = Math.max(margin, Math.min(rect.left, viewportWidth - dropdownWidth - margin));
  picker.style.position = 'fixed'; picker.style.left = dropdownLeft + 'px';
  picker.style.top = (rect.bottom + 4) + 'px'; picker.style.width = dropdownWidth + 'px';
  picker.classList.remove('hidden');
  var search = document.getElementById('cg-lipid-picker-search');
  if (search) search.value = '';
  renderCgLipidPicker('');
  keepDropdownInViewport(picker, rect);
}

function closeCgLipidPicker() {
  document.getElementById('cg-lipid-picker-dropdown')?.classList.add('hidden');
  _cgPickerTarget = null;
}

function renderCgLipidPicker(filter) {
  var container = document.getElementById('cg-lipid-picker-list');
  if (!container) return;
  container.replaceChildren();
  var query = String(filter || '').trim().toLowerCase();
  var families = {};
  _cgLipidCatalog.forEach(function(lipid) {
    if (query && ![lipid.name, lipid.family].some(function(value) { return String(value).toLowerCase().includes(query); })) return;
    (families[lipid.family] ||= []).push(lipid);
  });
  Object.keys(families).sort().forEach(function(familyName) {
    var header = document.createElement('div'); header.className = 'lipid-cat-header'; header.textContent = familyName;
    var grid = document.createElement('div'); grid.className = 'lipid-cat-grid';
    families[familyName].forEach(function(lipid) {
      var card = document.createElement('button'); card.type = 'button'; card.className = 'lipid-card';
      var title = document.createElement('strong'); title.textContent = lipid.name;
      var details = document.createElement('span'); details.className = 'hint';
      details.textContent = 'Initial area ' + Number(lipid.apl_nm2).toFixed(2) + ' nm² · charge ' + (Number(lipid.charge) >= 0 ? '+' : '') + lipid.charge;
      card.append(title, details);
      card.addEventListener('click', function() {
        if (!_cgPickerTarget) return;
        var mix = _cgPickerTarget.leaflet === 'upper' ? _cgMixUpper : _cgMixLower;
        mix[_cgPickerTarget.index].name = lipid.name;
        if (!_cgAsymmetric && _cgPickerTarget.leaflet === 'upper') _cgMixLower = cgCompositionFor('upper');
        closeCgLipidPicker(); renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
        invalidateCoarseGrainedFrom('cg_environment');
      });
      grid.appendChild(card);
    });
    container.append(header, grid);
  });
}

function updateCgLipidCounts() {
  var count = Number(document.getElementById('cg-n-lipids-per-leaflet')?.value || 150);
  function exactCounts(mix) {
    var total = mix.reduce(function(sum, item) { return sum + Number(item.ratio || 0); }, 0);
    if (total <= 0 || !Number.isInteger(count)) return mix.map(function(item) { return [item.name, 0]; });
    var rows = mix.map(function(item, index) {
      var exact = count * Number(item.ratio || 0) / total;
      return {name: item.name, index: index, value: Math.floor(exact), fraction: exact - Math.floor(exact)};
    });
    var remaining = count - rows.reduce(function(sum, item) { return sum + item.value; }, 0);
    rows.slice().sort(function(a, b) {
      return b.fraction - a.fraction || a.name.localeCompare(b.name) || a.index - b.index;
    }).slice(0, remaining).forEach(function(item) { rows[item.index].value += 1; });
    return rows.map(function(item) { return [item.name, item.value]; });
  }
  function render(leaflet, mix) {
    var target = document.getElementById('cg-' + leaflet + '-lipid-counts');
    if (!target || !_cgLipidCatalog.length) return;
    var totalRatio = mix.reduce(function(sum, item) { return sum + Number(item.ratio || 0); }, 0);
    var apl = mix.reduce(function(sum, item) {
      var lipid = _cgLipidCatalog.find(function(candidate) { return candidate.name === item.name; });
      return sum + (lipid ? Number(lipid.apl_nm2) * Number(item.ratio || 0) : 0);
    }, 0) / Math.max(totalRatio, 1);
    var rows = exactCounts(mix).map(function(item) { return item[0] + ': ' + item[1]; }).join(' · ');
    target.textContent = count + ' lipids/leaflet · initial weighted area ' + apl.toFixed(3) + ' nm² · ' + rows;
  }
  render('upper', _cgMixUpper);
  render('lower', _cgAsymmetric ? _cgMixLower : _cgMixUpper);
}

function collectCoarseGrainedSimulationParams() {
  var config = {
    minimization_steps: Number(document.getElementById('cg-mini-steps')?.value || 20000),
    minimization_tolerance: Number(document.getElementById('cg-mini-tolerance')?.value || 200),
    minimization_step_nm: Number(document.getElementById('cg-mini-step')?.value || 0.005),
    eq1_duration_ns: Number(document.getElementById('cg-eq1-duration')?.value || 1),
    eq1_timestep_fs: Number(document.getElementById('cg-eq1-dt')?.value || 10),
    eq1_temperature: Number(document.getElementById('cg-eq1-temperature')?.value || 310),
    eq1_tau_t: Number(document.getElementById('cg-eq1-tau-t')?.value || 1),
    eq2_duration_ns: Number(document.getElementById('cg-eq2-duration')?.value || 10),
    eq2_timestep_fs: Number(document.getElementById('cg-eq2-dt')?.value || 20),
    eq2_temperature: Number(document.getElementById('cg-eq2-temperature')?.value || 310),
    eq2_tau_t: Number(document.getElementById('cg-eq2-tau-t')?.value || 1),
    eq2_pressure: Number(document.getElementById('cg-eq2-pressure')?.value || 1),
    eq2_tau_p: Number(document.getElementById('cg-eq2-tau-p')?.value || 5),
    production_ns: Number(document.getElementById('cg-production-ns')?.value || 1000),
    production_timestep_fs: Number(document.getElementById('cg-production-dt')?.value || 20),
    production_temperature: Number(document.getElementById('cg-production-temperature')?.value || 310),
    production_tau_t: Number(document.getElementById('cg-production-tau-t')?.value || 1),
    production_pressure: Number(document.getElementById('cg-production-pressure')?.value || 1),
    production_tau_p: Number(document.getElementById('cg-production-tau-p')?.value || 5),
    output_interval_ps: Number(document.getElementById('cg-output-ps')?.value || 100),
    energy_interval_ps: Number(document.getElementById('cg-energy-ps')?.value || 20),
    log_interval_ps: Number(document.getElementById('cg-log-ps')?.value || 20),
    comm_mode: String(document.getElementById('cg-comm-mode')?.value || 'Linear'),
    comm_interval: Number(document.getElementById('cg-comm-interval')?.value || 100),
    equilibration_1: document.getElementById('cg-eq1')?.checked !== false,
    equilibration_2: document.getElementById('cg-eq2')?.checked !== false,
  };
  var ranges = [
    ['Minimization steps', config.minimization_steps, 100, 1000000],
    ['Minimization tolerance', config.minimization_tolerance, 1, 10000],
    ['Minimization step size', config.minimization_step_nm, 0.0001, 0.1],
    ['NVT duration', config.eq1_duration_ns, 0.001, 1000],
    ['NVT timestep', config.eq1_timestep_fs, 1, 20],
    ['NVT temperature', config.eq1_temperature, 250, 370],
    ['NVT thermostat tau', config.eq1_tau_t, 0.1, 20],
    ['NPT duration', config.eq2_duration_ns, 0.001, 10000],
    ['NPT timestep', config.eq2_timestep_fs, 1, 20],
    ['NPT temperature', config.eq2_temperature, 250, 370],
    ['NPT thermostat tau', config.eq2_tau_t, 0.1, 20],
    ['NPT pressure', config.eq2_pressure, 0.1, 100],
    ['NPT barostat tau', config.eq2_tau_p, 0.1, 50],
    ['Production length', config.production_ns, 1, 100000],
    ['Production timestep', config.production_timestep_fs, 1, 20],
    ['Production temperature', config.production_temperature, 250, 370],
    ['Production thermostat tau', config.production_tau_t, 0.1, 20],
    ['Production pressure', config.production_pressure, 0.1, 100],
    ['Production barostat tau', config.production_tau_p, 0.1, 50],
    ['Trajectory interval', config.output_interval_ps, 1, config.production_ns * 1000],
    ['Energy interval', config.energy_interval_ps, 0.02, config.production_ns * 1000],
    ['Log interval', config.log_interval_ps, 0.02, config.production_ns * 1000],
    ['COM interval', config.comm_interval, 1, 1000000],
  ];
  ranges.forEach(function(item) {
    if (!Number.isFinite(item[1]) || item[1] < item[2] || item[1] > item[3]) {
      throw new Error(item[0] + ' must be between ' + item[2] + ' and ' + item[3] + '.');
    }
  });
  if (!Number.isInteger(config.minimization_steps) || !Number.isInteger(config.comm_interval)) {
    throw new Error('CG minimization steps and COM interval must be integers.');
  }
  return config;
}

function collectCoarseGrainedExecutionHardware() {
  var threads = Number(document.getElementById('cg-threads')?.value || 8);
  var mpiRanks = Number(document.getElementById('cg-mpi-ranks')?.value || 1);
  var useGpu = document.getElementById('cg-use-gpu')?.checked !== false;
  var gpuIds = String(document.getElementById('cg-gpu-ids')?.value || '0').trim();
  if (!Number.isInteger(threads) || threads < 1 ||
      !Number.isInteger(mpiRanks) || mpiRanks < 1 || threads % mpiRanks !== 0) {
    throw new Error('CG CPU threads must be positive integers and exactly divisible by thread-MPI ranks.');
  }
  if (useGpu) {
    if (!/^\d+(?:,\d+)*$/.test(gpuIds)) {
      throw new Error('CG GPU IDs must be unique comma-separated integers, for example 0 or 0,1.');
    }
    var selected = gpuIds.split(',');
    if (new Set(selected).size !== selected.length || selected.length > mpiRanks) {
      throw new Error('CG GPU IDs must be unique and their count cannot exceed thread-MPI ranks.');
    }
  }
  return {
    mode: 'thread-mpi',
    cpu_threads: threads,
    mpi_ranks: mpiRanks,
    use_gpu: useGpu,
    gpu_count: useGpu ? gpuIds.split(',').length : 0,
    gpu_ids: useGpu ? gpuIds : '',
    gmx_command: 'gmx',
    mpi_launcher: 'mpirun',
    pin: 'auto'
  };
}

function restoreCoarseGrainedConfig(taskState) {
  if (!isCoarseGrainedWorkflow() || !taskState) return;
  function value(id, raw) {
    var element = document.getElementById(id);
    if (element && raw !== undefined && raw !== null) element.value = String(raw);
  }
  function checked(id, raw) {
    var element = document.getElementById(id);
    if (element && raw !== undefined) element.checked = raw !== false;
  }
  var input = taskState.step_input_config || {};
  checked('cg-include-protein', input.include_protein);
  var mapping = taskState.step_cg_mapping_config || {};
  value('cg-protein-model', mapping.protein_model);
  value('cg-secondary', mapping.secondary_structure);
  value('cg-secondary-string', mapping.secondary_structure_string);
  checked('cg-elastic', mapping.elastic);
  value('cg-elastic-force', mapping.elastic_force);
  value('cg-elastic-lower', mapping.elastic_lower);
  value('cg-elastic-upper', mapping.elastic_upper);
  var orientation = taskState.step_cg_orientation_config || {};
  _cgOrientMode = orientation.method === 'manual' ? 'manual' : 'ppm';
  value('cg-orientation-half-thickness', orientation.half_thickness);
  _cgOrientZOffset = Number(orientation.z_offset || 0);
  _cgOrientTilt = Number(orientation.tilt || 0);
  _cgOrientPhi = Number(orientation.phi || 0);
  [
    ['cg-orient-manual-z', _cgOrientZOffset],
    ['cg-orient-manual-z-num', _cgOrientZOffset],
    ['cg-orient-manual-tilt', _cgOrientTilt],
    ['cg-orient-manual-tilt-num', _cgOrientTilt],
    ['cg-orient-manual-phi', _cgOrientPhi],
    ['cg-orient-manual-phi-num', _cgOrientPhi],
  ].forEach(function(entry) { value(entry[0], entry[1]); });
  var environment = taskState.step_cg_environment_config || {};
  value('cg-n-lipids-per-leaflet', environment.n_lipids_per_leaflet);
  if (Array.isArray(environment.upper_leaflet) && environment.upper_leaflet.length) {
    _cgMixUpper = environment.upper_leaflet.map(function(item) { return {name: item.name, ratio: item.ratio}; });
  }
  if (Array.isArray(environment.lower_leaflet) && environment.lower_leaflet.length) {
    _cgMixLower = environment.lower_leaflet.map(function(item) { return {name: item.name, ratio: item.ratio}; });
  } else {
    _cgMixLower = cgCompositionFor('upper');
  }
  _cgAsymmetric = environment.asymmetric === true;
  checked('cg-asymmetric-bilayer', _cgAsymmetric);
  document.getElementById('cg-lower-leaflet-section')?.classList.toggle('hidden', !_cgAsymmetric);
  renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
  var solvation = taskState.step_cg_solvation_config || {};
  checked('cg-include-solvent', solvation.include_solvent);
  value('cg-padding', solvation.padding_nm);
  var finalSystem = taskState.step_cg_system_config || {};
  value('cg-salt', finalSystem.salt_molarity == null ? solvation.salt_molarity : finalSystem.salt_molarity);
  var simulation = taskState.step_simparams_config || taskState.simparams || {};
  value('cg-mini-steps', simulation.minimization_steps);
  value('cg-mini-tolerance', simulation.minimization_tolerance);
  value('cg-mini-step', simulation.minimization_step_nm);
  value('cg-eq1-duration', simulation.eq1_duration_ns);
  value('cg-eq1-dt', simulation.eq1_timestep_fs);
  value('cg-eq1-temperature', simulation.eq1_temperature);
  value('cg-eq1-tau-t', simulation.eq1_tau_t);
  value('cg-eq2-duration', simulation.eq2_duration_ns);
  value('cg-eq2-dt', simulation.eq2_timestep_fs);
  value('cg-eq2-temperature', simulation.eq2_temperature);
  value('cg-eq2-tau-t', simulation.eq2_tau_t);
  value('cg-eq2-pressure', simulation.eq2_pressure);
  value('cg-eq2-tau-p', simulation.eq2_tau_p);
  value('cg-production-ns', simulation.production_ns);
  value('cg-production-dt', simulation.production_timestep_fs);
  value('cg-production-temperature', simulation.production_temperature);
  value('cg-production-tau-t', simulation.production_tau_t);
  value('cg-production-pressure', simulation.production_pressure);
  value('cg-production-tau-p', simulation.production_tau_p);
  value('cg-output-ps', simulation.output_interval_ps);
  value('cg-energy-ps', simulation.energy_interval_ps);
  value('cg-log-ps', simulation.log_interval_ps);
  value('cg-comm-mode', simulation.comm_mode);
  value('cg-comm-interval', simulation.comm_interval);
  checked('cg-eq1', simulation.equilibration_1);
  checked('cg-eq2', simulation.equilibration_2);
  var execution = taskState.step_execution_config || taskState.execution || {};
  checked('cg-use-gpu', execution.use_gpu);
  value('cg-gpu-ids', Array.isArray(execution.gpu_ids) ? execution.gpu_ids.join(',') : execution.gpu_ids);
  value('cg-threads', execution.cpu_threads);
  value('cg-mpi-ranks', execution.mpi_ranks);
  syncCoarseGrainedInputControls(false);
  updateCgOrientationControlLabels();
  syncCgOrientationMode();
}

function pureMembraneIncludesSolvent() {
  return !(state.taskType && state.taskType.pipeline === 'pure_membrane') ||
    document.getElementById('pure-membrane-include-solvent')?.checked !== false;
}

function syncPureMembraneSolvationOption(invalidate) {
  if (!state.taskType || state.taskType.pipeline !== 'pure_membrane') return;
  var includeSolvent = pureMembraneIncludesSolvent();
  var controls = document.getElementById('solvation-controls');
  var notice = document.getElementById('pure-membrane-dry-notice');
  if (controls) controls.classList.toggle('hidden', !includeSolvent);
  if (notice) notice.classList.toggle('hidden', includeSolvent);

  var desired = ['forcefield', 'membrane', 'solvation'];
  if (includeSolvent) desired.push('ions');
  desired.push('final_review', 'simparams');
  var changed = JSON.stringify(state.wizardSteps) !== JSON.stringify(desired);
  state.wizardSteps = desired;
  if (changed) renderStepNav();

  if (!includeSolvent) {
    _solvChecked = true;
    _checkedSteps.add('solvation');
    _checkedSteps.delete('ions');
    if (_checkedConfig) {
      delete _checkedConfig.solvation;
      delete _checkedConfig.ions;
    }
  } else if (invalidate) {
    resetSolvCheck();
  }
  if (invalidate) {
    window._setIonsChecked ? window._setIonsChecked(false) : null;
    updateNextButtonState();
    updateStepNavHighlight();
  }
}

function loadTaskDefaults() {
  const defaults = state.taskType.default_config || {};
  function setDefaultValue(id, raw) {
    var element = document.getElementById(id);
    if (element && raw !== undefined && raw !== null) element.value = String(raw);
  }
  if (isCoarseGrainedWorkflow()) {
    var cgInput = defaults.input || {};
    var cgProtein = document.getElementById('cg-include-protein');
    if (cgProtein) cgProtein.checked = cgInput.include_protein !== false;
    var cgMapping = defaults.cg_mapping || {};
    var cgProteinModel = document.getElementById('cg-protein-model');
    var cgSecondary = document.getElementById('cg-secondary');
    var cgElastic = document.getElementById('cg-elastic');
    if (cgProteinModel) cgProteinModel.value = cgMapping.protein_model || 'folded';
    if (cgSecondary) cgSecondary.value = cgMapping.secondary_structure || 'auto';
    if (cgElastic) cgElastic.checked = cgMapping.elastic !== false;
    var cgOrientation = defaults.cg_orientation || {};
    _cgOrientMode = cgOrientation.method === 'manual' ? 'manual' : 'ppm';
    setDefaultValue(
      'cg-orientation-half-thickness',
      cgOrientation.half_thickness == null ? 1.4 : cgOrientation.half_thickness
    );
    _cgOrientZOffset = Number(cgOrientation.z_offset || 0);
    _cgOrientTilt = Number(cgOrientation.tilt || 0);
    _cgOrientPhi = Number(cgOrientation.phi || 0);
    var cgEnvironmentDefaults = defaults.cg_environment || {};
    var cgCount = document.getElementById('cg-n-lipids-per-leaflet');
    if (cgCount) cgCount.value = cgEnvironmentDefaults.n_lipids_per_leaflet || 150;
    _cgMixUpper = (cgEnvironmentDefaults.upper_leaflet || [{name: 'POPC', ratio: 100}]).map(function(item) { return {name: item.name, ratio: item.ratio}; });
    _cgMixLower = (cgEnvironmentDefaults.lower_leaflet || _cgMixUpper).map(function(item) { return {name: item.name, ratio: item.ratio}; });
    _cgAsymmetric = cgEnvironmentDefaults.asymmetric === true;
    var cgAsymmetric = document.getElementById('cg-asymmetric-bilayer');
    if (cgAsymmetric) cgAsymmetric.checked = _cgAsymmetric;
    document.getElementById('cg-lower-leaflet-section')?.classList.toggle('hidden', !_cgAsymmetric);
    renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
    var cgSolvation = defaults.cg_solvation || {};
    var cgIncludeSolvent = document.getElementById('cg-include-solvent');
    var cgPadding = document.getElementById('cg-padding');
    var cgSalt = document.getElementById('cg-salt');
    if (cgIncludeSolvent) cgIncludeSolvent.checked = cgSolvation.include_solvent !== false;
    if (cgPadding) cgPadding.value = coarseGrainedEnvironment() === 'bilayer' ? 2.0 : 1.5;
    if (cgSalt) cgSalt.value = cgSolvation.salt_molarity == null ? 0.15 : cgSolvation.salt_molarity;
    syncCoarseGrainedInputControls(false);
    syncCgOrientationMode();
    return;
  }
  // Apply defaults to form fields
  if (defaults.membrane) {
    const m = defaults.membrane;
    selectLipid(m.lipid_type || 'POPC', {restoreDefault: true});
    const nLipidsEl = document.getElementById('n-lipids-per-leaflet');
    if (nLipidsEl) nLipidsEl.value = m.n_lipids_per_leaflet || 150;
  }
  if (defaults.solvation) {
    const s = defaults.solvation;
    const el = document.getElementById('ff-water-model');
    if (el && el.options.length > 0) el.value = s.water_model || 'tip3p';
    syncLockedWaterModelDisplay();
    document.getElementById('box-padding').value = s.box_padding || 1.5;
    var includeSolvent = document.getElementById('pure-membrane-include-solvent');
    if (includeSolvent && state.taskType && state.taskType.pipeline === 'pure_membrane') {
      includeSolvent.checked = s.enabled !== false;
    }
  }
  if (defaults.ions) {
    const i = defaults.ions;
    document.getElementById('ion-neutralize').checked = i.neutralize !== false;
    const np2 = document.getElementById("ion-neutralize-pair"); const nc2 = document.getElementById("ion-neutralize"); if (np2 && nc2) { if (nc2.checked) np2.classList.remove("hidden"); else np2.classList.add("hidden"); }
  }
  if (defaults.forcefield) {
    var ffDefaults = defaults.forcefield;
    var proteinFF = document.getElementById('ff-protein');
    if (proteinFF && ffDefaults.name) proteinFF.value = ffDefaults.name;
    updateWaterModelOptions(true);
    var waterFF = document.getElementById('ff-water-model');
    if (waterFF && ffDefaults.water_model &&
        Array.from(waterFF.options).some(function(o) { return o.value === ffDefaults.water_model; })) {
      waterFF.value = ffDefaults.water_model;
    }
    syncLockedWaterModelDisplay();
  }
}

// ===================================================================
// Dynamic Step Navigation
// ===================================================================

function renderStepNav() {
  const nav = document.getElementById('step-nav');
  nav.innerHTML = '';

  // Title step
  const titleBtn = document.createElement('button');
  titleBtn.className = 'step active';
  titleBtn.innerHTML = `<span class="step-num">&#9664;</span> Task Type`;
  titleBtn.addEventListener('click', () => goToTaskSelect());
  nav.appendChild(titleBtn);

  // Divider
  const divider = document.createElement('span');
  divider.className = 'step-divider';
  divider.textContent = '›';
  divider.style.cssText = 'align-self:center;color:var(--text-muted);font-size:18px;margin:0 4px;';
  nav.appendChild(divider);

  // Module steps
  state.wizardSteps.forEach((modName, idx) => {
    const meta = STEP_META[modName] || { title: modName };
    const icon = meta.icon || (idx + 1);
    const btn = document.createElement('button');
    btn.className = 'step';
    btn.dataset.stepModule = modName;
    btn.innerHTML = `<span class="step-num">${icon}</span> ${meta.title}`;
    btn.addEventListener('click', () => goToWizardStep(idx));
    nav.appendChild(btn);
  });

  nav.classList.add('dynamic');
  updateStepNavHighlight();
}

// ===================================================================
// Step locking logic
// ===================================================================

/** Step is unlocked only when ALL previous steps are completed. */
function canGoToStep(idx) {
  if (state.customLipidBusy && idx > state.currentStepIdx) return false;
  // Check both _checkedSteps (set by Check buttons) and completedSteps
  // (set by markStepComplete).  They must be consistent — the actual gate
  // is whether each preceding step has a checkpoint saved on the server.
  for (let i = 0; i < idx; i++) {
    var modName = state.wizardSteps[i];
    if (!modName) continue;
    // Skip steps that don't require explicit checks
    if (modName === 'topology' || modName === 'simparams' || modName === 'export') continue;
    if (modName === 'membrane' && v4CompositionErrors().length) return false;
    if (modName === 'final_review' && !window.isFinalReviewConfirmed()) {
      return false;
    }
    if (!_checkedSteps.has(modName) && !state.completedSteps.has(i)) return false;
  }
  return true;
}

/** Mark a step as completed and checked. */
function markStepComplete(idx) {
  state.completedSteps.add(idx);
  var modName = state.wizardSteps[idx];
  if (modName) _checkedSteps.add(modName);
  updateStepNavHighlight();
}

// Steps that have been "checked" (checkpoint saved to disk).
// Next button is disabled until the current step is checked.
var _checkedSteps = new Set();
// Snapshot of valid config per step (prevents DOM drift between Check and Build)
var _checkedConfig = null;

/** Check if current step's minimal requirements are met. */
function isCurrentStepFulfilled() {
  if (state.customLipidBusy) return false;
  const modName = state.wizardSteps[state.currentStepIdx];
  if (!modName) return true;
  // Step must be "checked" (checkpoint saved) before Next is allowed
  if (modName === 'topology' || modName === 'simparams' || modName === 'export') return true;
  if (modName === 'ions') {
    return _checkedSteps.has('ions') &&
      (window._isIonsChecked ? window._isIonsChecked() : false);
  }
  if (modName === 'membrane') return _checkedSteps.has(modName) && !v4CompositionErrors().length;
  if (modName === 'final_review') return window.isFinalReviewConfirmed();
  if (modName === 'forcefield') return _checkedSteps.has(modName) && _ffCompatibilityValid;
  return _checkedSteps.has(modName);
}

function goToTaskSelect() {
  forgetCurrentTask();
  // Always allowed back
  state.currentStepIdx = -1;
  clearStepProgress();
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  var panel0 = document.getElementById('panel-task-type');
  if (panel0) panel0.classList.add('active');
  updateStepNavHighlight();
  if (!_restoringRoute && window.location.pathname !== '/') history.pushState({}, '', '/');
}

// ---- Step execution on server (incremental checkpoint build) ----
var _stepRunning = false;
var _viewerLoadTimer = null;  // cleared before setting new setTimeout in goToWizardStep

/** Centralized fetch helper — throws on non-2xx, parses JSON on success. */
async function _apiFetch(url, options) {
  // Step handlers consume structured domain errors, warnings and repair details.
  return GMXHttp.json(url, options, {allowErrorPayload: true});
}

// A queued preview must not keep a completed Check locked indefinitely.
var _VIEWER_REQUEST_TIMEOUT_MS = 30000;

async function _loadStepViewerPdb(stepName) {
  if (!state.taskId) return null;
  var controller = new AbortController();
  var timeout = setTimeout(function() { controller.abort(); }, _VIEWER_REQUEST_TIMEOUT_MS);
  try {
    var resp = await fetch('/api/step/' + state.taskId + '/' + stepName + '/viewer.pdb', {
      signal: controller.signal
    });
    if (resp.ok) return await resp.text();
    console.warn('_loadStepViewerPdb: HTTP', resp.status, 'for step', stepName);
  } catch (e) { console.warn('_loadStepViewerPdb: preview unavailable for step', stepName, e); }
  finally { clearTimeout(timeout); }
  return null;
}

async function refreshCheckedInputPreview() {
  var taskId = state.taskId;
  var report = document.getElementById('input-check-report');
  var previous = document.getElementById('input-preview-feedback');
  if (previous) previous.remove();
  var feedback = document.createElement('p');
  feedback.id = 'input-preview-feedback';
  feedback.className = 'hint';
  feedback.textContent = 'Loading the checked structure preview…';
  report.appendChild(feedback);
  try {
    var pdb = await _loadStepViewerPdb('input');
    if (taskId !== state.taskId || !feedback.isConnected) return;
    if (!pdb) throw new Error('Saved preview unavailable');
    if (state.pdbInfo) {
      state.pdbInfo.pdb_content = pdb;
      var box = pdb.match(/^CRYST1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/m);
      if (box) state.pdbInfo.box_nm = box.slice(1).map(value => Number(value) / 10);
    }
    redrawPDBViewerWithChainFilter();
    feedback.remove();
  } catch (error) {
    if (taskId !== state.taskId || !feedback.isConnected) return;
    // Input validation is authoritative. A display failure does not revoke it;
    // the separate final-system confirmation still requires a rendered viewer.
    feedback.textContent = 'Input check passed, but the saved preview could not be displayed. You can continue or reload the preview. ';
    var retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn';
    retry.textContent = 'Reload preview';
    retry.addEventListener('click', function() { refreshCheckedInputPreview(); });
    feedback.appendChild(retry);
  }
}

async function renderCoarseGrainedViewer(stepName, pdbOverride, preserveCamera) {
  var targetMap = {
    cg_mapping: 'cg-mapping-viewer',
    cg_orientation: 'cg-orientation-viewer',
    cg_environment: 'cg-environment-viewer',
    cg_solvation: 'cg-solvation-viewer',
    cg_system: 'cg-system-viewer',
  };
  var targetId = targetMap[stepName];
  if (!pdbOverride && ['cg_environment', 'cg_solvation', 'cg_system'].includes(stepName)) {
    if (!_checkedSteps.has(stepName)) return false;
    return Boolean(await GMXViewer.render(targetId, stepName));
  }
  var target = targetId ? document.getElementById(targetId) : null;
  if (!target) return false;
  await GMXAssets.viewer();
  // 3Dmol reads the element's client rectangle when its WebGL canvas is
  // created.  Rendering a hidden wizard panel previously produced a viewport
  // anchored to the document rather than this box.  Wait until the target is
  // both visible and laid out, then render in its own stacking context.
  if (!target.closest('.panel.active') || target.clientWidth < 2 || target.clientHeight < 2) {
    await new Promise(function(resolve) {
      requestAnimationFrame(function() { requestAnimationFrame(resolve); });
    });
  }
  if (!target.closest('.panel.active') || target.clientWidth < 2 || target.clientHeight < 2) {
    return false;
  }
  window._cgViewers = window._cgViewers || {};
  var cached = window._cgViewers[targetId];
  if (cached && (cached.taskId !== state.taskId || cached.stepName !== stepName)) {
    try { cached.viewer.clear(); } catch (e) {}
    target.replaceChildren();
    delete window._cgViewers[targetId];
    cached = null;
  }
  var pdb = pdbOverride || await _loadStepViewerPdb(stepName);
  if (!pdb) {
    if (cached) {
      try { cached.viewer.clear(); } catch (e) {}
      target.replaceChildren();
      delete window._cgViewers[targetId];
    }
    return false;
  }
  // Protein-free mapping checkpoints intentionally contain only CRYST1/END.
  // 3Dmol cannot create a model from that empty coordinate set; skipping the
  // viewer is the correct successful state, not a failed Check.
  if (!/^\s*(?:ATOM|HETATM)/m.test(pdb)) {
    target.replaceChildren();
    return false;
  }
  var viewer = cached && cached.viewer;
  if (!viewer || !target.querySelector('canvas')) {
    target.replaceChildren();
    try {
      viewer = $3Dmol.createViewer(target, {backgroundColor: window.gmxViewerBackground()});
    } catch (error) {
      target.textContent = '3D viewer unavailable: this browser does not provide a working WebGL context.';
      target.classList.add('viewer-unavailable');
      return false;
    }
    window._cgViewers[targetId] = {viewer: viewer, taskId: state.taskId, stepName: stepName};
  }
  target.classList.remove('viewer-unavailable');
  viewer.removeAllModels();
  if (viewer.removeAllShapes) viewer.removeAllShapes();
  if (viewer.removeAllLabels) viewer.removeAllLabels();
  var model = viewer.addModel(pdb, 'pdb');
  GMXStyle.apply(viewer, GMXStyle.pdbAtoms(viewer, pdb), {coarse: true});
  if (stepName === 'cg_orientation') {
    var half = Number(document.getElementById('cg-orientation-half-thickness')?.value || 1.4);
    addCgOrientationPlaneMarkers(viewer, pdb, half);
  }
  // The mapping checkpoint box is only an internal envelope estimate.  The
  // user-defined physical PBC cell begins at CG Environment.
  if (stepName !== 'cg_mapping' && stepName !== 'cg_orientation' && viewer.addUnitCell) {
    viewer.addUnitCell(model, {boxColor: '#64748b'});
  }
  if (viewer.resize) viewer.resize();
  if (!preserveCamera) viewer.zoomTo();
  viewer.render();
  viewer.setSlab(-10000, 10000);
  requestAnimationFrame(function() {
    if (!target.closest('.panel.active')) return;
    if (viewer.resize) viewer.resize();
    viewer.render();
  });
  return true;
}

function addCgOrientationPlaneMarkers(viewer, pdb, halfThicknessNm) {
  // PDB/3Dmol coordinates are Angstrom, while the Martini workflow stores
  // scientific distances in nm. Keep this conversion local and explicit.
  var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  (pdb || '').split('\n').forEach(function(line) {
    if (line.indexOf('ATOM') !== 0 && line.indexOf('HETATM') !== 0) return;
    var x = Number.parseFloat(line.substring(30, 38));
    var y = Number.parseFloat(line.substring(38, 46));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
  });
  var halfSpanA = 30.0;
  if (Number.isFinite(xMin) && Number.isFinite(yMin)) {
    halfSpanA = Math.max(
      halfSpanA,
      Math.max(xMax - xMin, yMax - yMin) / 2.0 + 20.0
    );
  }
  var interfaceA = Number(halfThicknessNm) * 10.0;
  var spacingA = 15.0;
  if (!Number.isFinite(interfaceA) || interfaceA <= 0 || !viewer.addSphere) return;
  for (var xA = -halfSpanA; xA <= halfSpanA + 0.01; xA += spacingA) {
    for (var yA = -halfSpanA; yA <= halfSpanA + 0.01; yA += spacingA) {
      [-interfaceA, interfaceA].forEach(function(zA) {
        viewer.addSphere({
          center: {x: xA, y: yA, z: zA},
          radius: 1.2, color: '#94a3b8', opacity: 0.55,
        });
      });
    }
  }
}

function cgOrientationConfig() {
  var config = {
    method: _cgOrientMode,
    half_thickness: Number(
      document.getElementById('cg-orientation-half-thickness')?.value || 1.4
    ),
  };
  if (_cgOrientMode === 'manual') {
    config.z_offset = _cgOrientZOffset;
    config.tilt = _cgOrientTilt;
    config.phi = _cgOrientPhi;
  }
  return config;
}

function updateCgOrientationControlLabels() {
  var z = document.getElementById('cg-orient-manual-z-val');
  var tilt = document.getElementById('cg-orient-manual-tilt-val');
  var phi = document.getElementById('cg-orient-manual-phi-val');
  if (z) z.textContent = Number(_cgOrientZOffset).toFixed(2);
  if (tilt) tilt.textContent = Number(_cgOrientTilt).toFixed(0);
  if (phi) phi.textContent = Number(_cgOrientPhi).toFixed(0);
}

async function requestCgOrientationPreview(config, preserveCamera) {
  if (!state.taskId) return null;
  var requestId = ++_cgOrientPreviewRequestId;
  var response = await fetch('/api/cg-orient-preview/' + state.taskId, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: config}),
  });
  var data = await response.json();
  if (requestId !== _cgOrientPreviewRequestId) return null;
  if (!response.ok || data.status !== 'ok') {
    throw new Error(data.error || 'Martini orientation preview failed');
  }
  _cgOrientedPdbContent = data.oriented_pdb || null;
  if (!_cgOrientedPdbContent) throw new Error('Orientation preview returned no coordinates');
  data.viewer_rendered = await renderCoarseGrainedViewer(
    'cg_orientation', _cgOrientedPdbContent, preserveCamera === true
  );
  return data;
}

async function runCgAutomaticOrientationPreview() {
  var status = document.getElementById('cg-orient-preview-status');
  var zResult = document.getElementById('cg-orient-result-z');
  var tiltResult = document.getElementById('cg-orient-result-tilt');
  if (status) {
    status.textContent = 'Computing exact Martini backend preview...';
    status.style.color = '#d97706';
  }
  if (zResult) zResult.textContent = 'Computing...';
  if (tiltResult) tiltResult.textContent = '—';
  try {
    var data = await requestCgOrientationPreview({
      method: 'ppm',
      half_thickness: Number(
        document.getElementById('cg-orientation-half-thickness')?.value || 1.4
      ),
    }, false);
    if (!data) return;
    var metrics = data.orientation || {};
    if (zResult) zResult.textContent = Number(metrics.z_offset_nm || 0).toFixed(2) + ' nm (PPM)';
    if (tiltResult) tiltResult.textContent = Number(metrics.refinement_degrees || 0).toFixed(1) + '°';
    if (status) {
      status.textContent = data.viewer_rendered
        ? 'Preview matches the coordinates Check will save'
        : 'Exact coordinates are ready, but this browser cannot display the 3D preview';
      status.style.color = data.viewer_rendered ? '#059669' : '#d97706';
    }
  } catch (error) {
    if (zResult) zResult.textContent = 'Unavailable';
    if (status) {
      status.textContent = error.message || 'Preview failed';
      status.style.color = '#dc2626';
    }
  }
}

async function runCgManualOrientationPreview() {
  var status = document.getElementById('cg-orient-preview-status');
  if (status) {
    status.textContent = 'Updating exact preview...';
    status.style.color = '#d97706';
  }
  try {
    var data = await requestCgOrientationPreview(cgOrientationConfig(), true);
    if (data && status) {
      status.textContent = data.viewer_rendered
        ? 'Live preview matches the coordinates Check will save'
        : 'Exact coordinates updated, but this browser cannot display the 3D preview';
      status.style.color = data.viewer_rendered ? '#059669' : '#d97706';
    }
  } catch (error) {
    if (status) {
      status.textContent = error.message || 'Preview failed';
      status.style.color = '#dc2626';
    }
  }
}

function scheduleCgManualOrientationPreview(delayMs) {
  if (_cgOrientPreviewTimer !== null) clearTimeout(_cgOrientPreviewTimer);
  ++_cgOrientPreviewRequestId;
  _cgOrientPreviewTimer = setTimeout(
    runCgManualOrientationPreview, delayMs == null ? 120 : delayMs
  );
}

function showCgOrientationPreview() {
  if (_cgOrientMode === 'manual') scheduleCgManualOrientationPreview(0);
  else runCgAutomaticOrientationPreview();
}

function initCgOrientationControls() {
  var controls = {
    z: {
      slider: document.getElementById('cg-orient-manual-z'),
      number: document.getElementById('cg-orient-manual-z-num'),
    },
    tilt: {
      slider: document.getElementById('cg-orient-manual-tilt'),
      number: document.getElementById('cg-orient-manual-tilt-num'),
    },
    phi: {
      slider: document.getElementById('cg-orient-manual-phi'),
      number: document.getElementById('cg-orient-manual-phi-num'),
    },
  };

  function syncPair(name, source, userChanged) {
    var pair = controls[name];
    if (!pair.slider || !pair.number) return;
    if (source === 'slider') pair.number.value = pair.slider.value;
    else pair.slider.value = pair.number.value;
    var value = Number(source === 'slider' ? pair.slider.value : pair.number.value);
    if (!Number.isFinite(value)) return;
    if (name === 'z') _cgOrientZOffset = value;
    else if (name === 'tilt') _cgOrientTilt = value;
    else _cgOrientPhi = value;
    updateCgOrientationControlLabels();
    if (userChanged) {
      invalidateCoarseGrainedFrom('cg_orientation');
      scheduleCgManualOrientationPreview(120);
    }
  }

  Object.keys(controls).forEach(function(name) {
    var pair = controls[name];
    if (pair.slider) pair.slider.addEventListener('input', function() {
      syncPair(name, 'slider', true);
    });
    if (pair.number) pair.number.addEventListener('input', function() {
      syncPair(name, 'number', true);
    });
  });

  document.querySelectorAll('.cg-orient-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      var nextMode = tab.dataset.method === 'manual' ? 'manual' : 'ppm';
      if (_cgOrientMode !== nextMode) {
        _cgOrientMode = nextMode;
        invalidateCoarseGrainedFrom('cg_orientation');
      }
      syncCgOrientationMode();
      showCgOrientationPreview();
    });
  });
  document.getElementById('cg-orient-rerun-btn')?.addEventListener(
    'click', runCgAutomaticOrientationPreview
  );
  updateCgOrientationControlLabels();
  syncCgOrientationMode();
}

function initCoarseGrainedControls() {
  initCgOrientationControls();
  var includeProtein = document.getElementById('cg-include-protein');
  if (includeProtein) includeProtein.addEventListener('change', function() {
    syncCoarseGrainedInputControls(true);
  });
  var cgAsymmetric = document.getElementById('cg-asymmetric-bilayer');
  if (cgAsymmetric) cgAsymmetric.addEventListener('change', function() {
    _cgAsymmetric = cgAsymmetric.checked;
    document.getElementById('cg-lower-leaflet-section')?.classList.toggle('hidden', !_cgAsymmetric);
    if (!_cgAsymmetric) _cgMixLower = cgCompositionFor('upper');
    var label = document.getElementById('cg-upper-leaflet-label');
    if (label) label.innerHTML = _cgAsymmetric
      ? 'Upper Leaflet <span class="hint">(extracellular / outer)</span>'
      : 'Bilayer <span class="hint">(same composition in both leaflets)</span>';
    renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
    invalidateCoarseGrainedFrom('cg_environment');
  });
  document.querySelectorAll('.cg-add-lipid-btn').forEach(function(button) {
    button.addEventListener('click', async function() {
      await loadMartiniLipidCapabilities();
      var leaflet = button.dataset.leaflet;
      var mix = leaflet === 'upper' ? _cgMixUpper : _cgMixLower;
      var existing = new Set(mix.map(function(item) { return item.name; }));
      var next = _cgLipidCatalog.find(function(item) { return !existing.has(item.name); });
      if (!next) { alert('All installed Martini lipids are already selected.'); return; }
      mix.push({name: next.name, ratio: 0});
      normalizeCgRatios(mix);
      if (!_cgAsymmetric && leaflet === 'upper') _cgMixLower = cgCompositionFor('upper');
      renderCgMix('upper'); renderCgMix('lower'); updateCgLipidCounts();
      invalidateCoarseGrainedFrom('cg_environment');
    });
  });
  var cgSearch = document.getElementById('cg-lipid-picker-search');
  if (cgSearch) cgSearch.addEventListener('input', function() { renderCgLipidPicker(cgSearch.value); });
  document.addEventListener('click', function(event) {
    if (!event.target.closest('#cg-lipid-picker-dropdown') && !event.target.closest('#panel-cg_environment .mix-lipid-trigger')) closeCgLipidPicker();
  });
  // Capabilities are loaded when entering a coarse-grained workflow.

  var checks = [
    ['cg-model-check', 'cg_model', 'cg-model-status'],
    ['cg-mapping-check', 'cg_mapping', 'cg-mapping-status'],
    ['cg-orientation-check', 'cg_orientation', 'cg-orientation-status'],
    ['cg-environment-check', 'cg_environment', 'cg-environment-status'],
    ['cg-solvation-check', 'cg_solvation', 'cg-solvation-status'],
    ['cg-system-check', 'cg_system', 'cg-system-status'],
  ];
  checks.forEach(function(spec) {
    var button = document.getElementById(spec[0]);
    if (!button) return;
    button.addEventListener('click', async function() {
      await _doCheckStep(spec[1], spec[2], spec[0]);
    });
  });
  var inputIds = [
    'cg-protein-model', 'cg-secondary', 'cg-secondary-string', 'cg-elastic',
    'cg-elastic-force', 'cg-elastic-lower', 'cg-elastic-upper',
    'cg-n-lipids-per-leaflet',
    'cg-include-solvent', 'cg-padding', 'cg-salt'
  ];
  inputIds.forEach(function(id) {
    var element = document.getElementById(id);
    if (!element) return;
    element.addEventListener('change', function() {
      var step = id.indexOf('cg-protein') === 0 || id.indexOf('cg-secondary') === 0 ||
        id.indexOf('cg-elastic') === 0 ? 'cg_mapping' :
        id.indexOf('cg-orient') === 0 ? 'cg_orientation' :
        id === 'cg-salt' ? 'cg_system' :
        id === 'cg-include-solvent' || id === 'cg-padding' ? 'cg_solvation' :
        'cg_environment';
      invalidateCoarseGrainedFrom(step);
      if (id === 'cg-include-solvent') {
        var padding = document.getElementById('cg-padding');
        var salt = document.getElementById('cg-salt');
        if (padding) padding.disabled = !element.checked;
        if (salt) salt.disabled = !element.checked;
      }
      if (id === 'cg-padding') element.dataset.userEdited = 'true';
      if (id === 'cg-n-lipids-per-leaflet') updateCgLipidCounts();
    });
  });
}

function goToWizardStep(idx) {
  const wanted = state.wizardSteps[idx];
  if (!window._optionsReady && wanted !== 'input') {
    const task = state.taskType;
    ensureOptionsLoaded().then(ok => { if (ok && task === state.taskType) goToWizardStep(idx); });
    return;
  }

  if (idx < 0 || idx >= state.wizardSteps.length || state.uploadRunning) return;

  // Lock check: forward steps must have all previous completed
  if (idx > state.currentStepIdx && !canGoToStep(idx)) {
    shakeStepNav();
    return;
  }

  // Structure processing blocking: must compute or skip protonation before proceeding
  const structIdx = state.wizardSteps.indexOf('structure');
  const ionIdx = state.wizardSteps.indexOf("ions");
  var solvIdx = state.wizardSteps.indexOf("solvation");
  if (solvIdx >= 0 && idx > solvIdx && !_solvChecked) {
    shakeStepNav();
    alert("Please run Compute Solvent Volume before proceeding.");
    return;
  }
  if (ionIdx >= 0 && idx > ionIdx && (
      !(window._isIonsChecked ? window._isIonsChecked() : false)
  )) {
    shakeStepNav();
    alert("Run Check Ion Counts before continuing to Final Structure Review.");
    return;
  }
  if (structIdx >= 0 && idx > structIdx && !_protonationComputed) {
    shakeStepNav();
    alert('Please compute protonation (or check "Skip protonation") before proceeding.');
    return;
  }

  document.getElementById('compute-queue-status')?.classList.add('hidden');
  state.currentStepIdx = idx;
  const modName = state.wizardSteps[idx];
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById(`panel-${modName}`);
  if (panel) panel.classList.add('active');

  // Navigation only — NO server-side execution.
  // Check buttons handle all checkpoint creation.
  // Next button is gated by isCurrentStepFulfilled() which
  // checks _checkedSteps (set by Check button handlers).

  // When moving backward, clear checked state for later steps
  for (var i = idx + 1; i < state.wizardSteps.length; i++) {
    state.completedSteps.delete(i);
    _checkedSteps.delete(state.wizardSteps[i]);
    clearStepProgress(state.wizardSteps[i]);
  }
  if (idx <= state.wizardSteps.indexOf('membrane')) { _compositionChecked = false; _membraneCheckpointPdb = null; _membraneActualBox = null; _membraneActualCounts = null; }
  if (idx <= state.wizardSteps.indexOf('solvation')) {
    _solvChecked = pureMembraneIncludesSolvent() ? false : true;
  }
  if (idx <= state.wizardSteps.indexOf('structure')) _protonationComputed = false;

  // When navigating TO Step 4, restore the authoritative config and viewer
  // checkpoint. The legacy `orient` field is normally null; step_orient_config
  // is the value persisted by the incremental Step API.
  if (modName === 'orient' && state.taskId) {
    fetch('/api/task/' + state.taskId).then(function(r) { return r.json(); }).then(function(ts) {
      var restored = _restoreOrientationConfig(ts);
      if (restored && window._loadOrientationCheckpointPreview) {
        window._loadOrientationCheckpointPreview().then(function(loaded) {
          if (!loaded && _orientMode === 'ppm') runPPMAuto();
          if (!loaded && _orientMode === 'manual' && window._scheduleManualOrientationPreview) {
            window._scheduleManualOrientationPreview(0);
          }
        });
      } else if (_orientMode === 'ppm') {
        runPPMAuto();
      } else if (window._scheduleManualOrientationPreview) {
        window._scheduleManualOrientationPreview(0);
      }
    }).catch(function(){
      if (_orientMode === 'ppm') runPPMAuto();
    });
  }

  if (modName === 'forcefield') {
    refreshForceFieldCompatibility();
    watchLigandPreparation();
  }

  // Load viewer data from previous step's checkpoint (not current step —
  // current step's checkpoint is created by the Check button).
  if (modName === "membrane") {
    refreshV4LipidAvailability();
    if (_viewerLoadTimer !== null) clearTimeout(_viewerLoadTimer);
    _viewerLoadTimer = setTimeout(async function() {
      renderMembraneViewer();
    }, 400);
  }

  if (modName === "solvation") {
    setTimeout(async function() {
      renderSolvationViewer();
    }, 400);
  }

  if (modName === 'final_review') window.renderFinalReview();

  if (modName === 'cg_orientation') {
    setTimeout(showCgOrientationPreview, 250);
  } else if (modName && modName.indexOf('cg_') === 0 && modName !== 'cg_model') {
    setTimeout(function() { renderCoarseGrainedViewer(modName); }, 250);
  }

  // Structure step: auto-compute PROPKA with defaults on first visit
  if (modName === 'structure' && !_protonationComputed) {
    setTimeout(function() { runProtonation(); }, 500);
  }

  updateStepNavHighlight();
  updateNextButtonState();
  syncTaskRoute(idx);
}

function goToNextStep() {
  // Can't advance if current step unfulfilled
  if (!isCurrentStepFulfilled()) {
    shakeStepNav();
    return;
  }
  if (state.currentStepIdx + 1 < state.wizardSteps.length) {
    var prevIdx = state.currentStepIdx;
    var nextIdx = state.currentStepIdx + 1;
    // Navigate first — goToWizardStep validates and may reject
    goToWizardStep(nextIdx);
    // Only mark previous step complete if navigation actually succeeded
    if (state.currentStepIdx === nextIdx) {
      markStepComplete(prevIdx);
    }
  }
}

function goToPrevStep() {
  // Always allowed — no lock on backward navigation
  if (state.currentStepIdx > 0) {
    goToWizardStep(state.currentStepIdx - 1);
  } else {
    goToTaskSelect();
  }
}

function updateStepNavHighlight() {
  const btns = document.querySelectorAll('#step-nav .step[data-step-module]');
  btns.forEach((btn, i) => {
    btn.classList.remove('active', 'done', 'locked');
    const unlocked = canGoToStep(i) || i <= state.currentStepIdx;  // past steps always accessible
    if (!unlocked && i > state.currentStepIdx) {
      btn.classList.add('locked');
    }
    if (i < state.currentStepIdx) btn.classList.add('done');
    if (i === state.currentStepIdx) btn.classList.add('active');
  });

  // Title step
  const titleBtn = document.querySelector('#step-nav .step:first-child');
  if (titleBtn && !titleBtn.dataset.stepModule) {
    titleBtn.classList.toggle('active', state.currentStepIdx === -1);
  }
}

/** Brief shake animation when user tries to skip ahead. */
function shakeStepNav() {
  const nav = document.getElementById('step-nav');
  if (!nav) return;
  nav.classList.add('shake');
  setTimeout(function() { nav.classList.remove('shake'); }, 400);
}

// ===================================================================
// Button Wiring (delegated — no per-panel inline data-step needed)
// ===================================================================

document.addEventListener('click', (e) => {
  // Next buttons
  if (e.target.closest('.next-btn') && !e.target.closest('#panel-task-type')) {
    e.preventDefault();
    if (state.currentStepIdx === state.wizardSteps.length - 1) {
      // On last step, this is the Run button — do nothing (handled by runBtn)
    } else {
      goToNextStep();
    }
  }
  // Back buttons
  if (e.target.closest('.back-btn')) {
    e.preventDefault();
    goToPrevStep();
  }
});

document.addEventListener('keydown', function(event) {
  var option = event.target.closest('[role="option"]');
  if (!option || !['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
  var listbox = option.closest('[role="listbox"]');
  if (!listbox) return;
  var options = Array.from(listbox.querySelectorAll('[role="option"]:not(:disabled)'));
  var index = options.indexOf(option);
  if (index < 0 || !options.length) return;
  event.preventDefault();
  if (event.key === 'Home') index = 0;
  else if (event.key === 'End') index = options.length - 1;
  else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') index = (index + 1) % options.length;
  else index = (index - 1 + options.length) % options.length;
  options[index].focus();
});

// ===================================================================
// Options Loading
// ===================================================================

async function resumeTask(taskId, requestedStepIdx) {
  if (!await ensureOptionsLoaded()) return false;
  _resumeError='';document.getElementById('task-resume-error')?.remove();
  try {
    state.customLipidBusy = false;
    var res = await fetch("/api/task/" + taskId + "/resume");
    if (!res.ok) throw new Error("Task not found, expired, or temporarily unavailable. Enter the Task ID to retry.");
    var taskState = await res.json();
    var resumeStepData = null;
    try {
      var resumeStepResponse = await fetch('/api/steps/' + taskId);
      if (resumeStepResponse.ok) resumeStepData = await resumeStepResponse.json();
    } catch(e) { /* task state remains a valid fallback */ }

    var savedTaskType = taskState.task_type || {};
    if (savedTaskType.requires_input !== false &&
        !taskState.pdb_info_full && !taskState.pdb_info) {
      throw new Error("Task has no structure data yet. Retry after its upload finishes.");
    }

    // Initialize wizard steps from saved state; fall back to task type defaults
    var savedTypeId = taskState.task_type_id ||
      ((taskState.task_type || {}).id) ||
      (resumeStepData && resumeStepData.pipeline_type);
    if (!state.wizardSteps || state.wizardSteps.length === 0 ||
        !state.taskType || state.taskType.id !== savedTypeId) {
      // Try to use the actual task type from saved state
      if (savedTypeId) {
        try {
          var typeRes = await fetch('/api/task-type/' + savedTypeId);
          if (typeRes.ok) {
            var typeDetail = await typeRes.json();
            state.wizardSteps = typeDetail.visible_modules || [];
            state.taskType = typeDetail;
          }
        } catch(e) { /* fall through to hardcoded default */ }
      }
      // Final fallback
      if (!state.wizardSteps || state.wizardSteps.length === 0) {
        // Fallback if no task type found — include all possible steps
        state.wizardSteps = taskState.visible_modules || ["input","forcefield","structure","solvation","ions","final_review","simparams"];
        state.taskType = { id: savedTypeId || "solvator", visible_modules: state.wizardSteps, pipeline: "solvator" };
      }
    }
    renderStepNav();
    state.completedSteps = new Set();
    _checkedSteps.clear();
    _checkedConfig = null;
    configureTaskSpecificControls();
    // Restore the force field before initializing simulation parameters.
    // CHARMM and Amber require different non-bonded defaults; initializing
    // against the page's Amber default would leave a resumed CHARMM task with
    // the invalid combination Force-switch + DispCorr=EnerPres.
    var savedForceFieldConfig = taskState.step_forcefield_config || taskState.forcefield || {};
    _charmmCompatSmiles = savedForceFieldConfig.charmm_compat_smiles || {};
    _charmmIdentitySources = {};
    Object.keys(_charmmCompatSmiles).forEach(function(name) {
      if (_charmmCompatSmiles[name]) _charmmIdentitySources[name] = 'smiles';
    });
    Object.keys(savedForceFieldConfig.charmm_compat_mol2 || {}).forEach(function(name) {
      _charmmIdentitySources[name] = 'mol2';
    });
    _charmmMol2Uploads = taskState.ligand_chemistry_uploads || {};
    _charmmCompatResearch = savedForceFieldConfig.charmm_compat_allow_research === true;
    _restoredLigandBackend = savedForceFieldConfig.ligand_ff || null;
    var resumedProteinForceField = document.getElementById('ff-protein');
    if (resumedProteinForceField && typeof savedForceFieldConfig.name === 'string') {
      var savedForceFieldOption = Array.from(resumedProteinForceField.options).some(function(option) {
        return option.value === savedForceFieldConfig.name;
      });
      if (savedForceFieldOption) resumedProteinForceField.value = savedForceFieldConfig.name;
    }
    if (isCoarseGrainedWorkflow()) {
      restoreCoarseGrainedConfig(taskState);
    } else {
      initSimParams();
      restoreSimulationParams(
        taskState.step_simparams_config || taskState.simparams,
        taskState.step_execution_config || taskState.execution
      );
    }

    var pdb = taskState.pdb_info_full || taskState.pdb_info || {};
    var info = {
      filename: pdb.filename || "",
      num_atoms: pdb.num_atoms || 0,
      chains: pdb.chains || [],
      box_nm: pdb.box_nm || [10, 10, 10],
      task_id: taskId,
      pdb_content: taskState.pdb_content || "",
      sequences: taskState.sequences || [],
      small_molecules: pdb.small_molecules || taskState.small_molecules || [],
      validation_warnings: taskState.validation_warnings || [],
      input_status: taskState.input_status,
      cell_info: taskState.cell_info,
      input_validation: taskState.input_validation,
      chain_mapping: taskState.chain_mapping || {},
      selection_info: taskState.input_selection_summary || null,
      fragment_choices: taskState.input_fragment_choices || null,
      task_id: taskId,
    };

    // Set global state
    state.pdbInfo = info;
    state.taskId = taskId;
    await loadTaskCustomLipids();
    _cgenffUploads = taskState.cgenff_uploads || {};
    _smallMolState = {};
    Object.entries(taskState.small_molecule_labels || {}).forEach(function(entry) {
      _smallMolState[String(entry[0]).toUpperCase()] = {
        included: true,
        name: String(entry[1]),
      };
    });
    window._smallMolState = _smallMolState;

    // Restore step-specific state where possible
    if (taskState.structure) {
      // Restore protonation if we had assignments
    }
    _restoreOrientationConfig(taskState);
    if (taskState.membrane) {
      if (taskState.membrane.upper_mix) _mixUpper = taskState.membrane.upper_mix;
      if (taskState.membrane.lower_mix) _mixLower = taskState.membrane.lower_mix;
    }
    if (taskState.ions) {
      // Restore ion config if available
    }
    // Browser form restoration is not authoritative. Restore the value saved
    // by Step 3 and reject stale/out-of-range values such as 0.0.
    var savedStructureConfig = taskState.step_structure_config || taskState.structure || {};
    var savedPH = savedStructureConfig.pH;
    if (savedForceFieldConfig.ligand_pH !== undefined) savedPH = savedForceFieldConfig.ligand_pH;
    if (savedPH === undefined) savedPH = taskState.pH;
    savedPH = Number(savedPH === undefined ? 7.0 : savedPH);
    if (!Number.isFinite(savedPH) || savedPH < 1.0 || savedPH > 13.0) savedPH = 7.0;
    _systemPH = savedPH;
    _ligandChargeRequest++;
    _computedLigandCharges = {}; _computedLigandChargePH = null;
    _ligandChargeDrafts = {}; _ligandChargeOrigins = {};
    Object.entries(savedForceFieldConfig.ligand_charges || {}).forEach(function(entry) {
      _ligandChargeDrafts[entry[0]] = String(entry[1]);
      _ligandChargeOrigins[entry[0]] = 'manual';
    });
    var resumedPHInput = document.getElementById('proc-pH');
    if (resumedPHInput) resumedPHInput.value = savedPH.toFixed(1);

    // Show task ID in header
    var tidEl = document.getElementById("task-id-display");
    var tidBox = document.getElementById("header-task-id");
    if (tidEl) tidEl.textContent = taskId;
    if (tidBox) tidBox.classList.remove("hidden");

    // Show upload info and navigate
    if (info.num_atoms > 0) showUploadInfo(info);
    restoreInputSelection(taskState.input_selection || {});
    var savedInput = taskState.step_input_config || {};
    document.getElementById('input-allow-incomplete').checked = savedInput.allow_incomplete_protein === true;
    document.getElementById('input-renumber-residues').checked = savedInput.renumber_residues === true;
    Object.keys(savedInput.chain_names || {}).forEach(function(chain) {
      if (_chainState[chain]) {
        _chainState[chain].name = savedInput.chain_names[chain];
        document.querySelectorAll('.chain-rename').forEach(function(button) {
          if (button.dataset.chain === chain) button.textContent = savedInput.chain_names[chain];
        });
      }
    });
    _savedFragmentConfig = savedInput;
    setInputModificationReport(taskState.input_modifications || {});
    restoreStructureProcessingConfig(savedStructureConfig);
    renderModificationGeometryReport(taskState.modification_geometry || []);

    // Restore step-specific flags from actual checkpoint data
    var _resumeCheckedSteps = [];
    if (resumeStepData) {
      _resumeCheckedSteps = (resumeStepData.steps || []).filter(function(step) {
        return typeof step === 'string' || step.has_checkpoint;
      }).map(function(step) {
        return typeof step === 'string' ? step : step.name;
      });
    }
    if (!resumeStepData && !taskState.input_check_required) {
      _resumeCheckedSteps = (taskState.steps_completed || []).slice();
    }
    if (taskState.input_check_required) {
      _resumeCheckedSteps = [];
      revokeStepPass('input');
      var recheckStatus = document.getElementById('input-check-status');
      if (recheckStatus) {
        recheckStatus.textContent = 'Run Check Upload again: this task has not passed the current input validation.';
        recheckStatus.style.color = '#d97706';
      }
    }
    _protonationComputed = _resumeCheckedSteps.indexOf('structure') >= 0;
    _solvChecked = _resumeCheckedSteps.indexOf('solvation') >= 0;
    _compositionChecked = _resumeCheckedSteps.indexOf('membrane') >= 0;
    var membraneRecord = (resumeStepData && resumeStepData.steps || []).find(function(step) {
      return step && typeof step === 'object' && ['membrane', 'cg_environment'].includes(step.name);
    });
    renderMembraneCompositionWarnings(membraneRecord && membraneRecord.membrane_metrics && membraneRecord.membrane_metrics.membrane_composition_warnings || []);
    _membraneActualCounts = membraneRecord && membraneRecord.membrane_metrics && membraneRecord.membrane_metrics.membrane || null;
    _membraneActualBox = membraneRecord && membraneRecord.membrane_metrics && membraneRecord.membrane_metrics.box_dimensions_nm || null;
    var membraneConfig = taskState.step_membrane_config || {};
    if (membraneConfig.lipid_composition && membraneConfig.lipid_composition.upper) {
      _mixUpper = membraneConfig.lipid_composition.upper;
      _asymmetric = Array.isArray(membraneConfig.lipid_composition.lower);
      _mixLower = _asymmetric ? membraneConfig.lipid_composition.lower : _mixUpper.map(m => ({...m}));
      var asymmetryToggle = document.getElementById('asymmetric-bilayer');
      if (asymmetryToggle) asymmetryToggle.checked = _asymmetric;
      document.getElementById('lower-leaflet-section').classList.toggle('hidden', !_asymmetric);
      updateLeafletLabels();
    }
    var savedLipidCount = document.getElementById('n-lipids-per-leaflet');
    if (savedLipidCount && membraneConfig.n_lipids_per_leaflet) savedLipidCount.value = membraneConfig.n_lipids_per_leaflet;

    var hasIonCheckpoint = _resumeCheckedSteps.indexOf('ions') >= 0;
    if (window._setIonsChecked) window._setIonsChecked(hasIonCheckpoint);
    if (window._setSystemConfirmed) window._setSystemConfirmed(false);
    if (window.invalidateFinalReview) window.invalidateFinalReview();
    // Pre-mark completed steps
    var completedSteps = _resumeCheckedSteps;
    completedSteps.forEach(function(stepName) {
      var si = state.wizardSteps.indexOf(stepName);
      if (si >= 0) markStepComplete(si);
    });

    if (_resumeCheckedSteps.includes('input') && info.num_atoms > 0) showCheckedInputSummary(info);

    if (isCoarseGrainedWorkflow()) {
      var cgSystemRecord = (resumeStepData && resumeStepData.steps || []).find(function(step) {
        return step && typeof step === 'object' && step.name === 'cg_system';
      });
      var cgConfirmation = document.getElementById('cg-confirm-system');
      if (cgConfirmation && cgSystemRecord) {
        cgConfirmation.checked = cgSystemRecord.confirmed === true;
        cgConfirmation.disabled = cgSystemRecord.preview_available !== true;
        if (cgSystemRecord.confirmed === true) _checkedSteps.add('cg_system');
      }
    }

    // Navigation still checks the restored server checkpoints.
    state.currentStepIdx = 0;

    var resumedBuild = taskState.build_status || {};
    var completedBuild = resumedBuild.status === "completed" &&
      resumedBuild.download_available === true && resumedBuild.result &&
      !taskState.input_check_required;

    // Navigate to the first incomplete step. A completed finalization resumes
    // directly on the Simulation Parameters result panel unless the URL
    // explicitly requested another step.
    var currentStep = taskState.resume_step || taskState.current_step ||
      state.wizardSteps[0] || "input";
    if (currentStep === 'simparams' && !completedBuild) currentStep = 'final_review';
    var stepIdx = state.wizardSteps.indexOf(currentStep);
    if (taskState.input_check_required) {
      stepIdx = state.wizardSteps.indexOf('input');
    } else if (Number.isInteger(requestedStepIdx)) {
      stepIdx = Math.max(0, Math.min(requestedStepIdx, state.wizardSteps.length - 1));
    } else if (completedBuild) {
      var resultStepIdx = state.wizardSteps.indexOf("simparams");
      if (resultStepIdx >= 0) stepIdx = resultStepIdx;
    }
    if (completedBuild && state.wizardSteps[stepIdx] === 'simparams') {
      // Restore access to an existing download without fabricating fresh approval.
      state.currentStepIdx=stepIdx;
      document.querySelectorAll('.panel').forEach(panel=>panel.classList.toggle('active',panel.id==='panel-simparams'));
      updateStepNavHighlight();
    } else {
      while (stepIdx > 0 && !canGoToStep(stepIdx)) stepIdx--;
      if (stepIdx >= 0) goToWizardStep(stepIdx);
    }

    // Show success message after navigation
    var taskLink = document.getElementById("resume-task-id");
    if (taskLink) taskLink.value = taskId;
    // Remove alert — use non-blocking notification instead
    var headerSub = document.getElementById("header-task-title");
    if (headerSub) {
      headerSub.textContent = "Task " + taskId + " resumed — " + completedSteps.length + " steps restored.";
      setTimeout(function() { headerSub.textContent = state.taskType?.title || ""; }, 5000);
    }
    if (completedBuild && state.wizardSteps[stepIdx] === "simparams") {
      var progressSection = document.getElementById("progress-section");
      if (progressSection) progressSection.classList.remove("hidden");
      _showBuildResult(resumedBuild.result);
      syncTaskRoute(stepIdx, true);
    } else if (resumedBuild.status === "queued" || resumedBuild.status === "running") {
      state.buildRunning = true;
      startBuildProgress(true);
      setBuildProgressPhase('Reconnected to build — waiting for status…');
      watchBuildResult({task_id:taskId}, 2000, null);
      try {
        var queueResponse = await fetch("/api/build/" + taskId + "/queue-status");
        var queueState = queueResponse.ok ? await queueResponse.json() : resumedBuild;
        queueState.task_id = taskId;
        showComputeQueueStatus(queueState);
      } catch (error) {
        resumedBuild.task_id = taskId;
        showComputeQueueStatus(resumedBuild);
      }
    }
    rememberCurrentTask();
    return true;
  } catch (e) {
    _resumeError='Could not restore task: '+e.message;
    showResumeError(_resumeError);
    return false;
  }
}


let _optionsPromise = null;
window._optionsReady = false;
function optionNotice(message) {
  let notice = document.getElementById('options-load-status');
  if (!notice) {
    notice = document.createElement('div'); notice.id = 'options-load-status';
    notice.setAttribute('role', 'status');
    document.getElementById('task-grid').before(notice);
  }
  notice.replaceChildren(document.createTextNode(message + ' '));
  return notice;
}
function ensureOptionsLoaded() {
  return window._optionsReady ? Promise.resolve(true) : loadOptions();
}
function loadOptions() {
  if (_optionsPromise) return _optionsPromise;
  _optionsPromise = (async () => {
    try {
      optionNotice('Loading parameter choices… You can open a workflow now.');
      const deadline = performance.now() + 60000;
      let opts;
      do {
        opts = await GMXHttp.json('/api/options', {}, {
          validate: data => Array.isArray(data.lipids) && Array.isArray(data.force_fields) &&
            Array.isArray(data.water_models)
        });
        if (opts.status !== 'checking') break;
        if (performance.now() >= deadline) throw new Error('Parameter choices are still preparing.');
        await new Promise(resolve => setTimeout(resolve, 750));
      } while (true);
      buildLipidPicker(opts.lipids, opts.lipid_categories, opts.availability);
      initLipidMixing();
      if (state.taskId) setTimeout(loadTaskCustomLipids, 0);
      window._allWaterModels = opts.water_models || [];
      window._allSolvents = opts.solvents || [];
      window._forceFieldOptions = opts.force_fields || [];
      renderProteinForceFieldOptions();
      updateWaterModelOptions(true);
      syncLockedWaterModelDisplay();
      window._optionsReady = true;
      document.getElementById('options-load-status')?.remove();
      return true;
    } catch (err) {
      const notice = optionNotice('Parameter choices could not be loaded.');
      const retry = document.createElement('button'); retry.type = 'button';
      retry.textContent = 'Retry loading choices'; retry.addEventListener('click', loadOptions);
      notice.appendChild(retry);
      return false;
    } finally { _optionsPromise = null; }
  })();
  return _optionsPromise;
}

function populateSelect(id, items) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = '';
  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item.value;
    opt.textContent = item.label;
    if (item.disabled) {
      opt.disabled = true;
      // The label says "Coming Soon"; the tooltip says what would actually
      // make it available, which is usually running the installer.
      if (item.title) opt.title = item.title;
    }
    sel.appendChild(opt);
  });
}

// Force fields the interface knows about but cannot build with are shown
// greyed out rather than hidden. Hiding them makes a capability the project
// has look like one it lacks, and leaves a user who was told to select it
// with nowhere to look.
function renderProteinForceFieldOptions() {
  var select = document.getElementById('ff-protein');
  var options = window._forceFieldOptions || [];
  if (!select || !options.length) return;

  var previous = select.value;
  populateSelect('ff-protein', options.map(function(ff) {
    // The catalog label already carries the release and whether it is legacy;
    // repeating that here produced "(legacy) — legacy".
    var label = ff.label || ff.name;
    if (!ff.installed) label += ' — Coming Soon';
    return {
      value: ff.name,
      label: label,
      disabled: !ff.installed,
      title: ff.installed ? '' :
        'Not installed on this deployment. Run ./install-local.sh to fetch and ' +
        'verify its parameters.'
    };
  }));

  var installed = options.filter(function(ff) { return ff.installed; });
  var wanted = [previous, 'amber14sb'].concat(installed.map(function(ff) { return ff.name; }));
  for (var i = 0; i < wanted.length; i++) {
    if (installed.some(function(ff) { return ff.name === wanted[i]; })) {
      select.value = wanted[i];
      return;
    }
  }
}

function updateWaterModelOptions(useForceFieldDefault) {
  var select = document.getElementById('ff-water-model');
  var ffSelect = document.getElementById('ff-protein');
  if (!select || !ffSelect || !window._allWaterModels) return;
  var ffName = ffSelect.value;
  var previous = select.value;
  var models = window._allWaterModels.filter(function(model) {
    return !model.supported_force_fields || model.supported_force_fields.indexOf(ffName) >= 0;
  });
  populateSelect('ff-water-model', models.map(function(model) {
    return {value: model.name, label: model.full_name + ' (' + model.n_atoms + '-site)'};
  }));
  var ffInfo = (window._forceFieldOptions || []).find(function(ff) { return ff.name === ffName; });
  var preferred = ffInfo ? ffInfo.water_model : 'tip3p';
  var allowed = models.map(function(model) { return model.name; });
  if (!useForceFieldDefault && allowed.indexOf(previous) >= 0) select.value = previous;
  else if (allowed.indexOf(preferred) >= 0) select.value = preferred;
  else if (allowed.length) select.value = allowed[0];
  syncLockedWaterModelDisplay();
}

function syncLockedWaterModelDisplay() {
  var selected = document.getElementById('ff-water-model');
  var display = document.getElementById('water-model-display');
  if (display) display.value = selected && selected.value ? selected.value.toUpperCase() : '—';
}

function currentMembraneLipidNames() {
  if (!state.taskType || (state.taskType.visible_modules || []).indexOf('membrane') < 0) return [];
  var names = [];
  var mixes = [_mixUpper || []];
  if (_asymmetric) mixes.push(_mixLower || []);
  mixes.forEach(function(mix) {
    mix.forEach(function(item) {
      if (Number(item.ratio) > 0 && names.indexOf(item.name) < 0) names.push(item.name);
    });
  });
  return names.sort();
}

window._ffCompatibility = null;
let _ffCompatibilityRequest = 0;
let _ffCompatibilityController = null;
let _ffCompatibilityValid = false;
let _forceFieldRevision = 0;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, function(character) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character];
  });
}

function populateCompatibilitySelect(id, options) {
  var select = document.getElementById(id);
  if (!select) return false;
  var previous = select.value;
  select.innerHTML = '';
  var firstEnabled = null;
  (options || []).forEach(function(item) {
    var option = document.createElement('option');
    option.value = item.value;
    option.disabled = !item.enabled;
    option.textContent = item.label + (item.enabled ? '' : ' — unavailable: ' + (item.reason || 'incompatible'));
    select.appendChild(option);
    if (item.enabled && firstEnabled === null) firstEnabled = item.value;
  });
  var previousAllowed = (options || []).some(function(item) {
    return item.value === previous && item.enabled;
  });
  if (previousAllowed) select.value = previous;
  else if (firstEnabled !== null) select.value = firstEnabled;
  else select.selectedIndex = -1;
  return firstEnabled !== null;
}

async function refreshForceFieldCompatibility() {
  if (!state.taskId) return;
  _ffCompatibilityController?.abort();
  const controller = _ffCompatibilityController = new AbortController();
  const request = ++_ffCompatibilityRequest, task = state.taskId;
  var protein = document.getElementById('ff-protein');
  var status = document.getElementById('ff-compatibility-status');
  var confirm = document.getElementById('forcefield-check-btn');
  if (!protein) return;
  const proteinName = protein.value, lipids = JSON.stringify(currentMembraneLipidNames());
  const current = () => request === _ffCompatibilityRequest && task === state.taskId &&
    proteinName === protein.value && lipids === JSON.stringify(currentMembraneLipidNames());
  const selects = ['ff-lipid', 'ff-ligand'].map(id => document.getElementById(id));
  _ffCompatibilityValid = false;
  if (confirm) confirm.disabled = true;
  selects.forEach(select => { if (select) select.disabled = true; });
  updateNextButtonState();
  try {
    if (status) {
      status.textContent = 'Checking installed parameter families...';
      status.style.color = '';
    }
    var report = await GMXHttp.json('/api/forcefield-compatibility/' + task, {
      signal: controller.signal,
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        protein_ff: proteinName,
        lipid_names: JSON.parse(lipids),
      }),
    }, {
      current,
      validate: report => typeof report.family === 'string' &&
        Array.isArray(report.lipid_options) && Array.isArray(report.ligand_options) &&
        [...report.lipid_options, ...report.ligand_options].every(item =>
          item && typeof item.value === 'string' && typeof item.label === 'string' &&
          typeof item.enabled === 'boolean')
    });
    if (!current()) return;
    var previousFamily = window._ffCompatibility && window._ffCompatibility.family;
    if (previousFamily !== report.family) {
      document.getElementById('ff-ligand').value = '';
    }
    window._ffCompatibility = report;
    var lipidOk = populateCompatibilitySelect('ff-lipid', report.lipid_options);
    var ligandOk = populateCompatibilitySelect('ff-ligand', report.ligand_options);
    if (_restoredLigandBackend) {
      if ((report.ligand_options || []).some(function(item) {
        return item.enabled && item.value === _restoredLigandBackend;
      })) document.getElementById('ff-ligand').value = _restoredLigandBackend;
      _restoredLigandBackend = null;
    }
    renderLigandChargeInputs();
    var warnings = [];
    (report.lipid_options || []).forEach(function(item) {
      if (!item.enabled && item.reason) warnings.push('Lipid: ' + item.reason);
    });
    (report.ligand_options || []).forEach(function(item) {
      if (!item.enabled && item.reason) warnings.push('Small molecule: ' + item.reason);
    });
    (report.ligands || []).forEach(function(item) {
      if (report.family !== 'charmm' && item.rtp_reason) {
        warnings.push((item.display_name || item.name) + ': ' + item.rtp_reason);
      }
    });
    var nucleicOk = !report.nucleic_acid || report.nucleic_acid.enabled;
    if (report.nucleic_acid && report.nucleic_acid.present) {
      var nucleic = report.nucleic_acid;
      warnings.push(
        'Nucleic acid (' + (nucleic.polymer_types || []).join('/') + ', ' +
        nucleic.residues + ' residues): ' + nucleic.reason
      );
    }
    var valid = lipidOk && ligandOk && nucleicOk;
    _ffCompatibilityValid = valid;
    if (status) {
      status.innerHTML = '<strong>' + (valid ? '✓ Compatible family: ' : '✗ No complete compatible combination for ') +
        String(report.family || '').toUpperCase() + '</strong>' +
        (warnings.length ? '<ul>' + warnings.map(function(item) { return '<li>' + escapeHtml(item) + '</li>'; }).join('') + '</ul>' : '');
      status.style.color = valid ? '#166534' : '#b91c1c';
    }
    if (confirm) confirm.disabled = !valid || _stepRunning;
    updateV4CompositionAvailability();
  } catch (error) {
    if (!current()) return;
    window._ffCompatibility = null;
    _ffCompatibilityValid = false;
    if (status) {
      status.textContent = '✗ ' + error.message + ' ';
      status.style.color = '#b91c1c';
      const retry = document.createElement('button');
      retry.type = 'button'; retry.className = 'btn';
      retry.id = 'ff-compatibility-retry'; retry.textContent = 'Retry compatibility check';
      retry.addEventListener('click', refreshForceFieldCompatibility);
      status.appendChild(retry);
    }
    if (confirm) confirm.disabled = true;
  } finally {
    if (current()) {
      selects.forEach(select => { if (select) select.disabled = false; });
      updateNextButtonState();
    }
  }
}

function resetLigandPHState() {
  _charmmIdentityRequest++;
  _charmmIdentitySources = {}; _charmmMol2Uploads = {};
  clearTimeout(_charmmIdentityTimer);
  _ligandChargeRequest++;
  _computedLigandCharges = {}; _computedLigandChargePH = null;
  _ligandChargeDrafts = {}; _ligandChargeOrigins = {};
  _systemPH = 7.0;
  ['ff-ligand-ph', 'proc-pH'].forEach(function(id) {
    var input = document.getElementById(id);
    if (input) input.value = '7.0';
  });
}

function setLigandEnvironmentPH(value) {
  var pH = Number(value);
  if (String(value).trim() === '' || !Number.isFinite(pH) || pH < 1 || pH > 13) pH = NaN;
  if (Object.is(pH, _systemPH)) return;
  _systemPH = pH;
  scheduleCharmmIdentity();
  _ligandChargeRequest++; // A late response for an earlier pH cannot overwrite this state.
  _computedLigandCharges = {}; _computedLigandChargePH = null;
  ['ff-ligand-ph', 'proc-pH'].forEach(function(id) {
    var input = document.getElementById(id);
    if (input && input !== document.activeElement) input.value = Number.isFinite(pH) ? String(pH) : '';
  });
  Object.keys(_ligandChargeDrafts).forEach(function(name) {
    if (_ligandChargeOrigins[name] !== 'manual') {
      delete _ligandChargeDrafts[name];
      var input = document.querySelector('[data-ligand-charge="' + name + '"]');
      if (input) input.value = '';
    }
    _computedLigandCharges[name] = {status: 'stale', error: 'pH changed; recalculate the suggestion and review any manual charge.'};
    updateLigandChargeStatus(name);
  });
  resetForceFieldCheck();
  if (typeof invalidateProtonationState === 'function') {
    invalidateProtonationState('Environment pH changed; recompute protein protonation.');
  }
}

function renderLigandChargeInputs() {
  var container = document.getElementById('ff-ligand-charges');
  var source = document.getElementById('ff-ligand');
  if (!container) return;
  container.innerHTML = '';
  var report = window._ffCompatibility;
  if (!report || !source) return;
  if ((report.ligand_names || []).length && ['gaff2', 'charmm_compat', 'cgenff'].includes(source.value)) {
    var phLabel = document.createElement('label');
    phLabel.className = 'field'; phLabel.textContent = 'Solution pH for ligand protonation (1.0–13.0)';
    var phInput = document.createElement('input');
    phInput.type = 'number'; phInput.id = 'ff-ligand-ph';
    phInput.min = '1'; phInput.max = '13'; phInput.step = '0.1'; phInput.required = true;
    phInput.value = Number.isFinite(_systemPH) ? String(_systemPH) : '';
    phInput.addEventListener('input', function() { setLigandEnvironmentPH(phInput.value); });
    phInput.addEventListener('change', function() {
      setLigandEnvironmentPH(phInput.value);
      if (source.value === 'gaff2') loadGaffChargeSuggestions(true);
    });
    phLabel.appendChild(phInput); container.appendChild(phLabel);
    var phHint = document.createElement('p'); phHint.className = 'hint';
    phHint.textContent = source.value === 'gaff2'
      ? 'This pH controls charge suggestions and GAFF2 protonation, and is shared with protein processing. Changing pH invalidates previous checks.'
      : source.value === 'charmm_compat'
        ? 'Automatic identification uses a solution-pH protonation model. Explicit MOL2/SMILES overrides retain their supplied state. This pH is shared with protein processing.'
        : 'CHARMM uses the explicit state in the uploaded CGenFF package; pH does not rewrite its hydrogens or charges.';
    container.appendChild(phHint);
  }
  if (source.value === 'charmm_compat') {
    var localIntro = document.createElement('p');
    localIntro.className = 'hint';
    localIntro.textContent = 'Ligands are identified automatically from the uploaded structure, ' +
      'public chemical definitions and coordinate perception. Provide MOL2 or SMILES only when ' +
      'identification needs clarification, or to override the molecular state. Identification ' +
      'does not guarantee local force-field coverage; unsupported parameters still require CGenFF import.';
    container.appendChild(localIntro);
    (report.ligand_names || []).forEach(function(name) {
      var row = document.createElement('div');
      row.className = 'field charmm-compat-row';
      var label = document.createElement('label');
      label.textContent = ((report.ligand_labels || {})[name] || name) + ' (' + name + ')';
      var mode = document.createElement('select'); mode.dataset.charmmIdentitySource = name;
      [['auto', 'Identify automatically'], ['mol2', 'Provide MOL2'], ['smiles', 'Provide SMILES']].forEach(function(item) {
        var option = document.createElement('option'); option.value = item[0]; option.textContent = item[1];
        mode.appendChild(option);
      });
      mode.value = _charmmIdentitySources[name] || 'auto';
      mode.addEventListener('change', function() {
        _charmmIdentitySources[name] = mode.value;
        _charmmIdentityRequest++; resetForceFieldCheck(); renderLigandChargeInputs();
      });
      label.appendChild(mode); row.appendChild(label);
      var input = document.createElement('input');
      input.type = 'text'; input.maxLength = 4096;
      input.dataset.charmmSmiles = name;
      input.placeholder = 'Exact molecular state, e.g. CCO for ethanol';
      input.value = _charmmCompatSmiles[name] || '';
      input.hidden = mode.value !== 'smiles';
      input.addEventListener('input', function() {
        _charmmCompatSmiles[name] = input.value.trim();
        resetForceFieldCheck(); scheduleCharmmIdentity();
      });
      row.appendChild(input);
      var smilesHint = document.createElement('p'); smilesHint.className = 'hint';
      smilesHint.hidden = mode.value !== 'smiles';
      smilesHint.textContent = 'Include charge and stereochemistry. If atom identities are ambiguous, ' +
        'use MOL2 with matching atom names, or map every SMILES heavy atom 1..N to retained atom order.';
      row.appendChild(smilesHint);
      var fileLabel = document.createElement('label'); fileLabel.textContent = 'MOL2 with bonds and the intended protonation state';
      fileLabel.hidden = mode.value !== 'mol2';
      var file = document.createElement('input'); file.type = 'file'; file.accept = '.mol2';
      file.dataset.charmmMol2 = name;
      file.addEventListener('change', function() { uploadCharmmIdentity(name, file); });
      fileLabel.appendChild(file); row.appendChild(fileLabel);
      var status = document.createElement('p'); status.className = 'hint';
      status.dataset.ligandIdentityStatus = name;
      status.textContent = 'Identifying ligand chemistry…'; row.appendChild(status);
      container.appendChild(row);
    });
    var researchLabel = document.createElement('label');
    researchLabel.className = 'experimental-assignment-warning';
    var research = document.createElement('input');
    research.type = 'checkbox'; research.id = 'ff-charmm-research';
    research.checked = _charmmCompatResearch;
    research.addEventListener('change', function() {
      _charmmCompatResearch = research.checked; resetForceFieldCheck();
    });
    researchLabel.appendChild(research);
    var researchText = document.createElement('span');
    researchText.textContent = 'Allow experimental assignment: atom types and ' +
      'charges are read from environments the installed CHARMM release itself contains, so the range is ' +
      'no longer a fixed list -- aldehydes, amides, nitroaromatics and heterocycles are accepted where ' +
      'the release covers them. Out-of-plane (improper) terms are NOT assigned, so any planar centre is ' +
      'unrestrained. Every export reports which atoms had no corroborating environment and how far the ' +
      'charges of the matched ones spread. Physical accuracy has not been validated.';
    researchLabel.appendChild(researchText);
    container.appendChild(researchLabel);
    loadCharmmIdentity();
    return;
  }
  if (source.value === 'cgenff') {
    var intro = document.createElement('div');
    intro.className = 'validation-warnings';
    intro.innerHTML = '<strong>External CGenFF parameters required.</strong> Submit each retained molecule to ' +
      '<a href="https://cgenff.com/" target="_blank" rel="noopener noreferrer">CGenFF/ParamChem</a>, ' +
      'then upload the exact MOL2 submitted to the website and its returned STR file. ' +
      'Atom names and CGenFF release must match; Check will reject incomplete or mismatched packages.';
    container.appendChild(intro);
    (report.ligand_names || []).forEach(function(name) {
      var displayName = (report.ligand_labels || {})[name] || name;
      var row = document.createElement('div');
      row.className = 'field cgenff-upload-row';
      var label = document.createElement('strong');
      label.textContent = displayName + ' (' + name + ')';
      function createFileControl(title, buttonText, accept, datasetName) {
        var control = document.createElement('div');
        control.className = 'field cgenff-file-control';
        var fileLabel = document.createElement('span');
        fileLabel.textContent = title;
        var input = document.createElement('input');
        input.type = 'file'; input.accept = accept; input.hidden = true;
        input.dataset[datasetName] = name;
        var choose = document.createElement('button');
        choose.type = 'button'; choose.className = 'btn'; choose.textContent = buttonText;
        var filename = document.createElement('span');
        filename.className = 'hint'; filename.textContent = 'No file selected';
        choose.addEventListener('click', function() { input.click(); });
        input.addEventListener('change', function() {
          filename.textContent = input.files.length ? input.files[0].name : 'No file selected';
        });
        control.appendChild(fileLabel); control.appendChild(input);
        control.appendChild(choose); control.appendChild(filename);
        return { element: control, input: input };
      }
      var mol2Control = createFileControl(
        'Submitted MOL2 — the exact molecule file uploaded to ParamChem',
        'Choose submitted MOL2', '.mol2', 'cgenffMol2'
      );
      var streamControl = createFileControl(
        'Returned STR — the parameter stream downloaded from ParamChem',
        'Choose returned STR', '.str', 'cgenffStr'
      );
      var mol2 = mol2Control.input;
      var stream = streamControl.input;
      var button = document.createElement('button');
      button.type = 'button'; button.className = 'btn'; button.textContent = 'Upload and validate';
      var status = document.createElement('span');
      status.className = 'hint'; status.dataset.cgenffStatus = name;
      var saved = _cgenffUploads[name];
      if (saved && saved.force_field === document.getElementById('ff-protein').value) {
        status.textContent = '✓ Validated package already uploaded';
        status.style.color = '#059669';
      } else {
        status.textContent = 'MOL2 + STR required';
      }
      button.addEventListener('click', async function() {
        if (!mol2.files.length || !stream.files.length) {
          status.textContent = '✗ Select both the submitted MOL2 and returned STR file';
          status.style.color = '#dc2626';
          return;
        }
        button.disabled = true;
        status.textContent = 'Uploading and validating...'; status.style.color = '#d97706';
        var form = new FormData();
        form.append('ligand_name', name);
        form.append('force_field', document.getElementById('ff-protein').value);
        form.append('mol2_file', mol2.files[0]);
        form.append('str_file', stream.files[0]);
        try {
          var response = await fetch('/api/cgenff-upload/' + state.taskId, { method: 'POST', body: form });
          var result = await response.json();
          if (!response.ok || result.error) throw new Error(result.error || 'CGenFF upload failed');
          _cgenffUploads[name] = result;
          status.textContent = '✓ Validated CGenFF ' + (result.cgenff_version || 'version not declared') +
            (result.maximum_penalty == null ? '' : ', max penalty ' + result.maximum_penalty) +
            (result.warning ? ' — ' + result.warning : '');
          status.style.color = result.warning ? '#d97706' : '#059669';
          resetForceFieldCheck();
        } catch (error) {
          delete _cgenffUploads[name];
          status.textContent = '✗ ' + error.message; status.style.color = '#dc2626';
          resetForceFieldCheck();
        } finally {
          button.disabled = false;
        }
      });
      row.appendChild(label); row.appendChild(mol2Control.element); row.appendChild(streamControl.element);
      row.appendChild(button); row.appendChild(status);
      container.appendChild(row);
    });
    return;
  }
  if (source.value !== 'gaff2') return;
  (report.ligand_names || []).forEach(function(name) {
    var displayName = (report.ligand_labels || {})[name] || name;
    var row = document.createElement('label');
    row.className = 'field';
    row.textContent = displayName + ' integer net charge ';
    var input = document.createElement('input');
    input.type = 'number'; input.step = '1'; input.value = '';
    input.required = true; input.placeholder = 'Required';
    input.dataset.ligandCharge = name;
    input.style.width = '80px';
    if (Object.prototype.hasOwnProperty.call(_ligandChargeDrafts, name)) {
      input.value = _ligandChargeDrafts[name];
    }
    input.addEventListener('input', function() {
      _ligandChargeDrafts[name] = input.value;
      _ligandChargeOrigins[name] = 'manual';
      updateLigandChargeStatus(name);
      resetForceFieldCheck();
    });
    row.appendChild(input);
    var chargeStatus = document.createElement('span');
    chargeStatus.className = 'hint';
    chargeStatus.dataset.ligandChargeStatus = name;
    row.appendChild(chargeStatus);
    container.appendChild(row);
  });
  if ((report.ligand_names || []).length) {
    var hint = document.createElement('span');
    hint.className = 'hint';
    hint.textContent = 'Required: enter the integer net charge for each retained molecule at the intended protonation state. GAFF2 then assigns AM1-BCC partial charges.';
    container.appendChild(hint);
    var recompute = document.createElement('button');
    recompute.type = 'button'; recompute.className = 'btn';
    recompute.textContent = 'Recalculate charge suggestions at target pH';
    recompute.addEventListener('click', function() {
      loadGaffChargeSuggestions(true);
    });
    container.appendChild(recompute);
    loadGaffChargeSuggestions(false);
  }
}

function formatIntegerCharge(value) {
  return value > 0 ? '+' + value : String(value);
}

function updateLigandChargeStatus(name) {
  var status = document.querySelector('[data-ligand-charge-status="' + name + '"]');
  var input = document.querySelector('[data-ligand-charge="' + name + '"]');
  if (!status || !input) return;
  var suggestion = _computedLigandCharges[name];
  if (!suggestion || suggestion.status !== 'ok') {
    status.textContent = suggestion && suggestion.error
      ? 'Charge calculation unavailable: ' + suggestion.error
      : 'Computing pH-dependent charge suggestion...';
    status.style.color = '#d97706';
    return;
  }
  var computed = formatIntegerCharge(suggestion.net_charge);
  var base = 'Computed suggestion: ' + computed + ' at pH ' + Number(suggestion.pH).toFixed(1) +
    ' (' + suggestion.formula + '). ';
  if (_ligandChargeOrigins[name] === 'manual') {
    status.textContent = base + 'User override: ' + (input.value || 'blank') + '.';
    status.style.color = '#b45309';
  } else {
    status.textContent = base + 'Edit the integer field to override.';
    status.style.color = '#059669';
  }
}

async function loadGaffChargeSuggestions(force) {
  if (!state.taskId) return;
  var targetPH = Number(_systemPH);
  var names = ((window._ffCompatibility || {}).ligand_names || []);
  if (!names.length) return;
  if (!Number.isFinite(targetPH) || targetPH < 1.0 || targetPH > 13.0) {
    names.forEach(function(name) {
      _computedLigandCharges[name] = {status: 'error', error: 'Enter a solution pH between 1.0 and 13.0.'};
      updateLigandChargeStatus(name);
    });
    return;
  }
  if (!force && _computedLigandChargePH === targetPH && Object.keys(_computedLigandCharges).length) {
    names.forEach(updateLigandChargeStatus);
    return;
  }
  names.forEach(function(name) {
    var status = document.querySelector('[data-ligand-charge-status="' + name + '"]');
    if (status) { status.textContent = 'Computing at pH ' + targetPH.toFixed(1) + '...'; status.style.color = '#d97706'; }
  });
  var requestId = ++_ligandChargeRequest;
  var requestTask = state.taskId;
  try {
    var result = await GMXHttp.json('/api/ligand-charge-suggestions/' + requestTask, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pH: targetPH}),
    }, {
      current: () => requestId === _ligandChargeRequest && requestTask === state.taskId && targetPH === _systemPH,
      validate: data => data.suggestions && typeof data.suggestions === 'object' && Number.isFinite(Number(data.pH))
    });
    if (requestId !== _ligandChargeRequest || requestTask !== state.taskId || targetPH !== _systemPH) return;
    _computedLigandCharges = result.suggestions || {};
    _computedLigandChargePH = Number(result.pH);
    names.forEach(function(name) {
      var suggestion = _computedLigandCharges[name];
      var input = document.querySelector('[data-ligand-charge="' + name + '"]');
      if (suggestion && suggestion.status === 'ok' && input && _ligandChargeOrigins[name] !== 'manual') {
        input.value = String(suggestion.net_charge);
        _ligandChargeDrafts[name] = input.value;
        _ligandChargeOrigins[name] = 'computed';
      }
      updateLigandChargeStatus(name);
    });
    resetForceFieldCheck();
  } catch (error) {
    if (requestId !== _ligandChargeRequest || requestTask !== state.taskId || targetPH !== _systemPH) return;
    names.forEach(function(name) {
      _computedLigandCharges[name] = {status: 'error', error: error.message};
      updateLigandChargeStatus(name);
    });
  }
}

function collectLigandCharges() {
  var charges = {};
  document.querySelectorAll('[data-ligand-charge]').forEach(function(input) {
    if (/^-?\d+$/.test(input.value.trim())) {
      charges[input.dataset.ligandCharge] = parseInt(input.value, 10);
    }
  });
  return charges;
}

function collectCharmmCompatSmiles() {
  if (document.getElementById('ff-ligand')?.value !== 'charmm_compat') return {};
  var result = {};
  ((window._ffCompatibility || {}).ligand_names || []).forEach(function(name) {
    if (_charmmIdentitySources[name] === 'smiles') result[name] = _charmmCompatSmiles[name] || '';
  });
  return result;
}

function collectCharmmCompatMol2() {
  var result = {};
  if (document.getElementById('ff-ligand')?.value !== 'charmm_compat') return result;
  ((window._ffCompatibility || {}).ligand_names || []).forEach(function(name) {
    if (_charmmIdentitySources[name] === 'mol2') result[name] = {uploaded: true, sha256: _charmmMol2Uploads[name]?.sha256 || null};
  });
  return result;
}

function scheduleCharmmIdentity() {
  _charmmIdentityRequest++;
  clearTimeout(_charmmIdentityTimer);
  document.querySelectorAll('[data-ligand-identity-status]').forEach(function(item) {
    item.textContent = 'Chemistry input changed; identifying again…';
  });
  _charmmIdentityTimer = setTimeout(loadCharmmIdentity, 400);
}

async function loadCharmmIdentity() {
  if (document.getElementById('ff-ligand')?.value !== 'charmm_compat' || !state.taskId) return;
  var requestId = ++_charmmIdentityRequest, taskId = state.taskId, pH = _systemPH;
  var inputs = {charmm_compat_smiles: collectCharmmCompatSmiles(), charmm_compat_mol2: collectCharmmCompatMol2(), ligand_pH: pH};
  var inputKey = JSON.stringify(inputs);
  function current() {
    return requestId === _charmmIdentityRequest && taskId === state.taskId && Object.is(pH, _systemPH) &&
      document.getElementById('ff-ligand')?.value === 'charmm_compat' && inputKey === JSON.stringify({
        charmm_compat_smiles: collectCharmmCompatSmiles(), charmm_compat_mol2: collectCharmmCompatMol2(), ligand_pH: _systemPH});
  }
  try {
    if (!Number.isFinite(pH)) throw new Error('Enter a solution pH between 1 and 13.');
    var missing = Object.keys(inputs.charmm_compat_mol2).filter(function(name) { return !_charmmMol2Uploads[name]?.ready; });
    if (missing.length) throw new Error('Choose a MOL2 file for: ' + missing.join(', '));
    var empty = Object.keys(inputs.charmm_compat_smiles).filter(function(name) { return !inputs.charmm_compat_smiles[name]; });
    if (empty.length) throw new Error('Enter SMILES for: ' + empty.join(', '));
    var data = await GMXHttp.json('/api/ligand-chemistry/' + taskId, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: inputKey
    }, {
      current,
      validate: data => data.ligands && typeof data.ligands === 'object'
    });
    if (!current()) return;
    Object.entries(data.ligands || {}).forEach(function(entry) {
      var status = document.querySelector('[data-ligand-identity-status="' + entry[0] + '"]');
      if (!status) return;
      var record = entry[1];
      var sources = {user_smiles: 'supplied SMILES', user_mol2: 'supplied MOL2', input_mmcif: 'uploaded mmCIF', input_pdb: 'hydrogen-complete uploaded ligand',
        wwpdb_ccd: 'wwPDB chemical dictionary', coordinate_perception: 'coordinate perception'};
      status.textContent = record.status === 'ok'
        ? '✓ Identified from ' + (sources[record.source] || record.source) + ': ' + record.smiles +
          '; net charge ' + record.net_charge + '. ' + (record.warnings || []).join(' ')
        : 'Clarification required: ' + record.error;
    });
  } catch (error) {
    if (!current()) return;
    document.querySelectorAll('[data-ligand-identity-status]').forEach(function(item) { item.textContent = error.message; });
  }
}

async function uploadCharmmIdentity(name, fileInput) {
  if (!fileInput.files.length) return;
  var taskId = state.taskId, selected = fileInput.files[0];
  delete _charmmMol2Uploads[name];
  var status = document.querySelector('[data-ligand-identity-status="' + name + '"]');
  _charmmIdentityRequest++; resetForceFieldCheck();
  if (status) status.textContent = 'Checking MOL2 against the retained ligand…';
  var form = new FormData(); form.append('ligand_name', name); form.append('mol2_file', selected);
  try {
    var response = await fetch('/api/ligand-chemistry-upload/' + taskId, {method: 'POST', body: form});
    var result = await response.json();
    if (taskId !== state.taskId || fileInput.files[0] !== selected) return;
    if (!response.ok || result.error) throw new Error(result.error || 'MOL2 validation failed');
    _charmmMol2Uploads[name] = result;
    loadCharmmIdentity();
  } catch (error) {
    if (taskId !== state.taskId || fileInput.files[0] !== selected) return;
    if (status) status.textContent = error.message;
  }
}

function collectCGenFFParameters() {
  var result = {};
  if (document.getElementById('ff-ligand')?.value !== 'cgenff') return result;
  var report = window._ffCompatibility || {};
  var forceField = document.getElementById('ff-protein')?.value;
  (report.ligand_names || []).forEach(function(name) {
    var item = _cgenffUploads[name];
    if (item && item.force_field === forceField && item.ready) {
      // Paths remain server-owned and task-scoped.  The marker only declares
      // that this browser expects the previously validated package to be used.
      result[name] = { uploaded: true };
    }
  });
  return result;
}

function applyForceFieldResolution(metrics) {
  var resolution = metrics.forcefield_resolution || {};
  var status = document.getElementById('forcefield-check-status');
  if (status && resolution.effective_protein_ff) {
    status.textContent += ' · protein ' + String(resolution.effective_protein_ff).toUpperCase() +
      ' / lipid ' + String(resolution.effective_lipid_ff || '—').toUpperCase() +
      ' / water ' + String(resolution.water_model || '—').toUpperCase();
  }
}

function resetForceFieldCheck() {
  _forceFieldRevision++;
  _checkedSteps.delete('forcefield');
  if (_checkedConfig) delete _checkedConfig.forcefield;
  var status = document.getElementById('forcefield-check-status');
  if (status) status.textContent = '';
  updateNextButtonState();
  updateStepNavHighlight();
}

// ===================================================================
// Custom Lipid Picker and modal live in app_parts/custom_lipids.js.
// ===================================================================

// ===================================================================
// Orientation Step
// ===================================================================

let _dominantLipidDHH = null;  // nm, set when lipid selected; used for membrane plane rendering
// _orientMode: UI tab selection ('ppm' = auto tab, 'manual' = manual tab)
// _orientAlgorithm: which auto algorithm is selected (ppm/hmoment/tmd/com)
let _orientMode = 'ppm';
let _orientAlgorithm = 'ppm';
let _orientZOffset = 0.0;
let _orientTilt = 0.0;
let _orientPhi = 0.0;  // azimuthal tilt direction (degrees)
let _orientedPdbContent = null;  // backend-generated Step 4 coordinates for preview
let _orientPreviewRequestId = 0;
let _orientPreviewTimer = null;

function orientationHydrophobicHalfThickness() {
  function leafletDhh(entries) {
    var weighted = 0.0;
    var total = 0.0;
    (entries || []).forEach(function(entry) {
      var lipid = (_lipidPickerData.lipids || []).find(function(item) {
        return item.name === entry.name;
      });
      var dhh = Number(lipid && lipid.bilayer_thickness);
      var ratio = Number(entry && entry.ratio);
      if (Number.isFinite(dhh) && dhh > 0 && Number.isFinite(ratio) && ratio > 0) {
        weighted += dhh * ratio;
        total += ratio;
      }
    });
    return total > 0 ? weighted / total : null;
  }
  var upper = leafletDhh(_mixUpper);
  var lower = leafletDhh(_asymmetric ? _mixLower : _mixUpper);
  var values = [upper, lower].filter(function(value) { return Number.isFinite(value); });
  var dhh = values.length
    ? values.reduce(function(sum, value) { return sum + value; }, 0.0) / values.length
    : Number(_dominantLipidDHH || 3.8);
  // DHH measures headgroup-to-headgroup separation; the PPM transfer-energy
  // slab represents only the hydrocarbon core, approximately 80% of DHH.
  return Math.max(0.5, Math.min(3.0, dhh * 0.4));
}

const _ALGO_DESC = {
  'ppm': 'PPM-like Wimley-White whole-residue transfer free energy minimization. A confident hydrophobic transmembrane-helix consensus defines the membrane normal; whole-protein PCA is used only as a fallback. Hydrophobic residues favour the membrane core and charged/polar residues favour the aqueous phase. Review the physical-quality report because this is not the external OPM/PPM server.',
  'hmoment': 'Computes the 3D hydrophobic-moment vector (Eisenberg consensus scale) and aligns it to the membrane normal. Best for α-helical proteins with clear amphipathic character.',
  'tmd': 'Sliding-window Kyte-Doolittle hydropathy scan (window=19, threshold=1.6) to detect trans-membrane helices. Positions the membrane midplane at the centre of predicted TM segments.',
  'com': 'Places the protein centre of mass at the membrane midplane (z=0). Simplest method — no hydrophobicity information used.',
};

function _setOrientationModeUI() {
  document.querySelectorAll('.orient-tab').forEach(function(tab) {
    tab.classList.toggle('active', tab.dataset.method === _orientMode);
  });
  var autoResult = document.getElementById('orient-auto-result');
  var manual = document.getElementById('orient-manual');
  if (autoResult) autoResult.classList.toggle('hidden', _orientMode !== 'ppm');
  if (manual) manual.classList.toggle('hidden', _orientMode !== 'manual');
  var algo = document.getElementById('orient-algorithm');
  if (algo && _orientAlgorithm) algo.value = _orientAlgorithm;
}

function _restoreOrientationConfig(taskState) {
  var savedConfig = taskState && taskState.step_orient_config;
  var savedResult = taskState && taskState.orient;
  var saved = savedConfig || savedResult;
  if (!saved || !saved.method) return false;
  _orientMode = saved.method === 'manual' ? 'manual' : 'ppm';
  if (saved.method !== 'manual') _orientAlgorithm = saved.method;
  var z = Number(saved.z_offset != null ? saved.z_offset : savedResult && savedResult.z_offset);
  var tilt = Number(saved.tilt != null ? saved.tilt : savedResult && savedResult.tilt);
  var phi = Number(saved.phi != null ? saved.phi : savedResult && savedResult.phi);
  if (Number.isFinite(z)) _orientZOffset = z;
  if (Number.isFinite(tilt)) _orientTilt = tilt;
  if (Number.isFinite(phi)) _orientPhi = phi;

  var values = [
    ['orient-manual-z', _orientZOffset],
    ['orient-manual-z-num', _orientZOffset],
    ['orient-manual-tilt', _orientTilt],
    ['orient-manual-tilt-num', _orientTilt],
    ['orient-manual-phi', _orientPhi],
    ['orient-manual-phi-num', _orientPhi],
  ];
  values.forEach(function(entry) {
    var element = document.getElementById(entry[0]);
    if (element) element.value = entry[1];
  });
  var zValue = document.getElementById('orient-manual-z-val');
  var tiltValue = document.getElementById('orient-manual-tilt-val');
  var phiValue = document.getElementById('orient-manual-phi-val');
  if (zValue) zValue.textContent = _orientZOffset.toFixed(2);
  if (tiltValue) tiltValue.textContent = _orientTilt;
  if (phiValue) phiValue.textContent = _orientPhi;
  _setOrientationModeUI();
  return true;
}

function invalidateOrientationCheck(message) {
  var orientIndex = state.wizardSteps.indexOf('orient');
  if (orientIndex >= 0) {
    for (var i = orientIndex; i < state.wizardSteps.length; i++) {
      state.completedSteps.delete(i);
      _checkedSteps.delete(state.wizardSteps[i]);
      if (_checkedConfig) delete _checkedConfig[state.wizardSteps[i]];
    }
  }
  var status = document.getElementById('orient-check-status');
  if (status) {
    status.textContent = message || 'Orientation changed — run Check Orientation again';
    status.style.color = '#d97706';
  }
  updateNextButtonState();
  updateStepNavHighlight();
}

function renderOrientationQuality(quality) {
  var report = document.getElementById('orient-quality-report');
  if (!report) return;
  report.replaceChildren();
  var warnings = quality && Array.isArray(quality.warnings) ? quality.warnings : [];
  if (!quality || Object.keys(quality).length === 0) {
    report.className = 'hidden';
    return;
  }
  report.className = warnings.length ? 'validation-warnings' : 'input-check-report';
  var heading = document.createElement('h4');
  heading.textContent = warnings.length
    ? '\u26a0 Orientation requires scientific review'
    : '\u2713 Orientation geometry checks passed';
  report.appendChild(heading);
  if (warnings.length) {
    var list = document.createElement('ul');
    warnings.forEach(function(warning) {
      var item = document.createElement('li');
      item.textContent = warning;
      list.appendChild(item);
    });
    report.appendChild(list);
  }
  if (quality.core_fraction != null) {
    var metrics = document.createElement('p');
    var metricsText =
      'Core residues: ' + Math.round(Number(quality.core_fraction) * 100) + '%; ' +
      'hydrophobic in core: ' + Math.round(Number(quality.hydrophobic_core_fraction) * 100) + '%; ' +
      'charged in core: ' + Math.round(Number(quality.charged_core_fraction) * 100) + '%.';
    if (quality.tm_bundle_tilt_degrees != null) {
      metricsText += ' TM-bundle tilt: ' +
        Number(quality.tm_bundle_tilt_degrees).toFixed(1) + '°.';
    }
    if (quality.non_tm_residue_count != null) {
      metricsText += ' Non-TM residues in core: ' +
        Number(quality.non_tm_core_residue_count || 0) + '/' +
        Number(quality.non_tm_residue_count) + '.';
    }
    metrics.textContent = metricsText;
    report.appendChild(metrics);
  }
}

async function requestOrientationPreview(config) {
  if (!state.taskId) return null;
  config = Object.assign({}, config || {}, {
    half_thickness: orientationHydrophobicHalfThickness(),
  });
  var requestId = ++_orientPreviewRequestId;
  var response = await fetch('/api/orient-preview/' + state.taskId, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: config}),
  });
  var data = await response.json();
  if (requestId !== _orientPreviewRequestId) return null;
  if (!response.ok || data.status !== 'ok') {
    throw new Error(data.error || 'Orientation preview failed');
  }
  _orientedPdbContent = data.oriented_pdb || null;
  renderOrientationQuality(data.orientation_quality || {});
  if (window._redrawOrientViewer) window._redrawOrientViewer();
  return data;
}

// Diagnostic bodies have one owner. Progress and button labels contain only status.
function uniqueFeedbackMessages(messages) {
  var seen = new Set();
  var result = [];
  (messages || []).forEach(function(value) {
    var text = typeof value === 'string' ? value : (value && value.message) || '';
    String(text).split('\n').forEach(function(line) {
      line = line.trim();
      var key = line.replace(/^(?:[✗×]\s*|Step execution failed:\s*)/, '').replace(/\s+/g, ' ');
      if (key && !seen.has(key)) { seen.add(key); result.push(line); }
    });
  });
  return result;
}

function appendFeedbackMessages(panel, messages) {
  var lines = uniqueFeedbackMessages(messages);
  if (!lines.length) return;
  var list = document.createElement('ul');
  lines.forEach(function(message) {
    var item = document.createElement('li');
    item.textContent = message;
    list.appendChild(item);
  });
  panel.appendChild(list);
}

function renderInputReadiness(report, checked, errorMessage) {
  var panel = document.getElementById('input-readiness-report');
  if (!panel) return;
  panel.replaceChildren();
  panel.classList.toggle('hidden', !report && !errorMessage);
  if (!report && !errorMessage) return;
  report = report || {};
  var errors = uniqueFeedbackMessages((report.errors || []).concat(errorMessage || []));
  panel.classList.toggle('error', errors.length > 0);
  var heading = document.createElement('h4');
  heading.textContent = errors.length ? 'Input requires review' :
    (checked ? 'Input structure validation passed' : 'Input assessment — run Check Upload to validate your selection');
  panel.appendChild(heading);
  var messages = errors.concat(report.warnings || []);
  if ((report.repairable_residues || []).length) {
    messages.push(report.repairable_residues.length +
      (errors.length ? ' residue(s) have repairable missing side-chain atoms. Resolve the blocking issues before automatic repair can run.' :
       ' residue(s) have missing side-chain atoms eligible for automatic repair during Check Upload.'));
  }
  appendFeedbackMessages(panel, messages);
}

function stepFeedbackElement(handle) {
  var input = handle.stepName === 'input';
  var id = input ? 'input-readiness-report' : 'step-feedback-' + handle.buttonId;
  var panel = document.getElementById(id);
  if (!panel) {
    panel = document.createElement('div');
    panel.id = id;
    panel.className = 'input-check-report step-feedback hidden';
    panel.setAttribute('aria-live', 'polite');
  }
  if (handle.element.nextElementSibling !== panel) handle.element.after(panel);
  return panel;
}

function operationHasFeedbackOwner(queueState) {
  if (queueState.task_id && queueState.task_id !== state.taskId) return false;
  var path = queueState.request_url || '';
  if (state.uploadRunning && path === '/api/upload-pdb') return true;
  if (state.buildRunning && (path === '/api/build' || !queueState.operation_id)) return true;
  return /\/api\/(?:step|filter-pdb)\//.test(path) &&
    !!document.querySelector('.step-progress[data-state="running"]');
}

function revokeStepPass(stepName) {
  var start = state.wizardSteps.indexOf(stepName);
  if (start < 0) return;
  for (var i = start; i < state.wizardSteps.length; i++) {
    _checkedSteps.delete(state.wizardSteps[i]);
    state.completedSteps.delete(i);
    if (_checkedConfig) delete _checkedConfig[state.wizardSteps[i]];
  }
  updateNextButtonState();
  updateStepNavHighlight();
}

function renderInputCheckReport(repair, errorMessage, modificationReport, nucleicAcids) {
  var panel = document.getElementById('input-check-report');
  if (!panel) return;
  panel.replaceChildren();
  panel.classList.remove('hidden', 'error');

  var heading = document.createElement('h4');
  if (errorMessage) {
    panel.classList.add('hidden');
    renderInputReadiness(null, false, errorMessage);
    return;
  }

  repair = repair || {};
  var repaired = repair.status === 'repaired';
  heading.textContent = repaired
    ? '\u2713 Automatic protein repair completed'
    : '\u2713 Input structure check completed';
  panel.appendChild(heading);

  var summary = document.createElement('p');
  summary.textContent = repaired
    ? ((repair.residues_repaired || 0) + ' residue(s) repaired; ' +
       (repair.atoms_added || 0) + ' heavy atom(s) added with ' +
       (repair.backend || 'the configured repair backend') + '.')
    : (repair.validation || 'No missing standard protein heavy atoms detected.');
  panel.appendChild(summary);

  var nucleicRows = Array.isArray(nucleicAcids) ? nucleicAcids : [];
  if (nucleicRows.length) {
    var nucleicHeading = document.createElement('h4');
    nucleicHeading.textContent = '\u2713 Nucleic-acid polymer detected';
    panel.appendChild(nucleicHeading);
    var nucleicSummary = document.createElement('p');
    nucleicSummary.textContent = nucleicRows.map(function(item) {
      return (item.polymer_type || 'nucleic acid') + ' chain ' +
        (item.chain_id || '?') + ' (' + item.n_residues + ' residues)';
    }).join('; ') +
      '. Canonical DNA/RNA requires CHARMM36m; native GROMACS topology ' +
      'generation will validate termini, polymer bonds, hydrogens, and charge in Step 3. ' +
      'The uploaded nucleic-acid coordinates will be replaced by the hydrogen-complete ' +
      'pdb2gmx coordinates; review the Step 3 viewer before continuing.';
    panel.appendChild(nucleicSummary);
    var unsupportedNucleic = [];
    nucleicRows.forEach(function(item) {
      (item.unsupported_residues || []).forEach(function(name) {
        if (unsupportedNucleic.indexOf(name) < 0) unsupportedNucleic.push(name);
      });
    });
    if (unsupportedNucleic.length) {
      var warning = document.createElement('p');
      warning.className = 'validation-warnings';
      warning.textContent = 'Unsupported modified nucleotide residue(s): ' +
        unsupportedNucleic.join(', ') +
        '. They will be blocked rather than converted to canonical chemistry.';
      panel.appendChild(warning);
    }
  }

  var residues = Array.isArray(repair.residues) ? repair.residues : [];
  if (residues.length) {
    var table = document.createElement('table');
    var head = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Residue', 'Added heavy atoms'].forEach(function(label) {
      var th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    var body = document.createElement('tbody');
    residues.forEach(function(item) {
      var row = document.createElement('tr');
      var location = document.createElement('td');
      location.textContent = (item.chain || '?') + ':' + item.resid + ' ' + item.resname;
      var atoms = document.createElement('td');
      atoms.textContent = (item.added_atoms || []).join(', ');
      row.appendChild(location);
      row.appendChild(atoms);
      body.appendChild(row);
    });
    table.appendChild(body);
    panel.appendChild(table);
  }

  if (repaired && repair.validation) {
    var validation = document.createElement('p');
    validation.className = 'input-check-validation';
    validation.textContent = repair.validation;
    panel.appendChild(validation);
  }

  var modificationRecords = modificationReport && Array.isArray(modificationReport.records)
    ? modificationReport.records : [];
  if (modificationRecords.length) {
    var modificationHeading = document.createElement('h4');
    modificationHeading.textContent = '\u26a0 Modified protein residues detected';
    panel.appendChild(modificationHeading);
    var modificationSummary = document.createElement('p');
    modificationSummary.textContent =
      'Recognized modifications were converted to standard parent residues and recorded. ' +
      'They will be checked against the selected force field and proposed in Step 3.';
    panel.appendChild(modificationSummary);
    var modificationList = document.createElement('ul');
    modificationRecords.forEach(function(record) {
      var item = document.createElement('li');
      var location = (record.chain || '?') + ':' + record.resid + ' ' + record.original_resname;
      if (record.status === 'recognized') {
        item.textContent = location + ' → ' + record.standard_resname +
          '; recorded as ' + record.patch_id + '.';
      } else {
        item.textContent = record.warning || (location + ': requires manual review.');
      }
      modificationList.appendChild(item);
    });
    panel.appendChild(modificationList);
  }
}

function renderModificationGeometryReport(reports) {
  var panel = document.getElementById('modification-geometry-report');
  if (!panel) return;
  panel.replaceChildren();
  var rows = Array.isArray(reports) ? reports : [];
  if (!rows.length) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.remove('hidden');
  var heading = document.createElement('h4');
  heading.textContent = '\u2713 Modified-residue geometry validation passed';
  panel.appendChild(heading);
  var summary = document.createElement('p');
  summary.textContent =
    'Added or restored heavy atoms were checked against the selected force field\'s ' +
    'equilibrium bond lengths and angles. Hard heavy-atom overlaps are rejected.';
  panel.appendChild(summary);
  var table = document.createElement('table');
  table.innerHTML = '<thead><tr><th>Site</th><th>Patch</th><th>New atoms</th>' +
    '<th>Max bond error</th><th>Max angle error</th><th>Minimum clearance</th></tr></thead>';
  var body = document.createElement('tbody');
  rows.forEach(function(report) {
    var row = document.createElement('tr');
    var clearance = report.min_nonbonded_distance_nm;
    var values = [
      String(report.chain || '?') + ':' + String(report.resid),
      String(report.patch_id || ''),
      (report.added_atoms || []).join(', '),
      Number(report.max_bond_error_nm || 0).toFixed(4) + ' nm',
      Number(report.max_angle_error_deg || 0).toFixed(2) + '\u00b0',
      clearance == null ? 'n/a' : Number(clearance).toFixed(3) + ' nm',
    ];
    values.forEach(function(value) {
      var cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.appendChild(row);
    var deposited = report.deposited_geometry;
    if (deposited && (deposited.retained_deposited_atoms || deposited.rebuilt_atoms)) {
      var detail = document.createElement('p');
      var pieces = [];
      if ((deposited.retained_deposited_atoms || []).length) {
        pieces.push('preserved ' + deposited.retained_deposited_atoms.join(', '));
      }
      if ((deposited.rebuilt_atoms || []).length) {
        pieces.push('built missing ' + deposited.rebuilt_atoms.join(', '));
      }
      detail.textContent = report.chain + ':' + report.resid + ' uploaded geometry: ' +
        pieces.join('; ') + '. ' + (deposited.reason || '');
      panel.appendChild(detail);
    }
  });
  table.appendChild(body);
  panel.appendChild(table);
}

// ---- Generic Check button behavior ----
var _progressHandle = null;

// Every step whose Check writes a checkpoint to disk shows a progress bar
// directly beneath its own Check button. One shared element cannot do this:
// the button that starts a Check lives in a wizard panel that is hidden and
// shown as the user moves between steps, so the bar has to belong to the same
// panel as the button that was clicked. Bars are therefore created on demand —
// which also gives the required behaviour that no bar exists until a Check
// runs — and then kept, showing the outcome of the most recent Check.

var _STEP_PROGRESS_POLL_MS = 800;

var _STEP_PROGRESS_LABELS = {
  input: 'upload check',
  forcefield: 'force field setup',
  structure: 'structure processing',
  orient: 'orientation',
  membrane: 'membrane build',
  solvation: 'solvation',
  ions: 'ion placement',
  cg_model: 'Martini model check',
  cg_mapping: 'protein mapping',
  cg_orientation: 'orientation',
  cg_environment: 'environment build',
  cg_solvation: 'solvation',
  cg_system: 'final system build',
};

/** Find, or build on first use, the progress bar belonging to a Check button. */
function _stepProgressElement(btnId, statusElId) {
  var existing = document.getElementById('step-progress-' + btnId);
  if (existing) return existing;
  var button = document.getElementById(btnId);
  if (!button) return null;

  var element = document.createElement('div');
  element.id = 'step-progress-' + btnId;
  element.className = 'step-progress';
  element.setAttribute('aria-live', 'polite');
  element.innerHTML =
    '<div class="step-progress-row">' +
      '<div class="step-progress-track">' +
        '<div class="step-progress-bar" role="progressbar" aria-valuemin="0"' +
        ' aria-valuemax="100" aria-valuenow="0"></div>' +
      '</div>' +
      '<span class="step-progress-percent">0%</span>' +
    '</div>' +
    '<div class="step-progress-foot">' +
      '<span class="step-progress-phase"></span>' +
      '<span class="step-progress-elapsed"></span>' +
    '</div>';

  // Anchor on the status text where the button has one beside it: the status
  // is inline, so inserting after it puts the bar on the line below the button
  // instead of between the button and its own message. Where the status lives
  // in a different container (the solvation and ion panels wrap the button on
  // its own), the button itself is the anchor.
  var anchor = statusElId ? document.getElementById(statusElId) : null;
  if (!anchor || anchor.parentElement !== button.parentElement) anchor = button;
  anchor.insertAdjacentElement('afterend', element);
  return element;
}

function _paintStepProgress(handle, fraction) {
  var percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  var bar = handle.element.querySelector('.step-progress-bar');
  var readout = handle.element.querySelector('.step-progress-percent');
  if (bar) {
    bar.style.width = percent + '%';
    bar.setAttribute('aria-valuenow', String(percent));
  }
  if (readout) readout.textContent = percent + '%';
}

function _startProgressDisplay(stepName, btnId, statusElId) {
  var element = _stepProgressElement(btnId, statusElId);
  if (!element) return null;
  var label = _STEP_PROGRESS_LABELS[stepName] || stepName;
  var handle = {
    element: element,
    buttonId: btnId,
    stepName: stepName,
    statusId: statusElId,
    startedAt: Date.now(),
    fraction: 0.02,
    poll: null,
    clock: null,
  };

  element.classList.remove('hidden');
  element.setAttribute('data-state', 'running');
  var phase = element.querySelector('.step-progress-phase');
  var elapsed = element.querySelector('.step-progress-elapsed');
  if (phase) phase.textContent = 'Starting ' + label + '…';
  if (elapsed) elapsed.textContent = '0.0 s';
  _paintStepProgress(handle, handle.fraction);

  handle.clock = setInterval(function() {
    if (elapsed) elapsed.textContent = ((Date.now() - handle.startedAt) / 1000).toFixed(1) + ' s';
  }, 100);

  return handle;
}

/**
 * Show the bar under a Check button and keep it live until the Check returns.
 *
 * The percentage is only ever what the server reports: the runner publishes a
 * fraction as it passes each phase, and nothing here invents intermediate
 * values. While a phase is long the bar holds its position, and the animated
 * fill plus the running clock are what say the step is still alive.
 *
 * Returns a handle for finishStepProgress, or null when the button is absent.
 */
function startStepProgress(stepName, btnId, statusElId) {
  if (window.GMXViewer) GMXViewer.invalidate();
  var status = document.getElementById(statusElId);
  if (status) status.textContent = '';
  if (stepName === 'input') {
    document.getElementById('input-readiness-report').classList.add('hidden');
    document.getElementById('input-check-report').classList.add('hidden');
  }
  var oldFeedback = document.getElementById('step-feedback-' + btnId);
  if (oldFeedback) { oldFeedback.replaceChildren(); oldFeedback.classList.add('hidden'); }
  var handle = _startProgressDisplay(stepName, btnId, statusElId);
  if (!handle) return null;
  var element = handle.element;
  var phase = element.querySelector('.step-progress-phase');

  handle.poll = GMXPoll.start(function() {
    return GMXPoll.read('/api/step/' + state.taskId + '/progress').then(function(data) {
      if (handle.done || handle.awaitingPreview || !element.isConnected || !data || !data.running) return;
      // A step can only advance, but two Checks in quick succession can leave
      // a stale entry behind; never let the bar run backwards.
      var reported = Number(data.fraction) || 0;
      if (reported > handle.fraction) {
        handle.fraction = reported;
        _paintStepProgress(handle, reported);
      }
      if (phase && data.phase) phase.textContent = data.phase;
    }).catch(function() {
      // A missed poll is not worth surfacing; the next one catches up.
    });
  }, _STEP_PROGRESS_POLL_MS);

  beginCheckNotice(handle);
  return handle;
}

/**
 * Close out a bar: green and "Complete" with the total time on success, red
 * with a short status on failure. The diagnostic body owns the reason.
 * The bar stays on the page as the record of the
 * most recent Check for that step.
 */
function finishStepProgress(handle, ok, message, report) {
  if (!handle) return;
  if (handle.done && (ok || handle.failed)) return;
  handle.done = true;
  handle.failed = !ok;
  if (handle.poll) { GMXPoll.stop(handle.poll); handle.poll = null; }
  if (handle.clock) { GMXPoll.stop(handle.clock); handle.clock = null; }
  var seconds = (Date.now() - handle.startedAt) / 1000;
  var element = handle.element;
  var phase = element.querySelector('.step-progress-phase');
  var elapsed = element.querySelector('.step-progress-elapsed');
  element.setAttribute('data-state', ok ? 'done' : 'error');
  _paintStepProgress(handle, ok ? 1 : handle.fraction);
  if (phase) phase.textContent = ok ? 'Complete' : 'Check failed';
  if (elapsed) elapsed.textContent = seconds.toFixed(1) + ' s total';
  var status = document.getElementById(handle.statusId);
  if (status) status.textContent = '';
  if (!ok && handle.stepName !== 'build') renderStepFailure(handle, message, report);
}

function renderStepFailure(handle, message, report) {
  var feedback = stepFeedbackElement(handle);
  if (handle.stepName === 'input') {
    renderInputReadiness(report, false, message || 'The check could not finish.');
    document.getElementById('input-check-report').classList.add('hidden');
  } else {
    feedback.replaceChildren();
    feedback.classList.remove('hidden');
    feedback.classList.add('error');
    appendFeedbackMessages(feedback, [message || 'The check could not finish.']);
  }
}

function finishUnfinishedStepProgress(handle) {
  if (handle && !handle.done) finishStepProgress(handle, false, 'Stopped before completion');
  if (handle) handle.settled = true;
  updateNextButtonState();
}

/**
 * Drop the bars of steps whose checkpoints have just been invalidated.
 *
 * A green "Complete" is a claim about work that still stands. Going back and
 * re-checking an earlier step discards every later checkpoint, so the bars
 * that described them must go with the checkpoints rather than stay on screen
 * asserting a result the task no longer holds.
 */
// ---- Background ligand parameterization ----
// Started server-side when the upload Check succeeds, because it depends on
// nothing chosen later and is the slowest thing a Check does. This only
// reports it: a wait already being worked on should not look like a new one.

var _ligandPrepTimer = null;

function watchLigandPreparation() {
  var line = document.getElementById('ff-ligand-prep');
  if (!line || !state.taskId) return;
  if (_ligandPrepTimer) { GMXPoll.stop(_ligandPrepTimer); _ligandPrepTimer = null; }

  function render(data) {
    var molecules = (data && data.molecules) || {};
    var names = Object.keys(molecules);
    if (!data || data.state === 'idle' || !names.length) {
      line.classList.add('hidden');
      return;
    }
    var computing = names.filter(function(n) { return molecules[n] === 'computing'; });
    var ready = names.filter(function(n) { return molecules[n] === 'ready'; });
    line.classList.remove('hidden');
    if (computing.length) {
      line.setAttribute('data-state', 'running');
      line.textContent = 'Preparing GAFF2 parameters for ' + computing.join(', ') +
        ' in the background — Check will reuse the result rather than wait for it.';
      return;
    }
    if (data.state === 'done' && ready.length) {
      line.setAttribute('data-state', 'done');
      line.textContent = 'GAFF2 parameters ready for ' + ready.join(', ') + '.';
      if (_ligandPrepTimer) { GMXPoll.stop(_ligandPrepTimer); _ligandPrepTimer = null; }
      return;
    }
    // Ambiguous charges, an unavailable GAFF2 environment or a failure: the
    // Check itself reports those with the context needed to act on them.
    line.classList.add('hidden');
    if (_ligandPrepTimer) { GMXPoll.stop(_ligandPrepTimer); _ligandPrepTimer = null; }
  }

  function poll() {
    return GMXPoll.read('/api/ligand-prep/' + state.taskId)
      .then(render)
      .catch(function() { /* a missed poll is not worth surfacing */ });
  }

  _ligandPrepTimer = GMXPoll.start(poll, 2000, true);
}

function clearStepProgress(stepName) {
  var scope = stepName ? document.getElementById('panel-' + stepName) : document;
  if (!scope) return;
  scope.querySelectorAll('.step-progress, .step-feedback').forEach(function(element) {
    element.remove();
  });
  if (_checkNotice && (!stepName || _checkNotice.handle.stepName === stepName)) {
    _checkNotice = null;
    document.getElementById('compute-queue-status').classList.add('hidden');
  }
}

async function _doCheckStep(stepName, statusElId, btnId) {
  var statusEl = document.getElementById(statusElId);
  var btn = document.getElementById(btnId);
  if (!state.taskId) {
    if (statusEl) { statusEl.textContent = 'No task loaded'; statusEl.style.color = '#dc2626'; }
    return;
  }
  if (stepName === 'structure') {
    var protonationError = validateStructureProtonationReady();
    if (protonationError) {
      if (statusEl) { statusEl.textContent = '\u2717 ' + protonationError; statusEl.style.color = '#dc2626'; }
      return;
    }
  }
  // Guard against concurrent step execution
  if (_stepRunning) {
    if (statusEl) { statusEl.textContent = 'Please wait — another step is running'; statusEl.style.color = '#d97706'; }
    return;
  }
  try {
    if (statusEl) { statusEl.textContent = 'Running...'; statusEl.style.color = '#d97706'; }
    if (stepName === 'input' && !(isCoarseGrainedWorkflow() && !coarseGrainedIncludesProtein())) {
      var inputReport = document.getElementById('input-check-report');
      if (inputReport) inputReport.classList.add('hidden');
    }
    if (btn) btn.disabled = true;
    revokeStepPass(stepName);
    _stepRunning = true;
    _progressHandle = startStepProgress(stepName, btnId, statusElId);

    var inputRevision = _inputRevision;
    // Input step: first filter PDB by chain/molecule selections
    if (stepName === 'input' && !(isCoarseGrainedWorkflow() && !coarseGrainedIncludesProtein())) {
      var includedChains = [];
      for (var ch in _chainState) {
        if (_chainState[ch].included) includedChains.push(ch);
      }
      // Also include chains from small molecules (they may reside in
      // chains that have no protein residues, e.g. ligand in chain A
      // while protein is in chain B — without this the filter-pdb
      // endpoint deletes the ligand because its chain isn't in the list).
      var smols = (state.pdbInfo && (state.pdbInfo.selection_info || state.pdbInfo).small_molecules) || [];
      smols.forEach(function(m) {
        if (_smallMolState && _smallMolState[m.resname] && !_smallMolState[m.resname].included) return;
        if (typeof m.chain === 'string' && includedChains.indexOf(m.chain) < 0) includedChains.push(m.chain);
      });
      // Collect excluded molecule resnames: replacement water +
      // user-unchecked small molecules
      var excludedResn = ['HOH', 'SOL', 'WAT', 'TIP', 'TIP3', 'SPC', 'SPCE'];
      for (var _smr in _smallMolState) {
        if (_smallMolState.hasOwnProperty(_smr) && !_smallMolState[_smr].included) {
          if (excludedResn.indexOf(_smr) < 0) excludedResn.push(_smr);
        }
      }
      var smallMoleculeLabels = {};
      Object.keys(_smallMolState).forEach(function(key) {
        smallMoleculeLabels[key] = _smallMolState[key].name || key;
      });
      var filterResponse = await fetch('/api/filter-pdb/' + state.taskId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          include_chains: includedChains,
          exclude_resnames: excludedResn,
          small_molecule_labels: smallMoleculeLabels,
        }),
      });
      var filterResult = await filterResponse.json();
      if (!filterResponse.ok || filterResult.error) {
        throw new Error(filterResult.error || 'PDB filtering failed');
      }
    }

    var cfg = buildModuleConfig(stepName)[stepName] || {};
    var chemistryRevision = _structureRevision;
    var forceFieldRevision = _forceFieldRevision;
    var result = await _apiFetch('/api/step/' + state.taskId + '/' + stepName, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });
    if (result.status === 'ok') {
      if (stepName === 'input' && inputRevision !== _inputRevision) {
        throw new Error('Input selection changed while checking. Run Check Upload again.');
      }
      if (stepName === 'structure' && chemistryRevision !== _structureRevision) {
        throw new Error('Chemistry changed while checking. Run Check Structure again.');
      }
      if (stepName === 'forcefield' && forceFieldRevision !== _forceFieldRevision) {
        throw new Error('Force-field settings changed while checking. Confirm the current settings again.');
      }
      if (stepName === 'input') {
        _progressHandle.awaitingPreview = true;
        GMXPoll.stop(_progressHandle.poll);
        _progressHandle.poll = null;
      } else finishStepProgress(_progressHandle, true);
      if (stepName === 'cg_environment') renderMembraneCompositionWarnings((result.metrics || {}).membrane_composition_warnings || []);
      _checkedSteps.add(stepName);
      // Snapshot the config that was just validated (prevents drift when user
      // changes DOM values between "Check" and "Build")
      _checkedConfig = _checkedConfig || {};
      _checkedConfig[stepName] = JSON.parse(JSON.stringify(cfg));
      if (statusEl) {
        statusEl.textContent = stepName === 'cg_system'
          ? '✓ Quality gates passed; inspect and confirm below'
          : '';
        statusEl.style.color = '#059669';
      }
      var cgConfirmation = stepName === 'cg_system'
        ? document.getElementById('cg-confirm-system') : null;
      if (cgConfirmation) {
        cgConfirmation.checked = false;
        cgConfirmation.disabled = true;
      }
      updateNextButtonState();
      updateStepNavHighlight();

      // Refresh viewer to confirm WYSIWYG — what you see IS what was saved
      if (stepName === 'input') {
        renderInputReadiness(null);
        var inputWarnings = ((result.metrics || {}).input_validation || {}).warnings || [];
        var inputModificationReport = (result.metrics || {}).input_modifications || {};
        var standardizedSequences = (result.metrics || {}).input_sequences || [];
        _savedFragmentConfig = cfg;
        if ((result.metrics || {}).input_summary) showCheckedInputSummary(result.metrics.input_summary);
        if (standardizedSequences.length && state.pdbInfo) {
          state.pdbInfo.sequences = standardizedSequences;
          loadProcResidues();
        }
        setInputModificationReport(inputModificationReport);
        renderInputCheckReport(
          (result.metrics || {}).input_repair || {}, null, inputModificationReport,
          (result.metrics || {}).input_nucleic_acids || []
        );
        appendFeedbackMessages(document.getElementById('input-check-report'), inputWarnings);
        var reconstruction = (result.metrics || {}).input_reconstruction || {};
        var notes = [];
        if ((reconstruction.splits || []).length) notes.push({message:
          'Created ' + reconstruction.splits.length + ' additional chain fragment(s). ' +
          'No bonds cross the missing segments; review each fragment’s termini.'});
        if (reconstruction.renumber_residues) notes.push({message:
          'Residues are numbered from 1 within each chain. Original identifiers are retained in the saved mapping.'});
        appendFeedbackMessages(document.getElementById('input-check-report'), notes);
        if (!isCoarseGrainedWorkflow() || coarseGrainedIncludesProtein()) {
          _progressHandle.element.querySelector('.step-progress-phase').textContent =
            'Loading the checked structure preview…';
          await refreshCheckedInputPreview();
        }
        finishStepProgress(_progressHandle, true);
      } else if (stepName === 'forcefield') {
        applyForceFieldResolution(result.metrics || {});
      } else if (stepName === 'structure') {
        renderModificationGeometryReport(
          (result.metrics || {}).modification_geometry || []
        );
      } else if (stepName === 'solvation') {
        _solvChecked = true;
        applySolvationMetrics(result.metrics || {});
        updateNextButtonState();
        renderSolvationViewer();
      } else if (stepName.indexOf('cg_') === 0) {
        if (stepName === 'cg_mapping' && statusEl) {
          statusEl.textContent = 'Downstream box dimensions are derived automatically from this mapped extent.';
        }
        var cgViewerRendered;
        if (stepName === 'cg_orientation') {
          _cgOrientedPdbContent = await _loadStepViewerPdb('cg_orientation');
          cgViewerRendered = await renderCoarseGrainedViewer(
            stepName, _cgOrientedPdbContent, false
          );
          var cgPreviewStatus = document.getElementById('cg-orient-preview-status');
          if (cgPreviewStatus) {
            cgPreviewStatus.textContent = 'Showing saved Step 4 coordinates';
            cgPreviewStatus.style.color = '#059669';
          }
        } else {
          cgViewerRendered = await renderCoarseGrainedViewer(stepName);
        }
        if (cgConfirmation) {
          cgConfirmation.disabled = cgViewerRendered !== true;
          if (!cgViewerRendered && statusEl) {
            statusEl.textContent = '✓ Quality gates passed; WebGL viewer unavailable, so confirmation is disabled in this browser';
            statusEl.style.color = '#d97706';
          }
        }
      }
    } else {
      revokeStepPass(result.input_check_required ? 'input' : stepName);
      finishStepProgress(_progressHandle, false, result.error || 'Failed', result.input_validation);
      if (stepName === 'structure') renderModificationGeometryReport([]);
    }
  } catch(e) {
    revokeStepPass(stepName);
    finishStepProgress(_progressHandle, false, e.message || 'Network error');

    if (stepName === 'structure') renderModificationGeometryReport([]);
  } finally {
    _stepRunning = false;
    if (btn) btn.disabled = stepName === 'forcefield' && !_ffCompatibilityValid;
    // Safety net: any path that left the bar running closes it here, so a
    // Check can never leave a bar polling forever.
    finishUnfinishedStepProgress(_progressHandle);
    _progressHandle = null;
  }
}

function initCheckButtons() {
  // Upload step
  var ib = document.getElementById('input-check-btn');
  if (ib) ib.addEventListener('click', function() { _doCheckStep('input', 'input-check-status', 'input-check-btn'); });

  // Force field step
  var fb = document.getElementById('forcefield-check-btn');
  if (fb) fb.addEventListener('click', function() { _doCheckStep('forcefield', 'forcefield-check-status', 'forcefield-check-btn'); });

  // Structure step
  var sb = document.getElementById('structure-check-btn');
  if (sb) sb.addEventListener('click', function() { _doCheckStep('structure', 'structure-check-status', 'structure-check-btn'); });

  // Orient step — already handled in initOrientationStep
  // Membrane step — already handled in initLipidMixing (check-composition-btn)
  // Solvation step — already handled (solv-check-btn)
  // Ions step — already handled
}

/** Apply unified cartoon+stick style to a 3Dmol viewer. */
function _applyUnifiedStyle(viewer, pdbContent, onlyChains) {
  GMXStyle.apply(viewer, GMXStyle.pdbAtoms(viewer, pdbContent), {
    onlyChains: onlyChains,
    moleculeState: _smallMolState,
  });
}

function initOrientationStep() {
  // ====================================================================
  // Shared viewer — single 3Dmol instance on #orient-viewer.
  // Both Auto and Manual tabs use the same viewer.
  // ====================================================================
  async function _createOrientViewer() {
    if (window._orientViewer) return;
    var el = document.getElementById('orient-viewer');
    if (!el) return;
    await GMXAssets.viewer();
    if (window._orientViewer) return;
    window._orientViewer = $3Dmol.createViewer(el, {
      backgroundColor: window.gmxViewerBackground(), antialias: true,
    });
    window._orientViewer.setBackgroundColor(window.gmxViewerBackground());
    window._orientViewer.setSlab(-10000, 10000);
  }
  window._createOrientViewer = _createOrientViewer;

  // ====================================================================
  // Redraw backend-generated Step 4 coordinates against a fixed membrane.
  // Auto, Manual preview, Check and Step 5 now share the exact same module;
  // the browser never substitutes a moving plane for a protein transform.
  // ====================================================================
  function _redrawOrientViewer() {
    if (!window._orientViewer) return;
    var v = window._orientViewer;
    var pdb = _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);
    if (!pdb) return;
    v.removeAllModels();
    v.addModel(pdb, 'pdb');
    _applyUnifiedStyle(v, pdb);
    var halfThick = (_dominantLipidDHH || 3.8) * 0.5;
    drawMembranePlane(v, 0.0, halfThick, 0.0, 0.0);
    v.setStyle({elem: 'X'}, {sphere: {radius: 1.2, color: '0x6b7280', opacity: 0.55}});
    v.render();
    v.setSlab(-10000, 10000);
  }
  window._redrawOrientViewer = _redrawOrientViewer;

  window._loadOrientationCheckpointPreview = async function() {
    var requestId = ++_orientPreviewRequestId;
    var pdb = await _loadStepViewerPdb('orient');
    if (!pdb || requestId !== _orientPreviewRequestId) return false;
    _orientedPdbContent = pdb;
    await _createOrientViewer();
    _redrawOrientViewer();
    var previewStatus = document.getElementById('orient-preview-status');
    if (previewStatus) {
      previewStatus.textContent = 'Showing saved Step 4 coordinates';
      previewStatus.style.color = '#059669';
    }
    return true;
  };

  async function _runManualOrientationPreview() {
    var previewStatus = document.getElementById('orient-preview-status');
    if (previewStatus) {
      previewStatus.textContent = 'Updating exact preview...';
      previewStatus.style.color = '#d97706';
    }
    try {
      var data = await requestOrientationPreview({
        method: 'manual',
        z_offset: _orientZOffset,
        tilt: _orientTilt,
        phi: _orientPhi,
      });
      if (data && previewStatus) {
        previewStatus.textContent = 'Preview matches the coordinates Check will save';
        previewStatus.style.color = '#059669';
      }
    } catch (error) {
      if (previewStatus) {
        previewStatus.textContent = error.message || 'Preview failed';
        previewStatus.style.color = '#dc2626';
      }
    }
  }

  window._scheduleManualOrientationPreview = function(delayMs) {
    if (_orientPreviewTimer !== null) clearTimeout(_orientPreviewTimer);
    ++_orientPreviewRequestId;  // invalidate any in-flight auto/manual response
    _orientPreviewTimer = setTimeout(_runManualOrientationPreview, delayMs == null ? 180 : delayMs);
  };

  // ====================================================================
  // Slider sync
  // ====================================================================
  var _zSlider2 = document.getElementById('orient-manual-z');
  var _tiltSlider2 = document.getElementById('orient-manual-tilt');
  var _phiSlider2 = document.getElementById('orient-manual-phi');
  var _zNum2 = document.getElementById('orient-manual-z-num');
  var _tiltNum2 = document.getElementById('orient-manual-tilt-num');
  var _phiNum2 = document.getElementById('orient-manual-phi-num');

  function _syncOrientSliders(source, userChanged) {
    if (source !== 'num') { if (_zNum2) _zNum2.value = parseFloat(_zSlider2.value).toFixed(2); }
    if (source !== 'num') { if (_tiltNum2) _tiltNum2.value = _tiltSlider2.value; }
    if (source !== 'num') { if (_phiNum2) _phiNum2.value = _phiSlider2.value; }
    if (source !== 'slider') { if (_zSlider2) _zSlider2.value = _zNum2.value; }
    if (source !== 'slider') { if (_tiltSlider2) _tiltSlider2.value = _tiltNum2.value; }
    if (source !== 'slider') { if (_phiSlider2) _phiSlider2.value = _phiNum2.value; }
    _orientZOffset = parseFloat(_zSlider2 ? _zSlider2.value : 0);
    _orientTilt = parseFloat(_tiltSlider2 ? _tiltSlider2.value : 0);
    _orientPhi = parseFloat(_phiSlider2 ? _phiSlider2.value : 0);
    var zV = document.getElementById('orient-manual-z-val');
    var tV = document.getElementById('orient-manual-tilt-val');
    var pV = document.getElementById('orient-manual-phi-val');
    if (zV) zV.textContent = _orientZOffset.toFixed(2);
    if (tV) tV.textContent = _orientTilt;
    if (pV) pV.textContent = _orientPhi;
    if (userChanged !== false) {
      invalidateOrientationCheck();
      window._scheduleManualOrientationPreview(180);
    }
  }

  if (_zSlider2) _zSlider2.addEventListener('input', function() { _syncOrientSliders('slider', true); });
  if (_tiltSlider2) _tiltSlider2.addEventListener('input', function() { _syncOrientSliders('slider', true); });
  if (_phiSlider2) _phiSlider2.addEventListener('input', function() { _syncOrientSliders('slider', true); });
  if (_zNum2) _zNum2.addEventListener('input', function() { _syncOrientSliders('num', true); });
  if (_tiltNum2) _tiltNum2.addEventListener('input', function() { _syncOrientSliders('num', true); });
  if (_phiNum2) _phiNum2.addEventListener('input', function() { _syncOrientSliders('num', true); });

  // ====================================================================
  // Tab switching — show/hide controls only, viewer stays
  // ====================================================================
  document.querySelectorAll('.orient-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      var oldMode = _orientMode;
      _orientMode = tab.dataset.method;
      _setOrientationModeUI();
      if (oldMode !== _orientMode) invalidateOrientationCheck();
      if (_orientMode === 'ppm') {
        runPPMAuto();
      } else {
        // Force viewer resize after showing the manual div
        if (window._orientViewer) { window._orientViewer.resize(); }
        if (_zSlider2) { _zSlider2.value = _orientZOffset; if (_zNum2) _zNum2.value = parseFloat(_orientZOffset).toFixed(2); }
        if (_tiltSlider2) { _tiltSlider2.value = _orientTilt; if (_tiltNum2) _tiltNum2.value = _orientTilt; }
        if (_phiSlider2) { _phiSlider2.value = _orientPhi; if (_phiNum2) _phiNum2.value = _orientPhi; }
        _syncOrientSliders('slider', false);
        window._scheduleManualOrientationPreview(0);
      }
    });
  });

  // Algorithm selector
  var algoSelect = document.getElementById('orient-algorithm');
  if (algoSelect) {
    algoSelect.addEventListener('change', function() {
      _orientAlgorithm = algoSelect.value;
      document.getElementById('orient-algo-desc').textContent = _ALGO_DESC[_orientAlgorithm] || '';
      invalidateOrientationCheck();
      runPPMAuto();
    });
    document.getElementById('orient-algo-desc').textContent = _ALGO_DESC[_orientAlgorithm] || '';
  }

  // Re-run button
  var rerunBtn = document.getElementById('orient-rerun-btn');
  if (rerunBtn) rerunBtn.addEventListener('click', runPPMAuto);

  // ====================================================================
  // Check button
  // ====================================================================
  var checkBtn = document.getElementById('orient-check-btn');
  if (checkBtn) {
    checkBtn.addEventListener('click', async function() {
      var statusEl = document.getElementById('orient-check-status');
      if (!state.taskId) {
        if (statusEl) { statusEl.textContent = 'No task loaded'; statusEl.style.color = '#dc2626'; }
        return;
      }
      if (_stepRunning) {
        if (statusEl) { statusEl.textContent = 'Step already running — please wait'; statusEl.style.color = '#d97706'; }
        return;
      }
      revokeStepPass('orient');
      _stepRunning = true;
      var progress = startStepProgress('orient', 'orient-check-btn', 'orient-check-status');
      try {
        checkBtn.disabled = true;
        var orientConfig = buildModuleConfig().orient || { method: 'ppm' };
        var resp = await fetch('/api/step/' + state.taskId + '/orient', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: orientConfig }),
        });
        var result = await resp.json();
        if (result.status === 'ok') {
          finishStepProgress(progress, true);
          _checkedSteps.add('orient');
          _checkedConfig = _checkedConfig || {};
          _checkedConfig.orient = orientConfig;
          var quality = (result.metrics && result.metrics.orientation_quality) || {};
          var warnings = Array.isArray(quality.warnings) ? quality.warnings : [];
          renderOrientationQuality(quality);
          if (statusEl) {
            statusEl.textContent = warnings.length
              ? '⚠ Checked with ' + warnings.length + ' scientific warning(s)'
              : '';
            statusEl.style.color = warnings.length ? '#d97706' : '#059669';
          }
          updateNextButtonState();
          // The saved checkpoint is authoritative. Reload it even though the
          // preview used the same backend module, so Check visibly confirms
          // the exact coordinates that Step 5 will consume.
          await _createOrientViewer();
          var loaded = await window._loadOrientationCheckpointPreview();
          if (!loaded && statusEl) {
            statusEl.textContent = '⚠ Check saved, but the saved viewer could not be reloaded';
            statusEl.style.color = '#d97706';
          }
        } else {
          finishStepProgress(progress, false, result.error || 'Failed');

        }
      } catch(e) {
        revokeStepPass('orient');
        finishStepProgress(progress, false, e.message || 'Network error');
      } finally {
        _stepRunning = false;
        checkBtn.disabled = false;
        finishUnfinishedStepProgress(progress);
      }
    });
  }
}

async function runPPMAuto() {
  if (!state.pdbInfo || !state.taskId) {
    document.getElementById('orient-result-z').textContent = 'No PDB loaded';
    document.getElementById('orient-result-tilt').textContent = '—';
    return;
  }
  try {
    document.getElementById('orient-result-z').textContent = 'Computing...';
    var previewStatus = document.getElementById('orient-preview-status');
    if (previewStatus) {
      previewStatus.textContent = 'Computing exact backend preview...';
      previewStatus.style.color = '#d97706';
    }
    var algoSelect = document.getElementById('orient-algorithm');
    var algo = (algoSelect && algoSelect.value) || _orientAlgorithm || 'ppm';
    _orientAlgorithm = algo;
    var data = await requestOrientationPreview({method: algo});
    if (!data) return;
    var orientation = data.orientation || {};
    _orientZOffset = Number(orientation.z_offset || 0);
    _orientTilt = Number(orientation.tilt || 0);
    _orientPhi = Number(orientation.phi || 0);
    document.getElementById('orient-result-z').textContent = _orientZOffset.toFixed(2) + ' nm  (' + algo.toUpperCase() + ')';
    var quality = data.orientation_quality || {};
    var measuredTilt = Number(quality.tm_bundle_tilt_degrees);
    document.getElementById('orient-result-tilt').textContent =
      Number.isFinite(measuredTilt) ? measuredTilt.toFixed(1) + '°' : _orientTilt.toFixed(1) + '°';

    // Manual mode starts from the displayed automatic values, while its
    // backend preview treats them as explicit replacement values.
    var values = [
      ['orient-manual-z', _orientZOffset], ['orient-manual-z-num', _orientZOffset.toFixed(2)],
      ['orient-manual-tilt', _orientTilt], ['orient-manual-tilt-num', _orientTilt],
      ['orient-manual-phi', _orientPhi], ['orient-manual-phi-num', _orientPhi],
    ];
    values.forEach(function(entry) {
      var element = document.getElementById(entry[0]);
      if (element) element.value = entry[1];
    });
    var zV = document.getElementById('orient-manual-z-val');
    var tV = document.getElementById('orient-manual-tilt-val');
    var pV = document.getElementById('orient-manual-phi-val');
    if (zV) zV.textContent = _orientZOffset.toFixed(2);
    if (tV) tV.textContent = _orientTilt;
    if (pV) pV.textContent = _orientPhi;

    if (window._createOrientViewer) await window._createOrientViewer();
    if (window._redrawOrientViewer) window._redrawOrientViewer();
    if (previewStatus) {
      previewStatus.textContent = 'Preview matches the coordinates Check will save';
      previewStatus.style.color = '#059669';
    }
  } catch (e) {
    document.getElementById('orient-result-z').textContent = 'Error: ' + e.message;
    document.getElementById('orient-result-tilt').textContent = '—';
    var previewStatus2 = document.getElementById('orient-preview-status');
    if (previewStatus2) {
      previewStatus2.textContent = e.message || 'Preview failed';
      previewStatus2.style.color = '#dc2626';
    }
  }
}

function drawMembranePlane(viewer, zOffset, halfThickness, tiltDeg, phiDeg) {
  var thick = halfThickness;
  if (!thick || thick <= 0) {
    var lipidDHH = _dominantLipidDHH || 3.8;
    thick = lipidDHH * 0.5;
  }
  var padEl4 = document.getElementById('membrane-pad');
  var boxPad = 2.0;
  if (padEl4) { var pv4 = parseFloat(padEl4.value); if (!isNaN(pv4)) boxPad = pv4; }
  var pdbContent = _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content) || '';
  var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  var lines = pdbContent.split('\n');
  for (var li = 0; li < lines.length; li++) {
    var l = lines[li];
    if (l.indexOf('ATOM') === 0 || l.indexOf('HETATM') === 0) {
      var px = parseFloat(l.substring(30, 38)) / 10.0;
      var py = parseFloat(l.substring(38, 46)) / 10.0;
      if (!isNaN(px) && !isNaN(py)) {
        if (px < xMin) xMin = px; if (px > xMax) xMax = px;
        if (py < yMin) yMin = py; if (py > yMax) yMax = py;
      }
    }
  }
  var protXY = isFinite(xMin) ? Math.max(xMax - xMin, yMax - yMin) : 3.0;
  var boxXY = Math.max(protXY + 2 * boxPad, 4.0);
  var halfNm = boxXY / 2.0;
  var step = 1.5;

  var atoms = '';
  var serial = 1;
  for (var x = -halfNm; x <= halfNm + 0.001; x += step) {
    for (var y = -halfNm; y <= halfNm + 0.001; y += step) {
      for (var s = 0; s < 2; s++) {
        var cz = (s === 0) ? -thick : thick;
        var rx = x, ry = y, rz = cz;
        if (tiltDeg > 0.1) {
          var t = tiltDeg * Math.PI / 180;
          var p = (phiDeg || 0) * Math.PI / 180;
          var ax = -Math.sin(p), ay = Math.cos(p);
          var ct = Math.cos(t), st = Math.sin(t);
          var dot = ax*x + ay*y;
          var cx = ay*cz, cy = -ax*cz, cz2 = ax*y - ay*x;
          rx = x*ct + cx*st + ax*dot*(1-ct);
          ry = y*ct + cy*st + ay*dot*(1-ct);
          rz = cz*ct + cz2*st + zOffset;
        } else {
          rz = cz + zOffset;
        }
        atoms += 'ATOM  ' + String(serial).padStart(5) + '  X   MEM X' + String(serial).padStart(4) + '    ' + (rx*10).toFixed(1).padStart(8) + (ry*10).toFixed(1).padStart(8) + (rz*10).toFixed(1).padStart(8) + '  1.00  0.00          X  \n';
        serial++;
      }
    }
  }
  viewer.addModel(atoms, 'pdb');
}

function updateOrientSliderRanges() {
  if (!state.pdbInfo || !state.pdbInfo.box_nm) return;
  const box = state.pdbInfo.box_nm;
  const zSlider = document.getElementById('orient-manual-z');
  if (zSlider) {
    zSlider.min = -box[2];
    zSlider.max = box[2];
  }
}


// Compute protein extent from PDB content (returns {x, y, z} in nm)
function _proteinExtent(pdbContent) {
  var xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity,zMin=Infinity,zMax=-Infinity;
  var lines = (pdbContent||'').split('\n');
  for (var li=0;li<lines.length;li++) {
    var l=lines[li];
    if (l.indexOf('ATOM')===0||l.indexOf('HETATM')===0) {
      var px=parseFloat(l.substring(30,38))/10.0;
      var py=parseFloat(l.substring(38,46))/10.0;
      var pz=parseFloat(l.substring(46,54))/10.0;
      if (!isNaN(px)&&!isNaN(py)&&!isNaN(pz)) {
        if(px<xMin)xMin=px;if(px>xMax)xMax=px;
        if(py<yMin)yMin=py;if(py>yMax)yMax=py;
        if(pz<zMin)zMin=pz;if(pz>zMax)zMax=pz;
      }
    }
  }
  if (!isFinite(xMin)) return {x:3,y:3,z:6};
  return {x:xMax-xMin, y:yMax-yMin, z:zMax-zMin};
}

// Apply tilt + z_offset to a point (same transform as drawMembranePlane)
function _transformPoint(x, y, z, zOffset, tiltDeg, phiDeg) {
  if (tiltDeg > 0.1) {
    var t = tiltDeg * Math.PI / 180;
    var p = (phiDeg || 0) * Math.PI / 180;
    var ax = -Math.sin(p), ay = Math.cos(p);
    var ct = Math.cos(t), st = Math.sin(t);
    var dot = ax*x + ay*y;
    var cx = ay*z, cy = -ax*z, cz2 = ax*y - ay*x;
    return {
      x: x*ct + cx*st + ax*dot*(1-ct),
      y: y*ct + cy*st + ay*dot*(1-ct),
      z: z*ct + cz2*st + zOffset
    };
  }
  return { x: x, y: y, z: z + zOffset };
}

// Draw an axis-aligned orthogonal box from an explicit lower corner.
function drawOrthogonalBox(viewer, boxA_A, boxB_A, boxC_A, origin) {
  origin = origin || {x: 0, y: 0, z: 0};
  var x0 = origin.x, y0 = origin.y, z0 = origin.z;
  var corners = [
    [x0, y0, z0], [x0 + boxA_A, y0, z0],
    [x0 + boxA_A, y0 + boxB_A, z0], [x0, y0 + boxB_A, z0],
    [x0, y0, z0 + boxC_A], [x0 + boxA_A, y0, z0 + boxC_A],
    [x0 + boxA_A, y0 + boxB_A, z0 + boxC_A],
    [x0, y0 + boxB_A, z0 + boxC_A]
  ];
  var edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  edges.forEach(function(e) {
    viewer.addCylinder({
      start: {x: corners[e[0]][0], y: corners[e[0]][1], z: corners[e[0]][2]},
      end:   {x: corners[e[1]][0], y: corners[e[1]][1], z: corners[e[1]][2]},
      radius: 0.20, color: '0x6b7280', opacity: 0.70, fromCap: 0, toCap: 0,
    });
  });
}

// Draw box wireframe with tilt + z_offset applied to corners
function _drawTiltedBox(viewer, halfXY_A, halfZ_A, zOffset, tiltDeg, phiDeg) {
  var raw = [
    [-halfXY_A, -halfXY_A, -halfZ_A], [ halfXY_A, -halfXY_A, -halfZ_A],
    [ halfXY_A,  halfXY_A, -halfZ_A], [-halfXY_A,  halfXY_A, -halfZ_A],
    [-halfXY_A, -halfXY_A,  halfZ_A], [ halfXY_A, -halfXY_A,  halfZ_A],
    [ halfXY_A,  halfXY_A,  halfZ_A], [-halfXY_A,  halfXY_A,  halfZ_A]
  ];
  var corners = raw.map(function(c) {
    return _transformPoint(c[0], c[1], c[2], zOffset, tiltDeg, phiDeg);
  });
  var edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  edges.forEach(function(e) {
    viewer.addCylinder({
      start: {x: corners[e[0]].x, y: corners[e[0]].y, z: corners[e[0]].z},
      end:   {x: corners[e[1]].x, y: corners[e[1]].y, z: corners[e[1]].z},
      radius: 0.20, color: '0x6b7280', opacity: 0.70, fromCap: 0, toCap: 0,
    });
  });
}

// ===================================================================
// Structure Processing — Protonation + Termini + Modifications
// Implementation lives in app_parts/structure_processing.js.
// ===================================================================


// ===================================================================
// Solvation Check
// ===================================================================

// Reset solvation check when parameters change
function resetSolvCheck() {
  if (!pureMembraneIncludesSolvent()) {
    _solvChecked = true;
    _checkedSteps.add('solvation');
    return;
  }
  _solvChecked = false;
  _checkedSteps.delete('solvation');
  if (_checkedConfig) delete _checkedConfig.solvation;
  _waterVolume = 0;
  _waterCount = 0;
  var el = document.getElementById("solv-check-status");
  if (el) { el.textContent = ""; }
  var res = document.getElementById("solv-result");
  if (res) res.classList.add("hidden");
  updateNextButtonState();
  updateStepNavHighlight();
  if (state.wizardSteps[state.currentStepIdx] === 'solvation') {
    if (_solvationViewerTimer !== null) clearTimeout(_solvationViewerTimer);
    _solvationViewerTimer = setTimeout(function() {
      renderSolvationViewer();
    }, 180);
  }
}

function applySolvationMetrics(metrics) {
  var dims = metrics.box_dimensions_nm || [];
  var solvent = (metrics.components || []).find(function(c) { return c.kind === 'SOLVENT'; }) || {};
  var nWater = Number(solvent.n_molecules || (metrics.solvation && metrics.solvation.n_molecules) || 0);
  var waterModel = metrics.water_model || solvent.water_model || 'unknown';
  var boxVol = dims.length === 3 ? dims[0] * dims[1] * dims[2] : 0;
  _waterCount = nWater;
  _waterVolume = nWater * 0.0299;
  var dimsEl = document.getElementById('solv-box-dims');
  if (dimsEl && dims.length === 3) {
    dimsEl.textContent = dims.map(function(v) { return Number(v).toFixed(1); }).join(' × ') +
      ' = ' + boxVol.toFixed(0) + ' nm³';
  }
  var volEl = document.getElementById('solv-box-vol');
  if (volEl) volEl.textContent = boxVol.toFixed(0);
  var countEl = document.getElementById('solv-n-water');
  if (countEl) countEl.textContent = nWater.toLocaleString();
  var modelEl = document.getElementById('solv-water-model');
  if (modelEl) modelEl.textContent = String(waterModel).toUpperCase();
  var resultEl = document.getElementById('solv-result');
  if (resultEl) resultEl.classList.remove('hidden');
}

// ---- 3D viewer: solvation box ----
var _solvationViewer = null;
var _solvationViewerTimer = null;

function _pdbCoordinateBoundsAngstrom(pdbContent) {
  var bounds = {
    minX: Infinity, maxX: -Infinity,
    minY: Infinity, maxY: -Infinity,
    minZ: Infinity, maxZ: -Infinity,
  };
  pdbContent.split('\n').forEach(function(line) {
    if (line.indexOf('ATOM') !== 0 && line.indexOf('HETATM') !== 0) return;
    var x = parseFloat(line.substring(30, 38));
    var y = parseFloat(line.substring(38, 46));
    var z = parseFloat(line.substring(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
    bounds.minZ = Math.min(bounds.minZ, z);
    bounds.maxZ = Math.max(bounds.maxZ, z);
  });
  return Number.isFinite(bounds.minX) ? bounds : null;
}

function _pdbMembraneZBoundsAngstrom(pdbContent) {
  var selectedLipids = (_mixUpper || []).concat(_mixLower || []);
  var residueNames = new Set(selectedLipids.map(function(item) {
    return String(item.name || '').trim().toUpperCase().substring(0, 3);
  }).filter(Boolean));
  if (!residueNames.size) return null;
  var minZ = Infinity;
  var maxZ = -Infinity;
  pdbContent.split('\n').forEach(function(line) {
    if (line.indexOf('ATOM') !== 0 && line.indexOf('HETATM') !== 0) return;
    var residueName = line.substring(17, 20).trim().toUpperCase();
    if (!residueNames.has(residueName)) return;
    var z = parseFloat(line.substring(46, 54));
    if (!Number.isFinite(z)) return;
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  });
  return Number.isFinite(minZ) && maxZ > minZ ? {minZ: minZ, maxZ: maxZ} : null;
}

async function renderSolvationViewer() {
  if (!state.taskId) return;
  const checked = _checkedSteps.has('solvation') && pureMembraneIncludesSolvent();
  const source = checked ? 'solvation' :
    (state.taskType?.pipeline === 'solvator' ? 'structure' : 'membrane');
  const data = await GMXViewer.render('solvation-3d-viewer', source);
  if (data) {
    const label = document.getElementById('solvation-viewer-label');
    if (label) {
      label.textContent = source === 'structure'
        ? 'Checked solute; compute solvent to create the box with the requested padding.'
        : (checked ? 'Checked solvent box: ' : 'Checked membrane box; compute solvent to add padding: ') +
          data.box_nm.map(row=>Math.hypot(...row).toFixed(2)).join(' × ') + ' nm';
    }
  }
}


// ===================================================================
// Simulation Parameters live in app_parts/simulation.js.
// It is loaded immediately after this classic script to preserve shared globals.
// ===================================================================


// ===================================================================
// System Verification Viewer and membrane composition
// Implementation lives in app_parts/system_verification.js.
// ===================================================================
function setupUpload() {
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('pdb-file');
  const browseBtn = document.getElementById('browse-btn');
  if (!zone) return;

  browseBtn.addEventListener('click', () => input.click());
  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleFile(input.files[0]);
  });
}

async function handleFile(file) {
  if (state.uploadRunning) return;
  var fnameLower = file.name.toLowerCase();
  var structurePattern = /\.(?:pdb|ent|cif|mmcif)(?:\.gz)?$/i;
  if (!structurePattern.test(fnameLower)) {
    alert('Accepted structure formats: .pdb, .ent, .cif, .mmcif, and gzip-compressed variants.');
    return;
  }

  state.uploadRunning = true;
  state.pdbInfo = null;
  invalidateInputCheckpoint();
  document.getElementById('upload-info').classList.add('hidden');
  document.getElementById('validation-info').classList.add('hidden');
  const uploadControls = ['browse-btn', 'pdb-file', 'input-check-btn'].map(id => document.getElementById(id)).filter(Boolean);
  const disabledBeforeUpload = uploadControls.map(el => el.disabled);
  uploadControls.forEach(el => { el.disabled = true; });
  document.getElementById('upload-zone').setAttribute('aria-busy', 'true');
  // Show upload progress outside the initially hidden structure summary.
  var uploadSection = document.getElementById('upload-progress');
  var uploadBar = document.getElementById('upload-progress-fill');
  var uploadText = document.getElementById('upload-progress-text');
  if (!uploadSection) {
    // Create progress elements if they don't exist
    var infoSection = document.getElementById('upload-info');
    if (infoSection) {
      uploadSection = document.createElement('div');
      uploadSection.id = 'upload-progress';
      uploadSection.innerHTML = '<div class="progress-bar" style="height:6px;background:#e2e8f0;border-radius:3px;margin-top:8px">' +
        '<div id="upload-progress-fill" style="height:100%;width:0;background:#3b82f6;border-radius:3px;transition:width 0.2s"></div></div>' +
        '<div id="upload-progress-text" style="font-size:12px;color:#64748b;margin-top:4px"></div>';
      infoSection.before(uploadSection);
      uploadBar = document.getElementById('upload-progress-fill');
      uploadText = document.getElementById('upload-progress-text');
    }
  }
  if (uploadSection) uploadSection.style.display = 'block';
  if (uploadBar) { uploadBar.style.width = '0%'; uploadBar.style.background = '#3b82f6'; }
  if (uploadText) uploadText.textContent = 'Uploading ' + (file.size/1024/1024).toFixed(1) + ' MB...';

  const formData = new FormData();
  formData.append('file', file);
  formData.append('task_type', (state.taskType && state.taskType.id) || 'membrane-bilayer');
  if (isCoarseGrainedWorkflow() && state.taskId) {
    formData.append('task_id', state.taskId);
  }

  // Use XHR for upload progress tracking
  try {
    const initialResponse = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload-pdb');
      xhr.upload.onprogress = function(e) {
        if (e.lengthComputable && uploadBar) {
          var pct = Math.round(e.loaded / e.total * 100);
          uploadBar.style.width = pct + '%';
          if (uploadText) uploadText.textContent = 'Uploading... ' + pct + '%';
        }
      };
      xhr.onload = function() {
        if (!xhr.status) { reject(new Error('Upload connection closed')); return; }
        try {
          resolve(new Response(xhr.responseText, {status:xhr.status, headers:{
            'Content-Type':xhr.getResponseHeader('Content-Type') || 'application/json',
            'X-GMXBUILDER-Operation':xhr.getResponseHeader('X-GMXBUILDER-Operation') || ''
          }}));
        } catch (error) { reject(error); }
      };
      xhr.onerror = function() { reject(new Error('Upload failed')); };
      xhr.send(formData);
    });
    if (initialResponse.status === 202 && uploadText) uploadText.textContent = 'Upload received; processing structure…';
    const finalResponse = await window.resolveManagedResponse(initialResponse, {requestUrl:'/api/upload-pdb'});
    let data;
    try {
      data = await finalResponse.json();
    } catch (_error) {
      if (finalResponse.ok) throw new Error('The server returned an invalid upload response. Please retry.');
      data = {error:'Upload failed (HTTP ' + finalResponse.status + '). Please retry.'};
    }
    if (!finalResponse.ok || data.error || (data.validation_errors && data.validation_errors.length)) {
      if (uploadBar) uploadBar.style.background = '#dc2626';
      if (uploadText) uploadText.textContent = 'Upload failed';
      if (data.validation_errors && data.validation_errors.length) {
        showValidationErrors(data.validation_errors, data.validation_warnings || []);
      } else {
        const details = (data.parse_issues || []).map(issue => {
          const identity = issue.chain !== undefined
            ? `${issue.chain || '?'}:${issue.resid}${issue.insertion_code || ''}`
            : (issue.residue || []).join(':');
          return `${identity} ${issue.resname || ''} ${issue.atom || ''}: ` +
            `${issue.code === 'invalid_occupancy' ? 'invalid occupancy' : 'unknown element'} ` +
            `${issue.value ?? issue.element ?? ''}`;
        });
        if ((data.parse_issue_count || 0) > details.length)
          details.push(`Showing ${details.length} of ${data.parse_issue_count} affected atoms.`);
        showValidationErrors([data.error || 'Upload failed (HTTP ' + finalResponse.status + ')'].concat(details), []);
      }
      return;
    }
    if (!data.task_id || !Number.isFinite(data.num_atoms) || data.num_atoms <= 0 || !data.pdb_content) throw new Error('The server did not return a complete structure. Resume the Task ID or retry the upload.');
    if (uploadBar) { uploadBar.style.background = '#22c55e'; uploadBar.style.width = '100%'; }
    if (uploadText) uploadText.textContent = 'Upload complete — ' + (file.size/1024/1024).toFixed(1) + ' MB';
    setTimeout(function() { if (uploadSection) uploadSection.style.display = 'none'; }, 2000);

    state.pdbInfo = data;
    state.taskId = data.task_id;
    resetLigandPHState();
    _charmmCompatSmiles = {};
    _charmmCompatResearch = false;
    _restoredLigandBackend = null;
    setTimeout(loadTaskCustomLipids, 0);
    syncTaskRoute(state.currentStepIdx, true);
    _smallMolState = {};
    window._smallMolState = _smallMolState;
    var tidEl = document.getElementById('task-id-display');
    var tidBox = document.getElementById('header-task-id');
    if (tidEl && data.task_id) { tidEl.textContent = data.task_id; }
    if (tidBox && data.task_id) { tidBox.classList.remove('hidden'); }
    _protonationComputed = false;  // reset for new PDB
    window._setIonsChecked ? window._setIonsChecked(false) : null;
    updateNextButtonState();
    showUploadInfo(data);
  } catch (err) {
    if (uploadBar) uploadBar.style.background = '#dc2626';
    if (uploadText) uploadText.textContent = 'Upload failed';
    showValidationErrors([err.message], []);
  } finally {
    state.uploadRunning = false;
    uploadControls.forEach((el, idx) => { el.disabled = disabledBeforeUpload[idx]; });
    document.getElementById('upload-zone').removeAttribute('aria-busy');
    updateNextButtonState();
  }
}

function showValidationErrors(errors, warnings) {
  const valInfo = document.getElementById('validation-info');
  const errDiv = document.getElementById('validation-errors');
  const warnDiv = document.getElementById('validation-warnings');
  if (!valInfo) return;

  valInfo.classList.remove('hidden');
  [errDiv, warnDiv].forEach(function(el) { if (el) { el.replaceChildren(); el.classList.add('hidden'); } });
  errors = uniqueFeedbackMessages(errors);
  var errorKeys = new Set(errors);
  warnings = uniqueFeedbackMessages(warnings).filter(function(warning) { return !errorKeys.has(warning); });

  if (errDiv && errors.length) {
    errDiv.classList.remove('hidden');
    errDiv.innerHTML = '<h4 style="color:#dc2626;">&#10007; Cannot Read PDB File</h4><ul></ul>';
    const errList = errDiv.querySelector('ul');
    errors.forEach(function(e) { const li = document.createElement('li'); li.textContent = e; errList.appendChild(li); });
  }

  if (warnDiv && warnings.length) {
    warnDiv.classList.remove('hidden');
    warnDiv.innerHTML = '<h4 style="color:#d97706;">&#9888; Warnings</h4><ul></ul>';
    const warnList = warnDiv.querySelector('ul');
    warnings.forEach(function(w) { const li = document.createElement('li'); li.textContent = w; warnList.appendChild(li); });
  }

  // Hide upload info
  const uploadInfo = document.getElementById('upload-info');
  if (uploadInfo) uploadInfo.classList.add('hidden');
  const uploadZone = document.getElementById('upload-zone');
  if (uploadZone) uploadZone.classList.remove('hidden');
}

function showUploadInfo(info) {
  ['input-allow-incomplete', 'input-renumber-residues'].forEach(function(id) {
    var control = document.getElementById(id);
    if (control) {
      control.checked = false;
      control.addEventListener('change', resetInputFragmentEditor);
    }
  });
  renderInputReadiness(info.input_validation || (info.input_status?.readiness === 'not_checked'
    ? {warnings: ['File read successfully. Workflow preparation is not yet checked. Run Check Upload before continuing.']}
    : null));
  const zone = document.getElementById('upload-zone');
  const box = document.getElementById('upload-info');
  if (!zone || !box) return;
  zone.classList.add('hidden');

  // Hide validation errors from previous attempts
  const valInfo = document.getElementById('validation-info');
  if (valInfo) valInfo.classList.add('hidden');

  box.classList.remove('hidden');
  document.getElementById('info-filename').textContent = info.filename;
  document.getElementById('info-atoms').textContent = info.num_atoms;
  document.getElementById('info-chains').textContent = (info.chains || []).join(', ') || '—';
  document.getElementById('info-box').textContent =
    ((info.box_nm || []).map(v => v.toFixed(1) + ' nm').join(' × ') || '—') +
    (info.cell_info?.box_source === 'estimated' ? ' (estimated display envelope)' : '');

  // Show non-blocking validation warnings
  if (info.validation_warnings && info.validation_warnings.length) {
    const warnDiv = document.getElementById('validation-warnings');
    if (warnDiv) {
      warnDiv.classList.remove('hidden');
      warnDiv.innerHTML = '<h4 style="color:#d97706;">&#9888; Warnings</h4><ul></ul>';
      const warnList = warnDiv.querySelector('ul');
      info.validation_warnings.forEach(function(w) {
        const li = document.createElement('li');
        li.textContent = w;
        warnList.appendChild(li);
      });
    }
  }

  _fragmentState = {}; _savedFragmentConfig = {}; _inputFragmentEditing = false;
  setCheckedInputDisplay(false);
  document.getElementById('edit-input-selection').addEventListener('click', resetInputFragmentEditor);

  // Render chain sequences with checkboxes and rename inputs
  var selectionInfo = info.selection_info || info;
  renderChainSequences(selectionInfo.sequences || [], selectionInfo.chains || [], selectionInfo.chain_mapping || {});
  if (!info.selection_info) info.selection_info = {
    num_atoms: info.num_atoms, box_nm: info.box_nm,
    sequences: info.sequences || [], chains: info.chains || [], pdb_content: info.pdb_content,
    small_molecules: info.small_molecules || [], chain_mapping: info.chain_mapping || {},
  };

  // Render small molecules
  renderSmallMolecules(selectionInfo.small_molecules || []);

  // Render 3D viewer
  renderPDBViewer(info.pdb_content || '');

  // Load residues into structure processing
  loadProcResidues();

  // Input step requires explicit Check — do NOT auto-complete
  // Next button stays disabled until user clicks "Check Upload"
  updateNextButtonState();
  updateOrientSliderRanges();
}

function renderInputSummaryFields(info) {
  document.getElementById('info-atoms').textContent = info.num_atoms ?? '—';
  document.getElementById('info-chains').textContent = (info.chains || []).join(', ') || '—';
  const sequences = info.sequences || [];
  if (sequences.some(chain => chain.fragment_count > 1)) {
    const sources = new Set(sequences.map(chain => chain.source_chain || chain.chain_id));
    document.getElementById('info-chains').textContent +=
      ` — ${sources.size} source polymer(s), ${sequences.length} coordinate fragments`;
  }
  document.getElementById('info-box').textContent =
    ((info.box_nm || []).map(v => v.toFixed(1) + ' nm').join(' × ') || '—') +
    (info.cell_info?.box_source === 'estimated' ? ' (estimated display envelope)' : '');
}

function restoreInputSelection(selection) {
  Object.keys(_chainState).forEach(chain => {
    _chainState[chain].included = !Array.isArray(selection.include_chains) || selection.include_chains.includes(chain);
  });
  document.querySelectorAll('#chain-sequences input[data-chain]').forEach(control => {
    control.checked = _chainState[control.dataset.chain].included;
  });
  Object.keys(_smallMolState).forEach(name => {
    _smallMolState[name].included = !(selection.exclude_resnames || []).includes(name);
  });
  document.querySelectorAll('#small-molecules input[data-smres]').forEach(control => {
    control.checked = _smallMolState[control.dataset.smres].included;
  });
}

function setCheckedInputDisplay(checked) {
  document.getElementById('input-selection-details').classList.toggle('hidden', checked);
  document.getElementById('checked-input-details').classList.toggle('hidden', !checked);
  document.getElementById('edit-input-selection').classList.toggle('hidden', !checked);
  document.getElementById('input-details-hint').textContent = checked
    ? '— select fragments to include; double-click a name to rename. Changes require Check Upload.'
    : '— select which to include, double-click name to rename';
}

function showCheckedInputSummary(summary) {
  // Display saved results without replacing the source IDs used by the next Check.
  if (state.pdbInfo) {
    state.pdbInfo.checked_sequences = summary.sequences || [];
    state.pdbInfo.fragment_choices = summary.fragment_choices || summary.sequences || [];
    ['num_atoms', 'chains', 'box_nm', 'small_molecules'].forEach(key => {
      if (summary[key] !== undefined) state.pdbInfo[key] = summary[key];
    });
  }
  renderInputSummaryFields(summary);
  const choices = summary.fragment_choices || summary.sequences || [];
  const saved = _savedFragmentConfig;
  _fragmentState = {};
  choices.forEach(chain => {
    const key = chain.fragment_key || `${chain.source_chain || chain.chain_id}:${chain.fragment_index || 1}`;
    chain.fragment_key = key;
    _fragmentState[key] = {included: !(saved.exclude_fragments || []).includes(key),
      name: (saved.fragment_names || {})[key] || chain.chain_id, original: chain.chain_id};
  });
  _inputFragmentEditing = true;
  renderChainSequences(choices, summary.chains || [], null, true);
  // Keep excluded molecules available for re-inclusion.
  renderSmallMolecules(state.pdbInfo?.selection_info?.small_molecules || summary.small_molecules || [], true);
  setCheckedInputDisplay(true);
}

// ---- Chain inclusion / rename state ----
let _fragmentState = {};
let _savedFragmentConfig = {};
let _inputFragmentEditing = false;
let _inputRevision = 0;

function resetInputFragmentEditor() {
  if (state.pdbInfo?.selection_info) renderSmallMolecules(state.pdbInfo.selection_info.small_molecules || []);
  _fragmentState = {}; _savedFragmentConfig = {}; _inputFragmentEditing = false;
  invalidateInputCheckpoint();
}

let _chainState = {};  // { chain_id: { included: true, name: original_id } }
// Small-molecule visibility state (keyed by resname — small molecules are
// not protein chains; they have their own checkboxes in the Small Molecules section)
let _smallMolState = {};  // { resname: { included: true, name: original_name } }
// Safety: ensure _smallMolState is always initialised
window._smallMolState = _smallMolState;

function invalidateInputCheckpoint() {
  _inputRevision++;
  if (state.pdbInfo) delete state.pdbInfo.checked_sequences;
  if (!_inputFragmentEditing && state.pdbInfo && state.pdbInfo.selection_info) {
    Object.assign(state.pdbInfo, state.pdbInfo.selection_info);
    renderInputSummaryFields(state.pdbInfo);
  }
  setCheckedInputDisplay(_inputFragmentEditing);
  if (_inputFragmentEditing) document.getElementById('input-details-hint').textContent =
    '— selection changed; run Check Upload to update the saved structure and viewer';
  renderInputReadiness(null);
  document.getElementById('input-check-report').classList.add('hidden');
  clearStepProgress('input');
  var inputIndex = state.wizardSteps.indexOf('input');
  if (inputIndex < 0) inputIndex = 0;
  for (var i = inputIndex; i < state.wizardSteps.length; i++) {
    state.completedSteps.delete(i);
    _checkedSteps.delete(state.wizardSteps[i]);
    if (_checkedConfig) delete _checkedConfig[state.wizardSteps[i]];
  }
  window._ffCompatibility = null;
  var status = document.getElementById('input-check-status');
  if (status) {
    status.textContent = 'Selection changed — run Check Upload again';
    status.style.color = '#d97706';
  }
  updateNextButtonState();
  updateStepNavHighlight();
  if (!_inputFragmentEditing) redrawPDBViewerWithChainFilter();
}

function commitSmallMoleculeLabel(resname, candidate) {
  var current = (_smallMolState[resname] && _smallMolState[resname].name) || resname;
  var label = String(candidate || '').trim();
  if (!label) {
    alert('Small-molecule display name must not be empty.');
    return current;
  }
  if (label.length > 64 || /[\u0000-\u001f\u007f]/.test(label)) {
    alert('Small-molecule display name must be 1–64 printable characters.');
    return current;
  }
  var duplicate = Object.keys(_smallMolState).some(function(key) {
    return key !== resname &&
      String(_smallMolState[key].name || key).toLocaleLowerCase() === label.toLocaleLowerCase();
  });
  if (duplicate) {
    alert('Each different small molecule must have a unique display name.');
    return current;
  }
  if (!_smallMolState[resname]) _smallMolState[resname] = {included: true, name: resname};
  if (_smallMolState[resname].name !== label) {
    _smallMolState[resname].name = label;
    window._smallMolState = _smallMolState;
    invalidateInputCheckpoint();
  }
  return label;
}

function wireRenameTrigger(trigger, beginRename) {
  trigger.addEventListener('click', function(event) {
    // Keyboard and assistive-technology activation generates a click with no
    // mouse detail. Mouse users retain the existing double-click gesture.
    if (event.detail === 0) beginRename();
  });
  trigger.addEventListener('dblclick', function(event) {
    event.preventDefault();
    beginRename();
  });
}

function createSmallMoleculeRenameTrigger(resname, label) {
  var trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'smallmol-name';
  trigger.dataset.smres = resname;
  trigger.textContent = label;
  trigger.title = 'Double-click or press Enter to rename';
  trigger.setAttribute('aria-label', 'Rename small molecule ' + label);
  wireRenameTrigger(trigger, function() { beginSmallMoleculeRename(trigger); });
  return trigger;
}

function beginSmallMoleculeRename(span) {
  var resname = span.dataset.smres;
  var input = document.createElement('input');
  input.type = 'text';
  input.value = (_smallMolState[resname] && _smallMolState[resname].name) || resname;
  input.className = 'smallmol-rename-input';
  input.style.width = '120px';
  span.replaceWith(input);
  input.focus();
  input.select();
  var finished = false;
  function finish(cancelled, restoreFocus) {
    if (finished) return;
    finished = true;
    var label = cancelled
      ? ((_smallMolState[resname] && _smallMolState[resname].name) || resname)
      : commitSmallMoleculeLabel(resname, input.value);
    var replacement = createSmallMoleculeRenameTrigger(resname, label);
    input.replaceWith(replacement);
    if (restoreFocus) replacement.focus();
  }
  input.addEventListener('blur', function() { finish(false, false); });
  input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') { event.preventDefault(); finish(false, true); }
    if (event.key === 'Escape') { event.preventDefault(); finish(true, true); }
  });
}

function renderSmallMolecules(molecules, checkedView = false) {
  const container = document.getElementById(checkedView ? 'checked-small-molecules' : 'small-molecules');
  const header = document.getElementById(checkedView ? 'checked-smallmol-header' : 'smallmol-header');
  if (!container || !header) return;

  if (!molecules.length) {
    container.innerHTML = '';
    header.style.display = 'none';
    return;
  }

  header.style.display = '';
  container.innerHTML = '';

  // Group by resname
  const grouped = {};
  molecules.forEach(m => {
    const key = m.resname;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(m);
  });

  // Preserve existing visibility state; only auto-include newly discovered
  // molecules.  Previously unchecked molecules stay unchecked.
  var _newSmState = {};
  molecules.forEach(function(m) {
    var _existing = _smallMolState[m.resname];
    _newSmState[m.resname] = _existing || { included: true, name: m.resname };
  });
  if (!checkedView) {
    _smallMolState = _newSmState;
    window._smallMolState = _smallMolState;
  }

  const cards = Object.entries(grouped).map(([resname, instances]) => {
    const totalAtoms = instances.reduce((s, m) => s + m.atom_count, 0);
    const formula = instances[0].formula || '?';
    const chains = [...new Set(instances.map(m => m.chain))].sort().join(', ');
    const checked = (_smallMolState[resname] && _smallMolState[resname].included) ? 'checked' : '';

    const card = document.createElement('div');
    card.className = 'smallmol-card';
    card.innerHTML =
      '<div class="smallmol-header">' +
        '<label class="smallmol-check">' +
          '<input type="checkbox" ' + checked + ' data-smres="' + escapeHtml(resname) + '">' +
          '<b>' + escapeHtml(resname) + '</b>' +
        '</label>' +
        '<button type="button" class="smallmol-name" data-smres="' + escapeHtml(resname) + '" title="Double-click or press Enter to rename" aria-label="Rename small molecule ' + escapeHtml(_smallMolState[resname] ? _smallMolState[resname].name : resname) + '">' + escapeHtml(_smallMolState[resname] ? _smallMolState[resname].name : resname) + '</button>' +
      '</div>' +
      '<div class="smallmol-info">' +
        'Formula: ' + escapeHtml(formula) + ' | Copies: ' + instances.length +
        ' | Atoms: ' + totalAtoms + ' | Chain: ' + escapeHtml(chains) +
      '</div>';
    return card;
  });

  cards.forEach(c => container.appendChild(c));

  // Wire up small molecule checkbox changes → redraw viewer
  container.querySelectorAll('.smallmol-check input[type="checkbox"]').forEach(function(cb) {
    cb.addEventListener('change', function() {
      var smres = this.dataset.smres;
      if (_smallMolState[smres]) {
        _smallMolState[smres].included = this.checked;
      }
      invalidateInputCheckpoint();
      if (!_inputFragmentEditing) redrawPDBViewerWithChainFilter();
    });
  });

  // Keep the existing mouse gesture and add native keyboard activation.
  container.querySelectorAll('.smallmol-name').forEach(span => {
    wireRenameTrigger(span, function() { beginSmallMoleculeRename(span); });
  });
}

// ===================================================================
// Chain Sequence Rendering
// ===================================================================

function classifyResidue(resname) {
  const r = resname.trim().toUpperCase();
  if (['DA','DC','DG','DT','DA5','DC5','DG5','DT5','DA3','DC3','DG3','DT3',
       'A','C','G','U','RA','RC','RG','RU','RA5','RC5','RG5','RU5',
       'RA3','RC3','RG3','RU3'].includes(r)) return 'nucleic';
  if (window.GMX && window.GMX.PROTEIN_RESNAMES.has(r)) return 'protein';
  if (window.GMX && window.GMX.SOLVENT_RESNAMES.has(r)) return 'water';
  if (window.GMX && window.GMX.ION_RESNAMES.has(r)) return 'ion';
  if (window.GMX && window.GMX.LIPID_RESNAMES.has(r)) return 'lipid';
  // Fallback for when constants.js is not loaded:
  if (['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','ASH','GLH','CYX','HID','HIE','HIP','LYN','ACE','NME','MSE'].includes(r)) return 'protein';
  if (['HOH','SOL','WAT','TIP','TIP3','SPC','SPCE'].includes(r)) return 'water';
  if (['NA','CL','K','CA','ZN','MG'].includes(r)) return 'ion';
  return 'other';
}

function createChainRenameTrigger(chainId, label, fragmentKey) {
  var trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'chain-rename';
  trigger.dataset.chain = chainId;
  if (fragmentKey) trigger.dataset.fragmentKey = fragmentKey;
  trigger.textContent = label;
  trigger.title = 'Double-click or press Enter to rename';
  trigger.setAttribute('aria-label', 'Rename chain ' + (chainId || 'unnamed'));
  wireRenameTrigger(trigger, function() { beginChainRename(trigger); });
  return trigger;
}

function beginChainRename(trigger) {
  var chainId = trigger.dataset.chain;
  var fragmentKey = trigger.dataset.fragmentKey;
  var entries = fragmentKey ? _fragmentState : _chainState;
  var entryKey = fragmentKey || chainId;
  var original = trigger.textContent;
  var input = document.createElement('input');
  input.type = 'text';
  input.value = original;
  input.className = 'chain-rename-input';
  input.style.width = '40px';
  trigger.replaceWith(input);
  input.focus();
  input.select();
  var finished = false;
  function finish(cancelled, restoreFocus) {
    if (finished) return;
    finished = true;
    var label = cancelled ? original : (input.value.trim() || original);
    if (!cancelled && (!/^[A-Za-z0-9]$/.test(label) || Object.keys(entries).some(function(key) {
      return key !== entryKey && entries[key].name === label;
    }))) {
      input.setCustomValidity('Use one unique letter or digit for the chain ID.');
      input.reportValidity();
      finished = false;
      return;
    }
    if (entries[entryKey]) entries[entryKey].name = label;
    if (label !== original) invalidateInputCheckpoint();
    var replacement = createChainRenameTrigger(chainId, label, fragmentKey);
    input.replaceWith(replacement);
    if (restoreFocus) replacement.focus();
  }
  input.addEventListener('blur', function() { finish(false, false); });
  input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') { event.preventDefault(); finish(false, true); }
    if (event.key === 'Escape') { event.preventDefault(); finish(true, true); }
  });
}

function renderChainSequences(sequences, chains, mapping, checkedView = false) {
  const container = document.getElementById(checkedView ? 'checked-chain-sequences' : 'chain-sequences');
  if (!container) return;
  container.innerHTML = '';

  // Init chain state for new upload
  if (!checkedView) {
    _chainState = {};
    (chains || []).forEach(ch => { _chainState[ch] = { included: true, name: ch || 'Unnamed chain' }; });
  }

  if (!sequences.length) {
    container.innerHTML = '<p class="hint">No residue data detected.</p>';
    return;
  }

  const sourceGroups = new Map();
  sequences.forEach(chain => {
    const chId = chain.chain_id || '';
    const st = (checkedView ? _fragmentState[chain.fragment_key] : _chainState[chId]) || { included: true, name: chId };

    const card = document.createElement('div');
    card.className = 'chain-card';

    const header = document.createElement('div');
    header.className = 'chain-header';
    const isFragment = checkedView && chain.fragment_count > 1;
    let destination = container;
    if (isFragment) {
      if (!sourceGroups.has(chain.source_chain)) {
        const group = document.createElement('section');
        group.className = 'source-protein-group';
        const title = document.createElement('h4');
        title.textContent = `Source protein ${chain.source_chain} — ${chain.fragment_count} coordinate fragments`;
        const note = document.createElement('p');
        note.className = 'fragment-model-warning';
        note.textContent = 'These fragments belong to one source protein. Missing connections are not rebuilt. ' +
          'Independent topology ends form an approximate model; charged or capped ends do not restore the missing loop.';
        group.append(title, note);
        const parts = sequences.filter(row => row.source_chain === chain.source_chain);
        const gaps = document.createElement('p');
        gaps.className = 'hint';
        gaps.textContent = parts.slice(1).map((part, index) => {
          const left = parts[index].residues.slice(-1)[0], right = part.residues[0];
          return `Coordinate gap: ${left.resname} ${left.author_resid ?? left.resid}${left.insertion_code || ''} → ${right.resname} ${right.author_resid ?? right.resid}${right.insertion_code || ''} (deposited numbering)`;
        }).join('; ');
        group.appendChild(gaps);
        container.appendChild(group);
        sourceGroups.set(chain.source_chain, group);
      }
      destination = sourceGroups.get(chain.source_chain);
    }
    const chainLabel = chId ? `${isFragment ? 'Fragment' : 'Chain'} ${chId}` : 'Unnamed chain';
    header.innerHTML =
      `<label class="chain-check">` +
        `<input type="checkbox" data-chain="${escapeHtml(chId)}" ${st.included ? 'checked' : ''}>` +
        `<b>${escapeHtml(chainLabel)}</b>` +
      `</label>` +
      `<button type="button" class="chain-rename" data-chain="${escapeHtml(chId)}" title="Double-click or press Enter to rename" aria-label="Rename chain ${escapeHtml(chId || 'unnamed')}">${escapeHtml(st.name || chId)}</button>` +
      `<span class="chain-len">${chain.length} residues</span>`;
    if (checkedView) {
      header.querySelector('input').dataset.fragmentKey = chain.fragment_key;
      header.querySelector('button').dataset.fragmentKey = chain.fragment_key;
      if (chain.author_chain !== undefined) {
        const origin = document.createElement('span');
        origin.className = 'hint';
        origin.textContent = ` · Deposited chain: ${chain.author_chain || 'unnamed'}`;
        header.appendChild(origin);
      }
    }
    if (mapping) {
      var sourceChain = Object.keys(mapping).find(function(key) { return mapping[key] === chId; });
      if (sourceChain !== undefined && sourceChain !== chId) {
        var origin = document.createElement('span');
        origin.className = 'hint';
        origin.textContent = ' · Source chain: ' + (sourceChain || 'unnamed');
        header.appendChild(origin);
      }
    }
    card.appendChild(header);

    // Flex-wrap sequence display — no horizontal scrolling
    const residues = chain.residues || [];
    const GROUP_SIZE = 10;

    const seqWrap = document.createElement('div');
    seqWrap.className = 'seq-flex-wrap';

    const rowWrap = document.createElement('div');
    rowWrap.className = 'seq-flex-row';

    for (let g = 0; g < Math.ceil(residues.length / GROUP_SIZE); g++) {
      const groupStart = g * GROUP_SIZE;
      const group = document.createElement('div');
      group.className = 'seq-group';

      // Number label
      const numLabel = document.createElement('div');
      numLabel.className = 'seq-group-num';
      numLabel.textContent = groupStart < residues.length ? residues[groupStart].resid : '';
      group.appendChild(numLabel);

      // Residue tags
      const tagRow = document.createElement('div');
      tagRow.className = 'seq-group-tags';
      for (let r = 0; r < GROUP_SIZE; r++) {
        const idx = groupStart + r;
        const tag = document.createElement('span');
        tag.className = 'seq-tag';
        if (idx < residues.length) {
          const cls = residues[idx].is_nucleic
            ? 'nucleic'
            : classifyResidue(residues[idx].resname);
          tag.classList.add(cls);
          tag.textContent = residues[idx].resname;
          tag.title = residues[idx].resname + ' ' + residues[idx].resid;
          if (residues[idx].author_resid !== undefined) {
            tag.title += ` · Deposited: ${residues[idx].author_chain || 'unnamed'}:${residues[idx].author_resid}${residues[idx].insertion_code || ''}`;
          }
        }
        tagRow.appendChild(tag);
      }
      group.appendChild(tagRow);
      rowWrap.appendChild(group);
    }
    seqWrap.appendChild(rowWrap);
    card.appendChild(seqWrap);
    destination.appendChild(card);
  });

  // Wire up chain checkboxes to toggle 3D viewer visibility
  container.querySelectorAll('.chain-check input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', () => {
      const ch = cb.dataset.chain;
      const included = cb.checked;
      var entries = checkedView ? _fragmentState : _chainState;
      var key = checkedView ? cb.dataset.fragmentKey : ch;
      if (entries[key]) entries[key].included = included;
      invalidateInputCheckpoint();
      // Refresh the PDB viewer
      if (!_inputFragmentEditing) redrawPDBViewerWithChainFilter();
    });
  });

  // Keep the existing mouse gesture and add native keyboard activation.
  container.querySelectorAll('.chain-rename').forEach(span => {
    wireRenameTrigger(span, function() { beginChainRename(span); });
  });
}

// 3-letter → 1-letter conversion
const AA3TO1 = {
  ALA:'A',ARG:'R',ASN:'N',ASP:'D',CYS:'C',GLN:'Q',GLU:'E',GLY:'G',
  HIS:'H',ILE:'I',LEU:'L',LYS:'K',MET:'M',PHE:'F',PRO:'P',SER:'S',
  THR:'T',TRP:'W',TYR:'Y',VAL:'V',ASH:'D',GLH:'E',CYX:'C',HID:'H',
  HIE:'H',HIP:'H',LYN:'K',ACE:'X',NME:'X',MSE:'M',SEC:'U',PYL:'O',
  HOH:'w',SOL:'w',WAT:'w',NA:'+',CL:'-',K:'+',CA:'2',ZN:'2',MG:'2',
};

// ===================================================================
// 3Dmol.js Viewer
// ===================================================================

async function renderPDBViewer(pdbContent) {
  const viewerEl = document.getElementById('pdb-viewer');
  if (!viewerEl) return;

  // Clear any previous viewer
  viewerEl.innerHTML = '';

  if (!pdbContent) {
    viewerEl.innerHTML = '<p style="color:#888;text-align:center;padding-top:180px;">No structure data</p>';
    return;
  }

  try { await GMXAssets.viewer(); }
  catch (error) { viewerEl.textContent = error.message; return; }

  try {
    const viewer = $3Dmol.createViewer(viewerEl, {
      backgroundColor: window.gmxViewerBackground(),
      antialias: true,
    });
    viewer.setBackgroundColor(window.gmxViewerBackground()); viewer.setSlab(-10000, 10000);

    viewer.addModel(pdbContent, 'pdb');
    // Records what is loaded so a later visibility change can restyle rather
    // than reparse. See redrawPDBViewerWithChainFilter.
    viewer.__gmxLoadedPdb = pdbContent;
    _applyUnifiedStyle(viewer, pdbContent);

    viewer.zoomTo();
    viewer.render();
  viewer.setSlab(-10000, 10000);

    window._pdbViewer = viewer;

    // Add controls hint
    const hint = document.createElement('div');
    hint.style.cssText = 'position:absolute;bottom:8px;right:12px;color:#888;font-size:11px;pointer-events:none;';
    hint.textContent = '🖱 drag: rotate | scroll: zoom | right-drag: pan';
    viewerEl.style.position = 'relative';
    viewerEl.appendChild(hint);

  } catch (err) {
    console.error('3Dmol viewer error:', err);
    viewerEl.innerHTML = '';  // clear previous content
    const errP = document.createElement('p');
    errP.style.cssText = 'color:#c00;text-align:center;padding-top:180px;';
    errP.textContent = 'Viewer error: ' + (err.message || 'Unknown error');
    viewerEl.appendChild(errP);
  }
}

/** Re-render the PDB viewer respecting chain visibility selections. */
function redrawPDBViewerWithChainFilter() {
  var pdbContent = state.pdbInfo && state.pdbInfo.pdb_content;
  var viewer = window._pdbViewer;
  if (!pdbContent || !viewer) return;

  // Ticking a chain checkbox changes what is *shown*, not what is loaded, and
  // this runs on every tick. Re-parsing the structure and rebuilding all of
  // its geometry to hide one chain was the entire cost of the operation, so
  // the model is reloaded only when the coordinates themselves changed.
  var reloaded = viewer.__gmxLoadedPdb !== pdbContent;
  if (reloaded) {
    viewer.removeAllModels();
    viewer.addModel(pdbContent, 'pdb');
    viewer.__gmxLoadedPdb = pdbContent;
  } else {
    // Styles set for the previous selection would otherwise survive on atoms
    // the new selection does not mention.
    viewer.setStyle({}, {});
  }

  // Build set of included protein chains.
  // Small molecules in non-protein chains are handled independently
  // in _applyUnifiedStyle (they don't need to be in includedChains).
  var includedChains = new Set();
  for (var c in _chainState) {
    if (_chainState[c].included) includedChains.add(c);
  }

  if (_checkedSteps.has('input') && state.pdbInfo) {
    // The checkpoint already contains the selected and reconstructed fragments.
    (state.pdbInfo.sequences || []).forEach(function(chain) { includedChains.add(chain.chain_id); });
  }

  if (includedChains.size === 0) {
    viewer.setStyle({}, {cartoon: {hidden: true}, stick: {hidden: true}, sphere: {hidden: true}, line: {hidden: true}});
  } else {
    // Pass onlyChains so _applyUnifiedStyle only styles included chains
    _applyUnifiedStyle(viewer, pdbContent, includedChains);
  }
  // Framing follows the structure, not the selection: re-zooming would throw
  // away the user's camera every time they tick a box.
  if (reloaded) viewer.zoomTo();
  viewer.render();
  viewer.setSlab(-10000, 10000);
}

// ===================================================================
// Build
// ===================================================================

function setupRunButton() {
  const runBtn = document.getElementById('run-btn');
  if (runBtn) runBtn.addEventListener('click', runBuild);
}

function mergeCheckedModuleConfig(config) {
  if (config.structure && _checkedConfig && _checkedConfig.structure &&
      JSON.stringify(config.structure) !== JSON.stringify(_checkedConfig.structure)) {
    invalidateStructureChemistry();
    throw new Error('Chemistry differs from the checked structure. Run Check Structure again.');
  }

  if (_checkedConfig) {
    for (var key in _checkedConfig) {
      if (_checkedConfig.hasOwnProperty(key)) {
        config[key] = Object.assign({}, config[key] || {}, _checkedConfig[key]);
      }
    }
  }
  return config;
}

function inputReconstructionConfig() {
  var names = {};
  Object.keys(_chainState).forEach(function(chain) {
    var item = _chainState[chain];
    if (item.included && item.name !== chain && item.name !== 'Unnamed chain') names[chain] = item.name;
  });
  return {
    allow_incomplete_protein: Boolean(document.getElementById('input-allow-incomplete')?.checked),
    renumber_residues: Boolean(document.getElementById('input-renumber-residues')?.checked),
    chain_names: names,
    fragment_names: Object.fromEntries(Object.entries(_fragmentState)
      .map(([key, item]) => [key, item.name])),
    exclude_fragments: Object.keys(_fragmentState).filter(key => !_fragmentState[key].included),
  };
}

function buildModuleConfig(focusStep) {
  const config = {};
  const taskModules = state.taskType ? state.taskType.visible_modules : [];

  if (isCoarseGrainedWorkflow()) {
    var environment = coarseGrainedEnvironment();
    var includeProtein = coarseGrainedIncludesProtein();
    var wants = function(step) { return !focusStep || focusStep === step; };
    if (wants('input')) {
      config.input = {include_protein: includeProtein, environment: environment, ...inputReconstructionConfig()};
    }
    if (wants('cg_model')) {
      config.cg_model = {model: 'martini3', water_model: 'W'};
    }
    if (wants('cg_mapping')) {
      config.cg_mapping = {
        protein_model: document.getElementById('cg-protein-model')?.value || 'folded',
        secondary_structure: document.getElementById('cg-secondary')?.value || 'auto',
        secondary_structure_string: document.getElementById('cg-secondary-string')?.value || '',
        elastic: document.getElementById('cg-elastic')?.checked !== false,
        elastic_force: Number(document.getElementById('cg-elastic-force')?.value || 700),
        elastic_lower: Number(document.getElementById('cg-elastic-lower')?.value || 0.5),
        elastic_upper: Number(document.getElementById('cg-elastic-upper')?.value || 0.9),
      };
    }
    if (wants('cg_orientation') && environment === 'bilayer') {
      config.cg_orientation = {
        method: _cgOrientMode,
        half_thickness: Number(document.getElementById('cg-orientation-half-thickness')?.value || 1.4),
      };
      if (_cgOrientMode === 'manual') {
        config.cg_orientation.z_offset = _cgOrientZOffset;
        config.cg_orientation.tilt = _cgOrientTilt;
        config.cg_orientation.phi = _cgOrientPhi;
      }
    }
    if (wants('cg_environment')) {
      config.cg_environment = {
        environment: environment,
      };
      if (environment === 'bilayer') {
        config.cg_environment.n_lipids_per_leaflet = Number(document.getElementById('cg-n-lipids-per-leaflet')?.value || 150);
        config.cg_environment.upper_leaflet = cgCompositionFor('upper');
        config.cg_environment.lower_leaflet = _cgAsymmetric ? cgCompositionFor('lower') : cgCompositionFor('upper');
        config.cg_environment.asymmetric = _cgAsymmetric;
      }
    }
    var includeSolvent = document.getElementById('cg-include-solvent')?.checked !== false;
    var saltMolarity = Number(document.getElementById('cg-salt')?.value || 0);
    if (wants('cg_solvation')) {
      config.cg_solvation = {
        include_solvent: includeSolvent,
        padding_nm: Number(document.getElementById('cg-padding')?.value || (environment === 'bilayer' ? 2 : 1.5)),
      };
    }
    if (wants('cg_system')) {
      config.cg_system = {
        salt_molarity: saltMolarity,
        confirm_system: document.getElementById('cg-confirm-system')?.checked === true,
      };
    }
    if (wants('topology')) config.topology = {};
    if (wants('simparams')) {
      config.simparams = collectCoarseGrainedSimulationParams();
      config.execution = collectCoarseGrainedExecutionHardware();
    }
    if (wants('export')) config.export = {write_mdp: includeSolvent};
    return mergeCheckedModuleConfig(config);
  }

  // Input
  if (taskModules.includes('input')) {
    if (state.taskId) {
      config.input = { task_id: state.taskId, ...inputReconstructionConfig() };
    }
  }

  // Structure Processing
  if (taskModules.includes('structure')) {
    var skipProtonation = document.getElementById("proc-skip-protonation") ? document.getElementById("proc-skip-protonation").checked : false;
    config.structure = {
      protonation: skipProtonation ? [] : _procAssignments.filter(function(a) { return a.is_titratable; }).map(function(a) {
        return { index: a.index, target: procResidueTarget(a.index), original: a.original, prediction_source: a.prediction_source, assigned_name: a.assigned_name, charge: a.charge, force_field_lacks_state: Boolean(a.force_field_lacks_state), state_override: Boolean(a.state_override) };
      }),
      modifications: serializeStructureModifications(),
      input_modification_decisions: _procInputRemovals,
      crosslinks: serializeStructureCrosslinks(),
      termini: _procTermini,
      pH: _systemPH,
      skip_protonation: skipProtonation,
    };
  }

  // Orientation
  if (taskModules.includes('orient')) {
    if (_orientMode === 'ppm') {
      // Auto algorithms compute their pose on the backend. Sending the
      // displayed result back as an override would be a silently ignored input.
      config.orient = {
        method: _orientAlgorithm || 'ppm',
        half_thickness: orientationHydrophobicHalfThickness(),
      };
    } else {
      config.orient = {
        method: 'manual',
        z_offset: _orientZOffset,
        tilt: _orientTilt,
        phi: _orientPhi,
        half_thickness: orientationHydrophobicHalfThickness(),
      };
    }
  }

  // Membrane
  if (taskModules.includes('membrane')) {
    const nLipidsEl = document.getElementById('n-lipids-per-leaflet');
    const nLipids = nLipidsEl ? parseInt(nLipidsEl.value) : 150;
    if (!Number.isInteger(nLipids) || nLipids < 64 || nLipids > 5000) {
      throw new Error('Lipids per leaflet must be a whole number from 64 to 5000.');
    }
    config.membrane = {
      lipid_composition: {
        upper: _mixUpper.map(m => ({...m})),
        lower: _asymmetric ? _mixLower.map(m => ({...m})) : null,
      },
      n_lipids_per_leaflet: nLipids,
    };
  }

  // Solvation
  if (taskModules.includes('solvation') && pureMembraneIncludesSolvent()) {
    if (state.taskType && state.taskType.pipeline === 'liquid') {
      var bx = (function(){var v=parseFloat(document.getElementById('box-padding')?.value);return isNaN(v)?5.0:v;})();
      config.solvation = {
        water_model: document.getElementById('ff-water-model')?.value || 'tip3p',
        box_size: [bx, bx, bx],
        box_padding: 0.0,
      };
    } else {
      config.solvation = {
        box_padding: (function(){var v=parseFloat(document.getElementById('box-padding')?.value);return isNaN(v)?1.5:v;})(),
        overlap_scale: parseFloat(document.getElementById('overlap-scale')?.value) || 0.8,
      };
    }
  }


// ===================================================================
  // Ions
  if (taskModules.includes('ions') && pureMembraneIncludesSolvent()) {
    config.ions = {
      cations: window._getIonCations ? window._getIonCations() : (console.warn('ions.js not loaded - falling back to default cations ["NA"]'), ["NA"]),
      anions: window._getIonAnions ? window._getIonAnions() : (console.warn('ions.js not loaded - falling back to default anions ["CL"]'), ["CL"]),
      concentration: window._getIonConcs ? window._getIonConcs() : (console.warn("ions.js not loaded - falling back to default concentration 0.15M NaCl"), {"NA":0.15,"CL":0.15}),
      neutralize: document.getElementById("ion-neutralize")?.checked !== false,
      neutralize_cation: document.getElementById("ion-neutralize-cation")?.value || "NA",
      neutralize_anion: document.getElementById("ion-neutralize-anion")?.value || "CL",
      ion_method: document.getElementById("ion-method")?.value || "random",
      exclusion_radius: parseFloat(document.getElementById("ion-exclusion")?.value) || 0.35,
    };
  }

  // Force field selection (early step — saves to metadata)
  if (taskModules.includes('forcefield')) {
    if (!Number.isFinite(_systemPH) || _systemPH < 1 || _systemPH > 13) {
      throw new Error('Enter a solution pH between 1.0 and 13.0 before checking the force field.');
    }
    var isPureMembrane = state.taskType && state.taskType.pipeline === 'pure_membrane';
    var isSolution = state.taskType && state.taskType.pipeline === 'solvator';
    config.forcefield = {
      name: document.getElementById('ff-protein')?.value || 'amber14sb',
      lipid_ff: isSolution ? 'none' : (document.getElementById('ff-lipid')?.value || 'none'),
      ligand_ff: isPureMembrane ? 'none' : (document.getElementById('ff-ligand')?.value || 'none'),
      ligand_charges: isPureMembrane ? {} : collectLigandCharges(),
      ligand_pH: _systemPH,
      cgenff_parameters: isPureMembrane ? {} : collectCGenFFParameters(),
      charmm_compat_smiles: isPureMembrane ? {} : collectCharmmCompatSmiles(),
      charmm_compat_mol2: isPureMembrane ? {} : collectCharmmCompatMol2(),
      charmm_compat_allow_research: !isPureMembrane &&
        document.getElementById('ff-ligand')?.value === 'charmm_compat' && _charmmCompatResearch,
      water_model: document.getElementById('ff-water-model')?.value || 'tip3p',
      lipid_names: isSolution ? [] : currentMembraneLipidNames(),
      system_name: document.getElementById('system-name')?.value || 'membrane_system',
    };
  }

  // Topology assignment (late step — reads from metadata, config is pass-through)
  if (taskModules.includes('topology')) {
    config.topology = {};
  }

  // Forward sim params to MDP generation
  // Collect per-stage simulation parameters
  if (taskModules.includes('forcefield') || taskModules.includes('topology')) {
    config.simparams = collectSimulationParams();
    config.execution = collectExecutionHardware();

    config.export = {
      write_mdp: pureMembraneIncludesSolvent(),
      mdp_params: {},
    };
  }

  // Merge checked/validated config snapshots — checked values always
  // take precedence over current DOM values (prevents drift between
  // "Check" and "Build").
  return mergeCheckedModuleConfig(config);
}

function _showBuildResult(result) {
  const resultSection = document.getElementById('result-section');
  finishBuildProgress(true);
  resultSection.classList.remove('hidden');

  const details = document.getElementById('result-details');
  details.innerHTML = '';
  details.classList.remove('input-check-report', 'error');
  document.getElementById('result-heading').textContent = 'Build complete';
  resultSection.dataset.state = 'done';
  document.getElementById('download-link').classList.remove('hidden');

  // ---- Verification warnings ----
  var verifyWarnings = [];
  var verifyInfo = null;
  (result.log || []).forEach(function(l) {
    if (l.indexOf('verification error') >= 0 || l.indexOf('FAILED') >= 0 || l.indexOf('mismatch') >= 0) {
      verifyWarnings.push(l);
    }
  });
  if (verifyWarnings.length > 0) {
    var warnDiv = document.createElement('div');
    warnDiv.style.cssText = 'background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:12px 16px;margin-bottom:16px';
    warnDiv.innerHTML = '<strong style="color:#d97706;">⚠ System Verification Warning</strong>' +
      '<p style="color:#92400e;margin:4px 0 0 0;font-size:13px">' +
      'The built system may differ from the 3D viewer preview. Review the verification metrics below.</p>';
    details.appendChild(warnDiv);
  }

  const table = document.createElement('table');
  const thead = document.createElement('tr');
  ['Component','Atoms','Molecules','Kind'].forEach(function(h) { const th=document.createElement('th'); th.textContent=h; thead.appendChild(th); });
  table.appendChild(thead);
  (result.components || []).forEach(function(c) {
    const tr = document.createElement('tr');
    var molStr = c.n_molecules ? c.n_molecules.toLocaleString() : (c.kind === 'SOLVENT' ? '—' : '—');
    var atomsStr = (c.atoms != null) ? c.atoms.toLocaleString() : '—';
    [c.name || '?', atomsStr, molStr, c.kind || '?'].forEach(function(v) {
      const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  const trTotal = document.createElement('tr');
  var totalStr = (result.num_atoms != null) ? String(result.num_atoms.toLocaleString()) : '?';
  [['Total','strong'], [totalStr,'strong'], ['',''], ['','']].forEach(function(p) {
    const td=document.createElement('td');
    if (p[1]==='strong') { const s=document.createElement('strong'); s.textContent=p[0]; td.appendChild(s); }
    else td.textContent=p[0];
    trTotal.appendChild(td);
  });
  table.appendChild(trTotal);

  details.appendChild(table);
  // The complete log has one retained, expandable home; no second result log.
  var logContent = document.getElementById('build-log-content');
  var lines = Array.from(logContent.children).map(function(el) { return el.textContent; });
  logContent.replaceChildren();
  uniqueFeedbackMessages(lines.concat(result.log || [])).forEach(function(line) {
    var item = document.createElement('div'); item.textContent = line; logContent.appendChild(item);
  });
  var logPanel = document.getElementById('build-log-panel');
  logPanel.classList.toggle('hidden', !logContent.children.length);
  logPanel.open = false;
  document.getElementById('build-log').style.display = 'block';
  document.getElementById('compute-queue-status').classList.add('hidden');

  const dlLink = document.getElementById('download-link');
  dlLink.href = result.download_url || '/api/task/' + (result.task_id || state.taskId) + '/download';
  dlLink.textContent = 'Download ZIP';
  const runBtn = document.getElementById('run-btn');
  if (runBtn) { runBtn.disabled = false; runBtn.textContent = '▶ Build System'; }
  state.buildRunning = false;
}

function showBuildFailure(message, inputCheckRequired) {
  document.getElementById('progress-section').classList.remove('hidden');
  finishBuildProgress(false);
  document.getElementById('result-section').classList.remove('hidden');
  document.getElementById('result-section').dataset.state = 'error';
  document.getElementById('result-heading').textContent = 'Build failed';
  var details = document.getElementById('result-details');
  details.replaceChildren();
  details.classList.add('input-check-report', 'error');
  appendFeedbackMessages(details, [message || 'The build could not finish.']);
  if (inputCheckRequired) {
    // A page left open across an upgrade can still hold obsolete Check passes.
    // Keep user settings, but revoke navigation and final-review authorization.
    revokeStepPass('input');
    if (window.invalidateFinalReview) window.invalidateFinalReview();
    var returnButton = document.createElement('button');
    returnButton.type = 'button';
    returnButton.textContent = 'Return to Check Upload';
    returnButton.addEventListener('click', function() {
      var inputIndex = state.wizardSteps.indexOf('input');
      if (inputIndex < 0) return;
      goToWizardStep(inputIndex);
      renderInputReadiness(null, false, message);
    });
    details.appendChild(returnButton);
  }
  document.getElementById('download-link').classList.add('hidden');
  document.getElementById('build-log-panel').open = false;
  document.getElementById('compute-queue-status').classList.add('hidden');
  var button = document.getElementById('run-btn');
  button.disabled = false;
  button.textContent = '▶ Build System';
  state.buildRunning = false;
}

function formatQueueWait(seconds) {
  var value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return Math.ceil(value) + " seconds";
  if (value < 3600) return Math.ceil(value / 60) + " minutes";
  return (value / 3600).toFixed(1) + " hours";
}

// A Check can submit several managed operations and then reload its viewer.
// Keep one notice alive for that whole interaction, not each individual ticket.
var _checkNotice = null;

function beginCheckNotice(handle) {
  _checkNotice = {handle:handle, taskId:state.taskId, queue:{
    task_id:state.taskId, status:'running', message:'Starting this check…'
  }};
  var notice = document.getElementById('compute-queue-status');
  if (notice && handle.element.nextElementSibling !== notice) handle.element.after(notice);
  showComputeQueueStatus(_checkNotice.queue);
  updateNextButtonState();
}

function currentCheckNotice() {
  var panel = document.querySelector('.panel.active');
  return _checkNotice && _checkNotice.taskId === state.taskId &&
    panel && panel.contains(_checkNotice.handle.element) ? _checkNotice : null;
}

function checkOwnsQueue(check, queueState) {
  if (queueState.task_id && queueState.task_id !== check.taskId) return false;
  var path = queueState.request_url || '';
  return !path || path === '/api/step/' + check.taskId + '/' + check.handle.stepName ||
    (check.handle.stepName === 'input' && path === '/api/filter-pdb/' + check.taskId);
}

function syncCheckNotice() {
  var check = currentCheckNotice();
  if (!check) return;
  var panel = document.querySelector('.panel.active');
  if (!panel || !panel.contains(check.handle.element)) return;
  updateComputeQueueStatus(check.queue);
}

function updateComputeQueueStatus(queueState) {
  var notice = document.getElementById("compute-queue-status");
  if (!notice || !queueState) return;
  var check = currentCheckNotice();
  if (check) {
    if (!checkOwnsQueue(check, queueState)) return;
    check.queue = queueState;
    if (check.handle.settled) {
      notice.classList.add('hidden');
      _checkNotice = null;
      return;
    }
    // Never expose a ticket's terminal state while the Check still has work.
    queueState = Object.assign({}, queueState);
    if (!['queued', 'running'].includes(queueState.status)) {
      queueState.status = 'running';
      queueState.message = 'Continuing this check…';
    }
    notice.classList.remove('hidden');
  } else if (['failed', 'cancelled'].includes(queueState.status) && operationHasFeedbackOwner(queueState)) {
    notice.classList.add('hidden');
    return;
  }
  notice.dataset.state = queueState.status || 'queued';
  var taskId = queueState.task_id || state.taskId || "";
  var taskEl = document.getElementById("compute-queue-task-id");
  var positionEl = document.getElementById("compute-queue-position");
  var estimateEl = document.getElementById("compute-queue-estimate");
  var titleEl = document.getElementById("compute-queue-title");
  var messageEl = document.getElementById("compute-queue-message");
  var totalEl = document.getElementById("compute-queue-total");
  var waitedEl = document.getElementById("compute-queue-waited");
  var expiresEl = document.getElementById("compute-queue-expires");
  var resourceEl = document.getElementById("compute-queue-resource");
  if (totalEl) totalEl.textContent = queueState.queue_length == null ? '—' :
    queueState.queue_length + ' waiting / ' + queueState.ahead + ' ahead';
  if (waitedEl) waitedEl.textContent = queueState.waited_seconds == null ? '—' :
    formatQueueWait(queueState.waited_seconds);
  if (expiresEl) expiresEl.textContent = queueState.expires_at ?
    new Date(typeof queueState.expires_at === 'number' ?
      queueState.expires_at * 1000 : queueState.expires_at).toLocaleString() : '—';
  if (resourceEl) resourceEl.textContent = queueState.pause_reason || 'Scheduled automatically';
  if (taskEl) taskEl.textContent = taskId;
  if (queueState.status === 'review') {
    if (titleEl) titleEl.textContent = 'Check finished — review required';
    if (messageEl) messageEl.textContent = 'Complete this step’s confirmation before continuing.';
    if (positionEl) positionEl.textContent = 'No longer waiting';
    if (estimateEl) estimateEl.textContent = '—';
  } else if (['completed', 'failed', 'cancelled'].includes(queueState.status)) {
    if (titleEl) titleEl.textContent = queueState.status === 'completed' ?
      'Operation finished' : 'Operation could not finish';
    if (messageEl) messageEl.textContent = queueState.error ||
      'The operation finished. Its result is shown in the workflow.';
    if (positionEl) positionEl.textContent = 'No longer waiting';
    if (estimateEl) estimateEl.textContent = '—';
  } else if (queueState.status === "running") {
    if (titleEl) titleEl.textContent = "Task processing has started";
    if (messageEl) messageEl.textContent =
      queueState.message || "Processing this step…";
    if (positionEl) positionEl.textContent = "Processing now";
    if (estimateEl) estimateEl.textContent = "Started";
  } else {
    if (titleEl) titleEl.textContent = "Task added to the compute queue";
    if (messageEl) messageEl.textContent =
      "Your operation has been saved and will start when resources are available. " +
      "Queue time counts toward the task's lifetime.";
    if (positionEl) positionEl.textContent = String(queueState.queue_position || "—");
    if (estimateEl) {
      var stamp = queueState.estimated_start_at ?
        new Date(queueState.estimated_start_at).toLocaleString() : "Pending estimate";
      estimateEl.textContent = queueState.estimated_wait_seconds == null ?
        'Not enough comparable history to estimate yet' :
        (queueState.estimated_start_at ? stamp + ' — ' : '') + 'about ' +
        formatQueueWait(queueState.estimated_wait_seconds) + ' remaining (estimate)';
    }
  }
}

function showComputeQueueStatus(queueState) {
  var notice = document.getElementById('compute-queue-status');
  if (!notice) return;
  var check = currentCheckNotice();
  if (check && !checkOwnsQueue(check, queueState)) return;
  // Automatic preparation and preview loads have local feedback, never a page-level receipt.
  if (!check && !state.buildRunning) {
    notice.classList.add('hidden');
    return;
  }
  if (!check && ['completed', 'failed', 'cancelled', 'review'].includes(queueState.status)) {
    notice.classList.add('hidden');
    return;
  }
  initComputeQueueStatus();
  // Keep the recovery guidance beside the operation the user is watching.
  var panel = document.querySelector('.panel.active');
  var progress = panel && panel.querySelector('.step-progress[data-state="running"]');
  var anchor = check ? check.handle.element :
    (state.uploadRunning ? document.getElementById('upload-progress') : progress);
  anchor = anchor || document.getElementById('progress-section');
  if (!anchor) return;
  // Polls update text in place. Reinsert only when the owning panel changes.
  if (notice.parentElement !== anchor.parentElement ||
      (!check && anchor.nextElementSibling !== notice)) anchor.after(notice);
  if (!check) notice.classList.remove('hidden');
  updateComputeQueueStatus(queueState);
}

function initComputeQueueStatus() {
  if (initComputeQueueStatus._done) return;
  var copy = document.getElementById('compute-queue-copy');
  if (!copy) return;
  initComputeQueueStatus._done = true;
  copy.addEventListener('click', async function() {
    var value = document.getElementById('compute-queue-task-id').textContent || '';
    try {
      await navigator.clipboard.writeText(value);
      copy.textContent = 'Copied';
    } catch (_error) { window.prompt('Copy this Task ID:', value); }
    setTimeout(function() { copy.textContent = 'Copy'; }, 1600);
  });
}

// Final export has queue states but no measured percentage. Share the Check
// widget and clock, with an indeterminate readout until the result arrives.
var _buildProgressHandle = null;

function startBuildProgress(reconnected) {
  if (_buildProgressHandle && _buildProgressHandle.clock) GMXPoll.stop(_buildProgressHandle.clock);
  _buildProgressHandle = _startProgressDisplay('build', 'run-btn', null);
  if (!_buildProgressHandle) return;
  _buildProgressHandle.reconnected = Boolean(reconnected);
  var section = document.getElementById('progress-section');
  section.prepend(_buildProgressHandle.element);
  section.classList.remove('hidden');
  var bar = _buildProgressHandle.element.querySelector('.step-progress-bar');
  bar.style.width = '35%';
  bar.removeAttribute('aria-valuenow');
  bar.setAttribute('aria-valuetext', 'In progress');
  _buildProgressHandle.element.querySelector('.step-progress-percent').textContent = '—';
}

function setBuildProgressPhase(message) {
  if (_buildProgressHandle) {
    _buildProgressHandle.element.querySelector('.step-progress-phase').textContent = message;
  }
}

function finishBuildProgress(ok) {
  var observedStart = _buildProgressHandle && _buildProgressHandle.element.isConnected;
  if (!observedStart) startBuildProgress();
  finishStepProgress(_buildProgressHandle, ok);
  var elapsed = _buildProgressHandle.element.querySelector('.step-progress-elapsed');
  if (!observedStart) elapsed.textContent = '';
  else if (_buildProgressHandle.reconnected) {
    elapsed.textContent = ((Date.now() - _buildProgressHandle.startedAt) / 1000).toFixed(1) +
      ' s since reconnect';
  }
  var bar = _buildProgressHandle.element.querySelector('.step-progress-bar');
  if (ok) bar.removeAttribute('aria-valuetext');
  else {
    bar.removeAttribute('aria-valuenow');
    bar.setAttribute('aria-valuetext', 'Build stopped');
    bar.style.width = '35%';
    _buildProgressHandle.element.querySelector('.step-progress-percent').textContent = '—';
    setBuildProgressPhase('Build stopped');
  }
}

function watchBuildResult(result, intervalMs, logTimer) {
  var handle = _buildProgressHandle;
  var poll = GMXPoll.start(async function() {
    if (handle !== _buildProgressHandle) { GMXPoll.stop(poll); return; }
    try {
      var status = await GMXPoll.read('/api/build/' + result.task_id + '/queue-status');
      if (handle !== _buildProgressHandle) return;
      if (['completed', 'failed', 'cancelled'].includes(status.status)) {
        GMXPoll.stop(poll);
        if (logTimer) GMXPoll.stop(logTimer);
        if (status.status === 'completed') _showBuildResult(status.result || result);
        else showBuildFailure(status.error || 'The build could not finish', status.input_check_required);
      } else {
        showComputeQueueStatus(Object.assign({task_id:result.task_id}, status));
        setBuildProgressPhase(status.status === 'running'
          ? 'Build started — waiting for completion...'
          : 'Waiting for resources');
      }
    } catch (error) {
      setBuildProgressPhase('Connection interrupted — reconnecting to build…');
    }
  }, intervalMs);
  (window._buildPollTimers = window._buildPollTimers || []).push(poll);
}

async function runBuild() {
  if (state.buildRunning || !state.taskType) return;
  state.buildRunning = true;

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  runBtn.textContent = 'Building...';

  const resultSection = document.getElementById('result-section');

  startBuildProgress();
  resultSection.classList.add('hidden');
  document.getElementById('build-log-content').replaceChildren();
  document.getElementById('build-log-panel').classList.add('hidden');
  // Build payload FIRST so log timer can reference task_id
  let modules;
  try {
    modules = buildModuleConfig();
  } catch (error) {
    state.buildRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '▶ Build System';
    showBuildFailure(error && error.message ? error.message : String(error));
    return;
  }
  const payload = {
    task_id: state.taskId || '',
    task_type: (state.taskType && state.taskType.id) || 'membrane-bilayer',
    system_name: document.getElementById('system-name')?.value || ((state.taskType && state.taskType.pipeline === 'solvator') ? 'solvator_system' : 'membrane_system'),
    modules: modules,
  };

  setBuildProgressPhase('Assembling pipeline configuration...');
  // Show log box and start polling AFTER payload is ready
  var logBox = document.getElementById("build-log");
  var logContent = document.getElementById("build-log-content");
  if (logBox) logBox.style.display = "block";
  document.getElementById('build-log-panel').classList.remove('hidden');
  document.getElementById('build-log-panel').open = true;
  if (logContent) logContent.innerHTML = "";
  // Track all polling intervals for cleanup on page unload
  var _timers = (window._buildPollTimers = window._buildPollTimers || []);
  var logSince = 0, logTimer = null;
  logTimer = GMXPoll.start(async function() {
    try {
      var ld = await GMXPoll.read("/api/build/" + payload.task_id + "/log?since=" + logSince);
      if (ld.lines && ld.lines.length > 0) {
        ld.lines.forEach(function(line) {
          if (logContent) {
            var div = document.createElement('div');
            div.textContent = line;
            logContent.appendChild(div);
          }
        });
        if (logBox) logBox.scrollTop = logBox.scrollHeight;
        logSince = ld.total;
      }
      if (ld.done) { GMXPoll.stop(logTimer); logTimer = null; }
    } catch(e) { /* network errors are transient — keep polling */ }
  }, 500);
  _timers.push(logTimer);

  setBuildProgressPhase(`Finalizing checked system: ${state.taskType.title}...`);

  try {
    var taskId = state.taskId || '';
    const res = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      var buildError = new Error(err.error || 'Build failed');
      buildError.inputCheckRequired = err.input_check_required === true;
      throw buildError;
    }

    const result = await res.json();

    if (result.status === 'completed') {
      if (logTimer) GMXPoll.stop(logTimer);
      _showBuildResult(result.result || result);
    } else {
      if (result.status === 'queued') {
        showComputeQueueStatus(result);
        setBuildProgressPhase('Waiting for resources');
      }
      // Restored, queued and immediately started builds share terminal handling.
      watchBuildResult(result, result.status === 'queued' ? 3000 : 2000, logTimer);
    }

  } catch (err) {
    if (logTimer) { GMXPoll.stop(logTimer); logTimer = null; }
    showBuildFailure(err.message, err.inputCheckRequired);
  }
}

// Load-order manifest: records that this file ran to completion.
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push("app.js");

function renderMembraneCompositionWarnings(items) {
  ['panel-membrane', 'panel-cg_environment', 'panel-final_review', 'panel-cg_system'].forEach(function(id) {
    var panel = document.getElementById(id);
    if (!panel) return;
    var box = panel.querySelector('.membrane-composition-advice');
    if (!box) {
      box = document.createElement('div');
      box.className = 'membrane-composition-advice validation-warnings';
      box.setAttribute('role', 'status');
      box.style.gridColumn = '1 / -1';
      // Keep advice in the reading flow before navigation; it never changes
      // Check/Next availability or asks for another confirmation.
      panel.insertBefore(box, panel.querySelector(':scope > .panel-actions'));
    }
    box.replaceChildren();
    box.hidden = !items.length;
    items.forEach(function(item) {
      var line = document.createElement('p');
      line.textContent = '⚠ ' + item.message;
      box.appendChild(line);
    });
  });
}

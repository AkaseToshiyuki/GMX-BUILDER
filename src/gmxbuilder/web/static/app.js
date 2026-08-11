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
var _cgenffUploads = {};
var _ligandChargeDrafts = {};
var _ligandChargeOrigins = {};
var _computedLigandCharges = {};
var _computedLigandChargePH = null;

// Step metadata — title + icon for each module name
const STEP_META = {
  input:      { title: 'PDB Upload' },
  structure:  { title: 'Structure Processing' },
  membrane:   { title: 'Membrane Builder' },
  orient:     { title: 'Orientation' },
  solvation:  { title: 'Solvent & Box' },
  ions:       { title: 'Ions' },
  forcefield: { title: 'Force Field' },
  topology:   { title: 'Topology & Parameters' },
  simparams:  { title: 'Simulation Params' },
  cg_model: { title: 'Martini 3 Model' },
  cg_mapping: { title: 'Protein Mapping' },
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
  'coarse-grained': 'CoarseGrainedBuilder',
};
const ROUTE_WORKFLOWS = Object.fromEntries(
  Object.entries(WORKFLOW_ROUTES).map(function(entry) { return [entry[1], entry[0]]; })
);
let _restoringRoute = false;

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
  // Task identifiers are deliberately kept out of browser history, address
  // bars, referrer headers, screenshots, and shared links.  Task recovery is
  // explicit through the Resume-by-ID form on the home page.
  const target = '/' + slug + '/Step' + (stepIdx + 1);
  if (window.location.pathname === target) return;
  if (_restoringRoute || forceReplace) history.replaceState({}, '', target);
  else history.pushState({}, '', target);
}

function parseTaskRoute(pathname) {
  const match = String(pathname || '').match(
    /^\/(BilayerBuilder|PureBilayerSystem|Solvator|CoarseGrainedBuilder)\/Step(\d+)\/?$/
  );
  if (!match) return null;
  return {
    workflow: ROUTE_WORKFLOWS[match[1]],
    stepIdx: Math.max(0, parseInt(match[2], 10) - 1),
  };
}

async function restoreRouteFromLocation() {
  const route = parseTaskRoute(window.location.pathname);
  if (!route) return;
  _restoringRoute = true;
  try {
    await selectTaskType(route.workflow);
    goToWizardStep(Math.min(route.stepIdx, state.wizardSteps.length - 1));
  } finally {
    _restoringRoute = false;
    if (state.currentStepIdx >= 0) syncTaskRoute(state.currentStepIdx, true);
  }
}

// ----- Init -----
document.addEventListener('DOMContentLoaded', async () => {
  console.log('DOMContentLoaded START');
  try {
  await Promise.all([loadTaskTypes(), loadOptions()]);
  setupUpload();
  setupRunButton();
  initOrientationStep();
  initStructureProcessing();
  
  initCustomLipidModal();
  initComputeQueueModal();
  initSimParams();
  initCheckButtons();
  initCoarseGrainedControls();
    // Solvation Check button
  var solvBtn = document.getElementById("solv-check-btn");
  if (solvBtn) {
    solvBtn.addEventListener("click", async function() {
      await _doCheckStep('solvation', 'solv-check-status', 'solv-check-btn');
      setTimeout(function() { renderSolvationViewer(); }, 300);
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
  // A full page load/reload never reconstructs task state from a URL.  Return
  // to the task selector and require the user to enter the saved Task ID.
  // In-page navigation still uses clean /Workflow/StepN history entries.
  if (window.location.pathname !== '/') history.replaceState({}, '', '/');
  window.addEventListener('popstate', function() {
    restoreRouteFromLocation();
  });
  console.log('DOMContentLoaded END — all init functions called');
  } catch(e) { console.error('DOMContentLoaded error:', e.message, e.stack); }

  // Cleanup build-polling intervals on page unload (prevent stale HTTP requests)
  window.addEventListener('beforeunload', function() {
    if (window._buildPollTimers) {
      window._buildPollTimers.forEach(function(t) { clearInterval(t); });
      window._buildPollTimers = [];
    }
  });
});

// ===================================================================
// Task Type Loading
// ===================================================================

async function loadTaskTypes() {
  try {
    const res = await fetch('/api/task-types');
    const data = await res.json();
    renderTaskCards(data.task_types);
  } catch (err) {
    console.error('Failed to load task types:', err);
  }
}

function renderTaskCards(taskTypes) {
  const grid = document.getElementById('task-grid');
  grid.innerHTML = '';

  // Group by category
  const grouped = {};
  taskTypes.forEach(t => {
    if (!grouped[t.category]) grouped[t.category] = [];
    grouped[t.category].push(t);
  });

  // Render with category headers
  for (const [cat, types] of Object.entries(grouped)) {
    // Category header
    const header = document.createElement('div');
    header.className = 'card-category-header';
    header.style.cssText = `grid-column:1/-1;font-size:13px;font-weight:700;color:${CAT_COLORS[cat]||'#64748b'};margin-top:${Object.keys(grouped).indexOf(cat)>0?'16px':'0'};`;
    header.textContent = cat;
    grid.appendChild(header);

    types.forEach(t => {
      const card = document.createElement('div');
      card.className = 'task-card' + (t.enabled ? '' : ' disabled');
      card.dataset.taskId = t.id;

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
      grid.appendChild(card);
    });
  }
}

async function selectTaskType(taskId) {
  try {
    // Show selection
    document.querySelectorAll('.task-card').forEach(c => {
      c.classList.toggle('selected', c.dataset.taskId === taskId);
    });

    const res = await fetch(`/api/task-type/${taskId}`);
    state.taskType = await res.json();
    state.wizardSteps = state.taskType.visible_modules || [];
    state.currentStepIdx = 0;
    state.completedSteps = new Set();   // reset locks on new task type
    state.pdbInfo = null;               // reset PDB info
    state.taskId = null;
    state.customLipidBusy = false;
    _orientedPdbContent = null;
    _membraneCheckpointPdb = null;
    _membraneActualBox = null;
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
      var taskIdText = document.getElementById('task-id-display');
      var taskIdBox = document.getElementById('header-task-id');
      if (taskIdText) taskIdText.textContent = state.taskId;
      if (taskIdBox) taskIdBox.classList.remove('hidden');
      setTimeout(loadTaskCustomLipids, 0);
      syncTaskRoute(0, true);
    }

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
}

function isCoarseGrainedWorkflow() {
  return !!(state.taskType && state.taskType.id === 'coarse-grained');
}

function coarseGrainedIncludesProtein() {
  if (!isCoarseGrainedWorkflow()) return false;
  return document.getElementById('cg-include-protein')?.checked !== false;
}

function syncCoarseGrainedInputControls(invalidate) {
  if (!isCoarseGrainedWorkflow()) return;
  var environment = document.getElementById('cg-environment')?.value || 'bilayer';
  var include = document.getElementById('cg-include-protein');
  if (environment === 'solution' && include) {
    include.checked = true;
    include.disabled = true;
  } else if (include) {
    include.disabled = false;
  }
  var includeProtein = include ? include.checked : true;
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
  ['cg-rotate-x', 'cg-rotate-y', 'cg-rotate-z', 'cg-z-offset'].forEach(function(id) {
    var element = document.getElementById(id);
    if (element) element.disabled = !includeProtein;
  });
  var solvent = document.getElementById('cg-include-solvent');
  if (solvent) {
    if (environment === 'solution') solvent.checked = true;
    solvent.disabled = environment === 'solution';
  }
  if (invalidate) invalidateCoarseGrainedFrom('input');
}

const _CG_STEP_ORDER = [
  'input', 'cg_model', 'cg_mapping', 'cg_environment', 'cg_solvation', 'cg_system'
];

function invalidateCoarseGrainedFrom(stepName) {
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

function parseCoarseGrainedComposition(value, label) {
  var entries = String(value || '').split(',').map(function(item) { return item.trim(); }).filter(Boolean);
  if (!entries.length) throw new Error(label + ' leaflet needs at least one lipid.');
  return entries.map(function(entry) {
    var match = entry.match(/^([A-Za-z0-9]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)$/);
    if (!match || Number(match[2]) <= 0) {
      throw new Error(label + ' leaflet entries must use NAME:positive-ratio.');
    }
    return {name: match[1].toUpperCase(), ratio: Number(match[2])};
  });
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
    eq2_tau_p: Number(document.getElementById('cg-eq2-tau-p')?.value || 4),
    production_ns: Number(document.getElementById('cg-production-ns')?.value || 1000),
    production_timestep_fs: Number(document.getElementById('cg-production-dt')?.value || 20),
    production_temperature: Number(document.getElementById('cg-production-temperature')?.value || 310),
    production_tau_t: Number(document.getElementById('cg-production-tau-t')?.value || 1),
    production_pressure: Number(document.getElementById('cg-production-pressure')?.value || 1),
    production_tau_p: Number(document.getElementById('cg-production-tau-p')?.value || 4),
    output_interval_ps: Number(document.getElementById('cg-output-ps')?.value || 100),
    energy_interval_ps: Number(document.getElementById('cg-energy-ps')?.value || 20),
    log_interval_ps: Number(document.getElementById('cg-log-ps')?.value || 20),
    comm_mode: String(document.getElementById('cg-comm-mode')?.value || 'Linear'),
    comm_interval: Number(document.getElementById('cg-comm-interval')?.value || 100),
    equilibration_1: document.getElementById('cg-eq1')?.checked !== false,
    equilibration_2: document.getElementById('cg-eq2')?.checked !== false,
    use_gpu: document.getElementById('cg-use-gpu')?.checked !== false,
    gpu_ids: String(document.getElementById('cg-gpu-ids')?.value || '0').trim(),
    threads: Number(document.getElementById('cg-threads')?.value || 8),
    mpi_ranks: Number(document.getElementById('cg-mpi-ranks')?.value || 1),
    system_name: document.getElementById('system-name')?.value || 'martini3_system',
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
  if (!Number.isInteger(config.threads) || config.threads < 1 ||
      !Number.isInteger(config.mpi_ranks) || config.mpi_ranks < 1 ||
      config.threads % config.mpi_ranks !== 0) {
    throw new Error('CG CPU threads must be positive integers and exactly divisible by thread-MPI ranks.');
  }
  if (config.use_gpu) {
    if (!/^\d+(?:,\d+)*$/.test(config.gpu_ids)) {
      throw new Error('CG GPU IDs must be unique comma-separated integers, for example 0 or 0,1.');
    }
    var gpuIds = config.gpu_ids.split(',');
    if (new Set(gpuIds).size !== gpuIds.length || gpuIds.length > config.mpi_ranks) {
      throw new Error('CG GPU IDs must be unique and their count cannot exceed thread-MPI ranks.');
    }
  }
  return config;
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
  function composition(entries) {
    return Array.isArray(entries) ? entries.map(function(item) {
      return String(item.name || '') + ':' + String(item.ratio == null ? 1 : item.ratio);
    }).join(',') : '';
  }
  var input = taskState.step_input_config || {};
  value('cg-environment', input.environment || 'bilayer');
  checked('cg-include-protein', input.include_protein);
  var mapping = taskState.step_cg_mapping_config || {};
  value('cg-protein-model', mapping.protein_model);
  value('cg-secondary', mapping.secondary_structure);
  value('cg-secondary-string', mapping.secondary_structure_string);
  checked('cg-elastic', mapping.elastic);
  value('cg-elastic-force', mapping.elastic_force);
  value('cg-elastic-lower', mapping.elastic_lower);
  value('cg-elastic-upper', mapping.elastic_upper);
  var environment = taskState.step_cg_environment_config || {};
  ['box_xy', 'box_z', 'rotate_x', 'rotate_y', 'rotate_z', 'z_offset'].forEach(function(key) {
    value('cg-' + key.replace(/_/g, '-'), environment[key]);
  });
  var upper = composition(environment.upper_leaflet);
  var lower = composition(environment.lower_leaflet);
  if (upper) value('cg-upper-lipids', upper);
  if (lower) value('cg-lower-lipids', lower);
  var solvation = taskState.step_cg_solvation_config || {};
  checked('cg-include-solvent', solvation.include_solvent);
  value('cg-salt', solvation.salt_molarity);
  var simulation = taskState.step_simparams_config || taskState.simparams || {};
  value('cg-temperature', simulation.temperature);
  value('cg-pressure', simulation.pressure);
  value('cg-production-ns', simulation.production_ns);
  value('cg-output-ps', simulation.output_interval_ps);
  checked('cg-eq1', simulation.equilibration_1);
  checked('cg-eq2', simulation.equilibration_2);
  checked('cg-use-gpu', simulation.use_gpu);
  value('cg-gpu-ids', simulation.gpu_ids);
  value('cg-threads', simulation.threads);
  value('cg-mpi-ranks', simulation.mpi_ranks);
  if (simulation.system_name) value('system-name', simulation.system_name);
  syncCoarseGrainedInputControls(false);
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
  desired.push('simparams');
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
  if (isCoarseGrainedWorkflow()) {
    var cgInput = defaults.input || {};
    var cgEnvironment = document.getElementById('cg-environment');
    var cgProtein = document.getElementById('cg-include-protein');
    if (cgEnvironment) cgEnvironment.value = cgInput.environment || 'bilayer';
    if (cgProtein) cgProtein.checked = cgInput.include_protein !== false;
    var cgMapping = defaults.cg_mapping || {};
    var cgProteinModel = document.getElementById('cg-protein-model');
    var cgSecondary = document.getElementById('cg-secondary');
    var cgElastic = document.getElementById('cg-elastic');
    if (cgProteinModel) cgProteinModel.value = cgMapping.protein_model || 'folded';
    if (cgSecondary) cgSecondary.value = cgMapping.secondary_structure || 'auto';
    if (cgElastic) cgElastic.checked = cgMapping.elastic !== false;
    var cgBox = defaults.cg_environment || {};
    var cgBoxXY = document.getElementById('cg-box-xy');
    var cgBoxZ = document.getElementById('cg-box-z');
    if (cgBoxXY) cgBoxXY.value = cgBox.box_xy || 12;
    if (cgBoxZ) cgBoxZ.value = cgBox.box_z || 14;
    var cgSolvation = defaults.cg_solvation || {};
    var cgIncludeSolvent = document.getElementById('cg-include-solvent');
    var cgSalt = document.getElementById('cg-salt');
    if (cgIncludeSolvent) cgIncludeSolvent.checked = cgSolvation.include_solvent !== false;
    if (cgSalt) cgSalt.value = cgSolvation.salt_molarity == null ? 0.15 : cgSolvation.salt_molarity;
    syncCoarseGrainedInputControls(false);
    return;
  }
  // Apply defaults to form fields
  if (defaults.membrane) {
    const m = defaults.membrane;
    selectLipid(m.lipid_type || 'POPC');
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
    if (modName === 'ions' && !(window._isSystemConfirmed ? window._isSystemConfirmed() : false)) {
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
      (window._isIonsChecked ? window._isIonsChecked() : false) &&
      (window._isSystemConfirmed ? window._isSystemConfirmed() : false);
  }
  if (modName === 'cg_system') {
    return _checkedSteps.has('cg_system') &&
      document.getElementById('cg-confirm-system')?.checked === true;
  }
  return _checkedSteps.has(modName);
}

function goToTaskSelect() {
  // Always allowed back
  state.currentStepIdx = -1;
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  var panel0 = document.getElementById('panel-task-type');
  if (panel0) panel0.classList.add('active');
  updateStepNavHighlight();
  if (!_restoringRoute) history.pushState({}, '', '/');
}

// ---- Step execution on server (incremental checkpoint build) ----
var _stepRunning = false;
var _viewerLoadTimer = null;  // cleared before setting new setTimeout in goToWizardStep

/** Centralized fetch helper — throws on non-2xx, parses JSON on success. */
async function _apiFetch(url, options) {
  var res = await fetch(url, options);
  if (!res.ok) {
    var text = await res.text().catch(function() { return ''; });
    throw new Error('HTTP ' + res.status + ': ' + (text || res.statusText).substring(0, 200));
  }
  return res.json();
}

async function _runStepOnServer(stepName, config) {
  if (_stepRunning || !state.taskId) return null;
  _stepRunning = true;
  var statusEl = document.getElementById('step-run-status');
  if (statusEl) { statusEl.textContent = 'Running...'; statusEl.style.display = 'block'; }
  try {
    var result = await _apiFetch('/api/step/' + state.taskId + '/' + stepName, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: config || {} }),
    });
    if (result.status === 'ok') {
      if (statusEl) { statusEl.textContent = '✓ Done (' + (result.elapsed_s != null ? result.elapsed_s : '?') + 's)'; statusEl.style.color = '#059669'; }
      return result;
    } else {
      if (statusEl) { statusEl.textContent = '✗ ' + (result.error || 'Failed'); statusEl.style.color = '#dc2626'; }
      return null;
    }
  } catch (e) {
    if (statusEl) { statusEl.textContent = '✗ ' + (e.message || 'Network error'); statusEl.style.color = '#dc2626'; }
    return null;
  } finally {
    _stepRunning = false;
  }
}

async function _loadStepViewerPdb(stepName) {
  if (!state.taskId) return null;
  try {
    var resp = await fetch('/api/step/' + state.taskId + '/' + stepName + '/viewer.pdb');
    if (resp.ok) return await resp.text();
    console.warn('_loadStepViewerPdb: HTTP', resp.status, 'for step', stepName);
  } catch (e) { console.warn('_loadStepViewerPdb: network error for step', stepName, e); }
  return null;
}

async function renderCoarseGrainedViewer(stepName) {
  var targetMap = {
    cg_mapping: 'cg-mapping-viewer',
    cg_environment: 'cg-environment-viewer',
    cg_solvation: 'cg-solvation-viewer',
    cg_system: 'cg-system-viewer',
  };
  var targetId = targetMap[stepName];
  var target = targetId ? document.getElementById(targetId) : null;
  if (!target || typeof $3Dmol === 'undefined') return false;
  window._cgViewers = window._cgViewers || {};
  var cached = window._cgViewers[targetId];
  if (cached && cached.taskId !== state.taskId) {
    try { cached.viewer.clear(); } catch (e) {}
    target.replaceChildren();
    delete window._cgViewers[targetId];
    cached = null;
  }
  var pdb = await _loadStepViewerPdb(stepName);
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
      viewer = $3Dmol.createViewer(target, {backgroundColor: 'white'});
    } catch (error) {
      target.textContent = '3D viewer unavailable: this browser does not provide a working WebGL context.';
      target.classList.add('viewer-unavailable');
      return false;
    }
    window._cgViewers[targetId] = {viewer: viewer, taskId: state.taskId};
  }
  target.classList.remove('viewer-unavailable');
  viewer.removeAllModels();
  var model = viewer.addModel(pdb, 'pdb');
  viewer.setStyle({}, {sphere: {radius: 0.16, colorscheme: 'Jmol'}});
  if (stepName === 'cg_mapping') {
    viewer.setStyle({}, {sphere: {radius: 0.14, colorscheme: 'Jmol'}, stick: {radius: 0.08, colorscheme: 'Jmol'}});
  }
  viewer.setStyle(
    {atom: ['BB', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5']},
    {sphere: {radius: 0.20, color: '#9f6f8f'}, stick: {radius: 0.08, color: '#9f6f8f'}}
  );
  viewer.setStyle({resn: 'W'}, {sphere: {radius: 0.045, color: '#60a5fa', opacity: 0.14}});
  viewer.setStyle({resn: ['NA', 'CL']}, {sphere: {radius: 0.24, colorscheme: 'Jmol'}});
  // The mapping checkpoint box is only an internal envelope estimate.  The
  // user-defined physical PBC cell begins at CG Environment.
  if (stepName !== 'cg_mapping' && viewer.addUnitCell) {
    viewer.addUnitCell(model, {boxColor: '#64748b'});
  }
  if (viewer.resize) viewer.resize();
  viewer.zoomTo();
  viewer.render();
  viewer.setSlab(-10000, 10000);
  return true;
}

async function confirmCoarseGrainedSystem() {
  var checkbox = document.getElementById('cg-confirm-system');
  var status = document.getElementById('cg-system-status');
  if (!checkbox || !checkbox.checked || !state.taskId) return;
  checkbox.disabled = true;
  try {
    var response = await fetch('/api/step/' + state.taskId + '/cg_system/confirm', {
      method: 'POST'
    });
    var result = await response.json();
    if (!response.ok || result.confirmed !== true) {
      throw new Error(result.error || 'Confirmation failed');
    }
    _checkedSteps.add('cg_system');
    // Confirmation is checkpoint metadata, never a construction option.  Keep
    // the checked module snapshot unchanged so a later rebuild cannot smuggle
    // confirmation into a fresh random assembly.
    checkbox.disabled = false;
    if (status) {
      status.textContent = '✓ Exact CG checkpoint confirmed';
      status.style.color = '#059669';
    }
    updateNextButtonState();
    updateStepNavHighlight();
  } catch (error) {
    checkbox.checked = false;
    checkbox.disabled = false;
    if (status) {
      status.textContent = '✗ ' + (error.message || 'Confirmation failed');
      status.style.color = '#dc2626';
    }
  }
}

function initCoarseGrainedControls() {
  var environment = document.getElementById('cg-environment');
  var includeProtein = document.getElementById('cg-include-protein');
  if (environment) environment.addEventListener('change', function() {
    syncCoarseGrainedInputControls(true);
  });
  if (includeProtein) includeProtein.addEventListener('change', function() {
    syncCoarseGrainedInputControls(true);
  });

  var checks = [
    ['cg-model-check', 'cg_model', 'cg-model-status'],
    ['cg-mapping-check', 'cg_mapping', 'cg-mapping-status'],
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
  var confirmation = document.getElementById('cg-confirm-system');
  if (confirmation) {
    confirmation.disabled = true;
    confirmation.addEventListener('change', function() {
      if (confirmation.checked) confirmCoarseGrainedSystem();
      else {
        _checkedSteps.delete('cg_system');
        updateNextButtonState();
        updateStepNavHighlight();
      }
    });
  }

  var inputIds = [
    'cg-protein-model', 'cg-secondary', 'cg-secondary-string', 'cg-elastic',
    'cg-elastic-force', 'cg-elastic-lower', 'cg-elastic-upper',
    'cg-box-xy', 'cg-box-z', 'cg-rotate-x', 'cg-rotate-y', 'cg-rotate-z',
    'cg-z-offset', 'cg-upper-lipids', 'cg-lower-lipids',
    'cg-include-solvent', 'cg-salt'
  ];
  inputIds.forEach(function(id) {
    var element = document.getElementById(id);
    if (!element) return;
    element.addEventListener('change', function() {
      var step = id.indexOf('cg-protein') === 0 || id.indexOf('cg-secondary') === 0 ||
        id.indexOf('cg-elastic') === 0 ? 'cg_mapping' :
        id === 'cg-include-solvent' || id === 'cg-salt' ? 'cg_solvation' :
        'cg_environment';
      invalidateCoarseGrainedFrom(step);
      if (id === 'cg-include-solvent') {
        var salt = document.getElementById('cg-salt');
        if (salt) salt.disabled = !element.checked;
      }
    });
  });
}

function goToWizardStep(idx) {
  if (idx < 0 || idx >= state.wizardSteps.length) return;

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
      !(window._isIonsChecked ? window._isIonsChecked() : false) ||
      !(window._isSystemConfirmed ? window._isSystemConfirmed() : false)
  )) {
    shakeStepNav();
    alert("Run Check Ion Counts, inspect the complete system, and click Confirm Simulation System before proceeding.");
    return;
  }
  if (structIdx >= 0 && idx > structIdx && !_protonationComputed) {
    shakeStepNav();
    alert('Please compute protonation (or check "Skip protonation") before proceeding.');
    return;
  }

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
  }
  if (idx <= state.wizardSteps.indexOf('membrane')) { _compositionChecked = false; _membraneCheckpointPdb = null; _membraneActualBox = null; }
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
  }

  // Load viewer data from previous step's checkpoint (not current step —
  // current step's checkpoint is created by the Check button).
  if (modName === "membrane") {
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

  if (modName === "ions" && window._renderIonViewer) {
    setTimeout(function() { window._renderIonViewer(); }, 400);
  }

  if (modName && modName.indexOf('cg_') === 0 && modName !== 'cg_model') {
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
  // Step nav buttons (only module steps — task type is handled inline above)
  const stepBtn = e.target.closest('#step-nav .step[data-step-module]');
  if (stepBtn) {
    const mod = stepBtn.dataset.stepModule;
    const idx = state.wizardSteps.indexOf(mod);
    if (idx >= 0) goToWizardStep(idx);
  }
});

// ===================================================================
// Options Loading
// ===================================================================

async function resumeTask(taskId, requestedStepIdx) {
  try {
    state.customLipidBusy = false;
    _customLipidFailedName = null;
    if (_customLipidPollTimer) {
      clearTimeout(_customLipidPollTimer);
      _customLipidPollTimer = null;
    }
    var res = await fetch("/api/task/" + taskId + "/resume");
    if (!res.ok) { alert("Task not found or expired."); return; }
    var taskState = await res.json();
    var resumeStepData = null;
    try {
      var resumeStepResponse = await fetch('/api/steps/' + taskId);
      if (resumeStepResponse.ok) resumeStepData = await resumeStepResponse.json();
    } catch(e) { /* task state remains a valid fallback */ }

    var savedTaskType = taskState.task_type || {};
    if (savedTaskType.requires_input !== false &&
        !taskState.pdb_info_full && !taskState.pdb_info) {
      alert("Task has no PDB data — cannot resume.");
      return;
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
        state.wizardSteps = taskState.visible_modules || ["input","forcefield","structure","solvation","ions","simparams"];
        state.taskType = { id: savedTypeId || "solvator", visible_modules: state.wizardSteps, pipeline: "solvator" };
      }
      // Rebuild step nav
      var nav = document.getElementById("step-nav");
      if (nav) {
        nav.innerHTML = "";
        state.wizardSteps.forEach(function(modName, idx) {
          var meta = STEP_META[modName] || { title: modName, icon: "?" };
          var btn = document.createElement("button");
          btn.className = "step";
          btn.dataset.stepModule = modName;
          btn.innerHTML = '<span class="step-num">' + (meta.icon || (idx + 1)) + '</span>' + meta.title;
          btn.addEventListener("click", function() {
            var si = state.wizardSteps.indexOf(modName);
            if (si >= 0) goToWizardStep(si);
          });
          nav.appendChild(btn);
        });
      }
    }
    configureTaskSpecificControls();
    // Restore the force field before initializing simulation parameters.
    // CHARMM and Amber require different non-bonded defaults; initializing
    // against the page's Amber default would leave a resumed CHARMM task with
    // the invalid combination Force-switch + DispCorr=EnerPres.
    var savedForceFieldConfig = taskState.step_forcefield_config || taskState.forcefield || {};
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
        taskState.step_simparams_config || taskState.simparams
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
      validation_warnings: [],
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
    if (savedPH === undefined) savedPH = taskState.pH;
    savedPH = Number(savedPH === undefined ? 7.0 : savedPH);
    if (!Number.isFinite(savedPH) || savedPH < 1.0 || savedPH > 13.0) savedPH = 7.0;
    _systemPH = savedPH;
    var resumedPHInput = document.getElementById('proc-pH');
    if (resumedPHInput) resumedPHInput.value = savedPH.toFixed(1);

    // Show task ID in header
    var tidEl = document.getElementById("task-id-display");
    var tidBox = document.getElementById("header-task-id");
    if (tidEl) tidEl.textContent = taskId;
    if (tidBox) tidBox.classList.remove("hidden");

    // Show upload info and navigate
    if (info.num_atoms > 0) showUploadInfo(info);
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
    if (_resumeCheckedSteps.length === 0) {
      _resumeCheckedSteps = (taskState.steps_completed || []).slice();
    }
    _protonationComputed = _resumeCheckedSteps.indexOf('structure') >= 0;
    _solvChecked = _resumeCheckedSteps.indexOf('solvation') >= 0;
    _compositionChecked = _resumeCheckedSteps.indexOf('membrane') >= 0;
    var hasIonCheckpoint = _resumeCheckedSteps.indexOf('ions') >= 0;
    if (window._setIonsChecked) window._setIonsChecked(hasIonCheckpoint);
    var persistedBuildStatus = (taskState.build_status || {}).status;
    var canRestoreSystemConfirmation = hasIonCheckpoint && (
      taskState.current_step === 'simparams' ||
      ['queued', 'running', 'completed'].indexOf(persistedBuildStatus) >= 0
    );
    if (window._setSystemConfirmed) {
      window._setSystemConfirmed(canRestoreSystemConfirmation);
    }
    // Pre-mark completed steps
    var completedSteps = _resumeCheckedSteps;
    completedSteps.forEach(function(stepName) {
      var si = state.wizardSteps.indexOf(stepName);
      if (si >= 0) markStepComplete(si);
    });
    // Input step is always complete when PDB data exists (step 0 lock bypass)
    var inputIdx = state.wizardSteps.indexOf("input");
    if (inputIdx >= 0 && state.pdbInfo && (
        state.pdbInfo.num_atoms > 0 || _resumeCheckedSteps.indexOf('input') >= 0
    )) markStepComplete(inputIdx);

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

    // Force-advance currentStepIdx past lock check (resume = all prior steps are done)
    state.currentStepIdx = 0;

    var resumedBuild = taskState.build_status || {};
    var completedBuild = resumedBuild.status === "completed" &&
      resumedBuild.download_available === true && resumedBuild.result;

    // Navigate to the first incomplete step. A completed finalization resumes
    // directly on the Simulation Parameters result panel unless the URL
    // explicitly requested another step.
    var currentStep = taskState.resume_step || taskState.current_step ||
      state.wizardSteps[0] || "input";
    var stepIdx = state.wizardSteps.indexOf(currentStep);
    if (Number.isInteger(requestedStepIdx)) {
      stepIdx = Math.max(0, Math.min(requestedStepIdx, state.wizardSteps.length - 1));
    } else if (completedBuild) {
      var resultStepIdx = state.wizardSteps.indexOf("simparams");
      if (resultStepIdx >= 0) stepIdx = resultStepIdx;
    }
    if (stepIdx >= 0) {
      // Navigate first, then show message (alert blocks UI)
    goToWizardStep(stepIdx);
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
    if (completedBuild && !Number.isInteger(requestedStepIdx)) {
      var progressSection = document.getElementById("progress-section");
      if (progressSection) progressSection.classList.remove("hidden");
      _showBuildResult(resumedBuild.result);
      syncTaskRoute(stepIdx, true);
    } else if (resumedBuild.status === "queued" || resumedBuild.status === "running") {
      state.buildRunning = true;
      try {
        var queueResponse = await fetch("/api/build/" + taskId + "/queue-status");
        var queueState = queueResponse.ok ? await queueResponse.json() : resumedBuild;
        queueState.task_id = taskId;
        showComputeQueueModal(queueState);
      } catch (error) {
        resumedBuild.task_id = taskId;
        showComputeQueueModal(resumedBuild);
      }
    }
  } catch (e) {
    alert("Failed to resume: " + e.message);
  }
}


async function loadOptions() {
  try {
    const res = await fetch('/api/options');
    const opts = await res.json();

    // Build custom lipid picker
    buildLipidPicker(opts.lipids, opts.lipid_categories);
    initLipidMixing();
    if (state.taskId) setTimeout(loadTaskCustomLipids, 0);

    // Store globally for task-type-dependent dropdown repopulation
    window._allWaterModels = opts.water_models || [];
    window._allSolvents = opts.solvents || [];

    window._forceFieldOptions = opts.force_fields || [];
    updateWaterModelOptions(true);

    // Sensible defaults
    syncLockedWaterModelDisplay();
  } catch (err) {
    console.error('Failed to load options:', err);
  }
}

function populateSelect(id, items) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = '';
  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item.value;
    opt.textContent = item.label;
    sel.appendChild(opt);
  });
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
  var protein = document.getElementById('ff-protein');
  var status = document.getElementById('ff-compatibility-status');
  var confirm = document.getElementById('forcefield-check-btn');
  if (!protein) return;
  try {
    if (status) status.textContent = 'Checking installed parameter families...';
    var response = await fetch('/api/forcefield-compatibility/' + state.taskId, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        protein_ff: protein.value,
        lipid_names: currentMembraneLipidNames(),
      }),
    });
    var report = await response.json();
    if (!response.ok || report.error) throw new Error(report.error || 'Compatibility check failed');
    window._ffCompatibility = report;
    var lipidOk = populateCompatibilitySelect('ff-lipid', report.lipid_options);
    var ligandOk = populateCompatibilitySelect('ff-ligand', report.ligand_options);
    renderLigandChargeInputs();
    var warnings = [];
    (report.lipid_options || []).forEach(function(item) {
      if (!item.enabled && item.reason) warnings.push('Lipid: ' + item.reason);
    });
    (report.ligand_options || []).forEach(function(item) {
      if (!item.enabled && item.reason) warnings.push('Small molecule: ' + item.reason);
    });
    (report.ligands || []).forEach(function(item) {
      warnings.push((item.display_name || item.name) + ': ' + item.rtp_reason);
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
    if (status) {
      status.innerHTML = '<strong>' + (valid ? '✓ Compatible family: ' : '✗ No complete compatible combination for ') +
        String(report.family || '').toUpperCase() + '</strong>' +
        (warnings.length ? '<ul>' + warnings.map(function(item) { return '<li>' + escapeHtml(item) + '</li>'; }).join('') + '</ul>' : '');
      status.style.color = valid ? '#166534' : '#b91c1c';
    }
    if (confirm) confirm.disabled = !valid;
  } catch (error) {
    window._ffCompatibility = null;
    if (status) { status.textContent = '✗ ' + error.message; status.style.color = '#b91c1c'; }
    if (confirm) confirm.disabled = true;
  }
}

function renderLigandChargeInputs() {
  var container = document.getElementById('ff-ligand-charges');
  var source = document.getElementById('ff-ligand');
  if (!container) return;
  container.innerHTML = '';
  var report = window._ffCompatibility;
  if (!report || !source) return;
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
  if (!Number.isFinite(targetPH) || targetPH < 1.0 || targetPH > 13.0) targetPH = 7.0;
  var names = ((window._ffCompatibility || {}).ligand_names || []);
  if (!names.length) return;
  if (!force && _computedLigandChargePH === targetPH && Object.keys(_computedLigandCharges).length) {
    names.forEach(updateLigandChargeStatus);
    return;
  }
  names.forEach(function(name) {
    var status = document.querySelector('[data-ligand-charge-status="' + name + '"]');
    if (status) { status.textContent = 'Computing at pH ' + targetPH.toFixed(1) + '...'; status.style.color = '#d97706'; }
  });
  try {
    var response = await fetch('/api/ligand-charge-suggestions/' + state.taskId, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pH: targetPH}),
    });
    var result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || 'Charge calculation failed');
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

function collectCGenFFParameters() {
  var result = {};
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
  _checkedSteps.delete('forcefield');
  if (_checkedConfig) delete _checkedConfig.forcefield;
  var status = document.getElementById('forcefield-check-status');
  if (status) status.textContent = '';
  updateNextButtonState();
  updateStepNavHighlight();
}

// ===================================================================
// Custom Lipid Picker
// ===================================================================

let _lipidPickerData = { lipids: [], categories: {} };
let _selectedLipid = 'POPC';
let _smilesDrawer = null;  // SmilesDrawer instance, initialized lazily

let _pickerTarget = null;  // { leaflet, idx } of which row triggered the picker

function buildLipidPicker(lipids, categories) {
  _lipidPickerData = { lipids, categories };

  const dropdown = document.getElementById('lipid-picker-dropdown');
  const searchInput = document.getElementById('lipid-picker-search');

  // Search filter
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      renderLipidList(searchInput.value.trim().toLowerCase());
    });
  }

  // Close on outside click (guard against duplicate listeners on re-init)
  if (!buildLipidPicker._outsideHandlerAttached) {
    buildLipidPicker._outsideHandlerAttached = true;
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#lipid-picker-dropdown') && !e.target.closest('.mix-lipid-trigger')) {
        closeLipidDropdown();
      }
    });
  }

  // Select a default
  selectLipid('POPC');
}

function openLipidDropdown(anchorEl) {
  const dropdown = document.getElementById('lipid-picker-dropdown');
  if (!dropdown) return;
  // Position dropdown near the anchor element
  if (anchorEl) {
    const rect = anchorEl.getBoundingClientRect();
    dropdown.style.position = 'fixed';
    dropdown.style.left = rect.left + 'px';
    dropdown.style.top = (rect.bottom + 4) + 'px';
    dropdown.style.width = Math.max(rect.width, 500) + 'px';
  }
  dropdown.classList.remove('hidden');
  const searchEl = document.getElementById('lipid-picker-search');
  if (searchEl) searchEl.value = '';
  renderLipidList('');
}

function closeLipidDropdown() {
  const dropdown = document.getElementById('lipid-picker-dropdown');
  if (dropdown) dropdown.classList.add('hidden');
  _pickerTarget = null;
}

function selectLipid(name) {
  const lipid = _lipidPickerData.lipids.find(l => l.name === name);
  if (!lipid) { closeLipidDropdown(); return; }
  const source = selectedLipidParameterSource();
  if (source && (lipid.parameterizations || []).indexOf(source) < 0 && !lipid._custom) {
    alert(lipidAvailabilityMessage(lipid, source));
    return;
  }
  _selectedLipid = name;
  // Save DHH for membrane plane rendering in orient 3D viewer
  if (lipid.bilayer_thickness) _dominantLipidDHH = lipid.bilayer_thickness;

  // If a specific row triggered the picker, update that row
  if (_pickerTarget) {
    const { leaflet, idx } = _pickerTarget;
    const mix = leaflet === 'upper' ? _mixUpper : _mixLower;
    if (idx < mix.length) {
      mix[idx] = lipidMixEntry(name, mix[idx].ratio);
      if (!_asymmetric && leaflet === 'upper') {
        _mixLower = _mixUpper.map(m => ({...m}));
      }
      _invalidateMembraneBuild();
      updateCompositionStatus();
      renderMixList('upper');
      renderMixList('lower');
    }
    closeLipidDropdown();
    return;
  }

  // Fallback: add/replace in upper leaflet mix
  const existing = _mixUpper.findIndex(m => m.name === name);
  if (existing >= 0) {
    // Already in mix — highlight it
  } else if (_mixUpper.length === 1 && _mixUpper[0].ratio === 100) {
    _mixUpper[0] = lipidMixEntry(name, 100);
  } else {
    _mixUpper.push(lipidMixEntry(name, 0));
    normalizeRatios(_mixUpper);
  }

  if (!_asymmetric) {
    _mixLower = _mixUpper.map(m => ({...m}));
  }
  renderMixList('upper');
  renderMixList('lower');

  closeLipidDropdown();
}

function lipidMixEntry(name, ratio) {
  const lipid = (_lipidPickerData.lipids || []).find(l => l.name === name);
  if (!lipid || !lipid._custom) return {name, ratio};
  return {
    name,
    ratio,
    category: lipid.category,
    common_name: lipid.common_name,
    formula: lipid.formula,
    tail1: lipid.tail1,
    tail2: lipid.tail2,
    area_per_lipid: lipid.area_per_lipid,
    bilayer_thickness: lipid.bilayer_thickness,
    vdw_radius: lipid.vdw_radius,
    charge: lipid.charge,
    mass: lipid.mass,
    smiles: lipid.smiles,
    canonical_smiles: lipid.canonical_smiles,
    inchi_key: lipid.inchi_key,
    _custom: true,
  };
}

function selectedLipidParameterSource() {
  const lipidSelect = document.getElementById('ff-lipid');
  if (lipidSelect && ['lipid21', 'gaff2', 'charmm36m', 'charmm36', 'oplsaa'].includes(lipidSelect.value)) {
    return lipidSelect.value;
  }
  const proteinSelect = document.getElementById('ff-protein');
  const protein = proteinSelect ? proteinSelect.value : '';
  if (protein.startsWith('amber')) return 'lipid21';
  if (protein === 'charmm36m' || protein === 'charmm36' || protein === 'oplsaa') return protein;
  return '';
}

function lipidParameterSourceLabel(source) {
  return ({
    lipid21: 'Amber Lipid21 v1.0 (exact)',
    gaff2: 'Amber14SB + GAFF2',
    charmm36m: 'CHARMM36m',
    charmm36: 'CHARMM36',
    oplsaa: 'OPLS-AA',
  })[source] || source || 'the selected force field';
}

function lipidAvailabilityMessage(lipid, selectedSource) {
  const alternatives = (lipid.parameterizations || [])
    .filter(source => source !== selectedSource)
    .map(lipidParameterSourceLabel);
  return lipid.name + ' is unavailable with ' + lipidParameterSourceLabel(selectedSource) + '. ' +
    (alternatives.length
      ? 'Available with: ' + alternatives.join(', ') + '.'
      : 'No validated alternative is installed.');
}

function renderLipidList(filter) {
  const container = document.getElementById('lipid-picker-list');
  container.innerHTML = '';

  const { lipids, categories } = _lipidPickerData;
  const filterLower = filter || '';
  const catNames = Object.keys(categories);
  const selectedSource = selectedLipidParameterSource();

  let anyVisible = false;

  catNames.forEach(cat => {
    const catLipids = categories[cat].lipids || [];
    const filtered = catLipids.filter(name => {
      if (!filterLower) return true;
      const l = lipids.find(ll => ll.name === name);
      if (!l) return false;
      return l.name.toLowerCase().includes(filterLower) ||
             l.common_name.toLowerCase().includes(filterLower) ||
             l.category.toLowerCase().includes(filterLower) ||
             l.formula.toLowerCase().includes(filterLower);
    });

    if (!filtered.length) return;
    anyVisible = true;

    // Category header
    const header = document.createElement('div');
    header.className = 'lipid-cat-header';
    header.textContent = cat;
    container.appendChild(header);

    // Category grid
    const grid = document.createElement('div');
    grid.className = 'lipid-cat-grid';

    filtered.forEach(name => {
      const l = lipids.find(ll => ll.name === name);
      if (!l) return;

      const card = document.createElement('div');
      card.className = 'lipid-card';
      card.dataset.lipidName = l.name;
      const supported = !selectedSource || (l.parameterizations || []).indexOf(selectedSource) >= 0;
      if (!supported) {
        card.classList.add('unavailable');
        card.setAttribute('aria-disabled', 'true');
        card.title = lipidAvailabilityMessage(l, selectedSource);
      }
      if (l.name === _selectedLipid) card.classList.add('selected');

      // Structure schematic (SVG)
      const imgDiv = document.createElement('div');
      imgDiv.className = 'lipid-card-img';
      imgDiv.innerHTML = lipidSchematicSVG(l);
      card.appendChild(imgDiv);

      // Info
      const info = document.createElement('div');
      info.className = 'lipid-card-info';

      const nameDiv = document.createElement('div');
      nameDiv.className = 'lipid-card-name';
      nameDiv.textContent = l.name;
      info.appendChild(nameDiv);

      const desc = document.createElement('div');
      desc.className = 'lipid-card-desc';
      desc.textContent = l.common_name;
      info.appendChild(desc);

      const meta = document.createElement('div');
      meta.className = 'lipid-card-meta';

      const areaTag = document.createElement('span');
      areaTag.className = 'lipid-card-tag';
      areaTag.textContent = `APL ${l.area_per_lipid} nm²`;
      meta.appendChild(areaTag);

      const chargeTag = document.createElement('span');
      chargeTag.className = 'lipid-card-tag';
      if (l.charge > 0) chargeTag.classList.add('charge-pos');
      else if (l.charge < 0) chargeTag.classList.add('charge-neg');
      else chargeTag.classList.add('charge-zero');
      chargeTag.textContent = l.charge === 0 ? 'neutral' : `charge ${l.charge > 0 ? '+' : ''}${l.charge}`;
      meta.appendChild(chargeTag);

      const parameterTag = document.createElement('span');
      parameterTag.className = 'lipid-card-tag';
      var availableSources = l.parameterizations || [];
      var sourceLabels = {
        gaff2: 'Amber/GAFF2', charmm36m: 'CHARMM36m RTP', charmm36: 'CHARMM36 RTP'
      };
      parameterTag.textContent = availableSources.indexOf(selectedSource) >= 0
        ? sourceLabels[selectedSource]
        : 'Unavailable';
      parameterTag.title = 'Topology parameter source';
      meta.appendChild(parameterTag);

      const tailTag = document.createElement('span');
      tailTag.className = 'lipid-card-tag';
      tailTag.textContent = `${l.tail1[0]}:${l.tail1[1]}/${l.tail2[0]}:${l.tail2[1]}`;
      meta.appendChild(tailTag);

      info.appendChild(meta);
      if (!supported) {
        const unavailable = document.createElement('div');
        unavailable.className = 'lipid-card-unavailable';
        unavailable.textContent = lipidAvailabilityMessage(l, selectedSource);
        info.appendChild(unavailable);
      }
      card.appendChild(info);

      if (supported) card.addEventListener('click', () => selectLipid(l.name));
      grid.appendChild(card);
    });

    container.appendChild(grid);
  });

  if (!anyVisible) {
    container.innerHTML = '<p class="hint" style="text-align:center;padding:20px;">No lipids match your search.</p>';
  }

  // "Custom Lipid" option at the bottom
  const customOpt = document.createElement('div');
  customOpt.className = 'lipid-card lipid-option-custom';
  const customSupported = selectedSource === 'gaff2';
  const selectedProteinFF = (document.getElementById('ff-protein') || {}).value || '';
  const gaffOption = ((window._ffCompatibility || {}).lipid_options || [])
    .find(option => option.value === 'gaff2');
  const canSwitchToCustom = selectedProteinFF.startsWith('amber') &&
    Boolean(gaffOption && gaffOption.enabled);
  customOpt.innerHTML = '<div class="lipid-card-info" style="text-align:center;padding:8px;"><span style="font-size:16px;">&#43;</span> Add Custom Lipid from SMILES' +
    (customSupported ? '' : canSwitchToCustom
      ? '<div class="lipid-card-unavailable">Switch this task from Lipid21 to Amber + GAFF2, then re-check Force Field.</div>'
      : '<div class="lipid-card-unavailable">Custom lipids currently require Amber + GAFF2.</div>') + '</div>';
  if (customSupported) {
    customOpt.style.cursor = 'pointer';
    customOpt.addEventListener('click', () => {
      closeLipidDropdown();
      openCustomLipidModal();
    });
  } else if (canSwitchToCustom) {
    customOpt.style.cursor = 'pointer';
    customOpt.addEventListener('click', () => {
      closeLipidDropdown();
      switchTaskToCustomLipidBackend();
    });
  } else {
    customOpt.classList.add('unavailable');
    customOpt.setAttribute('aria-disabled', 'true');
  }
  container.appendChild(customOpt);

}

function switchTaskToCustomLipidBackend() {
  const lipidFF = document.getElementById('ff-lipid');
  const forceFieldIdx = state.wizardSteps.indexOf('forcefield');
  if (!lipidFF || forceFieldIdx < 0) return;
  const accepted = window.confirm(
    'Custom lipids require Amber + GAFF2. Switch the lipid backend to GAFF2? ' +
    'The existing Force Field Check will be invalidated and must be run again.'
  );
  if (!accepted) return;
  lipidFF.value = 'gaff2';
  lipidFF.dispatchEvent(new Event('change', {bubbles: true}));
  _checkedSteps.delete('forcefield');
  state.completedSteps.delete(forceFieldIdx);
  for (let index = forceFieldIdx + 1; index < state.wizardSteps.length; index++) {
    _checkedSteps.delete(state.wizardSteps[index]);
    state.completedSteps.delete(index);
  }
  goToWizardStep(forceFieldIdx);
  alert(
    'Lipid backend changed to GAFF2. Run Force Field Check again, then return ' +
    'to Membrane Builder to submit the custom lipid.'
  );
}
// ===================================================================
// Custom Lipid Modal (SMILES input)
// ===================================================================

let _customLipidData = null;  // parsed lipid data, set on successful parse
let _customLipidPollTimer = null;
let _customLipidFailedName = null;

function openCustomLipidModal() {
  if (!state.taskId) {
    alert('Create or upload the task first. Custom lipid parameters must be bound to a task ID.');
    return;
  }
  const modal = document.getElementById('custom-lipid-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  // Reset state
  document.getElementById('custom-lipid-name').value = '';
  document.getElementById('custom-lipid-smiles').value = '';
  document.getElementById('custom-lipid-result').classList.add('hidden');
  document.getElementById('custom-lipid-confirm-btn').classList.add('hidden');
  document.getElementById('custom-lipid-build-status').classList.add('hidden');
  document.getElementById('custom-lipid-cancel-btn').disabled = false;
  _customLipidData = null;
  _customLipidFailedName = null;
}

function closeCustomLipidModal() {
  if (state.customLipidBusy) return;
  const modal = document.getElementById('custom-lipid-modal');
  if (modal) modal.classList.add('hidden');
  _customLipidData = null;
}

async function parseCustomLipid() {
  const nameEl = document.getElementById('custom-lipid-name');
  const smilesEl = document.getElementById('custom-lipid-smiles');
  const name = nameEl.value.trim();
  const smiles = smilesEl.value.trim();

  if (!name) { alert('Please enter a name for the lipid.'); return; }
  if (!smiles) { alert('Please enter a SMILES string.'); return; }

  try {
    const res = await fetch('/api/custom-lipid', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, smiles}),
    });
    if (!res.ok) {
      const err = await res.json();
      alert('Parse error: ' + (err.error || res.statusText));
      return;
    }
    const data = await res.json();
    _customLipidData = data;

    const exact = (data.registered_matches || []).find(match => match.match === 'exact');
    const confirmBtn = document.getElementById('custom-lipid-confirm-btn');
    if (exact) {
      confirmBtn.classList.add('hidden');
      alert('Submission rejected: this molecule already exists in the standard library as ' + exact.name + '. Custom submissions must be chemically distinct.');
    } else {
      confirmBtn.textContent = '\u2713 Build & Add Custom Lipid';
      const connectivity = (data.registered_matches || []).filter(match => match.match === 'connectivity');
      if (connectivity.length) {
        alert('A lipid with the same connectivity exists (' + connectivity.map(m => m.name).join(', ') + '), but stereochemistry differs. It will be treated as a new molecule.');
      }
    }

    // Show results
    document.getElementById('custom-lipid-result').classList.remove('hidden');
    if (!exact) document.getElementById('custom-lipid-confirm-btn').classList.remove('hidden');
    document.getElementById('cl-formula').textContent = data.formula;
    document.getElementById('cl-mass').textContent = data.mass.toFixed(1) + ' g/mol';
    document.getElementById('cl-category').textContent = data.category + ' (' + (data.headgroup || '') + ')';
    document.getElementById('cl-charge').textContent = (data.charge >= 0 ? '+' : '') + data.charge;
    document.getElementById('cl-apl').textContent = (data.area_per_lipid * 100).toFixed(1) + ' Å² (' + data.area_per_lipid.toFixed(3) + ' nm²)';
    document.getElementById('cl-dh').textContent = data.bilayer_thickness.toFixed(2) + ' nm';
    document.getElementById('cl-tails').textContent =
      'C' + data.tail1[0] + ':' + data.tail1[1] + ' / C' + data.tail2[0] + ':' + data.tail2[1];

    // Draw 2D structure with SmilesDrawer
    drawSmilesStructure(data.smiles);
  } catch (e) {
    alert('Failed to parse: ' + e.message);
  }
}

function drawSmilesStructure(smiles) {
  const canvas = document.getElementById('smiles-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (typeof SmilesDrawer === 'undefined') {
    window._cdmRetriesSD = (window._cdmRetriesSD || 0) + 1;
    if (window._cdmRetriesSD > 20) {
      ctx.fillStyle = '#c00'; ctx.font = '14px sans-serif';
      ctx.fillText('Structure renderer failed to load', 20, 100);
      return;
    }
    ctx.fillStyle = '#888'; ctx.font = '14px sans-serif';
    ctx.fillText('Structure renderer loading...', 20, 100);
    setTimeout(() => drawSmilesStructure(smiles), 800);
    return;
  }

  // Renderer is available — reset retry counter for future parses
  window._cdmRetriesSD = 0;

  try {
    // SmilesDrawer v2.x API: parse(smiles, successCb, errorCb)
    SmilesDrawer.parse(smiles, function(tree) {
      const drawer = new SmilesDrawer.Drawer({
        width: canvas.width,
        height: canvas.height,
        bondThickness: 1.4,
        shortBondLength: 0.82,
        bondSpacing: 0.18,
        terminalCarbons: true,
        explicitHydrogens: false,
        compactDrawing: true,
        fontSize: 12,
      });
      drawer.draw(tree, canvas, 'light', false);
    }, function(err) {
      ctx.fillStyle = '#888'; ctx.font = '14px sans-serif';
      ctx.fillText('(invalid SMILES)', 40, 100);
    });
  } catch (e) {
    ctx.fillStyle = '#888'; ctx.font = '14px sans-serif';
    ctx.fillText('(structure renderer error)', 40, 100);
  }
}

async function confirmCustomLipid() {
  if (_customLipidFailedName) {
    await retryCustomLipid(_customLipidFailedName);
    return;
  }
  if (!_customLipidData || !state.taskId) return;

  const data = _customLipidData;
  const exact = (data.registered_matches || []).find(match => match.match === 'exact');
  if (exact) {
    alert('This molecule is already present as ' + exact.name + ' and cannot be submitted as custom.');
    return;
  }
  const forceField = (document.getElementById('ff-protein') || {}).value || 'amber14sb';
  if (!forceField.startsWith('amber')) {
    alert('A new custom lipid currently requires the Amber + GAFF2 family. Select AMBER ff14SB in Force Field Selection, confirm it, then parse this lipid again. CHARMM/CGenFF and OPLS generators are not installed.');
    return;
  }
  const confirmBtn = document.getElementById('custom-lipid-confirm-btn');
  confirmBtn.disabled = true;
  confirmBtn.textContent = 'Submitting task-scoped calculation...';
  try {
    const build = await fetch('/api/task/' + state.taskId + '/custom-lipids', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name: data.name,
        smiles: data.canonical_smiles || data.smiles,
        force_field: forceField,
        lipid_ff: 'gaff2',
      }),
    });
    const result = await build.json();
    if (!build.ok) throw new Error(result.error || build.statusText);
    beginCustomLipidBlocking(result);
    await pollCustomLipid(result.name);
  } catch (error) {
    alert('Custom lipid submission failed: ' + error.message);
    confirmBtn.disabled = false;
    confirmBtn.textContent = '\u2713 Build & Add Custom Lipid';
  }
}

function addReadyTaskLipid(data) {
  const existing = _lipidPickerData.lipids.find(l => l.name === data.name);
  if (!existing) {
    _lipidPickerData.lipids.push({
      name: data.name,
      common_name: data.common_name,
      category: data.category,
      formula: data.formula,
      area_per_lipid: data.area_per_lipid,
      bilayer_thickness: data.bilayer_thickness,
      charge: data.charge,
      mass: data.mass,
      smiles: data.smiles,
      canonical_smiles: data.canonical_smiles,
      inchi_key: data.inchi_key,
      tail1: data.tail1,
      tail2: data.tail2,
      vdw_radius: data.vdw_radius,
      parameterizations: ['gaff2'],
      _custom: true,
      task_scoped: true,
    });
    if (!_lipidPickerData.categories['Custom Lipids']) {
      _lipidPickerData.categories['Custom Lipids'] = { lipids: [] };
    }
    if (!_lipidPickerData.categories['Custom Lipids'].lipids.includes(data.name)) {
      _lipidPickerData.categories['Custom Lipids'].lipids.push(data.name);
    }
  }
}

function beginCustomLipidBlocking(record) {
  state.customLipidBusy = true;
  _customLipidFailedName = null;
  const modal = document.getElementById('custom-lipid-modal');
  const status = document.getElementById('custom-lipid-build-status');
  if (modal) modal.classList.remove('hidden');
  if (status) status.classList.remove('hidden', 'failed');
  const cancel = document.getElementById('custom-lipid-cancel-btn');
  const confirm = document.getElementById('custom-lipid-confirm-btn');
  if (cancel) cancel.disabled = true;
  if (confirm) confirm.classList.add('hidden');
  renderCustomLipidStatus(record);
  updateNextButtonState();
  updateStepNavHighlight();
}

function renderCustomLipidStatus(record) {
  const progress = document.getElementById('custom-lipid-build-progress');
  const phase = document.getElementById('custom-lipid-build-phase');
  const message = document.getElementById('custom-lipid-build-message');
  if (progress) progress.style.width = Math.max(0, Math.min(100, Number(record.progress) || 0)) + '%';
  if (phase) phase.textContent = record.name + ' — ' + String(record.phase || record.state || '').replaceAll('_', ' ');
  if (message) message.textContent = record.message || '';
}

async function pollCustomLipid(name) {
  if (_customLipidPollTimer) clearTimeout(_customLipidPollTimer);
  try {
    const response = await fetch('/api/task/' + state.taskId + '/custom-lipids/' + encodeURIComponent(name));
    const record = await response.json();
    if (!response.ok) throw new Error(record.error || response.statusText);
    renderCustomLipidStatus(record);
    if (record.state === 'ready') {
      state.customLipidBusy = false;
      addReadyTaskLipid(record);
      selectLipid(record.name);
      const cancel = document.getElementById('custom-lipid-cancel-btn');
      if (cancel) { cancel.disabled = false; cancel.textContent = 'Close'; }
      const status = document.getElementById('custom-lipid-build-status');
      if (status) status.classList.remove('failed');
      updateNextButtonState();
      updateStepNavHighlight();
      setTimeout(loadTaskCustomLipids, 0);
      return;
    }
    if (record.state === 'failed') {
      state.customLipidBusy = true;
      _customLipidFailedName = record.name;
      const status = document.getElementById('custom-lipid-build-status');
      if (status) status.classList.add('failed');
      const confirm = document.getElementById('custom-lipid-confirm-btn');
      if (confirm) {
        confirm.classList.remove('hidden');
        confirm.disabled = false;
        confirm.textContent = '\u21bb Retry calculation';
      }
      return;
    }
    _customLipidPollTimer = setTimeout(function() { pollCustomLipid(name); }, 3000);
  } catch (error) {
    renderCustomLipidStatus({name, phase: 'connection error', progress: 0, message: error.message});
    _customLipidPollTimer = setTimeout(function() { pollCustomLipid(name); }, 5000);
  }
}

async function retryCustomLipid(name) {
  const response = await fetch('/api/task/' + state.taskId + '/custom-lipids/' + encodeURIComponent(name) + '/retry', {method: 'POST'});
  const record = await response.json();
  if (!response.ok) { alert(record.error || 'Retry failed'); return; }
  beginCustomLipidBlocking(record);
  pollCustomLipid(name);
}

async function loadTaskCustomLipids() {
  if (!state.taskId) return;
  try {
    const response = await fetch('/api/task/' + state.taskId + '/custom-lipids');
    if (!response.ok) return;
    const payload = await response.json();
    const records = payload.lipids || [];
    records.filter(r => r.state === 'ready').forEach(addReadyTaskLipid);
    const blocked = records.find(r => r.state !== 'ready');
    if (blocked) {
      beginCustomLipidBlocking(blocked);
      if (blocked.state === 'failed') {
        _customLipidFailedName = blocked.name;
        const status = document.getElementById('custom-lipid-build-status');
        if (status) status.classList.add('failed');
        const confirm = document.getElementById('custom-lipid-confirm-btn');
        if (confirm) {
          confirm.classList.remove('hidden');
          confirm.disabled = false;
          confirm.textContent = '\u21bb Retry calculation';
        }
      } else {
        pollCustomLipid(blocked.name);
      }
    }
  } catch (error) {
    console.warn('Could not restore task custom lipids', error);
  }
}

// Wire up modal buttons on page load
function initCustomLipidModal() {
  // Guard against double-initialisation (called from DOMContentLoaded)
  if (initCustomLipidModal._done) return;
  initCustomLipidModal._done = true;

  const parseBtn = document.getElementById('custom-lipid-parse-btn');
  const cancelBtn = document.getElementById('custom-lipid-cancel-btn');
  const confirmBtn = document.getElementById('custom-lipid-confirm-btn');

  if (parseBtn) parseBtn.addEventListener('click', parseCustomLipid);
  if (cancelBtn) cancelBtn.addEventListener('click', closeCustomLipidModal);
  if (confirmBtn) confirmBtn.addEventListener('click', confirmCustomLipid);

  // Close on overlay click
  const modal = document.getElementById('custom-lipid-modal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal && !state.customLipidBusy) closeCustomLipidModal();
    });
  }

  // Enter key in SMILES input triggers parse
  const smilesEl = document.getElementById('custom-lipid-smiles');
  if (smilesEl) {
    smilesEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') parseCustomLipid();
    });
  }
}

/** Generate a clean schematic SVG for a lipid. */
function lipidSchematicSVG(lipid) {
  const W = 90, H = 62;
  const headColors = {
    PC:'#4f46e5', PE:'#0891b2', PG:'#ea580c', PS:'#7c3aed',
    PA:'#dc2626', PI:'#ca8a04', SM:'#2563eb', ST:'#16a34a',
    PIP:'#9333ea', CL:'#db2777', LPC:'#6366f1', LPE:'#06b6d4',
    DG:'#78716c', CER:'#a16207', MGDG:'#22c55e', DGDG:'#15803d',
    GM1:'#d946ef',
  };
  const hc = headColors[lipid.category] || '#64748b';

  const t1Len = lipid.tail1[0] || 0;
  const t1Unsat = lipid.tail1[1] || 0;
  const t2Len = lipid.tail2[0] || 0;
  const t2Unsat = lipid.tail2[1] || 0;
  const isSterol = lipid.category === 'ST';
  const isSingleTail = ['LPC','LPE','SM','CER'].includes(lipid.category);
  const isGlycolipid = ['MGDG','DGDG','GM1'].includes(lipid.category);

  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<rect width="${W}" height="${H}" fill="#f8fafc" rx="4"/>`;

  if (isSterol) {
    // Sterol: 4-ring steroid backbone schematic
    svg += `<rect x="22" y="14" width="8" height="8" fill="none" stroke="${hc}" stroke-width="1.2" rx="1"/>`;
    svg += `<rect x="30" y="14" width="8" height="8" fill="none" stroke="${hc}" stroke-width="1.2" rx="1"/>`;
    svg += `<rect x="38" y="14" width="8" height="8" fill="none" stroke="${hc}" stroke-width="1.2" rx="1"/>`;
    svg += `<rect x="26" y="22" width="8" height="8" fill="none" stroke="${hc}" stroke-width="1.2" rx="1"/>`;
    svg += `<circle cx="55" cy="26" r="3" fill="${hc}" opacity="0.4"/>`;
    svg += `<circle cx="22" cy="18" r="4" fill="${hc}" opacity="0.7"/>`;
    svg += `<text x="45" y="48" text-anchor="middle" font-size="7" fill="#64748b">${lipid.name}</text>`;
  } else if (isSingleTail) {
    // Sphingolipids / lyso lipids: single tail from backbone
    svg += `<circle cx="20" cy="24" r="8" fill="${hc}" opacity="0.35"/>`;
    svg += `<text x="20" y="27" text-anchor="middle" font-size="6" font-weight="bold" fill="${hc}">${lipid.category}</text>`;
    svg += `<line x1="28" y1="20" x2="42" y2="10" stroke="#475569" stroke-width="1.5"/>`;
    svg += `<line x1="42" y1="10" x2="${42 + Math.max(t1Len*0.6, 8)}" y2="8" stroke="#475569" stroke-width="1.2"/>`;
    if (t1Len > 0) svg += `<text x="45" y="46" text-anchor="middle" font-size="6.5" fill="#94a3b8">C${t1Len}:${t1Unsat}</text>`;
  } else if (isGlycolipid) {
    // Glycolipids: sugar headgroup + two tails
    svg += `<circle cx="16" cy="20" r="6" fill="${hc}" opacity="0.25"/>`;
    svg += `<circle cx="22" cy="18" r="4" fill="${hc}" opacity="0.15"/>`;
    svg += `<text x="19" y="35" text-anchor="middle" font-size="6" fill="${hc}">${lipid.category}</text>`;
    svg += `<line x1="24" y1="24" x2="36" y2="16" stroke="#94a3b8" stroke-width="1"/>`;
    svg += `<line x1="24" y1="24" x2="36" y2="28" stroke="#94a3b8" stroke-width="1"/>`;
    const t1w = Math.max(6, Math.min(t1Len*1.0, 30));
    const t2w = Math.max(6, Math.min(t2Len*1.0, 30));
    svg += `<line x1="36" y1="14" x2="${36+t1w}" y2="11" stroke="#475569" stroke-width="1.3"/>`;
    svg += `<line x1="36" y1="28" x2="${36+t2w}" y2="30" stroke="#475569" stroke-width="1.3"/>`;
    svg += `<text x="40" y="46" text-anchor="start" font-size="6" fill="#94a3b8">${lipid.formula.substring(0,15)}...</text>`;
  } else {
    // Glycerophospholipid: headgroup circle + glycerol + two tails
    svg += `<circle cx="18" cy="22" r="8" fill="${hc}" opacity="0.35"/>`;
    svg += `<text x="18" y="25" text-anchor="middle" font-size="7" font-weight="bold" fill="${hc}">${lipid.category}</text>`;

    // Glycerol backbone
    svg += `<line x1="26" y1="22" x2="36" y2="18" stroke="#94a3b8" stroke-width="1"/>`;
    svg += `<line x1="26" y1="22" x2="36" y2="28" stroke="#94a3b8" stroke-width="1"/>`;

    // Tail 1 (upper)
    const t1w = Math.max(6, Math.min(t1Len * 1.2, 36));
    svg += `<line x1="36" y1="16" x2="${36 + t1w}" y2="13" stroke="#475569" stroke-width="1.5"/>`;
    if (t1Unsat > 0) {
      const bendX = 36 + t1w * 0.55;
      svg += `<line x1="${bendX}" y1="13" x2="${bendX + 5}" y2="10" stroke="#475569" stroke-width="1.5"/>`;
    }

    // Tail 2 (lower)
    const t2w = Math.max(6, Math.min(t2Len * 1.2, 36));
    svg += `<line x1="36" y1="28" x2="${36 + t2w}" y2="30" stroke="#475569" stroke-width="1.5"/>`;
    if (t2Unsat > 0) {
      const bendX = 36 + t2w * 0.55;
      svg += `<line x1="${bendX}" y1="30" x2="${bendX + 5}" y2="27" stroke="#475569" stroke-width="1.5"/>`;
    }

    // Tail labels
    svg += `<text x="42" y="46" text-anchor="middle" font-size="6.5" fill="#94a3b8">C${t1Len}:${t1Unsat} / C${t2Len}:${t2Unsat}</text>`;
  }

  // Formula below
  svg += `<text x="45" y="57" text-anchor="middle" font-size="6" fill="#cbd5e1">${lipid.formula}</text>`;
  svg += `</svg>`;
  return svg;
}

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

function renderInputCheckReport(repair, errorMessage, modificationReport, nucleicAcids) {
  var panel = document.getElementById('input-check-report');
  if (!panel) return;
  panel.replaceChildren();
  panel.classList.remove('hidden', 'error');

  var heading = document.createElement('h4');
  if (errorMessage) {
    panel.classList.add('error');
    heading.textContent = '\u2717 Input requires review';
    panel.appendChild(heading);
    var errorText = document.createElement('p');
    errorText.textContent = errorMessage;
    panel.appendChild(errorText);
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
      'generation will validate termini, polymer bonds, hydrogens, and charge in Step 3.';
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
    'New heavy atoms were constructed and checked against the selected force field\'s ' +
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
  });
  table.appendChild(body);
  panel.appendChild(table);
}

// ---- Generic Check button behavior ----
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
    _stepRunning = true;

    // Input step: first filter PDB by chain/molecule selections
    if (stepName === 'input' && !isCoarseGrainedWorkflow()) {
      var includedChains = [];
      for (var ch in _chainState) {
        if (_chainState[ch].included) includedChains.push(ch);
      }
      // Also include chains from small molecules (they may reside in
      // chains that have no protein residues, e.g. ligand in chain A
      // while protein is in chain B — without this the filter-pdb
      // endpoint deletes the ligand because its chain isn't in the list).
      var smols = (state.pdbInfo && state.pdbInfo.small_molecules) || [];
      smols.forEach(function(m) {
        if (_smallMolState && _smallMolState[m.resname] && !_smallMolState[m.resname].included) return;
        if (m.chain && includedChains.indexOf(m.chain) < 0) includedChains.push(m.chain);
      });
      // Collect excluded molecule resnames: always-strip (water, ions) +
      // user-unchecked small molecules
      var excludedResn = ['HOH', 'SOL', 'WAT', 'TIP', 'TIP3', 'SPC', 'SPCE', 'NA', 'CL', 'K', 'CA', 'ZN', 'MG'];
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
    var result = await _apiFetch('/api/step/' + state.taskId + '/' + stepName, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });
    if (result.status === 'ok') {
      if (stepName !== 'cg_system') _checkedSteps.add(stepName);
      // Snapshot the config that was just validated (prevents drift when user
      // changes DOM values between "Check" and "Build")
      _checkedConfig = _checkedConfig || {};
      _checkedConfig[stepName] = cfg;
      if (statusEl) {
        statusEl.textContent = stepName === 'cg_system'
          ? '✓ Quality gates passed; inspect and confirm below'
          : '✓ Checked (' + (result.elapsed_s != null ? result.elapsed_s : '?') + 's)';
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
        var inputModificationReport = (result.metrics || {}).input_modifications || {};
        var standardizedSequences = (result.metrics || {}).input_sequences || [];
        if (standardizedSequences.length && state.pdbInfo) {
          state.pdbInfo.sequences = standardizedSequences;
          loadProcResidues();
        }
        setInputModificationReport(inputModificationReport);
        renderInputCheckReport(
          (result.metrics || {}).input_repair || {}, null, inputModificationReport,
          (result.metrics || {}).input_nucleic_acids || []
        );
        // Reload viewer PDB from the saved checkpoint to confirm chain filter
        var vpdb = await _loadStepViewerPdb('input');
        if (vpdb) {
          // Update pdb_content and box_nm in state so subsequent
          // viewers use the module-validated checkpoint data.
          // (The upload response may contain an unreasonable CRYST1 box;
          // the module validates and fixes it.)
          if (state.pdbInfo) {
            state.pdbInfo.pdb_content = vpdb;
            // Parse CRYST1 from viewer.pdb to get the validated box
            var crystMatch = vpdb.match(/^CRYST1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/m);
            if (crystMatch) {
              state.pdbInfo.box_nm = [
                parseFloat(crystMatch[1]) / 10.0,
                parseFloat(crystMatch[2]) / 10.0,
                parseFloat(crystMatch[3]) / 10.0,
              ];
            }
          }
        }
        redrawPDBViewerWithChainFilter();
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
        if (stepName === 'cg_mapping') {
          var extent = ((result.metrics || {}).cg_mapping || {}).protein_extent_nm || [];
          if (extent.length === 3) {
            var recommendedXY = Math.ceil((Math.max(Number(extent[0]), Number(extent[1])) + 3.0) * 2) / 2;
            var recommendedZ = Math.ceil((Number(extent[2]) + 3.0) * 2) / 2;
            var boxXY = document.getElementById('cg-box-xy');
            var boxZ = document.getElementById('cg-box-z');
            if (boxXY && Number(boxXY.value) < recommendedXY) boxXY.value = String(recommendedXY);
            if (boxZ && Number(boxZ.value) < recommendedZ) boxZ.value = String(recommendedZ);
            if (statusEl) statusEl.textContent += ' — minimum safe box suggested: ' + recommendedXY.toFixed(1) + ' × ' + recommendedXY.toFixed(1) + ' × ' + recommendedZ.toFixed(1) + ' nm';
          }
        }
        var cgViewerRendered = await renderCoarseGrainedViewer(stepName);
        if (cgConfirmation) {
          cgConfirmation.disabled = cgViewerRendered !== true;
          if (!cgViewerRendered && statusEl) {
            statusEl.textContent = '✓ Quality gates passed; WebGL viewer unavailable, so confirmation is disabled in this browser';
            statusEl.style.color = '#d97706';
          }
        }
      }
    } else {
      if (statusEl) { statusEl.textContent = '✗ ' + (result.error || 'Failed'); statusEl.style.color = '#dc2626'; }
      if (stepName === 'input') renderInputCheckReport(null, result.error || 'Input check failed');
      if (stepName === 'structure') renderModificationGeometryReport([]);
    }
  } catch(e) {
    if (statusEl) { statusEl.textContent = '✗ ' + (e.message || 'Network error'); statusEl.style.color = '#dc2626'; }
    if (stepName === 'input') renderInputCheckReport(null, e.message || 'Network error');
    if (stepName === 'structure') renderModificationGeometryReport([]);
  } finally {
    _stepRunning = false;
    if (btn) btn.disabled = false;
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
  var chainSet = new Set();
  var chainColors = {};
  (pdbContent || '').split('\n').forEach(function(l) {
    if (l.indexOf('ATOM') === 0 || l.indexOf('HETATM') === 0) {
      var ch = (l.substring(21,22)||' ').trim();
      if (ch) chainSet.add(ch);
    }
  });
  var ci = 0;
  if (chainSet.size > 0) {
    chainSet.forEach(function(ch) {
      var macaron = (window.GMX && window.GMX.MACARON) || ['0xd4a5c7','0xa5c7d4','0xc7d4a5'];
      var color = macaron[ci % macaron.length];
      if (onlyChains && !onlyChains.has(ch)) {
        viewer.setStyle({chain: ch}, {cartoon: {hidden: true}, line: {hidden: true}});
      } else {
        viewer.setStyle({chain: ch}, {cartoon: {color: color, style: 'trace', thickness: 0.28}});
        chainColors[ch] = color;
        ci++;
      }
    });
  } else {
    viewer.setStyle({}, {cartoon: {color: 'spectrum', style: 'trace', thickness: 0.28}});
  }
  window._pdbChainColors = chainColors;
  // ---- small-molecule chain-awareness ----
  // Visibility is determined by two factors:
  //   1. The molecule's own checkbox (_smallMolState)
  //   2. Its parent chain — BUT only when that chain contains protein.
  //      Small molecules in their own chain (no protein residues) are
  //      controlled solely by their checkbox; they are not tied to any
  //      protein chain toggle.
  var _smVis = (state.pdbInfo && state.pdbInfo.small_molecules) || [];
  _smVis.forEach(function(m) {
    var _molOk = !_smallMolState[m.resname] || _smallMolState[m.resname].included;
    var _isProtChain = _chainState.hasOwnProperty(m.chain);
    // Only gate on chain inclusion when the chain actually has protein
    var _chOk = !onlyChains || !_isProtChain || onlyChains.has(m.chain);
    if (!_chOk || !_molOk) {
      viewer.setStyle({chain: m.chain, resn: m.resname}, {
        stick: {hidden: true}, sphere: {hidden: true},
        line: {hidden: true}, cartoon: {hidden: true}
      });
    } else {
      viewer.setStyle({chain: m.chain, resn: m.resname}, {});
    }
  });

  colorSmallMolecules(viewer, onlyChains);
  viewer.addStyle({resn: ['NA','CL','K','CA','ZN','MG']}, {sphere: {radius: 0.3, color: '0x94a3b8'}});
  viewer.addStyle({resn: ['HOH','SOL','WAT','TIP','TIP3']}, {sphere: {radius: 0.08, opacity: 0.15}});
}

function initOrientationStep() {
  // ====================================================================
  // Shared viewer — single 3Dmol instance on #orient-viewer.
  // Both Auto and Manual tabs use the same viewer.
  // ====================================================================
  function _createOrientViewer() {
    if (window._orientViewer) return;
    var el = document.getElementById('orient-viewer');
    if (!el || typeof $3Dmol === 'undefined') return;
    window._orientViewer = $3Dmol.createViewer(el, {
      backgroundColor: '0xffffff', antialias: true,
    });
    window._orientViewer.setBackgroundColor('0xffffff');
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
    _createOrientViewer();
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
      _stepRunning = true;
      try {
        if (statusEl) { statusEl.textContent = 'Running...'; statusEl.style.color = '#d97706'; }
        checkBtn.disabled = true;

        if (statusEl) { statusEl.textContent = 'Running orientation...'; }
        var orientConfig = buildModuleConfig().orient || { method: 'ppm' };
        var resp = await fetch('/api/step/' + state.taskId + '/orient', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: orientConfig }),
        });
        var result = await resp.json();
        if (result.status === 'ok') {
          _checkedSteps.add('orient');
          _checkedConfig = _checkedConfig || {};
          _checkedConfig.orient = orientConfig;
          var quality = (result.metrics && result.metrics.orientation_quality) || {};
          var warnings = Array.isArray(quality.warnings) ? quality.warnings : [];
          renderOrientationQuality(quality);
          if (statusEl) {
            statusEl.textContent = warnings.length
              ? '⚠ Checked with ' + warnings.length + ' scientific warning(s)'
              : '✓ Oriented (' + (result.elapsed_s != null ? result.elapsed_s : '?') + 's)';
            statusEl.style.color = warnings.length ? '#d97706' : '#059669';
          }
          updateNextButtonState();
          // The saved checkpoint is authoritative. Reload it even though the
          // preview used the same backend module, so Check visibly confirms
          // the exact coordinates that Step 5 will consume.
          _createOrientViewer();
          var loaded = await window._loadOrientationCheckpointPreview();
          if (!loaded && statusEl) {
            statusEl.textContent = '⚠ Check saved, but the saved viewer could not be reloaded';
            statusEl.style.color = '#d97706';
          }
        } else {
          if (statusEl) { statusEl.textContent = '✗ ' + (result.error || 'Failed'); statusEl.style.color = '#dc2626'; }
        }
      } catch(e) {
        if (statusEl) { statusEl.textContent = '✗ Network error'; statusEl.style.color = '#dc2626'; }
      } finally {
        _stepRunning = false;
        checkBtn.disabled = false;
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

    if (window._createOrientViewer) window._createOrientViewer();
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
// ===================================================================
// Structure Processing — Protonation + Termini + Modifications
// ===================================================================

let _procResidues = [];         // [{resname, chain, resid, index}]
let _procChains = [];           // sorted unique chain IDs
let _procAssignments = [];     // protonation results
// Authoritative fields are index + patch_id.  product_name and charge_shift
// are presentation metadata derived from the server-side, force-field-specific
// patch catalogue and must never be submitted as scientific input.
let _procModifications = [];   // [{index, patch_id, product_name, charge_shift}]
let _procCrosslinks = [];      // [{type:'disulfide', first_index, second_index}]
let _procTermini = {};          // { chain_id: { nter: 'ACE'|'', cter: 'NME'|'' } }
let _procPatchCatalog = [];
let _procCapCapabilities = {};
let _procCrosslinkCapabilities = {};
let _inputModificationReport = {detected: 0, recognized: 0, records: [], warnings: []};
let _inputModificationCapabilityWarnings = [];

function selectedProteinForceField() {
  return document.getElementById('ff-protein')?.value || 'charmm36';
}
let _procSelectedIdx = -1;
let _protonationComputed = false;  // blocks Next until true
let _protonationRunning = false;
let _protonationRequestId = 0;
let _computedProtonationInput = null;
let _lastProtonationResult = null;
let _systemPH = 7.0;               // recorded for later simulation setup
let _waterVolume = 0;               // nm³ — computed in solvation step
let _waterCount = 0;               // number of water molecules
let _solvChecked = false;

function initStructureProcessing() {
  // Tab switching
  document.querySelectorAll('.structproc-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.structproc-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const tabId = tab.dataset.tab;
      document.getElementById('structproc-protonation').classList.toggle('hidden', tabId !== 'protonation');
      document.getElementById('structproc-termini').classList.toggle('hidden', tabId !== 'termini');
      document.getElementById('structproc-modifications').classList.toggle('hidden', tabId !== 'modifications');
      if (tabId === 'modifications') renderModSequences();
      if (tabId === 'termini') renderTerminiTab();
    });
  });

  // pH input
  const phInput = document.getElementById('proc-pH');
  if (phInput) {
    const validatePHInput = (formatValue) => {
      let v = Number(phInput.value);
      if (!Number.isFinite(v) || v < 1.0 || v > 13.0) {
        invalidateProtonationState('Target pH must be between 1.0 and 13.0; enter a valid value and click Compute.');
        return false;
      }
      const changed = v !== _systemPH;
      _systemPH = v;
      if (formatValue) phInput.value = v.toFixed(1);
      if (changed) invalidateProtonationState('pH changed - click Compute to recalculate protonation.');
      return true;
    };
    phInput.addEventListener('input', () => validatePHInput(false));
    phInput.addEventListener('blur', () => validatePHInput(true));
    phInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        runProtonation();
      }
    });
    const initialPH = Number(phInput.value);
    _systemPH = Number.isFinite(initialPH) && initialPH >= 1.0 && initialPH <= 13.0 ? initialPH : 7.0;
    phInput.value = _systemPH.toFixed(1);
  }

  const hisSel = document.getElementById('proc-his-tautomer');
  if (hisSel) hisSel.addEventListener('change', () => {
    invalidateProtonationState('Histidine preference changed - click Compute to recalculate protonation.');
  });

  const runBtn = document.getElementById('proc-run-btn');
  if (runBtn) runBtn.addEventListener('click', runProtonation);

  const patchCancel = document.getElementById('proc-patch-cancel');
  const disulfideAdd = document.getElementById('proc-disulfide-add');
  if (disulfideAdd) disulfideAdd.addEventListener('click', addDisulfideCrosslink);
  // Skip protonation checkbox
  const skipCb = document.getElementById("proc-skip-protonation");
  if (skipCb) {
    skipCb.addEventListener("change", () => {
      invalidateProtonationState(
        skipCb.checked
          ? 'Protonation will be skipped. Run Check Structure to confirm this choice.'
          : 'Protonation enabled - click Compute before checking the structure.'
      );
      if (skipCb.checked) {
        _protonationComputed = true;
        document.getElementById('proc-chain-tables').innerHTML = '<p class="hint">Protonation skipped — original residue names and charges preserved.</p>';
      }
      updateNextButtonState();
    });
  }
  if (patchCancel) patchCancel.addEventListener('click', closePatchPicker);
}

function loadProcResidues() {
  if (!state.pdbInfo || !state.pdbInfo.sequences) return;
  _procResidues = [];
  _procModifications = [];
  _procCrosslinks = [];
  _procTermini = {};
  _procPatchCatalog = [];
  _procCapCapabilities = {};
  _procCrosslinkCapabilities = {};
  _lastProtonationResult = null;
  const seqs = state.pdbInfo.sequences || [];
  seqs.forEach(chain => {
    const ch = chain.chain_id || '';
    (chain.residues || []).forEach(res => {
      _procResidues.push({
        resname: res.resname, chain: ch, resid: res.resid,
        index: _procResidues.length,
      });
    });
  });
  // Unique sorted chains
  _procChains = [...new Set(_procResidues.map(r => r.chain))].sort();
  // Explicit cap atoms/bonds are not implemented yet. Use the supported
  // zwitterionic terminal chemistry by default instead of submitting hidden
  // ACE/NME values from disabled controls.
  _procChains.forEach(ch => {
    _procTermini[ch] = { nter: '', cter: '' };
  });
  // Load termini and modifications data (rendered on tab switch).
  // PROPKA is NOT auto-triggered here — that happens once when the
  // user navigates to the structure step (goToWizardStep), avoiding
  // duplicate runs from upload→loadProcResidues + step→auto trigger.
  if (_procResidues.length) {
    renderTerminiTab();
    renderModSequences();
    reloadModificationCatalog();
    reloadCrosslinkCapabilities();
    const forceField = encodeURIComponent(selectedProteinForceField());
    fetch('/api/terminal-capabilities?force_field=' + forceField)
      .then(function(response) { return response.json(); })
      .then(function(capabilities) {
        _procCapCapabilities = capabilities && !capabilities.error ? capabilities : {};
        renderTerminiTab();
      })
      .catch(function() { _procCapCapabilities = {}; renderTerminiTab(); });
  }
}

function restoreStructureProcessingConfig(savedConfig) {
  if (!savedConfig || typeof savedConfig !== 'object' || !_procResidues.length) return;
  if (savedConfig.termini && typeof savedConfig.termini === 'object') {
    _procChains.forEach(function(ch) {
      const saved = savedConfig.termini[ch];
      if (saved && typeof saved === 'object') {
        _procTermini[ch] = {
          nter: String(saved.nter || '').toUpperCase(),
          cter: String(saved.cter || '').toUpperCase(),
        };
      }
    });
  }
  if (Array.isArray(savedConfig.modifications)) {
    _procModifications = savedConfig.modifications.filter(function(mod) {
      return mod && Number.isInteger(mod.index) && mod.index >= 0 &&
        mod.index < _procResidues.length && typeof mod.patch_id === 'string';
    }).map(function(mod) {
      return {
        index: mod.index,
        patch_id: mod.patch_id,
        product_name: mod.product_name || '',
      };
    });
  }
  if (Array.isArray(savedConfig.crosslinks)) {
    _procCrosslinks = savedConfig.crosslinks.filter(function(item) {
      return item && item.type === 'disulfide' &&
        Number.isInteger(item.first_index) && Number.isInteger(item.second_index) &&
        item.first_index >= 0 && item.second_index >= 0 &&
        item.first_index < _procResidues.length && item.second_index < _procResidues.length &&
        item.first_index !== item.second_index;
    }).map(function(item) {
      return {
        type: 'disulfide',
        first_index: item.first_index,
        second_index: item.second_index,
      };
    });
  }
  if (Array.isArray(savedConfig.protonation)) {
    _procAssignments = [];
    savedConfig.protonation.forEach(function(assignment) {
      if (!assignment || !Number.isInteger(assignment.index)) return;
      _procAssignments[assignment.index] = Object.assign(
        {is_titratable: true}, assignment
      );
    });
  }
  const skip = document.getElementById('proc-skip-protonation');
  if (skip && typeof savedConfig.skip_protonation === 'boolean') {
    skip.checked = savedConfig.skip_protonation;
  }
  renderTerminiTab();
  renderModSequences();
  renderDisulfideControls();
  applyModification(-1, '', '');
}

function setInputModificationReport(report) {
  const safe = report && typeof report === 'object' ? report : {};
  _inputModificationReport = {
    detected: Number(safe.detected || 0),
    recognized: Number(safe.recognized || 0),
    records: Array.isArray(safe.records) ? safe.records : [],
    warnings: Array.isArray(safe.warnings) ? safe.warnings : [],
  };
  _inputModificationReport.records.forEach(function(record) {
    if (!record || !record.normalized || !record.standard_resname) return;
    var index = Number(record.residue_index);
    var residue = Number.isInteger(index) ? _procResidues[index] : null;
    if (!residue || Number(residue.resid) !== Number(record.resid) ||
        String(residue.chain || '?') !== String(record.chain || '?')) {
      residue = _procResidues.find(function(candidate) {
        return Number(candidate.resid) === Number(record.resid) &&
          String(candidate.chain || '?') === String(record.chain || '?');
      });
    }
    if (residue) residue.resname = String(record.standard_resname).toUpperCase();
  });
  applyDetectedInputModifications();
  renderDetectedModificationNotice();
  renderModSequences();
}

async function reloadModificationCatalog() {
  const forceField = encodeURIComponent(selectedProteinForceField());
  try {
    const response = await fetch('/api/patches?force_field=' + forceField);
    const patches = await response.json();
    if (!response.ok || !Array.isArray(patches)) throw new Error('Patch catalogue unavailable');
    _procPatchCatalog = patches;
    hydrateModificationMetadata();
    applyDetectedInputModifications();
    applyModification(-1, '', '');
  } catch (error) {
    _procPatchCatalog = [];
    _inputModificationCapabilityWarnings = [
      'The force-field modification catalogue could not be loaded; uploaded modifications were not auto-selected.'
    ];
    renderDetectedModificationNotice();
    renderModSequences();
  }
}

async function reloadCrosslinkCapabilities() {
  const forceField = encodeURIComponent(selectedProteinForceField());
  try {
    const response = await fetch('/api/crosslink-capabilities?force_field=' + forceField);
    const capabilities = await response.json();
    _procCrosslinkCapabilities = response.ok && !capabilities.error ? capabilities : {};
  } catch (_error) {
    _procCrosslinkCapabilities = {};
  }
  renderDisulfideControls();
}

function applyDetectedInputModifications() {
  _procModifications = _procModifications.filter(function(mod) {
    return mod.source !== 'input-detection';
  });
  _inputModificationCapabilityWarnings = [];
  if (!_procPatchCatalog.length) return;
  (_inputModificationReport.records || []).forEach(function(record) {
    if (!record || record.status !== 'recognized' || !record.patch_id) return;
    const patch = _procPatchCatalog.find(function(item) {
      return item && item.id === record.patch_id;
    });
    const location = (record.chain || '?') + ':' + record.resid + ' ' +
      record.original_resname;
    if (!patch) {
      _inputModificationCapabilityWarnings.push(
        location + ': recorded patch ' + record.patch_id + ' is absent from the installed catalogue.'
      );
      return;
    }
    if (patch.supported === false) {
      _inputModificationCapabilityWarnings.push(
        location + ': ' + record.patch_id + ' cannot be restored with ' +
        selectedProteinForceField() + ' — ' + (patch.support_reason || 'no validated topology is installed') + '.'
      );
      return;
    }
    const index = Number(record.residue_index);
    if (!Number.isInteger(index) || index < 0 || index >= _procResidues.length) {
      _inputModificationCapabilityWarnings.push(
        location + ': the standardized residue index could not be matched; select the modification manually.'
      );
      return;
    }
    const existing = _procModifications.find(function(mod) {
      return mod.index === index && mod.source !== 'input-detection';
    });
    if (existing) return;
    _procModifications = _procModifications.filter(function(mod) {
      return mod.index !== index;
    });
    _procModifications.push({
      index: index,
      patch_id: patch.id,
      product_name: patch.product_name || '',
      charge_shift: patch.charge_shift,
      source: 'input-detection',
    });
  });
  renderDetectedModificationNotice();
}

function renderDetectedModificationNotice() {
  const notice = document.getElementById('proc-upload-modification-notice');
  const tab = document.querySelector('.structproc-tab[data-tab="modifications"]');
  const records = _inputModificationReport.records || [];
  if (tab) tab.classList.toggle('detected-attention', records.length > 0);
  if (!notice) return;
  notice.replaceChildren();
  if (!records.length) {
    notice.classList.add('hidden');
    return;
  }
  notice.classList.remove('hidden');
  const heading = document.createElement('h4');
  heading.textContent = '\u26a0 Modified residues were detected in the uploaded protein';
  notice.appendChild(heading);
  const guidance = document.createElement('p');
  guidance.textContent =
    'Recognized residues were converted to their standard parents for safe processing. ' +
    'Open the Modifications tab and verify that every automatically selected type and site matches the uploaded structure.';
  notice.appendChild(guidance);
  const warnings = (_inputModificationReport.warnings || []).concat(
    _inputModificationCapabilityWarnings || []
  );
  if (warnings.length) {
    const list = document.createElement('ul');
    warnings.forEach(function(message) {
      const item = document.createElement('li');
      item.textContent = message;
      list.appendChild(item);
    });
    notice.appendChild(list);
  }
}

function hydrateModificationMetadata() {
  if (!Array.isArray(_procPatchCatalog) || !_procPatchCatalog.length) return;
  _procModifications = _procModifications.map(function(mod) {
    const patch = _procPatchCatalog.find(function(item) {
      return item && item.id === mod.patch_id;
    });
    if (!patch) return mod;
    return {
      index: mod.index,
      patch_id: mod.patch_id,
      product_name: patch.product_name || '',
      charge_shift: patch.charge_shift,
      source: mod.source,
    };
  });
}

function serializeStructureModifications() {
  return _procModifications.map(function(mod) {
    return { index: mod.index, patch_id: mod.patch_id };
  });
}

function serializeStructureCrosslinks() {
  return _procCrosslinks.map(function(item) {
    return {
      type: 'disulfide',
      first_index: item.first_index,
      second_index: item.second_index,
    };
  });
}

// -------------------------------------------------------------------
// Protonation
// -------------------------------------------------------------------

function invalidateProtonationState(message) {
  _protonationRequestId += 1;  // makes any in-flight response stale
  _protonationRunning = false;
  _protonationComputed = false;
  _computedProtonationInput = null;
  _procAssignments = [];
  var runButton = document.getElementById('proc-run-btn');
  var checkButton = document.getElementById('structure-check-btn');
  if (runButton) runButton.disabled = false;
  if (checkButton) checkButton.disabled = false;
  var structureIndex = state.wizardSteps.indexOf('structure');
  if (structureIndex >= 0) {
    for (var i = structureIndex; i < state.wizardSteps.length; i++) {
      state.completedSteps.delete(i);
      _checkedSteps.delete(state.wizardSteps[i]);
      if (_checkedConfig) delete _checkedConfig[state.wizardSteps[i]];
    }
  }
  var statusEl = document.getElementById('proc-propka-status');
  if (statusEl && message) {
    statusEl.classList.remove('hidden');
    statusEl.textContent = message;
    statusEl.style.color = '#d97706';
  }
  var checkStatus = document.getElementById('structure-check-status');
  if (checkStatus) checkStatus.textContent = '';
  updateNextButtonState();
  updateStepNavHighlight();
}

function validateStructureProtonationReady() {
  var skip = document.getElementById('proc-skip-protonation');
  if (skip && skip.checked) return '';
  if (_protonationRunning) return 'Protonation is still computing. Wait for it to finish before checking.';
  if (!_protonationComputed) return 'Run Compute after choosing the target pH and histidine state.';
  var displayedPH = Number(document.getElementById('proc-pH')?.value);
  var displayedHis = document.getElementById('proc-his-tautomer')?.value || 'HSE';
  if (!Number.isFinite(displayedPH) || displayedPH < 1.0 || displayedPH > 13.0) {
    return 'Target pH must be a number between 1.0 and 13.0.';
  }
  if (!_computedProtonationInput || displayedPH !== _computedProtonationInput.pH ||
      displayedHis !== _computedProtonationInput.his) {
    return 'The displayed pH or histidine preference differs from the last calculation. Run Compute again.';
  }
  var titratableNames = new Set(['HIS', 'ASP', 'GLU', 'CYS', 'LYS', 'TYR']);
  var assigned = new Set(
    _procAssignments.filter(function(a) { return a && a.is_titratable; })
      .map(function(a) { return Number(a.index); })
  );
  var missing = _procResidues.filter(function(r) {
    return titratableNames.has(String(r.resname || '').toUpperCase()) && !assigned.has(r.index);
  });
  if (missing.length) {
    var preview = missing.slice(0, 6).map(function(r) {
      return (r.chain || '?') + ':' + r.resid + ' ' + r.resname;
    }).join(', ');
    if (missing.length > 6) preview += ', ...';
    return 'Protonation results are incomplete (' + preview + '). Run Compute again.';
  }
  return '';
}

async function runProtonation() {
  const phInput = document.getElementById('proc-pH');
  const pH = Number(phInput ? phInput.value : NaN);
  const his = document.getElementById('proc-his-tautomer')?.value || 'HSE';
  const residues = _procResidues.map(r => r.resname);
  const taskIdParam = state.taskId || '';
  const statusEl = document.getElementById('proc-propka-status');
  const runBtn = document.getElementById('proc-run-btn');
  const checkBtn = document.getElementById('structure-check-btn');
  const checkStatusEl = document.getElementById('structure-check-status');

  if (!Number.isFinite(pH) || pH < 1.0 || pH > 13.0) {
    invalidateProtonationState('Target pH must be a number between 1.0 and 13.0; no calculation was run.');
    if (phInput) phInput.focus();
    return;
  }
  if (!residues.length) {
    invalidateProtonationState('No protein residues are available for protonation.');
    return;
  }
  if (_protonationRunning) {
    if (statusEl) {
      statusEl.classList.remove('hidden');
      statusEl.textContent = 'Protonation is already computing. Please wait.';
      statusEl.style.color = '#d97706';
    }
    return;
  }

  const requestId = ++_protonationRequestId;
  _systemPH = pH;
  _protonationRunning = true;
  _protonationComputed = false;
  _computedProtonationInput = null;
  _procAssignments = [];
  _checkedSteps.delete('structure');
  if (_checkedConfig) delete _checkedConfig.structure;
  if (checkStatusEl) checkStatusEl.textContent = '';
  if (runBtn) runBtn.disabled = true;
  if (checkBtn) checkBtn.disabled = true;
  if (statusEl) {
    statusEl.classList.remove('hidden');
    statusEl.textContent = 'Computing environment-sensitive protonation states...';
    statusEl.style.color = '#d97706';
  }
  updateNextButtonState();

  try {
    document.getElementById('proc-chain-tables').innerHTML = '<p class="hint">Computing protonation...</p>';
    const res = await fetch('/api/protonate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        residues, pH, his_tautomer: his,
        task_id: taskIdParam,
        structure_residues: _procResidues,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    if (requestId !== _protonationRequestId) return;
    if (Number(data.pH) !== pH) {
      throw new Error(`The server returned a result for pH ${data.pH}, not the requested pH ${pH}.`);
    }
    _procAssignments = data.assignments || [];
    const returnedIndices = new Set(
      _procAssignments.filter(function(a) { return a && a.is_titratable; })
        .map(function(a) { return Number(a.index); })
    );
    const expectedNames = new Set(['HIS', 'ASP', 'GLU', 'CYS', 'LYS', 'TYR']);
    const missingAssignments = _procResidues.filter(function(r) {
      return expectedNames.has(String(r.resname || '').toUpperCase()) && !returnedIndices.has(r.index);
    });
    if (missingAssignments.length) {
      throw new Error('The server returned incomplete protonation assignments. Run Compute again.');
    }
    _protonationComputed = true;
    _computedProtonationInput = {pH: pH, his: his};
    const previousResult = _lastProtonationResult;
    const currentStates = _procAssignments.filter(function(a) {
      return a && a.is_titratable;
    }).map(function(a) {
      return {
        index: Number(a.index),
        assigned_name: String(a.assigned_name || ''),
        charge: Number(a.charge || 0),
      };
    });
    let changedCount = null;
    if (previousResult) {
      const previousStates = new Map(previousResult.states.map(function(a) {
        return [a.index, a.assigned_name + '|' + a.charge];
      }));
      changedCount = currentStates.filter(function(a) {
        return previousStates.get(a.index) !== a.assigned_name + '|' + a.charge;
      }).length;
    }
    data.previous_pH = previousResult ? previousResult.pH : null;
    data.changed_count = changedCount;
    data.assigned_charge_e = currentStates.reduce(function(total, a) {
      return total + a.charge;
    }, 0);
    _lastProtonationResult = {pH: pH, his: his, states: currentStates};
    updateNextButtonState();
    if (statusEl) {
      const comparison = changedCount == null
        ? ''
        : `; ${changedCount} discrete residue state${changedCount === 1 ? '' : 's'} changed from pH ${previousResult.pH}`;
      statusEl.classList.remove('hidden');
      statusEl.textContent = `${data.propka_warning ? '⚠' : '✓'} Recalculated at pH ${pH.toFixed(1)} using ${data.method}${comparison}.` +
        (data.propka_warning ? ` ${data.propka_warning}` : '');
      statusEl.style.color = data.propka_warning ? '#d97706' : (data.used_propka ? '#059669' : '#64748b');
    }
    renderProtonationTables(data);
  } catch (e) {
    if (requestId !== _protonationRequestId) return;
    _procAssignments = [];
    _protonationComputed = false;
    _computedProtonationInput = null;
    document.getElementById('proc-chain-tables').innerHTML = '<p class="hint">No current protonation result. Correct the issue and click Compute again.</p>';
    if (statusEl) {
      statusEl.classList.remove('hidden');
      statusEl.textContent = 'Protonation failed: ' + (e.message || 'unknown error');
      statusEl.style.color = '#dc2626';
    }
    console.error('Protonation error:', e);
  } finally {
    if (requestId === _protonationRequestId) {
      _protonationRunning = false;
      if (runBtn) runBtn.disabled = false;
      if (checkBtn) checkBtn.disabled = false;
      updateNextButtonState();
    }
  }
}

function renderProtonationTables(data) {
  const container = document.getElementById('proc-chain-tables');
  if (!container) return;

  const titratable = new Set();
  (data.titratable_residues || []).forEach(t => titratable.add(t.index));

  let html = '';
  if (data.method) {
    html += `<p class="hint" style="margin-bottom:8px;">Method: <b>${data.method}</b> &nbsp;|&nbsp; pH ${data.pH} &nbsp;|&nbsp; ${data.titratable_count} titratable residues found</p>`;
    if (data.propka_warning) {
      html += `<p class="hint" style="margin-bottom:8px;color:#d97706;">⚠ ${data.propka_warning}</p>`;
    }
    html += `<p class="hint" style="margin-bottom:8px;">Assigned titratable-residue charge: <b>${data.assigned_charge_e >= 0 ? '+' : ''}${data.assigned_charge_e} e</b>`;
    if (data.changed_count != null) {
      html += ` &nbsp;|&nbsp; ${data.changed_count} discrete state${data.changed_count === 1 ? '' : 's'} changed since pH ${data.previous_pH}`;
      if (data.changed_count === 0 && Number(data.previous_pH) !== Number(data.pH)) {
        html += ' (the calculation did run; no predicted pKa threshold was crossed)';
      }
    }
    html += '</p>';
  }

  _procChains.forEach(ch => {
    const chainRes = _procResidues.filter(r => r.chain === ch);
    const chainTitr = chainRes.filter((_, i) => titratable.has(chainRes[i].index));
    if (!chainTitr.length) return;

    html += `<h4 style="margin-top:12px;color:#475569;">Chain ${ch || ' '} <span class="hint">${chainTitr.length} titratable residues</span></h4>`;
    html += `<table class="proc-table"><thead><tr><th>Resid</th><th>Original</th><th>Assigned</th><th>Charge</th><th>pKa</th><th>Shift</th><th>State</th><th>Override</th></tr></thead><tbody>`;

    chainRes.forEach((r, localIdx) => {
      const a = _procAssignments[r.index];
      if (!a || !a.is_titratable) return;
      const shift = a.pKa_shift;
      const shiftStr = shift != null ? (shift >= 0 ? '+' : '') + shift.toFixed(1) : '—';
      const altOpts = (a.alternatives || []).map(alt =>
        `<option value="${alt.name}" ${alt.name === a.assigned_name ? 'selected' : ''}>${alt.name} (${alt.charge >= 0 ? '+' : ''}${alt.charge})</option>`
      ).join('');
      html += `<tr>
        <td>${a.original} ${r.resid}</td>
        <td><b>${a.original}</b></td>
        <td style="color:#6366f1;font-weight:600;">${a.assigned_name}</td>
        <td>${a.charge >= 0 ? '+' : ''}${a.charge}</td>
        <td>${typeof a.pKa === 'number' ? a.pKa.toFixed(1) : a.pKa || '—'}</td>
        <td style="color:${shift > 0 ? '#dc2626' : shift < 0 ? '#059669' : '#94a3b8'};">${shiftStr}</td>
        <td style="font-size:11px;">${a.state_label}</td>
        <td><select class="proc-override" data-idx="${r.index}">${altOpts}</select></td>
      </tr>`;
    });

    html += '</tbody></table>';
  });

  container.innerHTML = html || '<p class="hint">No titratable residues found in any chain.</p>';

  // Wire overrides
  container.querySelectorAll('.proc-override').forEach(sel => {
    sel.addEventListener('change', () => {
      const idx = parseInt(sel.dataset.idx);
      const newName = sel.value;
      const a = _procAssignments[idx];
      if (!a) return;
      const alt = (a.alternatives || []).find(x => x.name === newName);
      if (alt) {
        a.assigned_name = alt.name; a.charge = alt.charge; a.state_label = alt.label;
        const row = sel.closest('tr');
        row.cells[2].textContent = alt.name;
        row.cells[3].textContent = (alt.charge >= 0 ? '+' : '') + alt.charge;
        row.cells[6].textContent = alt.label;
      }
    });
  });

  renderModSequences();
}

// -------------------------------------------------------------------
// Termini
// -------------------------------------------------------------------

function renderTerminiTab() {
  const container = document.getElementById('proc-termini-chains');
  if (!container) return;

  let html = '';
  _procChains.forEach(ch => {
    const t = _procTermini[ch] || { nter: '', cter: '' };
    const ace = _procCapCapabilities.ACE || {supported:false, reason:'Capability data is not loaded'};
    const formyl = _procCapCapabilities.FOR || {supported:false, reason:'Capability data is not loaded'};
    const nme = _procCapCapabilities.NME || {supported:false, reason:'Capability data is not loaded'};
    if (t.nter === 'ACE' && !ace.supported) t.nter = '';
    if (t.nter === 'FOR' && !formyl.supported) t.nter = '';
    if (t.cter === 'NME' && !nme.supported) t.cter = '';
    _procTermini[ch] = t;
    const firstRes = _procResidues.find(r => r.chain === ch);
    const lastRes = [..._procResidues].reverse().find(r => r.chain === ch);
    html += `<div class="termini-chain-row">
      <b style="min-width:60px;">Chain ${ch || ' '}</b>
      <span class="hint">N-ter: ${firstRes ? firstRes.resname + ' ' + firstRes.resid : '?'}</span>
      <select class="proc-nter-sel" data-chain="${ch}">
        <option value="" ${!t.nter ? 'selected' : ''}>Standard (NH₃⁺)</option>
        <option value="ACE" ${t.nter === 'ACE' ? 'selected' : ''} ${ace.supported ? '' : 'disabled'}>ACE${ace.supported ? ' — explicit acetyl cap' : ' — unavailable: ' + ace.reason}</option>
        <option value="FOR" ${t.nter === 'FOR' ? 'selected' : ''} ${formyl.supported ? '' : 'disabled'}>FOR${formyl.supported ? ' — explicit formyl cap' : ' — unavailable: ' + formyl.reason}</option>
      </select>
      <span class="hint">C-ter: ${lastRes ? lastRes.resname + ' ' + lastRes.resid : '?'}</span>
      <select class="proc-cter-sel" data-chain="${ch}">
        <option value="" ${!t.cter ? 'selected' : ''}>Standard (COO⁻)</option>
        <option value="NME" ${t.cter === 'NME' ? 'selected' : ''} ${nme.supported ? '' : 'disabled'}>NME${nme.supported ? ' — explicit methylamide cap' : ' — unavailable: ' + nme.reason}</option>
      </select>
    </div>`;
  });
  container.innerHTML = html || '<p class="hint">No chains detected.</p>';

  container.querySelectorAll('.proc-nter-sel').forEach(sel => {
    sel.addEventListener('change', () => {
      const ch = sel.dataset.chain;
      if (!_procTermini[ch]) _procTermini[ch] = { nter: '', cter: '' };
      _procTermini[ch].nter = sel.value;
    });
  });
  container.querySelectorAll('.proc-cter-sel').forEach(sel => {
    sel.addEventListener('change', () => {
      const ch = sel.dataset.chain;
      if (!_procTermini[ch]) _procTermini[ch] = { nter: '', cter: '' };
      _procTermini[ch].cter = sel.value;
    });
  });
}

// -------------------------------------------------------------------
// Modifications
// -------------------------------------------------------------------

function renderModSequences() {
  const container = document.getElementById('proc-mod-chains');
  if (!container) return;

  const titratableIdx = new Set();
  const modifiedIdx = new Set();
  const crosslinkedIdx = new Set();
  _procAssignments.forEach(a => { if (a.is_titratable) titratableIdx.add(a.index); });
  _procModifications.forEach(m => modifiedIdx.add(m.index));
  _procCrosslinks.forEach(function(item) {
    crosslinkedIdx.add(item.first_index);
    crosslinkedIdx.add(item.second_index);
  });

  let html = '';
  _procChains.forEach(ch => {
    const chainRes = _procResidues.filter(r => r.chain === ch);
    if (!chainRes.length) return;
    html += `<div class="proc-mod-chain-block"><h4 style="margin:8px 0 4px;color:#475569;">Chain ${ch || ' '} <span class="hint">${chainRes.length} residues</span></h4><div class="proc-mod-sequence">`;
    chainRes.forEach(r => {
      const i = r.index;
      const residuePatches = _procPatchCatalog.filter(function(patch) {
        return (patch.target_residues || []).indexOf(r.resname) >= 0;
      });
      const hasSupportedPatch = residuePatches.some(function(patch) {
        return patch.supported !== false;
      });
      let cls = 'proc-mod-res';
      if (titratableIdx.has(i)) cls += ' titratable';
      if (modifiedIdx.has(i)) cls += ' modified';
      if (crosslinkedIdx.has(i)) cls += ' modified';
      if (hasSupportedPatch && !crosslinkedIdx.has(i)) cls += ' modifiable';
      else if (residuePatches.length) cls += ' catalog-only';
      if (_procSelectedIdx === i) cls += ' selected';
      const capability = hasSupportedPatch ? 'simulation-ready modification available' :
        (residuePatches.length ? 'catalogue entries exist but are not simulation-ready' : 'no registered modification');
      html += `<span class="${cls}" data-idx="${i}" data-has-patches="${residuePatches.length ? '1' : '0'}" title="#${i+1} ${r.resname} ch ${ch} resid ${r.resid}; ${capability}">${r.resname}</span>`;
    });
    html += '</div></div>';
  });
  container.innerHTML = html || '<p class="hint">No residues loaded.</p>';

  container.querySelectorAll('.proc-mod-res').forEach(el => {
    el.addEventListener('click', () => {
      if (el.dataset.hasPatches !== '1') return;
      _procSelectedIdx = parseInt(el.dataset.idx);
      renderModSequences();
      openPatchPicker(_procSelectedIdx);
    });
  });
}

function renderDisulfideControls() {
  const first = document.getElementById('proc-disulfide-first');
  const second = document.getElementById('proc-disulfide-second');
  const add = document.getElementById('proc-disulfide-add');
  const capability = document.getElementById('proc-disulfide-capability');
  const list = document.getElementById('proc-disulfide-list');
  if (!first || !second || !add || !capability || !list) return;
  const support = _procCrosslinkCapabilities.disulfide || {};
  const used = new Set();
  _procCrosslinks.forEach(function(item) {
    used.add(item.first_index);
    used.add(item.second_index);
  });
  const cysteines = _procResidues.filter(function(residue) {
    return residue.resname === 'CYS' && !used.has(residue.index) &&
      !_procModifications.some(function(mod) { return mod.index === residue.index; });
  });
  const options = '<option value="">Select CYS…</option>' + cysteines.map(function(residue) {
    return '<option value="' + residue.index + '">Chain ' + (residue.chain || '?') +
      ' — CYS ' + residue.resid + ' (#' + (residue.index + 1) + ')</option>';
  }).join('');
  first.innerHTML = options;
  second.innerHTML = options;
  const enabled = support.supported === true && cysteines.length >= 2;
  first.disabled = !enabled;
  second.disabled = !enabled;
  add.disabled = !enabled;
  if (support.supported === true) {
    capability.textContent = 'Supported by ' + selectedProteinForceField() +
      '; SG–SG distance is validated against the force-field target before coordinates change.';
  } else {
    capability.textContent = 'Unavailable with ' + selectedProteinForceField() + ': ' +
      (support.reason || 'no validated paired-residue model is installed') + '.';
  }
  list.innerHTML = _procCrosslinks.map(function(item, index) {
    const left = _procResidues[item.first_index];
    const right = _procResidues[item.second_index];
    return '<div class="proc-mod-item">Chain ' + (left.chain || '?') + ':CYS ' + left.resid +
      ' — Chain ' + (right.chain || '?') + ':CYS ' + right.resid +
      ' <span style="color:#64748b;">(disulfide)</span>' +
      '<button data-crosslink="' + index + '" class="proc-crosslink-remove" title="Remove">×</button></div>';
  }).join('') || '<p class="hint">No disulfide crosslinks selected.</p>';
  list.querySelectorAll('.proc-crosslink-remove').forEach(function(button) {
    button.addEventListener('click', function() {
      _procCrosslinks.splice(parseInt(button.dataset.crosslink), 1);
      renderDisulfideControls();
      renderModSequences();
    });
  });
}

function addDisulfideCrosslink() {
  const first = document.getElementById('proc-disulfide-first');
  const second = document.getElementById('proc-disulfide-second');
  if (!first || !second) return;
  const firstIndex = Number(first.value);
  const secondIndex = Number(second.value);
  if (!Number.isInteger(firstIndex) || !Number.isInteger(secondIndex) ||
      first.value === '' || second.value === '' || firstIndex === secondIndex) {
    window.alert('Select two distinct cysteine residues for the disulfide.');
    return;
  }
  _procCrosslinks.push({
    type: 'disulfide', first_index: firstIndex, second_index: secondIndex,
  });
  renderDisulfideControls();
  renderModSequences();
}

async function openPatchPicker(idx) {
  const picker = document.getElementById('proc-patch-picker');
  const targetEl = document.getElementById('proc-patch-target');
  const optionsEl = document.getElementById('proc-patch-options');
  if (!picker || !targetEl || !optionsEl) return;
  const r = _procResidues[idx];
  if (!r) return;
  targetEl.textContent = `${r.resname} ${r.resid} (Chain ${r.chain || '?'}, #${idx+1})`;
  picker.classList.remove('hidden');
  try {
    const res = await fetch(`/api/patches/${encodeURIComponent(r.resname)}?force_field=${encodeURIComponent(selectedProteinForceField())}`);
    if (!res.ok) { optionsEl.innerHTML = '<p class="hint">No patches available.</p>'; return; }
    const patches = await res.json();
    if (!patches.length) { optionsEl.innerHTML = '<p class="hint">No modifications available.</p>'; return; }
    const supportedPatches = patches.filter(p => p.supported !== false);
    const capabilityNotice = supportedPatches.length ? '' :
      '<p class="hint" style="color:#b45309;margin-bottom:8px;">No simulation-ready modification is available for this residue. Catalogue entries below are disabled because complete atoms and bonded parameters are not yet implemented.</p>';
    optionsEl.innerHTML = capabilityNotice + patches.map(p =>
      `<div class="proc-patch-option" data-patch="${p.id}" data-supported="${p.supported !== false}"
            aria-disabled="${p.supported === false}" style="${p.supported === false ? 'opacity:.5;cursor:not-allowed;' : ''}">
        <span class="patch-name">${p.name}</span> → ${p.product_name}
        <span style="color:#64748b;font-size:11px;">(${p.description}; net charge shift ${p.charge_shift > 0 ? '+' : ''}${p.charge_shift}${p.supported === false ? '; unavailable: ' + p.support_reason : ''})</span>
      </div>`).join('');
    optionsEl.querySelectorAll('.proc-patch-option').forEach(opt => {
      opt.addEventListener('click', () => {
        const patch = patches.find(p => p.id === opt.dataset.patch);
        if (patch && patch.supported !== false) { applyModification(idx, patch.id, patch.product_name, patch.charge_shift); closePatchPicker(); }
      });
    });
  } catch (e) { optionsEl.innerHTML = '<p class="hint">Error.</p>'; }
}

function closePatchPicker() {
  const picker = document.getElementById('proc-patch-picker');
  if (picker) picker.classList.add('hidden');
}

function applyModification(idx, patchId, productName, chargeShift, source) {
  _procModifications = _procModifications.filter(m => m.index !== idx);
  if (patchId) _procModifications.push({ index: idx, patch_id: patchId, product_name: productName, charge_shift: chargeShift, source: source || 'user' });
  const listEl = document.getElementById('proc-mod-list');
  if (listEl) {
    listEl.innerHTML = _procModifications.map(m => {
      const rr = _procResidues[m.index];
      return `<div class="proc-mod-item">
        Chain ${rr.chain} — <b>${rr.resname} ${rr.resid}</b> → <b>${m.product_name}</b>
        <span style="color:#64748b;">(${m.patch_id})</span>
        <button data-idx="${m.index}" class="proc-mod-remove" title="Remove">×</button>
      </div>`;
    }).join('') || '<p class="hint">No modifications applied yet.</p>';
    listEl.querySelectorAll('.proc-mod-remove').forEach(btn => {
      btn.addEventListener('click', () => {
        const rmIdx = parseInt(btn.dataset.idx);
        _procModifications = _procModifications.filter(m => m.index !== rmIdx);
        applyModification(rmIdx, '', '');  // calls renderModSequences() internally
      });
    });
  }
  _procSelectedIdx = -1;
  renderModSequences();
}


/** Compute net system charge from protonation + modifications + termini. */
window._getSystemNetCharge = function() {
  var charge = 0;
  if (_procAssignments && _procAssignments.length) {
    _procAssignments.forEach(function(a) {
      if (typeof a.charge === "number") charge += a.charge;
    });
  }
  if (_procTermini) {
    Object.keys(_procTermini).forEach(function(ch) {
      var t = _procTermini[ch];
      if (!t.nter) charge += 1;
      if (!t.cter) charge -= 1;
    });
  }
  if (_procModifications && _procModifications.length) {
    _procModifications.forEach(function(m) {
      var pid = m.patch_id || "";
      if (typeof m.charge_shift === "number") charge += m.charge_shift;
      else if (pid.indexOf("PHOS1") === 0) charge -= 1;
      else if (pid.indexOf("PHOS") === 0) charge -= 2;
      else if (pid.indexOf("ACET") === 0 || pid.indexOf("SUCC") === 0 ||
               pid.indexOf("CBM") === 0 || pid.indexOf("CRO") === 0 ||
               pid.indexOf("BUT") === 0 || pid.indexOf("PRO") === 0 ||
               pid.indexOf("MAL") === 0 || pid.indexOf("GLR") === 0) charge -= 1;
      else if (pid.indexOf("CIT") === 0) charge -= 1;
      else if (pid.indexOf("CSO") === 0 || pid.indexOf("CSD") === 0 ||
               pid.indexOf("CSX") === 0) charge -= 1;
      else if (pid.indexOf("TYS") === 0) charge -= 1;
      else if (pid.indexOf("DEA") === 0 || pid.indexOf("DEG") === 0) charge -= 1;
      else if (pid.indexOf("PCA") === 0) charge -= 1;
    });
  }
  // Constitutively charged residues (always ionised at physiological pH)
  if (_procResidues && _procResidues.length) {
    _procResidues.forEach(function(r) {
      if (r.resname === "ARG") charge += 1;  // guanidinium, pKa ~12.5
    });
  }

  // Membrane lipid charges — estimate from composition
  if (_mixUpper && _mixUpper.length && _lipidPickerData && _lipidPickerData.lipids) {
    var lipids = _lipidPickerData.lipids;
    var nPerLeaflet = 100;
    // Try to read actual count from the membrane count table
    var countTables = document.querySelectorAll(".count-table");
    if (countTables.length > 0) {
      var boldEls = countTables[0].querySelectorAll("td b");
      boldEls.forEach(function(b) {
        var val = parseInt(b.textContent);
        if (!isNaN(val) && val > 0) { nPerLeaflet = val; }
      });
    }
    // Upper leaflet charge
    var upperCharge = 0;
    _mixUpper.forEach(function(m) {
      var lipid = lipids.find(function(l) { return l.name === m.name; });
      if (lipid && typeof lipid.charge === "number") {
        upperCharge += lipid.charge * (m.ratio / 100);
      }
    });
    charge += Math.round(upperCharge * nPerLeaflet);
    // Lower leaflet
    var lowerCharge = 0;
    var lowerMix = _asymmetric && _mixLower ? _mixLower : _mixUpper;
    lowerMix.forEach(function(m) {
      var lipid = lipids.find(function(l) { return l.name === m.name; });
      if (lipid && typeof lipid.charge === "number") {
        lowerCharge += lipid.charge * (m.ratio / 100);
      }
    });
    charge += Math.round(lowerCharge * nPerLeaflet);
  }

  return charge;
};


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
  var el = document.getElementById('solvation-3d-viewer');
  if (!el) return;
  if (el.offsetWidth === 0 || el.offsetHeight === 0) return;
  if (typeof $3Dmol === 'undefined') return;

  // Load the membrane system PDB from checkpoint — this contains the
  // protein at the correct (saved) orientation and lipids built around it.
  // Do NOT fall back to _orientedPdbContent which is the PPM reference
  // model (may differ from the actual checkpoint coordinates).
  var pdbContent = null;
  var checkpointStep = 'membrane';
  if (state.taskId) {
    // Do not show an older Solvation checkpoint after its inputs or an
    // upstream step changed.  Until the current config is checked, preview
    // the membrane checkpoint in the coordinate frame Solvator will create.
    if (_solvChecked) {
      pdbContent = await _loadStepViewerPdb('solvation');
      if (pdbContent) checkpointStep = 'solvation';
    }
    if (!pdbContent) pdbContent = await _loadStepViewerPdb('membrane');
  }
  if (!pdbContent) return;

  // Parse box from CRYST1 record in the checkpoint PDB (coordinates are in shifted frame)
  var boxA = 100, boxB = 100, boxC = 100;
  var lines = pdbContent.split('\n');
  for (var li = 0; li < lines.length; li++) {
    var l = lines[li];
    if (l.indexOf('CRYST1') === 0) {
      boxA = parseFloat(l.substring(6, 15)) || 100;
      boxB = parseFloat(l.substring(15, 24)) || 100;
      boxC = parseFloat(l.substring(24, 33)) || 100;
      break;
    }
  }
  var boxOrigin = {x: 0, y: 0, z: 0};
  if (checkpointStep === 'membrane') {
    // Match the future Solvator transform without moving the preview model:
    // XY is centred in the periodic box. Z padding starts at the two
    // lipid-water interfaces and the membrane midplane remains centred;
    // asymmetric protein protrusions do not move the membrane in the box.
    var bounds = _pdbCoordinateBoundsAngstrom(pdbContent);
    var paddingInput = document.getElementById('box-padding');
    var paddingNm = Number(paddingInput ? paddingInput.value : 2.0);
    if (!Number.isFinite(paddingNm) || paddingNm < 0) paddingNm = 2.0;
    var paddingA = paddingNm * 10.0;
    if (bounds) {
      boxOrigin.x = (bounds.minX + bounds.maxX) / 2.0 - boxA / 2.0;
      boxOrigin.y = (bounds.minY + bounds.maxY) / 2.0 - boxB / 2.0;
    }
    var membraneBounds = _pdbMembraneZBoundsAngstrom(pdbContent);
    if (membraneBounds) {
      var membraneMidZ = (membraneBounds.minZ + membraneBounds.maxZ) / 2.0;
      boxC = membraneBounds.maxZ - membraneBounds.minZ + 2.0 * paddingA;
      boxOrigin.z = membraneMidZ - boxC / 2.0;
    } else {
      var interfaceThicknessNm = Number(_dominantLipidDHH);
      if (!Number.isFinite(interfaceThicknessNm) || interfaceThicknessNm <= 0) {
        interfaceThicknessNm = 3.8;
      }
      boxC = interfaceThicknessNm * 10.0 + 2.0 * paddingA;
      boxOrigin.z = -boxC / 2.0;
    }
  }

  if (_solvationViewer) { try { _solvationViewer.clear(); } catch(e) {} _solvationViewer = null; }
  while (el.firstChild) { el.removeChild(el.firstChild); }

  _solvationViewer = $3Dmol.createViewer(el, {backgroundColor: '0xffffff', antialias: true});
  _solvationViewer.setBackgroundColor('0xffffff');
  _solvationViewer.setSlab(-100000, 100000);
  var v = _solvationViewer;

  // 1. Full solvated system
  v.addModel(pdbContent, 'pdb');
  _applyUnifiedStyle(v, pdbContent);

  // 2. Water: small semi-transparent blue spheres
  v.setStyle({resn: ['SOL', 'HOH', 'WAT', 'TIP', 'TIP3', 'SPC', 'SPCE']},
    {sphere: {radius: 0.15, opacity: 0.25, color: '0x3b82f6'}});

  // 3. Ions: visible spheres
  v.setStyle({resn: ['NA', 'CL', 'K', 'CA', 'ZN', 'MG']},
    {sphere: {radius: 0.4, opacity: 0.8}});

  // 4. Draw the physical periodic box in the checkpoint's coordinate frame.
  drawOrthogonalBox(v, boxA, boxB, boxC, boxOrigin);

  // Checked Solvation coordinates live in the positive [0,L] box frame.
  // Recompute the camera target after loading them so rotation is centred on
  // the system rather than on the global coordinate origin.
  v.zoomTo();
  v.render();
  v.setSlab(-100000, 100000);
  var label = document.getElementById('solvation-viewer-label');
  if (label) {
    var prefix = checkpointStep === 'solvation' ? 'Checked box: ' : 'Preview box: ';
    label.textContent = prefix + (boxA/10).toFixed(1) + '×' +
      (boxB/10).toFixed(1) + '×' + (boxC/10).toFixed(1) + ' nm';
  }
}


// ===================================================================
// Simulation Parameters — dynamic stage cards
// ===================================================================

var _simStages = [];       // stage configs
var _prodIters = [];       // production iteration configs
var _simHardware = {};

// Default restraint decay schedule (standard protocol, 6 stages)
var _DEFAULT_EM_BASE = {
  nsteps: 50000, emtol: 1000.0, emstep: 0.01, nstlist: 10,
  constraints: "h-bonds", bb: 4000, sc: 2000, lipid: 1000, dih: 1000,
  mdp_overrides_text: ""
};
var _DEFAULT_EM = Object.assign({}, _DEFAULT_EM_BASE);
var _DEFAULT_OUTPUT = {
  nstxout_compressed: 5000, nstxout: 0, nstvout: 0, nstfout: 0,
  nstcalcenergy: 100, nstenergy: 1000, nstlog: 1000,
  enabled: true, nstlist: 20, comm_mode: "linear", comm_grps: "System",
  constraints: "h-bonds", temperature: 310.15,
  mdp_overrides_text: ""
};
var _MEMBRANE_SCHEDULE = [
  { bb:4000, sc:2000, lipid:1000, dih:1000, dt:1.0, nsteps:125000, ensemble:"nvt",  tcoupl:"v-rescale", tau_t:"1.0", nstcomm:100, comm_grps:"SOLU_MEMB SOLV" },
  { bb:2000, sc:1000, lipid:400,  dih:400,  dt:1.0, nsteps:125000, ensemble:"nvt",  tcoupl:"v-rescale", tau_t:"1.0", nstcomm:100, comm_grps:"SOLU_MEMB SOLV" },
  { bb:1000, sc:500,  lipid:400,  dih:200,  dt:1.0, nsteps:125000, ensemble:"npt",  tcoupl:"v-rescale", tau_t:"1.0", tau_p:"5.0", ref_p:"1.0", compress:"4.5e-5", nstcomm:100, pcoupl:"C-rescale", comm_grps:"SOLU_MEMB SOLV" },
  { bb:500,  sc:200,  lipid:200,  dih:200,  dt:2.0, nsteps:250000, ensemble:"npt",  tcoupl:"v-rescale", tau_t:"1.0", tau_p:"5.0", ref_p:"1.0", compress:"4.5e-5", nstcomm:100, pcoupl:"C-rescale", comm_grps:"SOLU_MEMB SOLV" },
  { bb:200,  sc:50,   lipid:40,   dih:100,  dt:2.0, nsteps:250000, ensemble:"npt",  tcoupl:"v-rescale", tau_t:"1.0", tau_p:"5.0", ref_p:"1.0", compress:"4.5e-5", nstcomm:100, pcoupl:"C-rescale", comm_grps:"SOLU_MEMB SOLV" },
  { bb:50,   sc:0,    lipid:0,    dih:0,    dt:2.0, nsteps:250000, ensemble:"npt",  tcoupl:"v-rescale", tau_t:"1.0", tau_p:"5.0", ref_p:"1.0", compress:"4.5e-5", nstcomm:100, pcoupl:"C-rescale", comm_grps:"SOLU_MEMB SOLV" },
];
var _SOLUTION_SCHEDULE = [
  { bb:400, sc:40, lipid:0, dih:0, dt:1.0, nsteps:125000, ensemble:"nvt",
    tcoupl:"v-rescale", tau_t:"1.0", nstcomm:100, comm_grps:"SOLU SOLV" },
];

function isSolutionProtocol() {
  var pipeline = state.taskType && state.taskType.pipeline;
  return pipeline === "solvator" || pipeline === "liquid";
}

function syncMdpNonbondDefaults() {
  var forceField = String(document.getElementById("ff-protein")?.value || "amber14sb").toLowerCase();
  var isCharmm = forceField.indexOf("charmm") === 0;
  var defaults = {
    rlist: isCharmm ? 1.2 : 1.0,
    vdw_modifier: isCharmm ? "Force-switch" : "Potential-shift",
    rvdw_switch: isCharmm ? 1.0 : null,
    rvdw: isCharmm ? 1.2 : 1.0,
    rcoulomb: isCharmm ? 1.2 : 1.0,
    fourierspacing: 0.12,
    dispcorr: isCharmm ? "no" : "EnerPres"
  };
  [_DEFAULT_EM].concat(_simStages, _prodIters).forEach(function(stage) {
    Object.assign(stage, defaults);
  });
}

function initSimParams() {
  // Never carry one resumed task's minimization settings into a newly
  // selected task in the same browser session.
  var solution = isSolutionProtocol();
  _DEFAULT_EM = Object.assign({}, _DEFAULT_EM_BASE, {
    nsteps: solution ? 5000 : 50000,
    constraints: "h-bonds",
    bb: solution ? 400 : 4000,
    sc: solution ? 40 : 2000,
    lipid: solution ? 0 : 1000,
    dih: solution ? 0 : 1000
  });
  var schedule = solution ? _SOLUTION_SCHEDULE : _MEMBRANE_SCHEDULE;
  _simStages = schedule.map(function(stage) {
    return Object.assign({}, _DEFAULT_OUTPUT, {
      pcoupl_type: solution ? "isotropic" : "semisotropic",
      gen_seed: -1
    }, stage);
  });
  _prodIters = [Object.assign({}, _DEFAULT_OUTPUT, {
    nsteps: solution ? 500000 : 5000000,
    repeat: solution ? 10 : 5,
    dt: 2.0, nstxout_compressed: solution ? 50000 : 10000,
    tcoupl: "v-rescale", tau_t: "1.0", tau_p: "5.0", ref_p: "1.0",
    compress: "4.5e-5", nstcomm: 100, pcoupl: "C-rescale",
    pcoupl_type: solution ? "isotropic" : "semisotropic",
    comm_grps: solution ? "SOLU SOLV" : "SOLU_MEMB SOLV"
  })];
  _simHardware = {
    mode: "thread-mpi", cpu_threads: 1, mpi_ranks: 1,
    use_gpu: false, gpu_count: 1, gpu_ids: "0", gmx_command: "gmx",
    mpi_launcher: "mpirun", pin: "auto"
  };
  syncMdpNonbondDefaults();
  renderSimStages();
}

function mdpOverridesToText(value) {
  if (!value || typeof value !== "object") return "";
  return Object.keys(value).map(function(key) {
    return key + " = " + value[key];
  }).join("\n");
}

function restoreSimulationParams(saved) {
  if (!saved || typeof saved !== "object") {
    renderSimStages();
    return;
  }
  if (saved.hardware && typeof saved.hardware === "object") {
    _simHardware = Object.assign({}, _simHardware, saved.hardware);
    if (Array.isArray(_simHardware.gpu_ids)) {
      _simHardware.gpu_ids = _simHardware.gpu_ids.join(",");
    }
    if (!_simHardware.use_gpu && Number(_simHardware.gpu_count) === 0) {
      _simHardware.gpu_count = 1;
    }
  }
  var legacyStageDefaults = {};
  ["pcoupl_type", "gen_seed", "rlist", "vdw_modifier", "rvdw_switch",
   "rvdw", "rcoulomb", "fourierspacing", "dispcorr"].forEach(function(key) {
    if (saved[key] !== undefined) legacyStageDefaults[key] = saved[key];
  });
  var legacyOverrides = saved.mdp_overrides || {};
  if (Array.isArray(saved.eq_stages)) {
    _simStages = saved.eq_stages.map(function(stage) {
      var restored = Object.assign({}, _DEFAULT_OUTPUT, legacyStageDefaults, stage);
      restored.mdp_overrides_text = mdpOverridesToText(
        Object.assign({}, legacyOverrides, stage.mdp_overrides || {})
      );
      return restored;
    });
  }
  if (Array.isArray(saved.prod_iters)) {
    _prodIters = saved.prod_iters.map(function(stage) {
      var restored = Object.assign({}, _DEFAULT_OUTPUT, legacyStageDefaults, stage);
      restored.mdp_overrides_text = mdpOverridesToText(
        Object.assign({}, legacyOverrides, stage.mdp_overrides || {})
      );
      return restored;
    });
  }
  var minimization = saved.minimization || {};
  if (saved.em_nsteps !== undefined) minimization.nsteps = saved.em_nsteps;
  if (saved.em_ftol !== undefined) minimization.emtol = saved.em_ftol;
  if (saved.em_step !== undefined) minimization.emstep = saved.em_step;
  if (saved.em_nstlist !== undefined) minimization.nstlist = saved.em_nstlist;
  if (saved.em_constraints !== undefined) minimization.constraints = saved.em_constraints;
  Object.assign(_DEFAULT_EM, legacyStageDefaults, minimization);
  if (minimization.mdp_overrides || saved.em_overrides) {
    _DEFAULT_EM.mdp_overrides_text = mdpOverridesToText(
      minimization.mdp_overrides || saved.em_overrides
    );
  }
  renderSimStages();
}

function renderNonbondFields(prefix, values, includeNstlist) {
  var html = '<div class="param-row">';
  if (includeNstlist) {
    html += paramNumber(prefix + "-nstlist", "Neighbor-list interval", values.nstlist, 1, 1000, 1, "wide");
  }
  html += paramNumber(prefix + "-rlist", "rlist (nm)", values.rlist, 0.1, 5, 0.01, "narrow");
  html += paramSelect(prefix + "-vdw-modifier", "LJ modifier", [["Potential-shift","Potential shift"],["Force-switch","Force switch"],["Potential-switch","Potential switch"],["none","None"]], values.vdw_modifier);
  html += paramNumber(prefix + "-rvdw-switch", "LJ switch (nm)", values.rvdw_switch, 0, 5, 0.01, "narrow");
  html += paramNumber(prefix + "-rvdw", "LJ cutoff (nm)", values.rvdw, 0.1, 5, 0.01, "narrow");
  html += paramNumber(prefix + "-rcoulomb", "PME real cutoff (nm)", values.rcoulomb, 0.1, 5, 0.01, "narrow");
  html += paramNumber(prefix + "-fourierspacing", "PME spacing (nm)", values.fourierspacing, 0.01, 1, 0.01, "narrow");
  html += paramSelect(prefix + "-dispcorr", "Dispersion correction", [["EnerPres","Energy + pressure"],["Ener","Energy"],["no","None"]], values.dispcorr);
  return html + '</div>';
}

function renderSimStages() {
  var container = document.getElementById("simparams-stages");
  if (!container) return;

  var html = "";

  // ---- Execution hardware (run script only; never changes MDP physics) ----
  var ompThreads = Math.max(
    1, Math.floor(_simHardware.cpu_threads / Math.max(_simHardware.mpi_ranks, 1))
  );
  html += '<div class="sim-stage-card" style="border-color:#2563eb;">';
  html += '<div class="sim-stage-header open" onclick="toggleStageCard(this)">';
  html += '<span class="sim-stage-icon" style="color:#2563eb;">&#9889;</span>';
  html += '<span class="sim-stage-title" style="color:#2563eb;">Execution Hardware</span>';
  html += '<span class="sim-stage-summary">Defaults written to run_md.sh; MDP physics is unchanged</span></div>';
  html += '<div class="sim-stage-body open">';
  html += '<div class="param-row">';
  html += paramSelect("sim-hw-mode", "MPI mode", [["thread-mpi","Thread-MPI (single node)"],["external-mpi","External MPI / scheduler"]], _simHardware.mode);
  html += paramNumber("sim-hw-cpu", "Total CPU threads", _simHardware.cpu_threads, 1, 4096, 1, "wide");
  html += paramNumber("sim-hw-mpi", "MPI ranks", _simHardware.mpi_ranks, 1, 4096, 1, "wide");
  html += '<span class="param-item"><label>OpenMP threads/rank</label><output id="sim-hw-omp">' + ompThreads + '</output></span>';
  html += paramSelect("sim-hw-pin", "Thread pinning", [["auto","Automatic"],["on","On"],["off","Off"]], _simHardware.pin);
  html += '</div><div class="param-row">';
  html += '<span class="param-item"><label>GPU execution</label><label class="hint"><input type="checkbox" id="sim-hw-use-gpu"' + (_simHardware.use_gpu ? ' checked' : '') + '> Enable GPU IDs</label></span>';
  html += paramNumber("sim-hw-gpu-count", "GPU count", _simHardware.gpu_count, 1, 256, 1, "narrow");
  html += paramText("sim-hw-gpu-ids", "Logical GPU IDs", _simHardware.gpu_ids, "narrow");
  html += paramText("sim-hw-gmx", "GROMACS command", _simHardware.gmx_command, "wide");
  html += paramSelect("sim-hw-launcher", "External MPI launcher", [["mpirun","mpirun"],["mpiexec","mpiexec"],["srun","Slurm srun"]], _simHardware.mpi_launcher);
  html += '</div>';
  html += '<p class="hint">Total CPU threads must be exactly divisible by MPI ranks. For external MPI use an MPI-enabled GROMACS executable, usually gmx_mpi. GPU count must equal the number of unique logical GPU IDs and cannot exceed MPI ranks. GROMACS retains automatic task placement across the selected devices.</p>';
  html += '</div></div>';

  // ---- Energy Minimization ----
  var emTime = _DEFAULT_EM.nsteps + " steps";
  html += '<div class="sim-stage-card" style="border-color:#f59e0b;">';
  html += '<div class="sim-stage-header" data-stage="em" onclick="toggleStageCard(this)">';
  html += '<span class="sim-stage-icon" style="color:#f59e0b;">&#9673;</span>';
  html += '<span class="sim-stage-title" style="color:#f59e0b;">Energy Minimization</span>';
  html += '<span class="sim-stage-summary">' + _DEFAULT_EM.nsteps.toLocaleString() + ' steps &nbsp;|&nbsp; emtol=' + _DEFAULT_EM.emtol.toFixed(0) + '</span>';
  html += '</div>';
  html += '<div class="sim-stage-body" data-stage="em">';
  html += '<div class="param-row">';
  html += paramSelect("em-integrator", "Method", [["steep","Steepest Descent"],["cg","Conjugate Gradient"]], _DEFAULT_EM.integrator || "steep");
  html += paramNumber("em-nsteps", "Max Steps", _DEFAULT_EM.nsteps, 100, 100000, 1000, "wide");
  html += paramNumber("em-emtol", "Force Tolerance (kJ/mol/nm)", _DEFAULT_EM.emtol, 10, 10000, 100, "wide");
  html += paramNumber("em-emstep", "Step size (nm)", _DEFAULT_EM.emstep, 0.0001, 1, 0.001, "narrow");
  html += paramNumber("em-nstlist", "nstlist", _DEFAULT_EM.nstlist, 1, 100, 1, "narrow");
  html += paramSelect("em-constraints", "Constraints", [["none","None"],["h-bonds","H-bonds"],["all-bonds","All bonds"],["h-angles","H + angles"],["all-angles","All angles"]], _DEFAULT_EM.constraints);
  html += '</div>';
  html += '<div class="param-row">';
  html += paramNumber("em-bb", "BB restraint", _DEFAULT_EM.bb, 0, 10000, 50, "narrow");
  html += paramNumber("em-sc", "SC restraint", _DEFAULT_EM.sc, 0, 10000, 50, "narrow");
  html += paramNumber("em-lipid", "Lipid restraint", _DEFAULT_EM.lipid, 0, 10000, 50, "narrow");
  html += paramNumber("em-dih", "DIH restraint", _DEFAULT_EM.dih, 0, 10000, 50, "narrow");
  html += '</div>';
  html += renderNonbondFields("em", _DEFAULT_EM, false);
  html += paramTextarea("em-mdp-overrides", "Minimization overrides", _DEFAULT_EM.mdp_overrides_text,
    "Use for expert GROMACS directives that are not shown above.");
  html += '</div></div>';

  // ---- Equilibration stages ----
    // ---- Equilibration stages ----
  _simStages.forEach(function(st, i) {
    var stageEnabled = st.enabled !== false;
    var ensembleLabel = st.ensemble.toUpperCase();
    if (st.ensemble === "nvt") ensembleLabel = "NVT (no pressure coupling)";
    else if (st.ensemble === "npt") ensembleLabel = "NPT (semi-isotropic)";
    
    var timeNs = st.nsteps * st.dt / 1000000;
    var dtDisplay = st.dt.toFixed(1) + " fs";
    
    html += '<div class="sim-stage-card" data-run-card="eq' + i + '" style="opacity:' + (stageEnabled ? '1' : '0.55') + ';">';
    html += '<div class="sim-stage-header" data-stage="eq' + i + '" onclick="toggleStageCard(this)">';
    html += '<span class="sim-stage-icon">' + (i === 0 ? '&#9678;' : '&#9674;') + '</span>';
    html += '<span class="sim-stage-title">Equilibration ' + (i+1) + '</span>';
    html += '<span class="sim-stage-summary">' + st.nsteps.toLocaleString() + ' steps × ' + dtDisplay + ' = ' + timeNs.toFixed(1) + ' ns &nbsp;|&nbsp; BB=' + st.bb + ' SC=' + st.sc + ' Lipid=' + st.lipid + '</span>';
    html += stageEnabledControl("eq-enabled-" + i, stageEnabled);
    html += '</div>';
    html += '<div class="sim-stage-body" data-stage="eq' + i + '">';

    // Row 1: Basic
    html += '<div class="param-row">';
    html += paramNumber("eq-dt-" + i, "Timestep (fs)", st.dt, 0.5, 5.0, 0.5, "narrow");
    html += paramNumber("eq-nsteps-" + i, "Steps", st.nsteps, 1000, 10000000, 1000, "wide");
    html += paramNumber("eq-gen-seed-" + i, "Velocity seed (if first)", st.gen_seed, -1, 2147483647, 1, "wide");
    html += '<span class="hint" style="align-self:flex-end;margin-bottom:2px;" id="eq-time-' + i + '">' + timeNs.toFixed(1) + ' ns</span>';
    html += '</div>';

    // Row 2: Restraints
    html += '<div class="param-row">';
    html += paramNumber("eq-bb-" + i, "BB (kJ/mol/nm²)", st.bb, 0, 10000, 50, "narrow");
    html += paramNumber("eq-sc-" + i, "SC", st.sc, 0, 10000, 50, "narrow");
    html += paramNumber("eq-lipid-" + i, "Lipid", st.lipid, 0, 10000, 50, "narrow");
    html += paramNumber("eq-dih-" + i, "DIH", st.dih, 0, 10000, 50, "narrow");
    html += '</div>';

    // Row 3: Thermostat + COM removal
    html += '<div class="param-row">';
    html += paramSelect("eq-ensemble-" + i, "Ensemble", [["nvt","NVT"],["npt","NPT"]], st.ensemble);
    html += paramSelect("eq-tcoupl-" + i, "T-coupl", [["v-rescale","V-rescale"],["nose-hoover","Nose-Hoover"],["berendsen","Berendsen"]], st.tcoupl);
    html += paramText("eq-tau-t-" + i, "τ_t", st.tau_t, "narrow");
    html += paramNumber("eq-temperature-" + i, "Ref T (K)", st.temperature, 1, 1000, 1, "narrow");
    html += paramSelect("eq-comm-mode-" + i, "COM removal", [["linear","Linear translation"],["angular","Angular"],["none","Disabled"]], st.comm_mode);
    html += paramSelect("eq-comm-grps-" + i, "COM group(s)", commGroupOptions(), st.comm_grps);
    html += paramNumber("eq-nstcomm-" + i, "COM interval", st.nstcomm, 0, 1000000, 1, "wide");
    html += paramSelect("eq-constraints-" + i, "Constraints", [["none","None"],["h-bonds","H-bonds"],["all-bonds","All bonds"],["h-angles","H + angles"],["all-angles","All angles"]], st.constraints);
    html += '</div>';

    // NPT-only row (shown for NPT stages)
    if (st.ensemble === "npt") {
      html += '<div class="param-row npt-params">';
      html += paramSelect("eq-pcoupl-" + i, "P-coupl", [["C-rescale","C-rescale"],["berendsen","Berendsen"],["Parrinello-Rahman","P-R"]], st.pcoupl || "C-rescale");
      html += paramSelect("eq-pcoupl-type-" + i, "Pressure geometry", [["semisotropic","Semi-isotropic"],["isotropic","Isotropic"]], st.pcoupl_type);
      html += paramText("eq-tau-p-" + i, "τ_p", st.tau_p || "5.0", "narrow");
      html += paramText("eq-ref-p-" + i, "Ref P (bar)", st.ref_p || "1.0", "narrow");
      html += paramText("eq-compress-" + i, "Compress (bar⁻¹)", st.compress || "4.5e-5", "wide");
      html += '</div>';
    }

    html += '<div class="param-row">';
    html += paramNumber("eq-nstxout-compressed-" + i, "XTC interval", st.nstxout_compressed, 0, 100000000, 100, "wide");
    html += paramNumber("eq-nstxout-" + i, "Full coord interval", st.nstxout, 0, 100000000, 100, "wide");
    html += paramNumber("eq-nstvout-" + i, "Velocity interval", st.nstvout, 0, 100000000, 100, "wide");
    html += paramNumber("eq-nstfout-" + i, "Force interval", st.nstfout, 0, 100000000, 100, "wide");
    html += paramNumber("eq-nstcalcenergy-" + i, "Energy calc", st.nstcalcenergy, 1, 100000000, 1, "wide");
    html += paramNumber("eq-nstenergy-" + i, "Energy output", st.nstenergy, 0, 100000000, 100, "wide");
    html += paramNumber("eq-nstlog-" + i, "Log interval", st.nstlog, 0, 100000000, 100, "wide");
    html += '</div>';
    html += renderNonbondFields("eq-" + i, st, true);
    html += paramTextarea("eq-mdp-overrides-" + i, "Stage-specific advanced overrides", st.mdp_overrides_text || "",
      "One key = value per line. These values apply only to this stage.");

    html += '</div></div>';
  });

  // ---- Production ----
  _prodIters.forEach(function(pr, pi) {
    var productionEnabled = pr.enabled !== false;
    var repeats = Math.max(1, pr.repeat || 1);
    var segmentNs = pr.nsteps * pr.dt / 1000000;
    var timeNs = segmentNs * repeats;
    var frames = Math.floor(pr.nsteps / Math.max(pr.nstxout_compressed || 1, 1)) * repeats;
    html += '<div class="sim-stage-card" data-run-card="prod' + pi + '" style="border-color:#6366f1;opacity:' + (productionEnabled ? '1' : '0.55') + ';">';
    html += '<div class="sim-stage-header open" data-stage="prod' + pi + '" onclick="toggleStageCard(this)">';
    html += '<span class="sim-stage-icon" style="color:#6366f1;">&#9679;</span>';
    html += '<span class="sim-stage-title" style="color:#6366f1;">Production' + (_prodIters.length > 1 ? ' #' + (pi+1) : '') + '</span>';
    html += '<span class="sim-stage-summary">' + repeats + ' segment' + (repeats === 1 ? '' : 's') + ' × ' + segmentNs.toFixed(1) + ' ns = ' + timeNs.toFixed(1) + ' ns &nbsp;|&nbsp; ' + frames.toLocaleString() + ' frames</span>';
    html += stageEnabledControl("prod-enabled-" + pi, productionEnabled);
    html += '</div>';
    html += '<div class="sim-stage-body open" data-stage="prod' + pi + '">';

    html += '<div class="param-row">';
    html += paramNumber("prod-dt-" + pi, "Timestep (fs)", pr.dt, 0.5, 5.0, 0.5, "narrow");
    html += paramNumber("prod-nsteps-" + pi, "Steps per segment", pr.nsteps, 10000, 500000000, 10000, "wide");
    html += paramNumber("prod-repeat-" + pi, "Segments", repeats, 1, 100, 1, "narrow");
    html += '<span class="hint" style="align-self:flex-end;margin-bottom:2px;" id="prod-time-' + pi + '">' + timeNs.toFixed(1) + ' ns total</span>';
    html += '</div>';

    html += '<div class="param-row">';
    html += paramNumber("prod-nstxout-compressed-" + pi, "XTC interval", pr.nstxout_compressed, 0, 100000000, 100, "wide");
    html += '<span class="hint" style="align-self:flex-end;margin-bottom:2px;" id="prod-frames-' + pi + '">' + frames.toLocaleString() + ' frames</span>';
    html += paramSelect("prod-tcoupl-" + pi, "T-coupl", [["v-rescale","V-rescale"],["nose-hoover","Nose-Hoover"]], pr.tcoupl);
    html += paramText("prod-tau-t-" + pi, "τ_t", pr.tau_t, "narrow");
    html += paramNumber("prod-temperature-" + pi, "Ref T (K)", pr.temperature, 1, 1000, 1, "narrow");
    html += paramSelect("prod-constraints-" + pi, "Constraints", [["none","None"],["h-bonds","H-bonds"],["all-bonds","All bonds"],["h-angles","H + angles"],["all-angles","All angles"]], pr.constraints);
    html += '</div>';

    html += '<div class="param-row">';
    html += paramSelect("prod-pcoupl-" + pi, "P-coupl", [["C-rescale","C-rescale"],["Parrinello-Rahman","P-R"]], pr.pcoupl || "C-rescale");
    html += paramSelect("prod-pcoupl-type-" + pi, "Pressure geometry", [["semisotropic","Semi-isotropic"],["isotropic","Isotropic"]], pr.pcoupl_type);
    html += paramText("prod-tau-p-" + pi, "τ_p", pr.tau_p, "narrow");
    html += paramText("prod-ref-p-" + pi, "Ref P (bar)", pr.ref_p, "narrow");
    html += paramText("prod-compress-" + pi, "Compress", pr.compress, "wide");
    html += paramSelect("prod-comm-mode-" + pi, "COM removal", [["linear","Linear translation"],["angular","Angular"],["none","Disabled"]], pr.comm_mode);
    html += paramSelect("prod-comm-grps-" + pi, "COM group(s)", commGroupOptions(), pr.comm_grps);
    html += paramNumber("prod-nstcomm-" + pi, "COM interval", pr.nstcomm, 0, 1000000, 1, "wide");
    html += '</div>';

    html += '<div class="param-row">';
    html += paramNumber("prod-nstxout-" + pi, "Full coord interval", pr.nstxout, 0, 100000000, 100, "wide");
    html += paramNumber("prod-nstvout-" + pi, "Velocity interval", pr.nstvout, 0, 100000000, 100, "wide");
    html += paramNumber("prod-nstfout-" + pi, "Force interval", pr.nstfout, 0, 100000000, 100, "wide");
    html += paramNumber("prod-nstcalcenergy-" + pi, "Energy calc", pr.nstcalcenergy, 1, 100000000, 1, "wide");
    html += paramNumber("prod-nstenergy-" + pi, "Energy output", pr.nstenergy, 0, 100000000, 100, "wide");
    html += paramNumber("prod-nstlog-" + pi, "Log interval", pr.nstlog, 0, 100000000, 100, "wide");
    html += '</div>';
    html += renderNonbondFields("prod-" + pi, pr, true);
    html += paramTextarea("prod-mdp-overrides-" + pi, "Iteration-specific advanced overrides", pr.mdp_overrides_text || "",
      "One key = value per line. These values apply only to this production definition.");

    html += '</div></div>';
  });

  // Add iteration button
  html += '<button type="button" class="btn" id="add-prod-iter-btn" style="margin-top:4px;font-size:12px;">+ Add Production Iteration</button>';
  html += '<p class="hint">Segments run strictly in sequence. Each segment receives the previous segment checkpoint and writes a separate MDP/output prefix for safe restart.</p>';

  container.innerHTML = html;

  // Wire iteration button
  var addBtn = document.getElementById("add-prod-iter-btn");
  if (addBtn) {
    addBtn.addEventListener("click", function() {
      var last = _prodIters[_prodIters.length - 1] || Object.assign({}, _DEFAULT_OUTPUT, { nsteps: 5000000, repeat: 1, dt: 2.0, nstxout_compressed: 10000, tcoupl: "v-rescale", tau_t: "1.0", tau_p: "5.0", ref_p: "1.0", compress: "4.5e-5", nstcomm: 100 });
      var added = JSON.parse(JSON.stringify(last));
      added.enabled = true;
      added.repeat = 1;
      _prodIters.push(added);
      renderSimStages();
    });
  }

  // Wire all stage inputs for real-time updates
  wireStageInputs();
  updateAllTimeDisplays();
}

function paramNumber(id, label, value, min, max, step, cls) {
  return '<span class="param-item"><label>' + label + '</label><input type="number" id="' + id + '" value="' + escapeHtml(value) + '" min="' + min + '" max="' + max + '" step="' + step + '" class="' + (cls||'') + '"></span>';
}

function paramText(id, label, value, cls) {
  return '<span class="param-item"><label>' + label + '</label><input type="text" id="' + id + '" value="' + escapeHtml(value) + '" class="' + (cls||'') + '"></span>';
}

function paramSelect(id, label, options, selected) {
  var opts = options.map(function(o) {
    return '<option value="' + o[0] + '"' + (o[0] === selected ? ' selected' : '') + '>' + o[1] + '</option>';
  }).join("");
  return '<span class="param-item"><label>' + label + '</label><select id="' + id + '">' + opts + '</select></span>';
}

function stageEnabledControl(id, enabled) {
  return '<label class="hint" style="margin-left:auto;display:flex;align-items:center;gap:5px;" onclick="event.stopPropagation();">' +
    '<input type="checkbox" id="' + id + '"' + (enabled ? ' checked' : '') + '> Run stage</label>';
}

function commGroupOptions() {
  var options = [["System", "System — all atoms (recommended)"]];
  var modules = (state.taskType && state.taskType.visible_modules) || [];
  if (modules.indexOf("membrane") >= 0) {
    options.push(["SOLU_MEMB SOLV", "SOLU_MEMB + SOLV"]);
    options.push(["SOLU MEMB SOLV", "SOLU + MEMB + SOLV"]);
  } else if (!state.taskType || state.taskType.pipeline !== "liquid") {
    options.push(["SOLU SOLV", "SOLU + SOLV"]);
  }
  return options;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function paramTextarea(id, label, value, hint) {
  return '<div class="param-textarea"><label for="' + id + '">' + label + '</label>' +
    '<textarea id="' + id + '" rows="3" spellcheck="false" placeholder="example: nstlist = 20">' + escapeHtml(value) + '</textarea>' +
    '<span class="hint">' + hint + '</span></div>';
}

function toggleStageCard(header) {
  var body = header.nextElementSibling;
  if (!body) return;
  var open = body.classList.toggle("open");
  header.classList.toggle("open", open);
}

function wireStageInputs() {
  // Collect all number/text inputs in the simparams panel
  var container = document.getElementById("simparams-stages");
  if (!container) return;
  container.querySelectorAll("input, select, textarea").forEach(function(el) {
    el.addEventListener("input", function() { readStageParams(); updateAllTimeDisplays(); });
    el.addEventListener("change", function() {
      readStageParams();
      updateAllTimeDisplays();
      if (el.id.indexOf("eq-ensemble-") === 0) renderSimStages();
    });
  });
}

function readStageParams() {
  function numberValue(id, fallback) {
    var value = parseFloat(getVal(id));
    return Number.isFinite(value) ? value : fallback;
  }
  function integerValue(id, fallback) {
    var value = Number(getVal(id));
    return Number.isInteger(value) ? value : fallback;
  }
  function readNonbondFields(prefix, target, includeNstlist) {
    if (includeNstlist) {
      target.nstlist = integerValue(prefix + "-nstlist", target.nstlist);
    }
    target.rlist = numberValue(prefix + "-rlist", target.rlist);
    target.vdw_modifier = getVal(prefix + "-vdw-modifier") || target.vdw_modifier;
    target.rvdw_switch = numberValue(prefix + "-rvdw-switch", target.rvdw_switch);
    target.rvdw = numberValue(prefix + "-rvdw", target.rvdw);
    target.rcoulomb = numberValue(prefix + "-rcoulomb", target.rcoulomb);
    target.fourierspacing = numberValue(prefix + "-fourierspacing", target.fourierspacing);
    target.dispcorr = getVal(prefix + "-dispcorr") || target.dispcorr;
  }
  _simHardware.mode = getVal("sim-hw-mode") || _simHardware.mode;
  _simHardware.cpu_threads = integerValue(
    "sim-hw-cpu", _simHardware.cpu_threads
  );
  _simHardware.mpi_ranks = integerValue(
    "sim-hw-mpi", _simHardware.mpi_ranks
  );
  _simHardware.gpu_count = integerValue(
    "sim-hw-gpu-count", _simHardware.gpu_count
  );
  _simHardware.use_gpu =
    document.getElementById("sim-hw-use-gpu")?.checked === true;
  _simHardware.gpu_ids = getVal("sim-hw-gpu-ids") || "";
  _simHardware.gmx_command = getVal("sim-hw-gmx") || _simHardware.gmx_command;
  _simHardware.mpi_launcher =
    getVal("sim-hw-launcher") || _simHardware.mpi_launcher;
  _simHardware.pin = getVal("sim-hw-pin") || _simHardware.pin;
  var ompOutput = document.getElementById("sim-hw-omp");
  if (ompOutput) {
    ompOutput.textContent = (
      _simHardware.cpu_threads > 0 &&
      _simHardware.mpi_ranks > 0 &&
      _simHardware.cpu_threads % _simHardware.mpi_ranks === 0
    ) ? String(_simHardware.cpu_threads / _simHardware.mpi_ranks) : "invalid";
  }
  // Read EM
  _DEFAULT_EM.integrator = getVal("em-integrator") || _DEFAULT_EM.integrator || "steep";
  _DEFAULT_EM.nsteps = integerValue("em-nsteps", _DEFAULT_EM.nsteps);
  _DEFAULT_EM.emtol = numberValue("em-emtol", _DEFAULT_EM.emtol);
  _DEFAULT_EM.emstep = numberValue("em-emstep", _DEFAULT_EM.emstep);
  _DEFAULT_EM.nstlist = integerValue("em-nstlist", _DEFAULT_EM.nstlist);
  _DEFAULT_EM.constraints = getVal("em-constraints") || _DEFAULT_EM.constraints;
  _DEFAULT_EM.bb = numberValue("em-bb", _DEFAULT_EM.bb);
  _DEFAULT_EM.sc = numberValue("em-sc", _DEFAULT_EM.sc);
  _DEFAULT_EM.lipid = numberValue("em-lipid", _DEFAULT_EM.lipid);
  _DEFAULT_EM.dih = numberValue("em-dih", _DEFAULT_EM.dih);
  readNonbondFields("em", _DEFAULT_EM, false);
  _DEFAULT_EM.mdp_overrides_text = getVal("em-mdp-overrides") || "";
  // Read EQ stages
  for (var i = 0; i < _simStages.length; i++) {
    var st = _simStages[i];
    st.enabled = document.getElementById("eq-enabled-" + i)?.checked !== false;
    st.dt = numberValue("eq-dt-" + i, st.dt);
    st.nsteps = integerValue("eq-nsteps-" + i, st.nsteps);
    st.gen_seed = integerValue("eq-gen-seed-" + i, st.gen_seed);
    st.bb = numberValue("eq-bb-" + i, st.bb);
    st.sc = numberValue("eq-sc-" + i, st.sc);
    st.lipid = numberValue("eq-lipid-" + i, st.lipid);
    st.dih = numberValue("eq-dih-" + i, st.dih);
    st.ensemble = getVal("eq-ensemble-" + i) || st.ensemble;
    st.tcoupl = getVal("eq-tcoupl-" + i) || st.tcoupl;
    st.tau_t = getVal("eq-tau-t-" + i) || st.tau_t;
    st.temperature = numberValue("eq-temperature-" + i, st.temperature);
    st.comm_mode = getVal("eq-comm-mode-" + i) || st.comm_mode;
    st.comm_grps = getVal("eq-comm-grps-" + i) || st.comm_grps;
    st.nstcomm = integerValue("eq-nstcomm-" + i, st.nstcomm);
    st.constraints = getVal("eq-constraints-" + i) || st.constraints;
    ["nstxout_compressed","nstxout","nstvout","nstfout","nstcalcenergy","nstenergy","nstlog"].forEach(function(key) {
      st[key] = integerValue("eq-" + key.replace(/_/g, "-") + "-" + i, st[key]);
    });
    st.mdp_overrides_text = getVal("eq-mdp-overrides-" + i) || "";
    if (st.ensemble === "npt") {
      st.pcoupl = getVal("eq-pcoupl-" + i) || st.pcoupl;
      st.pcoupl_type = getVal("eq-pcoupl-type-" + i) || st.pcoupl_type;
      st.tau_p = getVal("eq-tau-p-" + i) || st.tau_p;
      st.ref_p = getVal("eq-ref-p-" + i) || st.ref_p;
      st.compress = getVal("eq-compress-" + i) || st.compress;
    }
    readNonbondFields("eq-" + i, st, true);
  }
  // Read prod stages
  for (var p = 0; p < _prodIters.length; p++) {
    var pr = _prodIters[p];
    pr.enabled = document.getElementById("prod-enabled-" + p)?.checked !== false;
    pr.dt = numberValue("prod-dt-" + p, pr.dt);
    pr.nsteps = integerValue("prod-nsteps-" + p, pr.nsteps);
    pr.repeat = integerValue("prod-repeat-" + p, pr.repeat || 1);
    ["nstxout_compressed","nstxout","nstvout","nstfout","nstcalcenergy","nstenergy","nstlog"].forEach(function(key) {
      pr[key] = integerValue("prod-" + key.replace(/_/g, "-") + "-" + p, pr[key]);
    });
    pr.tcoupl = getVal("prod-tcoupl-" + p) || pr.tcoupl;
    pr.tau_t = getVal("prod-tau-t-" + p) || pr.tau_t;
    pr.temperature = numberValue("prod-temperature-" + p, pr.temperature);
    pr.constraints = getVal("prod-constraints-" + p) || pr.constraints;
    pr.tau_p = getVal("prod-tau-p-" + p) || pr.tau_p;
    pr.ref_p = getVal("prod-ref-p-" + p) || pr.ref_p;
    pr.compress = getVal("prod-compress-" + p) || pr.compress;
    pr.pcoupl = getVal("prod-pcoupl-" + p) || pr.pcoupl;
    pr.pcoupl_type = getVal("prod-pcoupl-type-" + p) || pr.pcoupl_type;
    pr.comm_mode = getVal("prod-comm-mode-" + p) || pr.comm_mode;
    pr.comm_grps = getVal("prod-comm-grps-" + p) || pr.comm_grps;
    pr.nstcomm = integerValue("prod-nstcomm-" + p, pr.nstcomm);
    readNonbondFields("prod-" + p, pr, true);
    pr.mdp_overrides_text = getVal("prod-mdp-overrides-" + p) || "";
  }
}

function parseMdpOverrides(text, label) {
  var result = {};
  String(text || "").split(/\r?\n/).forEach(function(raw, index) {
    var line = raw.trim();
    if (!line || line.charAt(0) === ";" || line.charAt(0) === "#") return;
    var match = line.match(/^([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(\S(?:.*\S)?)$/);
    if (!match) throw new Error(label + ", line " + (index + 1) + ": expected key = value");
    if (Object.prototype.hasOwnProperty.call(result, match[1])) {
      throw new Error(label + ": duplicate key " + match[1]);
    }
    result[match[1]] = match[2];
  });
  return result;
}

function collectSimulationParams() {
  readStageParams();
  if (!Number.isInteger(_simHardware.cpu_threads) || _simHardware.cpu_threads < 1) {
    throw new Error("Total CPU threads must be a positive integer.");
  }
  if (!Number.isInteger(_simHardware.mpi_ranks) || _simHardware.mpi_ranks < 1 ||
      _simHardware.cpu_threads % _simHardware.mpi_ranks !== 0) {
    throw new Error("MPI ranks must be a positive exact divisor of total CPU threads.");
  }
  if (_simHardware.use_gpu &&
      !/^[0-9]+(,[0-9]+)*$/.test(String(_simHardware.gpu_ids).trim())) {
    throw new Error("GPU IDs must be comma-separated logical integers, for example 0 or 0,1.");
  }
  if (_simHardware.use_gpu) {
    var gpuIds = String(_simHardware.gpu_ids).trim().split(",");
    if (!Number.isInteger(_simHardware.gpu_count) || _simHardware.gpu_count < 1 ||
        _simHardware.gpu_count !== gpuIds.length) {
      throw new Error("GPU count must equal the number of selected GPU IDs.");
    }
    if (new Set(gpuIds).size !== gpuIds.length) {
      throw new Error("GPU IDs must be unique.");
    }
    if (_simHardware.gpu_count > _simHardware.mpi_ranks) {
      throw new Error("MPI ranks must be at least the selected GPU count.");
    }
  }
  if (!/^[A-Za-z0-9_./+-]+$/.test(String(_simHardware.gmx_command).trim())) {
    throw new Error("GROMACS command must be one executable name or path without shell syntax.");
  }
  var eq = _simStages.map(function(stage, index) {
    var copy = Object.assign({}, stage);
    copy.dt_unit = "fs";
    copy.mdp_overrides = copy.enabled === false ? {} : parseMdpOverrides(
      copy.mdp_overrides_text, "Equilibration " + (index + 1) + " overrides"
    );
    delete copy.mdp_overrides_text;
    return copy;
  });
  var prod = _prodIters.map(function(stage, index) {
    var copy = Object.assign({}, stage);
    copy.dt_unit = "fs";
    copy.mdp_overrides = copy.enabled === false ? {} : parseMdpOverrides(
      copy.mdp_overrides_text, "Production " + (index + 1) + " overrides"
    );
    delete copy.mdp_overrides_text;
    return copy;
  });
  if (!eq.some(function(stage) { return stage.enabled !== false; })) {
    throw new Error("Enable at least one equilibration stage so velocities and continuation are initialized safely.");
  }
  if (!prod.some(function(stage) { return stage.enabled !== false; })) {
    throw new Error("Enable at least one production stage.");
  }
  return {
    schema_version: 2,
    minimization: {
      integrator: _DEFAULT_EM.integrator || "steep",
      nsteps: _DEFAULT_EM.nsteps,
      emtol: _DEFAULT_EM.emtol,
      emstep: _DEFAULT_EM.emstep,
      nstlist: _DEFAULT_EM.nstlist,
      constraints: _DEFAULT_EM.constraints,
      bb: _DEFAULT_EM.bb,
      sc: _DEFAULT_EM.sc,
      lipid: _DEFAULT_EM.lipid,
      dih: _DEFAULT_EM.dih,
      rlist: _DEFAULT_EM.rlist,
      vdw_modifier: _DEFAULT_EM.vdw_modifier,
      rvdw_switch: _DEFAULT_EM.rvdw_switch,
      rvdw: _DEFAULT_EM.rvdw,
      rcoulomb: _DEFAULT_EM.rcoulomb,
      fourierspacing: _DEFAULT_EM.fourierspacing,
      dispcorr: _DEFAULT_EM.dispcorr,
      mdp_overrides: parseMdpOverrides(_DEFAULT_EM.mdp_overrides_text, "Minimization overrides")
    },
    eq_stages: eq,
    prod_iters: prod,
    hardware: {
      mode: _simHardware.mode,
      cpu_threads: _simHardware.cpu_threads,
      mpi_ranks: _simHardware.mpi_ranks,
      use_gpu: _simHardware.use_gpu,
      gpu_count: _simHardware.use_gpu ? _simHardware.gpu_count : 0,
      gpu_ids: _simHardware.use_gpu ? String(_simHardware.gpu_ids).trim() : "",
      gmx_command: String(_simHardware.gmx_command).trim(),
      mpi_launcher: _simHardware.mpi_launcher,
      pin: _simHardware.pin
    }
  };
}

function getVal(id) {
  var el = document.getElementById(id);
  return el ? el.value : null;
}

function updateAllTimeDisplays() {
  // EQ stages
  for (var i = 0; i < _simStages.length; i++) {
    var st = _simStages[i];
    var ns = st.nsteps * st.dt / 1000000;
    var el = document.getElementById("eq-time-" + i);
    if (el) el.textContent = ns.toFixed(1) + " ns";
    // Update header summary
    var card = document.querySelector('[data-stage="eq' + i + '"]');
    if (card && card.classList.contains("sim-stage-header")) {
      var runCard = card.closest(".sim-stage-card");
      if (runCard) runCard.style.opacity = st.enabled === false ? "0.55" : "1";
      var summary = card.querySelector(".sim-stage-summary");
      if (summary) summary.textContent = (st.enabled === false ? "SKIPPED | " : "") + st.nsteps.toLocaleString() + " steps × " + st.dt.toFixed(1) + " fs = " + ns.toFixed(1) + " ns  |  BB=" + st.bb + " SC=" + st.sc + " Lipid=" + st.lipid;
    }
  }
  // Prod stages
  for (var p = 0; p < _prodIters.length; p++) {
    var pr = _prodIters[p];
    var repeats = Math.max(1, pr.repeat || 1);
    var segmentNs = pr.nsteps * pr.dt / 1000000;
    var ns = segmentNs * repeats;
    var frames = Math.floor(pr.nsteps / Math.max(pr.nstxout_compressed || 1, 1)) * repeats;
    var timeEl = document.getElementById("prod-time-" + p);
    var frameEl = document.getElementById("prod-frames-" + p);
    if (timeEl) timeEl.textContent = ns.toFixed(1) + " ns total";
    if (frameEl) frameEl.textContent = frames.toLocaleString() + " frames";
    // Update header
    var card = document.querySelector('[data-stage="prod' + p + '"]');
    if (card && card.classList.contains("sim-stage-header")) {
      var productionCard = card.closest(".sim-stage-card");
      if (productionCard) productionCard.style.opacity = pr.enabled === false ? "0.55" : "1";
      var summary = card.querySelector(".sim-stage-summary");
      if (summary) summary.textContent = (pr.enabled === false ? "SKIPPED | " : "") + repeats + " segment" + (repeats === 1 ? "" : "s") + " × " + segmentNs.toFixed(1) + " ns = " + ns.toFixed(1) + " ns  |  " + frames.toLocaleString() + " frames";
    }
  }
}


// ===================================================================
// System Verification Viewer
// ===================================================================

// ---- Capture viewer metrics for comparison ----
let _previewConfig = null;  // stored for inclusion in build payload

function _captureViewerMetrics() {
  var pdbContent = _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);
  if (!pdbContent) return null;

  // ---- Box dimensions (matching renderSystemViewer computation) ----
  // Use CA-only atoms for protein metrics (robust against protonation/H addition)
  var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  var zMin = Infinity, zMax = -Infinity;
  var xMinCA = Infinity, xMaxCA = -Infinity, yMinCA = Infinity, yMaxCA = -Infinity;
  var zMinCA = Infinity, zMaxCA = -Infinity;
  var lines = pdbContent.split('\n');
  for (var li = 0; li < lines.length; li++) {
    var l = lines[li];
    if (l.indexOf('ATOM') === 0 || l.indexOf('HETATM') === 0) {
      var atomName = l.substring(12, 16).trim();
      var px = parseFloat(l.substring(30, 38)) / 10.0;  // Å → nm
      var py = parseFloat(l.substring(38, 46)) / 10.0;
      var pz = parseFloat(l.substring(46, 54)) / 10.0;
      if (!isNaN(px) && !isNaN(py) && !isNaN(pz)) {
        // All-atom extent (for box calculation)
        if (px < xMin) xMin = px; if (px > xMax) xMax = px;
        if (py < yMin) yMin = py; if (py > yMax) yMax = py;
        if (pz < zMin) zMin = pz; if (pz > zMax) zMax = pz;
        // CA-only extent (for protein metrics comparison)
        if (atomName === 'CA') {
          if (px < xMinCA) xMinCA = px; if (px > xMaxCA) xMaxCA = px;
          if (py < yMinCA) yMinCA = py; if (py > yMaxCA) yMaxCA = py;
          if (pz < zMinCA) zMinCA = pz; if (pz > zMaxCA) zMaxCA = pz;
        }
      }
    }
  }

  var isSolvator = state.taskType && state.taskType.pipeline === 'solvator';
  // Box dimensions use ALL-atom extent (for box calculation)
  var protXY = isFinite(xMin) ? Math.max(xMax - xMin, yMax - yMin) : 3.0;
  var protZ = isFinite(zMin) ? (zMax - zMin) : 6.0;

  // Protein metrics use CA-only (robust against protonation/H changes)
  // Fall back to all-atom if no CA atoms found
  var useCA = isFinite(xMinCA);
  var protComX = useCA ? (xMinCA + xMaxCA) / 2.0 : (xMin + xMax) / 2.0;
  var protComY = useCA ? (yMinCA + yMaxCA) / 2.0 : (yMin + yMax) / 2.0;
  var protComZ = useCA ? (zMinCA + zMaxCA) / 2.0 : (zMin + zMax) / 2.0;
  var protExtX = useCA ? (xMaxCA - xMinCA) : (xMax - xMin);
  var protExtY = useCA ? (yMaxCA - yMinCA) : (yMax - yMin);
  var protExtZ = useCA ? (zMaxCA - zMinCA) : (zMax - zMin);

  var protein = {
    center_of_mass_nm: [roundTo(protComX, 3), roundTo(protComY, 3), roundTo(protComZ, 3)],
    min_nm: [roundTo(useCA ? xMinCA : xMin, 3), roundTo(useCA ? yMinCA : yMin, 3), roundTo(useCA ? zMinCA : zMin, 3)],
    max_nm: [roundTo(useCA ? xMaxCA : xMax, 3), roundTo(useCA ? yMaxCA : yMax, 3), roundTo(useCA ? zMaxCA : zMax, 3)],
    extent_nm: [roundTo(protExtX, 3), roundTo(protExtY, 3), roundTo(protExtZ, 3)],
  };

  // Box dimensions
  var mPadEl = document.getElementById('membrane-pad');
  var mPad = 2.0; if (mPadEl) { var mpv = parseFloat(mPadEl.value); if (!isNaN(mpv)) mPad = mpv; }
  var zPad; { var zv = parseFloat(document.getElementById('box-padding')?.value); zPad = isNaN(zv) ? 2.0 : zv; }
  var dhZ = (_dominantLipidDHH || 3.8);
  var boxXY = Math.max(protXY + 2 * mPad, 4.0);
  var boxZ;
  if (isSolvator) {
    boxXY = Math.max(protXY + 2 * zPad, 4.0);
    boxZ = Math.max(protZ, 6.0) + 2 * zPad;
  } else {
    boxZ = Math.max(protZ, dhZ * 1.8) + 2 * zPad;
  }

  // Membrane metrics
  var membrane = null;
  if (!isSolvator) {
    var halfThick = dhZ * 0.5;
    membrane = {
      midplane_z_nm: roundTo(_orientZOffset || 0, 3),
      half_thickness_nm: roundTo(halfThick, 3),
    };
  }

  return {
    box_dimensions_nm: [roundTo(boxXY, 3), roundTo(boxXY, 3), roundTo(boxZ, 3)],
    protein: protein,
    membrane: membrane,
  };
}

function roundTo(val, decimals) {
  var p = Math.pow(10, decimals);
  return Math.round(val * p) / p;
}

function initSystemVerification() {
  var btn = document.getElementById("verify-check-btn");
  if (btn) {
    btn.addEventListener("click", async function() {
      // ---- Capture viewer metrics ----
      _previewConfig = _captureViewerMetrics();
      var pdbContent = _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);

      // ---- Send preview to backend ----
      if (_previewConfig && state.taskId) {
        try {
          var payload = {
            task_id: state.taskId,
            oriented_pdb: pdbContent || "",
            box_dimensions_nm: _previewConfig.box_dimensions_nm,
            protein: _previewConfig.protein,
            membrane: _previewConfig.membrane,
          };
          var resp = await fetch('/api/preview-pdb', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          var result = await resp.json();
          if (result.status === 'ok') {
            console.log('Preview PDB saved:', result.preview_resource);
          }
        } catch (e) {
          console.warn('Failed to save preview PDB:', e);
          // Non-fatal — build can still proceed without preview comparison
        }
      }

      _systemVerified = true;
      _checkedSteps.add('verify');
      document.getElementById("verify-status").textContent = "✓ System verified. You may now proceed to Force Field.";
      document.getElementById("verify-status").style.color = "#059669";
      updateNextButtonState();
    });
  }
}


/** Build a valid PDB HETATM line with correct column alignment. */
function pdbHetatm(serial, atomName, resName, chain, resid, x, y, z, element) {
  // PDB format columns:
  // 1-6: "HETATM", 7-11: serial, 12: space, 13-16: atomName, 17: altLoc,
  // 18-20: resName, 21: space, 22: chain, 23-26: resid, 27-30: spaces,
  // 31-38: x, 39-46: y, 47-54: z, 55-60: occ, 61-66: temp, 77-78: element
  var line = "HETATM" + String(serial).padStart(5, " ") +
    " " + String(atomName || "P").padEnd(4, " ") +
    String(resName || "LIP").padStart(3, " ") +
    " " + String(chain || "A") +
    String(resid || 1).toString().padStart(4, " ") +
    "    " +
    parseFloat(x).toFixed(3).padStart(8, " ") +
    parseFloat(y).toFixed(3).padStart(8, " ") +
    parseFloat(z).toFixed(3).padStart(8, " ") +
    "  1.00  0.00           " +
    String(element || "P").padStart(2, " ") +
    "\n";
  return line;
}

function renderSystemViewer() {
  var el = document.getElementById("verify-viewer");
  if (!el) return;
  if (!state.pdbInfo || !state.pdbInfo.pdb_content) {
    setTimeout(renderSystemViewer, 300);
    return;
  }
  if (typeof $3Dmol === "undefined") {
    window._cdmRetries3D = (window._cdmRetries3D || 0) + 1;
    if (window._cdmRetries3D > 30) { console.error('The bundled 3Dmol.js asset failed to load'); return; }
    setTimeout(renderSystemViewer, 500); return;
  }
  if (el.offsetWidth === 0 || el.offsetHeight === 0) {
    setTimeout(renderSystemViewer, 200);
    return;
  }

  // Destroy old viewer (cylinders/spheres are shapes, not cleared by removeAllModels)
  if (window._verifyViewer) {
    try { window._verifyViewer.clear(); } catch(e) {}
    window._verifyViewer = null;
  }
  while (el.firstChild) { el.removeChild(el.firstChild); }

  window._verifyViewer = $3Dmol.createViewer(el, { backgroundColor: "0xffffff", antialias: true });
  window._verifyViewer.setBackgroundColor('0xffffff');
  window._verifyViewer.setSlab(-100000, 100000);
  var v = window._verifyViewer;

  var isSolvator = state.taskType && state.taskType.pipeline === 'solvator';

  // ---- 1. Protein (oriented if PPM was run) ----
  var pdbForVerify = _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);
  v.addModel(pdbForVerify, "pdb");
  _applyUnifiedStyle(v, pdbForVerify);

  // ---- 2. Membrane (consistent with other viewers) ----
  if (!isSolvator) {
    var halfThick = (_dominantLipidDHH || 3.8) * 0.5;
    drawMembranePlane(v, 0.0, halfThick, 0.0, 0.0);
    v.setStyle({elem: 'X'}, {sphere: {radius: 1.2, color: '0x6b7280', opacity: 0.55}});
  }

  // ---- 3. Box wireframe (from actual configured parameters) ----
  var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  var lines = pdbForVerify.split('\n');
  for (var li = 0; li < lines.length; li++) {
    var l = lines[li];
    if (l.indexOf('ATOM') === 0 || l.indexOf('HETATM') === 0) {
      var px = parseFloat(l.substring(30, 38)) / 10.0;
      var py = parseFloat(l.substring(38, 46)) / 10.0;
      if (!isNaN(px) && !isNaN(py)) { if (px<xMin)xMin=px; if (px>xMax)xMax=px; if (py<yMin)yMin=py; if (py>yMax)yMax=py; }
    }
  }
  var protXY = isFinite(xMin) ? Math.max(xMax-xMin, yMax-yMin) : 3.0;
  var mPadEl = document.getElementById('membrane-pad');
  var mPad = 2.0; if (mPadEl) { var mpv=parseFloat(mPadEl.value); if (!isNaN(mpv)) mPad=mpv; }
  var boxXY = Math.max(protXY + 2*mPad, 4.0);
  var zPad; { var zv=parseFloat(document.getElementById('box-padding')?.value); zPad=isNaN(zv)?2.0:zv; }
  var protExtZ = _proteinExtent(pdbForVerify).z; var dhZ = (_dominantLipidDHH || 3.8); var boxZ = Math.max(protExtZ, dhZ * 1.8) + 2 * zPad;
  if (isSolvator) {
    boxXY = Math.max(protXY + 2*zPad, 4.0);
    boxZ = Math.max(protExtZ, 6.0) + 2*zPad;
  }
  var halfXY_A = (boxXY / 2.0) * 10.0;
  var halfZ_A = (boxZ / 2.0) * 10.0;

  // Apply PPM tilt + z_offset to box corners
  _drawTiltedBox(v, halfXY_A, halfZ_A, _orientZOffset * 10, _orientTilt, _orientPhi);

  // ---- 4. Ions (in water regions only — exclude membrane interior) ----
  if (!isSolvator) {
    var ionColors = {NA:'0x3b82f6',K:'0x8b5cf6',CL:'0xef4444',CA:'0x22c55e',MG:'0x10b981',ZN:'0x64748b'};
    var cations = window._getIonCations ? window._getIonCations() : (console.warn('ions.js not loaded - falling back to default cations ["NA"]'), ['NA']);
    var anions  = window._getIonAnions  ? window._getIonAnions()  : (console.warn('ions.js not loaded - falling back to default anions ["CL"]'), ['CL']);
    var nIon = 8;
    // Membrane occupies ±halfThick_A in Z; ions go above/below
    var halfThickA = halfThick * 10.0;  // Å
    function _randomIonZ() {
      // Pick upper or lower water region
      if (Math.random() < 0.5) {
        return -(halfThickA + Math.random() * (halfZ_A - halfThickA));  // below membrane
      } else {
        return halfThickA + Math.random() * (halfZ_A - halfThickA);     // above membrane
      }
    }
    for (var ci = 0; ci < nIon; ci++) {
      cations.forEach(function(cat) {
        var col = ionColors[cat] || '0x3b82f6';
        var rad = (cat==='CA'||cat==='MG'||cat==='ZN') ? 0.7 : 1.0;
        v.addSphere({center:{x:(Math.random()-0.5)*halfXY_A*2,y:(Math.random()-0.5)*halfXY_A*2,z:_randomIonZ()},radius:rad,color:col,opacity:0.65});
      });
      anions.forEach(function(ani) {
        var col = ionColors[ani] || '0xef4444';
        v.addSphere({center:{x:(Math.random()-0.5)*halfXY_A*2,y:(Math.random()-0.5)*halfXY_A*2,z:_randomIonZ()},radius:0.8,color:col,opacity:0.65});
      });
    }
  }

  v.zoomTo();
  v.render();
  v.setSlab(-100000, 100000);
}

// Lipid Mixing / Composition Editor
// ===================================================================

let _mixUpper = [{ name: 'POPC', ratio: 100 }];  // { name, ratio }
let _mixLower = [{ name: 'POPC', ratio: 100 }];
let _asymmetric = false;
let _compositionChecked = false;
let _compositionErrors = [];
function _invalidateMembraneBuild() {
  _compositionChecked = false;
  _membraneCheckpointPdb = null;
  _membraneActualBox = null;
}

function initLipidMixing() {
  const asymToggle = document.getElementById('asymmetric-bilayer');
  if (asymToggle) {
    asymToggle.addEventListener('change', () => {
      _asymmetric = asymToggle.checked;
      document.getElementById('lower-leaflet-section').classList.toggle('hidden', !_asymmetric);
      updateLeafletLabels();
      _invalidateMembraneBuild();
      updateCompositionStatus();
      if (!_asymmetric) {
        _mixLower = _mixUpper.map(m => ({...m}));
      }
      renderMixList('upper');
      renderMixList('lower');
      updateLipidCounts();
    });
  }

  // Add-lipid buttons
  document.querySelectorAll('.add-lipid-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const leaflet = btn.dataset.leaflet;
      const mix = leaflet === 'upper' ? _mixUpper : _mixLower;
      // Pick first lipid not already in the list
      const existing = new Set(mix.map(m => m.name));
      const selectedSource = selectedLipidParameterSource();
      const available = (_lipidPickerData.lipids || []).filter(l =>
        !existing.has(l.name) && (!selectedSource || (l.parameterizations || []).indexOf(selectedSource) >= 0)
      );
      if (!available.length) {
        alert('No additional validated lipids are available with ' + lipidParameterSourceLabel(selectedSource) + '.');
        return;
      }
      const pick = available[0].name;
      mix.push({ name: pick, ratio: 0 });
      _invalidateMembraneBuild();
      updateCompositionStatus();
      if (!_asymmetric && leaflet === 'upper') {
        _mixLower = _mixUpper.map(m => ({...m}));
      }
      normalizeRatios(leaflet === 'upper' ? _mixUpper : _mixLower);
      renderMixList('upper');
      renderMixList('lower');
    });
  });

  // Lipids-per-leaflet changes → clear check + refresh viewer
  const nLipidsEl = document.getElementById('n-lipids-per-leaflet');
  if (nLipidsEl) {
    nLipidsEl.addEventListener('input', () => { _invalidateMembraneBuild(); updateCompositionStatus(); renderMembraneViewer(); });
  }

  // Check button
  const checkBtn = document.getElementById('check-composition-btn');
  if (checkBtn) {
    checkBtn.addEventListener('click', () => checkComposition());
  }

  renderMixList('upper');
  renderMixList('lower');

  // Initialize 3D viewer after DOM settles
  setTimeout(function() { renderMembraneViewer(); }, 500);
}

// ---- 3D viewer: protein + box wireframe ----
var _membraneViewer = null;
var _membraneCheckpointPdb = null;  // set by checkComposition for WYSIWYG refresh
var _membraneActualBox = null;       // [box_x, box_y, box_z] in nm from checkpoint

async function renderMembraneViewer() {
  var el = document.getElementById('membrane-3d-viewer');
  if (!el) return;
  if (el.offsetWidth === 0 || el.offsetHeight === 0) return;
  if (typeof $3Dmol === 'undefined') return;

  // Use the membrane checkpoint PDB if available (set by checkComposition
  // after a successful build), otherwise the orient reference model.
  var hasCheckpoint = !!_membraneCheckpointPdb;
  var pdbContent = _membraneCheckpointPdb || _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);
  var isPureMembrane = state.taskType && state.taskType.pipeline === 'pure_membrane';
  if (!pdbContent && isPureMembrane) {
    pdbContent = 'CRYST1   40.000   40.000   70.000  90.00  90.00  90.00 P 1           1\nEND\n';
  }
  if (!pdbContent) return;

  // Box dimensions: use checkpoint values when available (WYSIWYG),
  // otherwise estimate from protein extent + padding (preview).
  var boxXY, boxZ;
  if (hasCheckpoint && _membraneActualBox) {
    boxXY = _membraneActualBox[0];
    boxZ = _membraneActualBox[2];
  } else {
    // Preview: compute box XY from user-specified lipids-per-leaflet
    var nLipids = 150;
    var nLipidsEl = document.getElementById('n-lipids-per-leaflet');
    if (nLipidsEl) { var nv = parseInt(nLipidsEl.value); if (!isNaN(nv) && nv >= 64) nLipids = nv; }
    // Weighted average APL
    var lipids = _lipidPickerData.lipids || [];
    var avgAPL = 0, totalRatio = 0;
    _mixUpper.forEach(function(m) {
      var ll = lipids.find(function(lll){return lll.name===m.name;});
      if (ll) { avgAPL += ll.area_per_lipid * m.ratio; totalRatio += m.ratio; }
    });
    if (totalRatio > 0) avgAPL /= totalRatio;
    if (avgAPL <= 0) avgAPL = 0.65;
    // Protein XY extent
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
    var protXY = isFinite(xMin) ? Math.max(xMax - xMin, yMax - yMin) : (isPureMembrane ? 0.0 : 3.0);
    // Reverse builder formula: n_lipids = boxXY² / APL * 1.30
    var lipidArea = nLipids * avgAPL / 1.30;
    var protArea = protXY * protXY;
    boxXY = Math.max(Math.sqrt(lipidArea + protArea), 4.0);
    var protExt = isPureMembrane ? {z: 0.0} : _proteinExtent(pdbContent);
    var dh = (_dominantLipidDHH || 3.8);
    boxZ = Math.max(protExt.z, dh * 1.8);
  }

  var halfXY_A = (boxXY / 2.0) * 10.0;
  var halfThick = (_dominantLipidDHH || 3.8) * 0.5;
  var membraneHalfZ_A = boxZ / 2.0 * 10.0;

  // Destroy old viewer (cylinders = shapes, not cleared by removeAllModels)
  if (_membraneViewer) {
    try { _membraneViewer.clear(); } catch(e) {}
    _membraneViewer = null;
  }
  while (el.firstChild) { el.removeChild(el.firstChild); }

  _membraneViewer = $3Dmol.createViewer(el, {backgroundColor: '0xffffff', antialias: true});
  _membraneViewer.setBackgroundColor('0xffffff');
  _membraneViewer.setSlab(-100000, 100000);
  var v = _membraneViewer;

  // Protein + lipids (checkpoint PDB) or just protein (preview)
  v.addModel(pdbContent, 'pdb');
  _applyUnifiedStyle(v, pdbContent);

  // Membrane plane — when checkpoint PDB is loaded the lipids are already
  // in their final positions (membrane midplane at Z=0 in the oriented
  // Membrane plane spheres — only in preview mode (before Check).
  // After Check the actual lipid molecules are visible in the PDB.
  if (!hasCheckpoint) {
    drawMembranePlane(v, 0.0, halfThick, 0.0, 0.0);
    v.setStyle({elem: 'X'}, {sphere: {radius: 1.2, color: '0x6b7280', opacity: 0.55}});
  }

  // Box wireframe
  // The preview protein coordinates already include orientation transforms;
  // the membrane box is the fixed laboratory-frame reference.
  var boxZOff = 0;
  var boxTilt = 0;
  var boxPhi = 0;
  _drawTiltedBox(v, halfXY_A, membraneHalfZ_A, boxZOff * 10, boxTilt, boxPhi);

  v.render();
  var label = document.getElementById('membrane-viewer-label');
  if (label) {
    var nLipidsLabel = 150;
    var nLipidsEl2 = document.getElementById('n-lipids-per-leaflet');
    if (nLipidsEl2) { var nv2 = parseInt(nLipidsEl2.value); if (!isNaN(nv2)) nLipidsLabel = nv2; }
    label.textContent = 'Box: ' + boxXY.toFixed(1) + '×' + boxXY.toFixed(1) + '×' + boxZ.toFixed(1)
      + ' nm  (n=' + (hasCheckpoint ? 'built' : nLipidsLabel) + '/leaflet)';
  }
}

function updateLeafletLabels() {
  const upperLabel = document.getElementById('leaflet-upper-label');
  if (upperLabel) {
    if (_asymmetric) {
      upperLabel.innerHTML = 'Upper Leaflet <span class="hint">(extracellular / outer)</span>';
    } else {
      upperLabel.innerHTML = 'Bilayer <span class="hint">(same composition both leaflets — count shown is per leaflet)</span>';
    }
  }
}

function renderMixList(leaflet) {
  const listEl = document.getElementById(`${leaflet}-lipid-list`);
  if (!listEl) return;
  const mix = leaflet === 'upper' ? _mixUpper : _mixLower;

  listEl.innerHTML = '';
  const lipids = _lipidPickerData.lipids || [];

  mix.forEach((entry, idx) => {
    const row = document.createElement('div');
    row.className = 'lipid-mix-row';

    // Lipid picker trigger (replaces plain <select>)
    const selWrap = document.createElement('div');
    selWrap.className = 'mix-lipid-picker';
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'mix-lipid-trigger';
    const lip = lipids.find(l => l.name === entry.name);
    trigger.innerHTML = `<span class="mix-lipid-trigger-name">${entry.name}</span><span class="mix-lipid-trigger-cat">${lip ? lip.category : ''}</span><span class="mix-lipid-trigger-arrow">&#9662;</span>`;
    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const dropdown = document.getElementById('lipid-picker-dropdown');
      const isOpen = dropdown && !dropdown.classList.contains('hidden');
      if (isOpen && _pickerTarget && _pickerTarget.leaflet === leaflet && _pickerTarget.idx === idx) {
        closeLipidDropdown();
      } else {
        _pickerTarget = { leaflet, idx };
        openLipidDropdown(trigger);
      }
    });
    selWrap.appendChild(trigger);
    row.appendChild(selWrap);

    // Ratio number input
    const ratioWrap = document.createElement('div');
    ratioWrap.className = 'mix-ratio';
    const numInput = document.createElement('input');
    numInput.type = 'number';
    numInput.min = 0;
    numInput.max = 100;
    numInput.step = 1;
    numInput.value = entry.ratio;
    numInput.className = 'mix-ratio-input';
    // On input: update data in-place, update display label, keep focus
    numInput.addEventListener('input', () => {
      const v = parseInt(numInput.value) || 0;
      mix[idx].ratio = Math.max(0, Math.min(100, v));
      _invalidateMembraneBuild();
      updateCompositionStatus();
    });
    // On blur/change: re-render to apply normalization if needed
    numInput.addEventListener('change', () => {
      renderMixList('upper');
      renderMixList('lower');
    });
    ratioWrap.appendChild(numInput);
    const pctLabel = document.createElement('span');
    pctLabel.className = 'mix-pct';
    pctLabel.textContent = '%';
    ratioWrap.appendChild(pctLabel);
    row.appendChild(ratioWrap);

    // Remove button (disabled if only 1)
    const rmBtn = document.createElement('button');
    rmBtn.className = 'mix-remove';
    rmBtn.textContent = '×';
    rmBtn.disabled = mix.length <= 1;
    rmBtn.addEventListener('click', () => {
      if (mix.length <= 1) return;
      mix.splice(idx, 1);
      _invalidateMembraneBuild();
      updateCompositionStatus();
      normalizeRatios(mix);
      if (!_asymmetric && leaflet === 'upper') {
        _mixLower = _mixUpper.map(m => ({...m}));
      }
      renderMixList('upper');
      renderMixList('lower');
    });
    row.appendChild(rmBtn);

    listEl.appendChild(row);
  });
}

function normalizeRatios(mix) {
  const total = mix.reduce((s, m) => s + m.ratio, 0);
  if (total === 0) {
    const eq = Math.floor(100 / mix.length);
    mix.forEach((m, i) => { m.ratio = i === mix.length - 1 ? 100 - eq * (mix.length - 1) : eq; });
  } else if (total !== 100) {
    const scale = 100 / total;
    let sum = 0;
    mix.forEach((m, i) => {
      if (i === mix.length - 1) {
        m.ratio = 100 - sum;
      } else {
        m.ratio = Math.round(m.ratio * scale);
        sum += m.ratio;
      }
    });
  }
}

/** Refresh all ratio display values for a leaflet and re-render. */
function refreshAllRatios() {
  renderMixList('upper');
  renderMixList('lower');
}

async function checkComposition() {
  var errors = [];
  const upperSum = _mixUpper.reduce((s, m) => s + m.ratio, 0);
  const lowerSum = _asymmetric ? _mixLower.reduce((s, m) => s + m.ratio, 0) : upperSum;
  if (upperSum !== 100 || lowerSum !== 100) {
    errors.push('Lipid ratios must sum to 100%');
  }

  // Validate lipids per leaflet
  var nLipidsEl = document.getElementById('n-lipids-per-leaflet');
  var totalLipids = nLipidsEl ? parseInt(nLipidsEl.value) : 150;
  if (isNaN(totalLipids) || totalLipids < 64) {
    errors.push('Minimum 64 lipids per leaflet required (recommended ≥100).');
  }

  if (errors.length > 0) {
    _invalidateMembraneBuild(); _compositionErrors = errors;
  } else {
    // Run membrane step on server
    var statusEl = document.getElementById('composition-status');
    if (statusEl) { statusEl.textContent = 'Running...'; statusEl.style.color = '#d97706'; }
    try {
      var cfg = buildModuleConfig().membrane || {};
      var resp = await fetch('/api/step/' + state.taskId + '/membrane', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: cfg }),
      });
      var result = await resp.json();
      if (result.status === 'ok') {
        _compositionChecked = true; _compositionErrors = [];
        _checkedSteps.add('membrane');
        var elapsed = (result.elapsed_s != null) ? result.elapsed_s + 's' : '?s';
        if (statusEl) { statusEl.textContent = '✓ Checked (' + elapsed + ')'; statusEl.style.color = '#059669'; }
        // Capture actual box dimensions from the server — the builder may
        // shrink the box after lipid placement and relaxation (WYSIWYG).
        if (result.metrics && result.metrics.box_dimensions_nm) {
          _membraneActualBox = result.metrics.box_dimensions_nm;
        }
        // Load the membrane checkpoint PDB so the viewer shows the actual
        // built membrane (WYSIWYG).  _membraneCheckpointPdb is used by
        // renderMembraneViewer — if set, it takes precedence over the
        // orient reference model.
        _membraneCheckpointPdb = await _loadStepViewerPdb('membrane') || null;
        if (typeof renderMembraneViewer === 'function') { renderMembraneViewer(); }
      } else {
        _invalidateMembraneBuild();
        _compositionErrors = [result.error || 'Server error'];
        if (statusEl) { statusEl.textContent = '✗ ' + (result.error || 'Failed'); statusEl.style.color = '#dc2626'; }
      }
    } catch(e) {
      _invalidateMembraneBuild();
      if (statusEl) { statusEl.textContent = '✗ Network error'; statusEl.style.color = '#dc2626'; }
    }
  }
  updateLipidCounts();
  updateCompositionStatus();
}

function updateCompositionStatus() {
  const el = document.getElementById('composition-status');
  if (!el) return;
  if (_compositionChecked && _compositionErrors.length === 0) {
    // Keep the elapsed time if already set by checkComposition
    if (!el.textContent || el.textContent.indexOf('✓') !== 0) {
      el.textContent = '✓ Composition valid';
      el.style.color = 'var(--success)';
    }
  } else if (_compositionErrors && _compositionErrors.length > 0) {
    el.innerHTML = _compositionErrors.map(function(e){return '⚠ '+e;}).join('<br>');
    el.style.color = 'var(--error, #dc2626)';
  } else {
    el.textContent = 'Click to validate ratios';
    el.style.color = 'var(--text-muted)';
  }
  updateNextButtonState();
}

/** Enable/disable the Next button based on current step requirements. */
function updateNextButtonState() {
  // Find the currently active panel's next button
  const activePanel = document.querySelector('.panel.active');
  if (!activePanel) return;
  const nextBtn = activePanel.querySelector('.next-btn');
  if (!nextBtn) return;

  const fulfilled = isCurrentStepFulfilled();
  nextBtn.disabled = !fulfilled;
  nextBtn.style.opacity = fulfilled ? '' : '0.45';
  nextBtn.style.cursor = fulfilled ? '' : 'not-allowed';
  nextBtn.title = fulfilled ? '' : 'Complete the current step before proceeding';
}

/** Compute and display per-lipid molecule counts for each leaflet. */
function updateLipidCounts() {
  var upperCountsEl=document.getElementById('upper-lipid-counts');
  var lowerCountsEl=document.getElementById('lower-lipid-counts');
  if(!upperCountsEl)return;

  var pdbContent=_orientedPdbContent||(state.pdbInfo&&state.pdbInfo.pdb_content)||'';
  var xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  var lines=pdbContent.split('\n');
  for(var li=0;li<lines.length;li++){
    var l=lines[li];
    if(l.indexOf('ATOM')===0||l.indexOf('HETATM')===0){
      var px=parseFloat(l.substring(30,38))/10.0;
      var py=parseFloat(l.substring(38,46))/10.0;
      if(!isNaN(px)&&!isNaN(py)){if(px<xMin)xMin=px;if(px>xMax)xMax=px;if(py<yMin)yMin=py;if(py>yMax)yMax=py;}
    }
  }
  var protXY=isFinite(xMin)?Math.max(xMax-xMin,yMax-yMin):3.0;
  var protArea=protXY*protXY;

  var nLipidsEl=document.getElementById('n-lipids-per-leaflet');
  var nLipids=nLipidsEl?parseInt(nLipidsEl.value):150;
  if(isNaN(nLipids)||nLipids<64)nLipids=150;

  var lipids=_lipidPickerData.lipids||[];

  function computeCounts(mix){
    var avgAPL=0,totalRatio=0;
    mix.forEach(function(m){
      var l=lipids.find(function(ll){return ll.name===m.name;});
      if(l){avgAPL+=l.area_per_lipid*m.ratio;totalRatio+=m.ratio;}
    });
    if(totalRatio>0)avgAPL/=totalRatio;
    if(avgAPL<=0)avgAPL=0.65;
    var counts=mix.map(function(m){return{name:m.name,ratio:m.ratio,count:Math.floor(nLipids*m.ratio/100)};});
    return{avgAPL:avgAPL,counts:counts};
  }

  // boxXY from n_lipids: reverse of builder formula n_lipids = boxXY²/APL*1.30
  var avgAPL2=computeCounts(_mixUpper).avgAPL;
  var lipidArea=nLipids*avgAPL2/1.30;
  var boxXY=Math.max(Math.sqrt(lipidArea+protArea),4.0);
  var memArea=boxXY*boxXY;

  function renderTable(el,mix,label){
    var r=computeCounts(mix);
    var sumOK=mix.reduce(function(s,m){return s+m.ratio;},0)===100;
    var html='<table class="count-table">';
    html+='<tr><td colspan="3" class="count-summary">';
    html+='Lipids/leaflet: <b>'+nLipids+'</b> &nbsp;|&nbsp; Box XY: <b>'+boxXY.toFixed(1)+' nm</b> &nbsp;|&nbsp; Area: <b>'+memArea.toFixed(1)+' nm²</b> &nbsp;|&nbsp; APL: <b>'+r.avgAPL.toFixed(3)+' nm²</b>';
    if(!sumOK)html+=' <span class="error-text">⚠ Ratios sum to '+mix.reduce(function(s,m){return s+m.ratio;},0)+'%, not 100%</span>';
    if(nLipids<64)html+=' <span class="error-text">⚠ Min 64 required</span>';
    html+='</td></tr>';
    html+='<tr><th>Lipid</th><th>Ratio</th><th>Count</th></tr>';
    r.counts.forEach(function(c){html+='<tr><td>'+c.name+'</td><td>'+c.ratio+'%</td><td><b>'+c.count+'</b></td></tr>';});
    html+='</table>';
    el.innerHTML=html;
    el.classList.remove('hidden');
  }

  renderTable(upperCountsEl,_mixUpper,'Upper');
  var totalBilayer = nLipids * 2;  // proteins per leaflet × 2 leaflets
  if(_asymmetric&&lowerCountsEl){renderTable(lowerCountsEl,_mixLower,'Lower');lowerCountsEl.classList.remove('hidden');}
  else if(lowerCountsEl){lowerCountsEl.classList.add('hidden');}
  if(!_asymmetric){
    var us=upperCountsEl.querySelector('.count-summary');
    if(us){us.innerHTML+=' &nbsp;|&nbsp; <b>Bilayer total: '+totalBilayer+' lipids</b>';}
  }
}
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
  var fnameLower = file.name.toLowerCase();
  var structurePattern = /\.(?:pdb|ent|cif|mmcif)(?:\.gz)?$/i;
  if (!structurePattern.test(fnameLower)) {
    alert('Accepted structure formats: .pdb, .ent, .cif, .mmcif, and gzip-compressed variants.');
    return;
  }

  // Show upload progress bar
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
      infoSection.appendChild(uploadSection);
      uploadBar = document.getElementById('upload-progress-fill');
      uploadText = document.getElementById('upload-progress-text');
    }
  }
  if (uploadSection) uploadSection.style.display = 'block';
  if (uploadBar) uploadBar.style.width = '0%';
  if (uploadText) uploadText.textContent = 'Uploading ' + (file.size/1024/1024).toFixed(1) + ' MB...';

  const formData = new FormData();
  formData.append('file', file);
  formData.append('task_type', (state.taskType && state.taskType.id) || 'membrane-bilayer');
  if (state.taskType && state.taskType.id === 'coarse-grained' && state.taskId) {
    formData.append('task_id', state.taskId);
  }

  // Use XHR for upload progress tracking
  try {
    const data = await new Promise((resolve, reject) => {
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
        try { resolve(JSON.parse(xhr.responseText)); }
        catch(e) { reject(new Error('Invalid response')); }
      };
      xhr.onerror = function() { reject(new Error('Upload failed')); };
      xhr.send(formData);
    });
    if (data.error || data.validation_errors) {
      if (uploadBar) uploadBar.style.background = '#dc2626';
      if (uploadText) uploadText.textContent = 'Upload failed';
      if (data.validation_errors) {
        showValidationErrors(data.validation_errors, data.validation_warnings || []);
      } else {
        alert('Upload error: ' + (data.error || 'Unknown error'));
      }
      return;
    }
    if (uploadBar) { uploadBar.style.background = '#22c55e'; uploadBar.style.width = '100%'; }
    if (uploadText) uploadText.textContent = 'Upload complete — ' + (file.size/1024/1024).toFixed(1) + ' MB';
    setTimeout(function() { if (uploadSection) uploadSection.style.display = 'none'; }, 2000);

    state.pdbInfo = data;
    state.taskId = data.task_id;
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
    alert('Upload failed: ' + err.message);
  }
}

function showValidationErrors(errors, warnings) {
  const valInfo = document.getElementById('validation-info');
  const errDiv = document.getElementById('validation-errors');
  const warnDiv = document.getElementById('validation-warnings');
  if (!valInfo) return;

  valInfo.classList.remove('hidden');

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
    (info.box_nm || []).map(v => v.toFixed(1) + ' nm').join(' × ') || '—';

  // Show non-blocking validation warnings
  if (info.validation_warnings && info.validation_warnings.length) {
    const warnDiv = document.getElementById('validation-warnings');
    if (warnDiv) {
      warnDiv.classList.remove('hidden');
      warnDiv.innerHTML = '<h4 style="color:#d97706;">&#9888; Warnings</h4><ul>' +
        info.validation_warnings.map(w => '<li>' + w + '</li>').join('') + '</ul>';
    }
  }

  // Render chain sequences with checkboxes and rename inputs
  renderChainSequences(info.sequences || [], info.chains || []);

  // Render small molecules
  renderSmallMolecules(info.small_molecules || []);

  // Render 3D viewer
  renderPDBViewer(info.pdb_content || '');

  // Load residues into structure processing
  loadProcResidues();

  // Input step requires explicit Check — do NOT auto-complete
  // Next button stays disabled until user clicks "Check Upload"
  updateNextButtonState();
  updateOrientSliderRanges();
}

// ---- Chain inclusion / rename state ----
let _chainState = {};  // { chain_id: { included: true, name: original_id } }
// Small-molecule visibility state (keyed by resname — small molecules are
// not protein chains; they have their own checkboxes in the Small Molecules section)
let _smallMolState = {};  // { resname: { included: true, name: original_name } }
// Safety: ensure _smallMolState is always initialised
window._smallMolState = _smallMolState;

function invalidateInputCheckpoint() {
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
  function finish(cancelled) {
    if (finished) return;
    finished = true;
    var label = cancelled
      ? ((_smallMolState[resname] && _smallMolState[resname].name) || resname)
      : commitSmallMoleculeLabel(resname, input.value);
    var replacement = document.createElement('span');
    replacement.className = 'smallmol-name';
    replacement.dataset.smres = resname;
    replacement.textContent = label;
    replacement.title = 'Double-click to rename';
    replacement.addEventListener('dblclick', function() {
      beginSmallMoleculeRename(replacement);
    });
    input.replaceWith(replacement);
  }
  input.addEventListener('blur', function() { finish(false); });
  input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') input.blur();
    if (event.key === 'Escape') finish(true);
  });
}

function renderSmallMolecules(molecules) {
  const container = document.getElementById('small-molecules');
  const header = document.getElementById('smallmol-header');
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
  _smallMolState = _newSmState;
  window._smallMolState = _smallMolState;

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
        '<span class="smallmol-name" data-smres="' + escapeHtml(resname) + '" title="Double-click to rename">' + escapeHtml(_smallMolState[resname] ? _smallMolState[resname].name : resname) + '</span>' +
      '</div>' +
      '<div class="smallmol-info">' +
        'Formula: ' + formula + ' | Copies: ' + instances.length +
        ' | Atoms: ' + totalAtoms + ' | Chain: ' + chains +
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
      redrawPDBViewerWithChainFilter();
    });
  });

  // Wire up rename on double-click
  container.querySelectorAll('.smallmol-name').forEach(span => {
    span.addEventListener('dblclick', function() {
      beginSmallMoleculeRename(span);
    });
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

function renderChainSequences(sequences, chains) {
  const container = document.getElementById('chain-sequences');
  if (!container) return;
  container.innerHTML = '';

  // Init chain state for new upload
  _chainState = {};
  (chains || []).forEach(ch => { _chainState[ch] = { included: true, name: ch }; });

  if (!sequences.length) {
    container.innerHTML = '<p class="hint">No residue data detected.</p>';
    return;
  }

  sequences.forEach(chain => {
    const chId = chain.chain_id || '';
    const st = _chainState[chId] || { included: true, name: chId };

    const card = document.createElement('div');
    card.className = 'chain-card';

    const header = document.createElement('div');
    header.className = 'chain-header';
    const chainLabel = chId ? `Chain ${chId}` : 'Chain';
    header.innerHTML =
      `<label class="chain-check">` +
        `<input type="checkbox" data-chain="${chId}" ${st.included ? 'checked' : ''}>` +
        `<b>${chainLabel}</b>` +
      `</label>` +
      `<span class="chain-rename" data-chain="${chId}" title="Double-click to rename">${st.name || chId}</span>` +
      `<span class="chain-len">${chain.length} residues</span>`;
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
        }
        tagRow.appendChild(tag);
      }
      group.appendChild(tagRow);
      rowWrap.appendChild(group);
    }
    seqWrap.appendChild(rowWrap);
    card.appendChild(seqWrap);
    container.appendChild(card);
  });

  // Wire up chain checkboxes to toggle 3D viewer visibility
  container.querySelectorAll('.chain-check input[type=checkbox]').forEach(cb => {
    cb.addEventListener('change', () => {
      const ch = cb.dataset.chain;
      const included = cb.checked;
      if (_chainState[ch]) _chainState[ch].included = included;
      // Refresh the PDB viewer
      redrawPDBViewerWithChainFilter();
    });
  });

  // Wire up chain rename on double-click
  container.querySelectorAll('.chain-rename').forEach(span => {
    span.addEventListener('dblclick', () => {
      const ch = span.dataset.chain;
      const orig = span.textContent;
      const input = document.createElement('input');
      input.type = 'text';
      input.value = orig;
      input.className = 'chain-rename-input';
      input.style.width = '40px';
      span.replaceWith(input);
      input.focus();
      input.select();
      const commit = () => {
        const val = input.value.trim() || orig;
        const newSpan = document.createElement('span');
        newSpan.className = 'chain-rename';
        newSpan.dataset.chain = ch;
        newSpan.textContent = val;
        newSpan.title = 'Double-click to rename';
        if (_chainState[ch]) _chainState[ch].name = val;
        input.replaceWith(newSpan);
        newSpan.addEventListener('dblclick', () => {
          const inp2 = document.createElement('input');
          inp2.type = 'text'; inp2.value = val;
          inp2.className = 'chain-rename-input'; inp2.style.width = '40px';
          newSpan.replaceWith(inp2); inp2.focus(); inp2.select();
          inp2.addEventListener('blur', () => {
            const v2 = inp2.value.trim() || val;
            const ns2 = document.createElement('span');
            ns2.className = 'chain-rename'; ns2.dataset.chain = ch;
            ns2.textContent = v2; ns2.title = 'Double-click to rename';
            if (_chainState[ch]) _chainState[ch].name = v2;
            inp2.replaceWith(ns2);
          });
          inp2.addEventListener('keydown', (e) => { if (e.key === 'Enter') inp2.blur(); });
        });
      };
      input.addEventListener('blur', commit);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    });
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

var _SMALLMOL_COLORS = [
  "0xf59e0b", "0xef4444", "0x10b981", "0x8b5cf6", "0x06b6d4",
  "0xf97316", "0xec4899", "0x6366f1", "0x14b8a6", "0x84cc16",
  "0xeab308", "0xd946ef", "0x0ea5e9", "0x78716c", "0x65a30d",
];

function colorSmallMolecules(viewer, onlyChains) {
  // onlyChains is accepted for API consistency but small-molecule visibility
  // is controlled solely by _smallMolState checkboxes — not by protein chain
  // toggles, because small molecules often reside in their own chain (different
  // from protein chains) and would be incorrectly hidden by the chain filter.
  if (!state.pdbInfo || !state.pdbInfo.small_molecules) return;
  var mols = state.pdbInfo.small_molecules;
  var seen = {};
  var ci = 0;
  if (typeof _smallMolState === 'undefined') _smallMolState = {};
  mols.forEach(function(m) {
    // Skip if this molecule's resname is unchecked in _smallMolState
    if (_smallMolState[m.resname] && !_smallMolState[m.resname].included) return;
    if (!seen[m.resname]) {
      seen[m.resname] = _SMALLMOL_COLORS[ci % _SMALLMOL_COLORS.length];
      ci++;
    }
    viewer.addStyle({resn: m.resname}, {stick: {radius: 0.18, color: seen[m.resname]}});
    viewer.addStyle({resn: m.resname}, {sphere: {radius: 0.25, color: seen[m.resname], opacity: 0.8}});
  });
}

// ===================================================================
// 3Dmol.js Viewer
// ===================================================================

function renderPDBViewer(pdbContent) {
  const viewerEl = document.getElementById('pdb-viewer');
  if (!viewerEl) return;

  // Clear any previous viewer
  viewerEl.innerHTML = '';

  if (!pdbContent) {
    viewerEl.innerHTML = '<p style="color:#888;text-align:center;padding-top:180px;">No structure data</p>';
    return;
  }

  if (typeof $3Dmol === 'undefined') {
    window._cdmRetriesPDB = (window._cdmRetriesPDB || 0) + 1;
    if (window._cdmRetriesPDB > 30) {
      viewerEl.innerHTML = '<p style="color:#c00;text-align:center;padding-top:180px;">The bundled 3Dmol.js viewer failed to load</p>';
      return;
    }
    viewerEl.innerHTML = '<p style="color:#888;text-align:center;padding-top:180px;">3Dmol.js loading...</p>';
    setTimeout(() => renderPDBViewer(pdbContent), 500);
    return;
  }

  try {
    const viewer = $3Dmol.createViewer(viewerEl, {
      backgroundColor: '0xffffff',
      antialias: true,
    });
    viewer.setBackgroundColor('0xffffff'); viewer.setSlab(-10000, 10000);

    viewer.addModel(pdbContent, 'pdb');
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
  viewer.removeAllModels();
  viewer.addModel(pdbContent, 'pdb');

  // Build set of included protein chains.
  // Small molecules in non-protein chains are handled independently
  // in _applyUnifiedStyle (they don't need to be in includedChains).
  var includedChains = new Set();
  for (var c in _chainState) {
    if (_chainState[c].included) includedChains.add(c);
  }

  if (includedChains.size === 0) {
    viewer.setStyle({}, {cartoon: {hidden: true}, stick: {hidden: true}, sphere: {hidden: true}, line: {hidden: true}});
  } else {
    // Pass onlyChains so _applyUnifiedStyle only styles included chains
    _applyUnifiedStyle(viewer, pdbContent, includedChains);
  }
  viewer.zoomTo();
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
  if (_checkedConfig) {
    for (var key in _checkedConfig) {
      if (_checkedConfig.hasOwnProperty(key)) {
        config[key] = Object.assign({}, config[key] || {}, _checkedConfig[key]);
      }
    }
  }
  return config;
}

function buildModuleConfig(focusStep) {
  const config = {};
  const taskModules = state.taskType ? state.taskType.visible_modules : [];

  if (isCoarseGrainedWorkflow()) {
    var environment = document.getElementById('cg-environment')?.value || 'bilayer';
    var includeProtein = coarseGrainedIncludesProtein();
    var wants = function(step) { return !focusStep || focusStep === step; };
    if (wants('input')) {
      config.input = {include_protein: includeProtein, environment: environment};
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
    if (wants('cg_environment')) {
      var upper = environment === 'bilayer'
        ? parseCoarseGrainedComposition(document.getElementById('cg-upper-lipids')?.value, 'Upper')
        : [];
      var lower = environment === 'bilayer'
        ? parseCoarseGrainedComposition(document.getElementById('cg-lower-lipids')?.value, 'Lower')
        : [];
      config.cg_environment = {
        environment: environment,
        box_xy: Number(document.getElementById('cg-box-xy')?.value || 12),
        box_z: Number(document.getElementById('cg-box-z')?.value || (environment === 'bilayer' ? 14 : 12)),
        rotate_x: Number(document.getElementById('cg-rotate-x')?.value || 0),
        rotate_y: Number(document.getElementById('cg-rotate-y')?.value || 0),
        rotate_z: Number(document.getElementById('cg-rotate-z')?.value || 0),
        z_offset: Number(document.getElementById('cg-z-offset')?.value || 0),
      };
      if (environment === 'bilayer') {
        config.cg_environment.upper_leaflet = upper;
        config.cg_environment.lower_leaflet = lower;
        config.cg_environment.asymmetric = JSON.stringify(upper) !== JSON.stringify(lower);
      }
    }
    var includeSolvent = document.getElementById('cg-include-solvent')?.checked !== false;
    var saltMolarity = Number(document.getElementById('cg-salt')?.value || 0);
    if (wants('cg_solvation')) {
      config.cg_solvation = {include_solvent: includeSolvent, salt_molarity: saltMolarity};
    }
    if (wants('cg_system')) {
      config.cg_system = {
        salt_molarity: saltMolarity,
        confirm_system: document.getElementById('cg-confirm-system')?.checked === true,
      };
    }
    if (wants('topology')) config.topology = {};
    if (wants('simparams')) config.simparams = collectCoarseGrainedSimulationParams();
    if (wants('export')) config.export = {write_mdp: includeSolvent};
    return mergeCheckedModuleConfig(config);
  }

  // Input
  if (taskModules.includes('input')) {
    if (state.taskId) {
      config.input = { task_id: state.taskId };
    }
  }

  // Structure Processing
  if (taskModules.includes('structure')) {
    var skipProtonation = document.getElementById("proc-skip-protonation") ? document.getElementById("proc-skip-protonation").checked : false;
    config.structure = {
      protonation: skipProtonation ? [] : _procAssignments.filter(function(a) { return a.is_titratable; }).map(function(a) {
        return { index: a.index, original: a.original, assigned_name: a.assigned_name, charge: a.charge };
      }),
      modifications: serializeStructureModifications(),
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
      config.orient = { method: _orientAlgorithm || 'ppm' };
    } else {
      config.orient = {
        method: 'manual',
        z_offset: _orientZOffset,
        tilt: _orientTilt,
        phi: _orientPhi,
      };
    }
  }

  // Membrane
  if (taskModules.includes('membrane')) {
    const nLipidsEl = document.getElementById('n-lipids-per-leaflet');
    const nLipids = nLipidsEl ? parseInt(nLipidsEl.value) : 150;
    config.membrane = {
      lipid_composition: {
        upper: _mixUpper.map(m => ({...m})),
        lower: _asymmetric ? _mixLower.map(m => ({...m})) : null,
      },
      n_lipids_per_leaflet: Math.max(nLipids || 150, 64),
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
    var isPureMembrane = state.taskType && state.taskType.pipeline === 'pure_membrane';
    var isSolution = state.taskType && state.taskType.pipeline === 'solvator';
    config.forcefield = {
      name: document.getElementById('ff-protein')?.value || 'amber14sb',
      lipid_ff: isSolution ? 'none' : (document.getElementById('ff-lipid')?.value || 'none'),
      ligand_ff: isPureMembrane ? 'none' : (document.getElementById('ff-ligand')?.value || 'none'),
      ligand_charges: isPureMembrane ? {} : collectLigandCharges(),
      ligand_pH: _systemPH,
      cgenff_parameters: isPureMembrane ? {} : collectCGenFFParameters(),
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
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const resultSection = document.getElementById('result-section');
  progressFill.style.width = '100%';
  progressText.textContent = 'Complete!';
  resultSection.classList.remove('hidden');

  const details = document.getElementById('result-details');
  details.innerHTML = '';

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

  const logP = document.createElement('p'); logP.innerHTML = '<strong>Build log:</strong>';
  const logUl = document.createElement('ul');
  (result.log || []).forEach(function(l) { const li=document.createElement('li'); li.textContent=l; logUl.appendChild(li); });
  details.appendChild(table);
  details.appendChild(logP);
  details.appendChild(logUl);

  const dlLink = document.getElementById('download-link');
  dlLink.href = '/api/download/' + (result.task_id || state.taskId);
  dlLink.textContent = 'Download ZIP';
  const runBtn = document.getElementById('run-btn');
  if (runBtn) { runBtn.disabled = false; runBtn.textContent = '▶ Build System'; }
  state.buildRunning = false;
}

function formatQueueWait(seconds) {
  var value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return Math.ceil(value) + " seconds";
  if (value < 3600) return Math.ceil(value / 60) + " minutes";
  return (value / 3600).toFixed(1) + " hours";
}

function updateComputeQueueModal(queueState) {
  var modal = document.getElementById("compute-queue-modal");
  if (!modal || !queueState) return;
  var taskId = queueState.task_id || state.taskId || "";
  var taskEl = document.getElementById("compute-queue-task-id");
  var positionEl = document.getElementById("compute-queue-position");
  var estimateEl = document.getElementById("compute-queue-estimate");
  var titleEl = document.getElementById("compute-queue-title");
  var messageEl = document.getElementById("compute-queue-message");
  if (taskEl) taskEl.textContent = taskId;
  if (queueState.status === "running") {
    if (titleEl) titleEl.textContent = "Task processing has started";
    if (messageEl) messageEl.textContent =
      "The task left the queue and is now being finalized. You may continue in the background.";
    if (positionEl) positionEl.textContent = "Processing now";
    if (estimateEl) estimateEl.textContent = "Started";
  } else {
    if (titleEl) titleEl.textContent = "Task added to the compute queue";
    if (messageEl) messageEl.textContent =
      "The server is busy. Your checked workflow has been saved and will start automatically.";
    if (positionEl) positionEl.textContent = String(queueState.queue_position || "—");
    if (estimateEl) {
      var stamp = queueState.estimated_start_at ?
        new Date(queueState.estimated_start_at).toLocaleString() : "Pending estimate";
      estimateEl.textContent = stamp + " (about " +
        formatQueueWait(queueState.estimated_wait_seconds) + ")";
    }
  }
}

function showComputeQueueModal(queueState) {
  var modal = document.getElementById("compute-queue-modal");
  if (!modal) return;
  updateComputeQueueModal(queueState);
  var saved = document.getElementById("compute-queue-saved");
  var close = document.getElementById("compute-queue-close");
  if (saved) saved.checked = false;
  if (close) close.disabled = true;
  modal.classList.remove("hidden");
}

function initComputeQueueModal() {
  var modal = document.getElementById("compute-queue-modal");
  var saved = document.getElementById("compute-queue-saved");
  var close = document.getElementById("compute-queue-close");
  var copy = document.getElementById("compute-queue-copy");
  if (saved && close) {
    saved.addEventListener("change", function() {
      close.disabled = !saved.checked;
    });
  }
  if (close && modal) {
    close.addEventListener("click", function() {
      if (!close.disabled) modal.classList.add("hidden");
    });
  }
  if (copy) {
    copy.addEventListener("click", async function() {
      var value = document.getElementById("compute-queue-task-id")?.textContent || "";
      try {
        await navigator.clipboard.writeText(value);
        copy.textContent = "Copied";
      } catch (error) {
        window.prompt("Copy this Task ID:", value);
      }
    });
  }
}

async function runBuild() {
  if (state.buildRunning || !state.taskType) return;
  state.buildRunning = true;

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  runBtn.textContent = 'Building...';

  const progressSection = document.getElementById('progress-section');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const resultSection = document.getElementById('result-section');

  progressSection.classList.remove('hidden');
  resultSection.classList.add('hidden');
  // Build payload FIRST so log timer can reference task_id
  let modules;
  try {
    modules = buildModuleConfig();
  } catch (error) {
    state.buildRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '▶ Build System';
    progressSection.classList.add('hidden');
    alert('Simulation parameter error: ' + (error && error.message ? error.message : String(error)));
    return;
  }
  const payload = {
    task_id: state.taskId || '',
    task_type: (state.taskType && state.taskType.id) || 'membrane-bilayer',
    system_name: document.getElementById('system-name')?.value || ((state.taskType && state.taskType.pipeline === 'solvator') ? 'solvator_system' : 'membrane_system'),
    modules: modules,
  };

  progressFill.style.width = '10%';
  progressText.textContent = 'Assembling pipeline configuration...';
  // Show log box and start polling AFTER payload is ready
  var logBox = document.getElementById("build-log");
  var logContent = document.getElementById("build-log-content");
  if (logBox) logBox.style.display = "block";
  if (logContent) logContent.innerHTML = "";
  // Track all polling intervals for cleanup on page unload
  var _timers = (window._buildPollTimers = window._buildPollTimers || []);
  var logSince = 0, logTimer = null;
  logTimer = setInterval(async function() {
    try {
      var lr = await fetch("/api/build/" + payload.task_id + "/log?since=" + logSince);
      var ld = await lr.json();
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
      if (ld.done) { clearInterval(logTimer); logTimer = null; }
    } catch(e) { /* network errors are transient — keep polling */ }
  }, 500);
  _timers.push(logTimer);

  progressFill.style.width = '20%';
  progressText.textContent = `Finalizing checked system: ${state.taskType.title}...`;

  try {
    var taskId = state.taskId || '';
    const res = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      if (res.status === 409) {
        alert('This task is already being built. Please wait for it to complete.');
        state.buildRunning = false; runBtn.disabled = false; runBtn.textContent = '▶ Build System';
        return;
      }
      throw new Error(err.error || 'Build failed');
    }

    const result = await res.json();

    // ---- Handle queued build ----
    if (result.status === 'queued') {
      showComputeQueueModal(result);
      progressFill.style.width = '10%';
      progressText.textContent = 'Queued — position ' + result.queue_position;
      resultSection.classList.remove('hidden');
      const details = document.getElementById('result-details');
      details.innerHTML = '<div style="padding:20px;text-align:center">' +
        '<h3 style="color:#d97706;">&#9201; Position <span id="queue-pos">' + result.queue_position + '</span> in build queue</h3>' +
        '<p style="color:#64748b;max-width:500px;margin:12px auto">' + result.message + '</p>' +
        '<p style="color:#64748b;font-size:13px;">Task ID: <code style="background:#f1f5f9;padding:2px 6px;border-radius:4px">' + result.task_id + '</code></p>' +
        '<p style="color:#94a3b8;font-size:12px;margin-top:20px">You can close this page. Return with the Task ID, or download later from <code>/api/task/' + result.task_id + '/download</code>.</p>' +
        '</div>';

      // Poll queue position
      var queuePoll = setInterval(async function() {
        try {
          var qr = await fetch('/api/build/' + result.task_id + '/queue-status');
          var qd = await qr.json();
          if (qd.status === 'running') {
            clearInterval(queuePoll);
            updateComputeQueueModal(qd);
            document.getElementById('queue-pos').textContent = '0 (now building)';
            progressText.textContent = 'Build started — waiting for completion...';
            // Switch to normal build log polling
            if (logTimer) { clearInterval(logTimer); }
            logTimer = setInterval(async function() {
              try {
                var lr = await fetch('/api/build/' + result.task_id + '/log?since=' + logSince);
                var ld = await lr.json();
                if (ld.lines && ld.lines.length > 0) {
                  ld.lines.forEach(function(line) {
                    if (logContent) { var div=document.createElement('div'); div.textContent=line; logContent.appendChild(div); }
                  });
                  if (logBox) logBox.scrollTop = logBox.scrollHeight;
                  logSince = ld.total;
                }
                if (ld.done) { clearInterval(logTimer); logTimer = null; }
              } catch(e) {}
            }, 500);
          } else if (qd.status === 'completed') {
            clearInterval(queuePoll);
            if (logTimer) { clearInterval(logTimer); }
            progressFill.style.width = '100%';
            progressText.textContent = 'Complete!';
            _showBuildResult(qd.result || result);
          } else if (qd.status === 'failed') {
            clearInterval(queuePoll);
            alert('Build failed: ' + (qd.error || 'unknown error'));
            state.buildRunning = false; runBtn.disabled = false;
            runBtn.textContent = '▶ Build System';
          } else {
            document.getElementById('queue-pos').textContent = qd.queue_position;
            progressText.textContent = 'Queued — position ' + qd.queue_position;
            updateComputeQueueModal(qd);
          }
        } catch(e) {}
      }, 3000);
      _timers.push(queuePoll);
      return;
    }

    // ---- Immediate build started — poll for completion ----
    var donePoll = setInterval(async function() {
      try {
        var qr = await fetch('/api/build/' + result.task_id + '/queue-status');
        var qd = await qr.json();
        if (qd.status === 'completed') {
          clearInterval(donePoll);
          if (logTimer) { clearInterval(logTimer); logTimer = null; }
          _showBuildResult(qd.result || result);
        } else if (qd.status === 'failed') {
          clearInterval(donePoll);
          if (logTimer) { clearInterval(logTimer); logTimer = null; }
          alert('Build failed: ' + (qd.error || 'unknown error'));
          state.buildRunning = false; runBtn.disabled = false;
          runBtn.textContent = '▶ Build System';
        }
      } catch(e) {}
    }, 2000);
    _timers.push(donePoll);

  } catch (err) {
    if (logTimer) { clearInterval(logTimer); logTimer = null; }
    progressSection.classList.add('hidden');
    alert('Build error: ' + err.message);
    state.buildRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '▶ Build System';
  }
}

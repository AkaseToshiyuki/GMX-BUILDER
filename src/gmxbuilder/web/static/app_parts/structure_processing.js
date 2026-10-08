// Loaded as a classic script after custom_lipids.js.
// Public globals are retained for workflow checks, resume, and build serialization.

// ===================================================================
// ===================================================================
// Structure Processing — Protonation + Termini + Modifications
// ===================================================================

let _procResidues = [];         // [{resname, chain, resid, index}]
let _procChains = [];           // sorted unique chain IDs
let _procAssignments = [];     // protonation results
// Authoritative fields are stable residue identity + patch_id.  product_name and charge_shift
// are presentation metadata derived from the server-side, force-field-specific
// patch catalogue and must never be submitted as scientific input.
let _structureRevision = 0;
let _procInputRemovals = [];
let _procModifications = [];   // [{index, patch_id, product_name, charge_shift}]
let _procCrosslinks = [];      // [{type:'disulfide', first_index, second_index}]
let _procTermini = {};          // { chain_id: { nter: 'ACE'|'', cter: 'NME'|'' } }
let _procPatchCatalog = [];
let _procCapCapabilities = {};
let _procCrosslinkCapabilities = {};
let _procCatalogRequest = 0, _procCrosslinkRequest = 0, _procTerminalRequest = 0;
let _procPickerRequest = 0;
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
        setLigandEnvironmentPH(phInput.value);
        invalidateProtonationState('Target pH must be between 1.0 and 13.0; enter a valid value and click Compute.');
        return false;
      }
      const changed = v !== _systemPH;
      setLigandEnvironmentPH(v);
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
  closePatchPicker();
  _procSelectedIdx = -1;
  _procResidues = [];
  _procModifications = [];
  _procInputRemovals = [];
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
      if (res.is_protein === false) return;
      _procResidues.push({
        resname: res.resname, chain: ch, resid: res.resid,
        index: _procResidues.length,
      });
    });
  });
  // Unique sorted chains
  _procChains = [...new Set(_procResidues.map(r => r.chain))].sort();
  // Keep the established default until the user selects a validated cap.
  // Capability loading below determines which force-field caps are available.
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
    reloadTerminalCapabilities();
  }
}

function restoreStructureProcessingConfig(savedConfig) {
  if (!savedConfig || typeof savedConfig !== 'object' || !_procResidues.length) return;
  _procInputRemovals = Array.isArray(savedConfig.input_modification_decisions)
    ? savedConfig.input_modification_decisions.filter(item => item.action === 'remove') : [];
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
    _procModifications = savedConfig.modifications.map(function(mod) {
      return Object.assign({}, mod, {index: resolveProcTarget(mod.target, mod.index)});
    }).filter(function(mod) {
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
    _procCrosslinks = savedConfig.crosslinks.map(function(item) {
      return Object.assign({}, item, {first_index: resolveProcTarget(item.first_target, item.first_index), second_index: resolveProcTarget(item.second_target, item.second_index)});
    }).filter(function(item) {
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
      if (!assignment) return;
      assignment = Object.assign({}, assignment, {index: resolveProcTarget(assignment.target, assignment.index)});
      if (!Number.isInteger(assignment.index) || assignment.index < 0) return;
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
  closePatchPicker();
  const forceField = selectedProteinForceField(), task = state.taskId, request = ++_procCatalogRequest;
  const current = () => request === _procCatalogRequest && task === state.taskId && forceField === selectedProteinForceField();
  try {
    const patches = await GMXHttp.json('/api/patches?force_field=' + encodeURIComponent(forceField), {}, {
      current, validate: Array.isArray
    });
    if (!current()) return;
    closePatchPicker();
    _procPatchCatalog = patches;
    hydrateModificationMetadata();
    applyDetectedInputModifications();
    applyModification(-1, '', '');
  } catch (error) {
    if (!current()) return;
    closePatchPicker();
    _procPatchCatalog = [];
    _inputModificationCapabilityWarnings = [
      'The force-field modification catalogue could not be loaded; uploaded modifications were not auto-selected.'
    ];
    renderDetectedModificationNotice();
    renderModSequences();
  }
}

async function reloadCrosslinkCapabilities() {
  const forceField = selectedProteinForceField(), task = state.taskId, request = ++_procCrosslinkRequest;
  const current = () => request === _procCrosslinkRequest && task === state.taskId && forceField === selectedProteinForceField();
  try {
    const capabilities = await GMXHttp.json('/api/crosslink-capabilities?force_field=' + encodeURIComponent(forceField), {}, {current});
    if (!current()) return;
    _procCrosslinkCapabilities = capabilities;
  } catch (_error) {
    if (!current()) return;
    _procCrosslinkCapabilities = {};
  }
  renderDisulfideControls();
}

async function reloadTerminalCapabilities() {
  const forceField = selectedProteinForceField(), task = state.taskId, request = ++_procTerminalRequest;
  const current = () => request === _procTerminalRequest && task === state.taskId && forceField === selectedProteinForceField();
  try {
    const capabilities = await GMXHttp.json('/api/terminal-capabilities?force_field=' + encodeURIComponent(forceField), {}, {current});
    if (!current()) return;
    _procCapCapabilities = capabilities;
  } catch (_error) {
    if (!current()) return;
    _procCapCapabilities = {};
  }
  renderTerminiTab();
}

function applyDetectedInputModifications() {
  _procModifications = _procModifications.filter(function(mod) {
    return mod.source !== 'input-detection';
  });
  _inputModificationCapabilityWarnings = [];
  if (!_procPatchCatalog.length) return;
  (_inputModificationReport.records || []).forEach(function(record) {
    if (!record || record.status !== 'recognized' || !record.patch_id) return;
    if (_procInputRemovals.some(item => item.target.chain === record.chain && item.target.resid === record.resid)) return;
    const index = _procResidues.findIndex(r => r.chain === record.chain && Number(r.resid) === Number(record.resid));
    let patch = _procPatchCatalog.find(function(item) {
      return item && item.id === record.patch_id;
    });
    if (patch && index >= 0) patch = patchForResidueContext(patch, _procResidues[index]);
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
  records.filter(record => record.status === 'recognized').forEach(record => {
    const label = document.createElement('label');
    label.className = 'input-reconstruction-option';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = _procInputRemovals.some(item => item.target.chain === record.chain && item.target.resid === record.resid);
    box.addEventListener('change', () => {
      _procInputRemovals = _procInputRemovals.filter(item => !(item.target.chain === record.chain && item.target.resid === record.resid));
      if (box.checked) {
        _procInputRemovals.push({action: 'remove', target: {chain: record.chain, resid: record.resid, resname: record.standard_resname}});
        const index = _procResidues.findIndex(r => r.chain === record.chain && Number(r.resid) === Number(record.resid));
        _procModifications = _procModifications.filter(mod => mod.index !== index || mod.patch_id !== record.patch_id);
      }
      invalidateStructureChemistry();
      applyDetectedInputModifications();
      applyModification(-1, '', '');
    });
    label.appendChild(box);
    const text = document.createElement('span');
    text.textContent = 'Explicitly remove uploaded ' + record.original_resname + ' at ' + record.chain + ':' + record.resid +
      ' and continue with its parent chemistry. This changes the molecule.';
    label.appendChild(text);
    notice.appendChild(label);
  });
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
    return { index: mod.index, target: procResidueTarget(mod.index), patch_id: mod.patch_id };
  });
}

function serializeStructureCrosslinks() {
  return _procCrosslinks.map(function(item) {
    return {
      type: 'disulfide',
      first_index: item.first_index,
      second_index: item.second_index,
      first_target: procResidueTarget(item.first_index),
      second_target: procResidueTarget(item.second_index),
    };
  });
}

function procResidueTarget(index) {
  const residue = _procResidues[index];
  if (!residue) throw new Error('The selected residue is no longer available. Reload the input.');
  return {chain: residue.chain, resid: Number(residue.resid), resname: residue.resname};
}

function resolveProcTarget(target, index) {
  if (!target) return index;
  return _procResidues.findIndex(r => r.chain === target.chain &&
    Number(r.resid) === Number(target.resid) && r.resname === target.resname);
}

function invalidateStructureChemistry() {
  _structureRevision += 1;
  if (window.invalidateFinalReview) window.invalidateFinalReview();
  const position = state.wizardSteps.indexOf('structure');
  if (position < 0) return;
  state.wizardSteps.slice(position).forEach(step => {
    _checkedSteps.delete(step);
    state.completedSteps.delete(state.wizardSteps.indexOf(step));
    if (_checkedConfig) delete _checkedConfig[step];
  });
  const status = document.getElementById('structure-check-status');
  if (status) status.textContent = 'Chemistry changed — run Check Structure again.';
  updateNextButtonState();
  updateStepNavHighlight();
}

function patchForResidueContext(patch, residue) {
  const sameChain = _procResidues.filter(r => r.chain === residue.chain);
  const caps = _procTermini[residue.chain] || {};
  const blocked = patch.unsupported_free_termini || [];
  if ((sameChain[0] === residue && !caps.nter && blocked.includes('N')) ||
      (sameChain[sameChain.length - 1] === residue && !caps.cter && blocked.includes('C'))) {
    return Object.assign({}, patch, {supported: false,
      support_reason: 'No validated free-terminal model for this residue in the selected force field.'});
  }
  return patch;
}

// -------------------------------------------------------------------
// Protonation
// -------------------------------------------------------------------

function invalidateProtonationState(message) {
  invalidateStructureChemistry();
  _protonationRequestId += 1;  // makes any in-flight response stale
  _protonationRunning = false;
  _protonationComputed = false;
  _computedProtonationInput = null;
  _procAssignments = [];
  var runButton = document.getElementById('proc-run-btn');
  var checkButton = document.getElementById('structure-check-btn');
  if (runButton) runButton.disabled = false;
  if (checkButton) checkButton.disabled = false;
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
  var displayedPH = Number(document.getElementById('proc-pH')?.value);
  var displayedHis = document.getElementById('proc-his-tautomer')?.value || 'HSE';
  if (!Number.isFinite(displayedPH) || displayedPH < 1.0 || displayedPH > 13.0) {
    return 'Target pH must be a number between 1.0 and 13.0.';
  }
  // The backend assesses free-terminal chemistry from its pKa model.
  // A covalent cap is not a substitute for a different free-terminal state.
  if (skip && skip.checked) return '';
  if (_protonationRunning) return 'Protonation is still computing. Wait for it to finish before checking.';
  if (!_protonationComputed) return 'Run Compute after choosing the target pH and histidine state.';
  if (!_computedProtonationInput || displayedPH !== _computedProtonationInput.pH ||
      displayedHis !== _computedProtonationInput.his) {
    return 'The displayed pH or histidine preference differs from the last calculation. Run Compute again.';
  }
  if (_procAssignments.some(a => a && a.force_field_lacks_state && !a.state_override)) {
    return 'The force field lacks a predicted protonation state. Review the flagged residue and explicitly select an available state only if chemically justified.';
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
        // Amber and CHARMM name the non-default states differently (ASH vs
        // ASPP, HID vs HSD). The server prefers the task's saved choice; this
        // covers a preview taken before that has been checked.
        force_field: (document.getElementById('ff-protein') || {}).value || '',
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
    html += `<p class="hint" style="margin-bottom:8px;">Method: <b>${escapeHtml(data.method)}</b> &nbsp;|&nbsp; pH ${escapeHtml(data.pH)} &nbsp;|&nbsp; ${escapeHtml(data.titratable_count)} titratable residues found</p>`;
    if (data.prediction_expected != null) {
      html += `<p class="hint">Environment-sensitive predictions matched: ${escapeHtml(data.prediction_matched)}/${escapeHtml(data.prediction_expected)}. Shift = predicted pKa − model pKa.</p>`;
    }
    if (data.propka_warning) {
      html += `<p class="hint" style="margin-bottom:8px;color:#d97706;">⚠ ${escapeHtml(data.propka_warning)}</p>`;
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

    html += `<h4 style="margin-top:12px;color:#475569;">Chain ${escapeHtml(ch || ' ')} <span class="hint">${chainTitr.length} titratable residues</span></h4>`;
    html += `<table class="proc-table"><thead><tr><th>Resid</th><th>Original</th><th>Assigned</th><th>Charge</th><th>pKa</th><th>Shift</th><th>Source</th><th>State</th><th>Override</th></tr></thead><tbody>`;

    chainRes.forEach((r, localIdx) => {
      const a = _procAssignments[r.index];
      if (!a || !a.is_titratable) return;
      const shift = a.pKa_shift;
      const shiftStr = shift != null ? (shift >= 0 ? '+' : '') + shift.toFixed(1) : '—';
      const unavailable = a.force_field_lacks_state && !a.state_override;
      const altOpts = (unavailable ? '<option value="" selected>Review unsupported prediction</option>' : '') + (a.alternatives || []).map(alt =>
        `<option value="${escapeHtml(alt.name)}" ${!unavailable && alt.name === a.assigned_name ? 'selected' : ''}>${escapeHtml(alt.name)} (${alt.charge >= 0 ? '+' : ''}${escapeHtml(alt.charge)})</option>`
      ).join('');
      html += `<tr>
        <td>${escapeHtml(a.original)} ${escapeHtml(r.resid)}</td>
        <td><b>${escapeHtml(a.original)}</b></td>
        <td style="color:#6366f1;font-weight:600;">${unavailable ? 'Review required' : escapeHtml(a.assigned_name)}</td>
        <td>${a.charge >= 0 ? '+' : ''}${a.charge}</td>
        <td>${escapeHtml(typeof a.pKa === 'number' ? a.pKa.toFixed(1) : a.pKa || '—')}</td>
        <td style="color:${shift > 0 ? '#dc2626' : shift < 0 ? '#059669' : '#94a3b8'};">${shiftStr}</td>
        <td>${escapeHtml(a.prediction_source || (a.predicted_pKa != null ? 'PROPKA' : 'Model pKa'))}</td>
        <td style="font-size:11px;">${escapeHtml(a.state_label)}</td>
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
        a.state_override = true;
        invalidateStructureChemistry();
        updateNextButtonState();
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
    const fragment = ((state.pdbInfo || {}).checked_sequences || []).find(row => row.chain_id === ch);
    if (fragment && fragment.fragment_count > 1) {
      const ends = [fragment.internal_n_terminus ? 'N' : '', fragment.internal_c_terminus ? 'C' : ''].filter(Boolean);
      html += `<p class="fragment-model-warning">Fragment ${escapeHtml(ch)} belongs to source protein ${escapeHtml(fragment.source_chain)}. ` +
        `${ends.join(' and ')} terminus created at an internal coordinate gap. Review the chemistry below: ` +
        'standard ends add local charges; caps do not reconstruct the missing connection.</p>';
    }
    const t = _procTermini[ch] || { nter: '', cter: '' };
    const ace = _procCapCapabilities.ACE || {supported:false, reason:'Capability data is not loaded'};
    const formyl = _procCapCapabilities.FOR || {supported:false, reason:'Capability data is not loaded'};
    const nme = _procCapCapabilities.NME || {supported:false, reason:'Capability data is not loaded'};
    // Capability refresh must not silently replace the saved terminal chemistry.
    // Unavailable selections remain visible and are rejected by Structure Check.
    _procTermini[ch] = t;
    const firstRes = _procResidues.find(r => r.chain === ch);
    const lastRes = [..._procResidues].reverse().find(r => r.chain === ch);
    html += `<div class="termini-chain-row">
      <b style="min-width:60px;">${fragment && fragment.fragment_count > 1 ? 'Fragment' : 'Chain'} ${escapeHtml(ch || ' ')}</b>
      <span class="hint">N-ter: ${escapeHtml(firstRes ? firstRes.resname + ' ' + firstRes.resid : '?')}</span>
      <select class="proc-nter-sel" data-chain="${escapeHtml(ch)}">
        <option value="" ${!t.nter ? 'selected' : ''}>Standard (NH₃⁺)</option>
        <option value="ACE" ${t.nter === 'ACE' ? 'selected' : ''} ${ace.supported ? '' : 'disabled'}>ACE${ace.supported ? ' — explicit acetyl cap' : ' — unavailable: ' + escapeHtml(ace.reason)}</option>
        <option value="FOR" ${t.nter === 'FOR' ? 'selected' : ''} ${formyl.supported ? '' : 'disabled'}>FOR${formyl.supported ? ' — explicit formyl cap' : ' — unavailable: ' + escapeHtml(formyl.reason)}</option>
      </select>
      <span class="hint">C-ter: ${escapeHtml(lastRes ? lastRes.resname + ' ' + lastRes.resid : '?')}</span>
      <select class="proc-cter-sel" data-chain="${escapeHtml(ch)}">
        <option value="" ${!t.cter ? 'selected' : ''}>Standard (COO⁻)</option>
        <option value="NME" ${t.cter === 'NME' ? 'selected' : ''} ${nme.supported ? '' : 'disabled'}>NME${nme.supported ? ' — explicit methylamide cap' : ' — unavailable: ' + escapeHtml(nme.reason)}</option>
      </select>
    </div>`;
  });
  container.innerHTML = html || '<p class="hint">No chains detected.</p>';

  container.querySelectorAll('.proc-nter-sel').forEach(sel => {
    sel.addEventListener('change', () => {
      const ch = sel.dataset.chain;
      if (!_procTermini[ch]) _procTermini[ch] = { nter: '', cter: '' };
      _procTermini[ch].nter = sel.value;
      invalidateStructureChemistry();
      applyDetectedInputModifications();
      applyModification(-1, '', '');
    });
  });
  container.querySelectorAll('.proc-cter-sel').forEach(sel => {
    sel.addEventListener('change', () => {
      const ch = sel.dataset.chain;
      if (!_procTermini[ch]) _procTermini[ch] = { nter: '', cter: '' };
      _procTermini[ch].cter = sel.value;
      invalidateStructureChemistry();
      applyDetectedInputModifications();
      applyModification(-1, '', '');
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
    html += `<div class="proc-mod-chain-block"><h4 style="margin:8px 0 4px;color:#475569;">Chain ${escapeHtml(ch || ' ')} <span class="hint">${chainRes.length} residues</span></h4><div class="proc-mod-sequence" role="listbox" aria-label="Residues in chain ${escapeHtml(ch || 'unnamed')}">`;
    chainRes.forEach(r => {
      const i = r.index;
      const residuePatches = _procPatchCatalog.filter(function(patch) {
        return (patch.target_residues || []).indexOf(r.resname) >= 0;
      });
      const hasSupportedPatch = residuePatches.some(function(patch) {
        return patchForResidueContext(patch, r).supported !== false;
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
      html += `<button type="button" role="option" class="${cls}" data-idx="${i}" data-has-patches="${residuePatches.length ? '1' : '0'}" aria-selected="${_procSelectedIdx === i ? 'true' : 'false'}" ${residuePatches.length ? '' : 'disabled aria-disabled="true"'} title="#${i+1} ${escapeHtml(r.resname)} ch ${escapeHtml(ch)} resid ${escapeHtml(r.resid)}; ${escapeHtml(capability)}" aria-label="Residue ${escapeHtml(r.resname)} ${escapeHtml(r.resid)}, chain ${escapeHtml(ch || 'unnamed')}; ${escapeHtml(capability)}">${escapeHtml(r.resname)}</button>`;
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
    return '<option value="' + residue.index + '">Chain ' + escapeHtml(residue.chain || '?') +
      ' — CYS ' + escapeHtml(residue.resid) + ' (#' + (residue.index + 1) + ')</option>';
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
    return '<div class="proc-mod-item">Chain ' + escapeHtml(left.chain || '?') + ':CYS ' + escapeHtml(left.resid) +
      ' — Chain ' + escapeHtml(right.chain || '?') + ':CYS ' + escapeHtml(right.resid) +
      ' <span style="color:#64748b;">(disulfide)</span>' +
      '<button data-crosslink="' + index + '" class="proc-crosslink-remove" title="Remove">×</button></div>';
  }).join('') || '<p class="hint">No disulfide crosslinks selected.</p>';
  list.querySelectorAll('.proc-crosslink-remove').forEach(function(button) {
    button.addEventListener('click', function() {
      _procCrosslinks.splice(parseInt(button.dataset.crosslink), 1);
      invalidateStructureChemistry();
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
  invalidateStructureChemistry();
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
  const forceField = selectedProteinForceField(), task = state.taskId;
  const request = ++_procPickerRequest;
  const current = () => request === _procPickerRequest && _procSelectedIdx === idx &&
    _procResidues[idx] === r && state.taskId === task &&
    selectedProteinForceField() === forceField && !picker.classList.contains('hidden');
  targetEl.textContent = `${r.resname} ${r.resid} (Chain ${r.chain || '?'}, #${idx+1})`;
  optionsEl.innerHTML = '<p class="hint">Loading modifications…</p>';
  picker.classList.remove('hidden');
  try {
    const res = await fetch(`/api/patches/${encodeURIComponent(r.resname)}?force_field=${encodeURIComponent(forceField)}`);
    if (!current()) return;
    if (!res.ok) { optionsEl.innerHTML = '<p class="hint">No patches available.</p>'; return; }
    const payload = await res.json();
    if (!current()) return;
    const patches = payload.map(p => patchForResidueContext(p, r));
    if (!patches.length) { optionsEl.innerHTML = '<p class="hint">No modifications available.</p>'; return; }
    const supportedPatches = patches.filter(p => p.supported !== false);
    const capabilityNotice = supportedPatches.length ? '' :
      '<p class="hint" style="color:#b45309;margin-bottom:8px;">No simulation-ready modification is available for this residue. Catalogue entries below are disabled because complete atoms and bonded parameters are not yet implemented.</p>';
    optionsEl.innerHTML = capabilityNotice + patches.map(p =>
      `<div class="proc-patch-option" data-patch="${escapeHtml(p.id)}" data-supported="${p.supported !== false}"
            aria-disabled="${p.supported === false}" style="${p.supported === false ? 'opacity:.5;cursor:not-allowed;' : ''}">
        <span class="patch-name">${escapeHtml(p.name)}</span> → ${escapeHtml(p.product_name)}
        <span style="color:#64748b;font-size:11px;">(${escapeHtml(p.description)}; net charge shift ${p.charge_shift > 0 ? '+' : ''}${escapeHtml(p.charge_shift)}${p.supported === false ? '; unavailable: ' + escapeHtml(p.support_reason) : ''})</span>
      </div>`).join('');
    optionsEl.querySelectorAll('.proc-patch-option').forEach(opt => {
      opt.addEventListener('click', () => {
        if (!current()) return;
        const patch = patches.find(p => p.id === opt.dataset.patch);
        if (patch && patch.supported !== false) { applyModification(idx, patch.id, patch.product_name, patch.charge_shift); closePatchPicker(); }
      });
    });
  } catch (e) { if (current()) optionsEl.innerHTML = '<p class="hint">Error.</p>'; }
}

function closePatchPicker() {
  ++_procPickerRequest;
  const picker = document.getElementById('proc-patch-picker');
  if (picker) picker.classList.add('hidden');
  const options = document.getElementById('proc-patch-options');
  if (options) options.replaceChildren();
}

function applyModification(idx, patchId, productName, chargeShift, source) {
  if (idx >= 0) invalidateStructureChemistry();
  _procModifications = _procModifications.filter(m => m.index !== idx);
  if (patchId) _procModifications.push({ index: idx, patch_id: patchId, product_name: productName, charge_shift: chargeShift, source: source || 'user' });
  const listEl = document.getElementById('proc-mod-list');
  if (listEl) {
    listEl.innerHTML = _procModifications.map(m => {
      const rr = _procResidues[m.index];
      return `<div class="proc-mod-item">
        Chain ${escapeHtml(rr.chain)} — <b>${escapeHtml(rr.resname)} ${escapeHtml(rr.resid)}</b> → <b>${escapeHtml(m.product_name)}</b>
        <span style="color:#64748b;">(${escapeHtml(m.patch_id)})</span>
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
  renderDetectedModificationNotice();
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
  // ARG contributes +1 in this UI estimate; the backend determines the final charge.
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

// Load-order manifest: records that this file ran to completion.
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push("app_parts/structure_processing.js");

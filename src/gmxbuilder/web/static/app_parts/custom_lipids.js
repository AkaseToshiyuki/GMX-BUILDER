// Loaded as a classic script after app.js.
// Public globals are retained because the main workflow calls them directly.

// ===================================================================
// Custom Lipid Picker
// ===================================================================

let _lipidPickerData = { lipids: [], categories: {} };
let _selectedLipid = 'POPC';
let _v4ListState = 'pending';
let _v4ListRequest = 0;

let _v4RefreshPromise = null;
let _v4RefreshTimer = null;
function applyV4Availability(data) {
  if (data.status === 'checking' || data.status === 'error') {
    _v4ListState = data.status === 'error' ? 'error' : 'loading';
  } else {
    _lipidPickerData.lipids.forEach(lipid => {
      if (lipid._custom) return;
      const entries = data.entries.filter(entry => entry.lipid_name === lipid.name && entry.ready === true);
      lipid.parameterizations = entries.map(entry => entry.lipid_ff);
      lipid.validation_scopes = Object.fromEntries(entries.map(entry => [entry.lipid_ff, entry.validation_scope]));
      const mixed = entries.find(entry => entry.amber_mixed_ready);
      if (mixed) {
        lipid.parameterizations.push('amber-mixed');
        lipid.validation_scopes['amber-mixed'] = mixed.validation_scope;
      }
    });
    _v4ListState = 'ready';
  }
  updateV4CompositionAvailability();
  clearTimeout(_v4RefreshTimer);
  if (_v4ListState === 'loading') _v4RefreshTimer = setTimeout(refreshV4LipidAvailability, 2000);
}
function refreshV4LipidAvailability() {
  if (_v4RefreshPromise) return _v4RefreshPromise;
  const request = ++_v4ListRequest;
  if (_v4ListState !== 'ready') { _v4ListState = 'loading'; updateV4CompositionAvailability(); }
  _v4RefreshPromise = (async () => {
    try {
      const data = await GMXHttp.json('/api/lipid-library-list', {cache: 'no-store'}, {
        current: () => request === _v4ListRequest,
        validate: data => data.library_version === 4 && Array.isArray(data.entries)
      });
      if (request !== _v4ListRequest) return false;
      applyV4Availability(data);
    } catch (error) {
      if (request !== _v4ListRequest) return false;
      _v4ListState = 'error';
      updateV4CompositionAvailability();
    } finally { _v4RefreshPromise = null; }
    const dropdown = document.getElementById('lipid-picker-dropdown');
    if (dropdown && !dropdown.classList.contains('hidden'))
      renderLipidList(document.getElementById('lipid-picker-search')?.value.trim().toLowerCase() || '');
    return _v4ListState === 'ready';
  })();
  return _v4RefreshPromise;
}

function v4CompositionErrors() {
  if (_v4ListState !== 'ready') return [_v4ListState === 'error'
    ? 'V4 availability could not be checked. Re-enter Membrane Builder to retry.'
    : 'Checking V4 lipid availability…'];
  const source = selectedLipidParameterSource();
  const mix = _mixUpper.concat(_asymmetric ? _mixLower : []);
  return [...new Set(mix.filter(entry => Number(entry.ratio) > 0).map(entry => entry.name))]
    .flatMap(name => {
      const lipid = _lipidPickerData.lipids.find(item => item.name === name);
      if (lipid && (lipid._custom || (lipid.parameterizations || []).includes(source))) return [];
      return [lipidAvailabilityMessage(lipid || {name}, source)];
    });
}

function updateV4CompositionAvailability() {
  const errors = v4CompositionErrors();
  const button = document.getElementById('check-composition-btn');
  if (button) button.disabled = _stepRunning || errors.length > 0;
  let notice = document.getElementById('v4-lipid-availability');
  if (!notice && button) {
    notice = document.createElement('div');
    notice.id = 'v4-lipid-availability';
    notice.setAttribute('role', 'status');
    button.parentNode.insertBefore(notice, button);
  }
  if (notice) notice.textContent = errors.join(' ');
  if (errors.length && _v4ListState === 'ready' && (_compositionChecked || _checkedSteps.has('membrane'))) {
    _invalidateMembraneBuild();
    revokeStepPass('membrane');
  }
  updateNextButtonState();
}

setInterval(() => {
  if (!document.hidden && document.getElementById('panel-membrane')?.classList.contains('active') && !_stepRunning) {
    refreshV4LipidAvailability();
  }
}, 30000);


let _pickerTarget = null;  // { leaflet, idx } of which row triggered the picker
let _lipidPickerReturnFocus = null;

function buildLipidPicker(lipids, categories, availability) {
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

  // Composition defaults are applied separately; refresh before allowing selection.
  if (availability) applyV4Availability(availability);
  else refreshV4LipidAvailability();
}

function openLipidDropdown(anchorEl) {
  const dropdown = document.getElementById('lipid-picker-dropdown');
  if (!dropdown) return;
  // Position dropdown near the anchor element
  if (anchorEl) {
    _lipidPickerReturnFocus = anchorEl;
    const rect = anchorEl.getBoundingClientRect();
    const margin = 8;
    const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
    const dropdownWidth = Math.min(Math.max(rect.width, 500), Math.max(0, viewportWidth - margin * 2));
    const dropdownLeft = Math.max(margin, Math.min(rect.left, viewportWidth - dropdownWidth - margin));
    dropdown.style.position = 'fixed';
    dropdown.style.left = dropdownLeft + 'px';
    dropdown.style.top = (rect.bottom + 4) + 'px';
    dropdown.style.width = dropdownWidth + 'px';
  }
  dropdown.classList.remove('hidden');
  const searchEl = document.getElementById('lipid-picker-search');
  if (searchEl) searchEl.value = '';
  renderLipidList('');
  if (anchorEl) keepDropdownInViewport(dropdown, anchorEl.getBoundingClientRect());
}

function keepDropdownInViewport(dropdown, anchorRect) {
  var margin = 8;
  var viewportHeight = document.documentElement.clientHeight || window.innerHeight;
  var desiredTop = anchorRect.bottom + 4;
  var top = Math.max(margin, Math.min(desiredTop, viewportHeight - dropdown.offsetHeight - margin));
  dropdown.style.top = top + 'px';
}

function closeLipidDropdown() {
  const dropdown = document.getElementById('lipid-picker-dropdown');
  if (dropdown) dropdown.classList.add('hidden');
  _pickerTarget = null;
}

function selectLipid(name, {restoreDefault = false} = {}) {
  const lipid = _lipidPickerData.lipids.find(l => l.name === name);
  if (!lipid) { closeLipidDropdown(); return; }
  const source = selectedLipidParameterSource();
  // Restoring a configured default must not depend on an in-flight availability read.
  // The composition check still requires current V4 readiness.
  if (!restoreDefault && !lipid._custom && (_v4ListState !== 'ready' || !source || !(lipid.parameterizations || []).includes(source))) {
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
    chains: lipid.chains,
    chain_label: lipid.chain_label,
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
  if (lipidSelect && ['lipid21', 'amber-mixed', 'gaff2', 'charmm36m', 'charmm36', 'oplsaa'].includes(lipidSelect.value)) {
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
    'amber-mixed': 'Lipid21 with GAFF2 for missing species',
    charmm36m: 'CHARMM36m',
    charmm36: 'CHARMM36',
    oplsaa: 'OPLS-AA',
  })[source] || source || 'the selected force field';
}

function lipidAvailabilityMessage(lipid, selectedSource) {
  const alternatives = (lipid.parameterizations || [])
    .filter(source => source !== selectedSource)
    .map(lipidParameterSourceLabel);
  return lipid.name + ' is awaiting V4 acceptance with ' + lipidParameterSourceLabel(selectedSource) + '. ' +
    (alternatives.length
      ? 'Available with: ' + alternatives.join(', ') + '.'
      : 'No validated alternative is installed.');
}

function renderLipidList(filter) {
  const container = document.getElementById('lipid-picker-list');
  container.innerHTML = '';
  container.setAttribute('role', 'listbox');
  container.setAttribute('aria-label', 'Available lipids');

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

      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'lipid-card';
      card.dataset.lipidName = l.name;
      card.setAttribute('role', 'option');
      const supported = _v4ListState === 'ready' && (l._custom || (l.parameterizations || []).includes(selectedSource));
      card.disabled = !supported;
      card.setAttribute('aria-selected', l.name === _selectedLipid ? 'true' : 'false');
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
      parameterTag.textContent = availableSources.indexOf(selectedSource) >= 0
        ? lipidParameterSourceLabel(selectedSource)
        : 'Unavailable';
      parameterTag.title = 'Topology parameter source';
      meta.appendChild(parameterTag);

      const tailTag = document.createElement('span');
      tailTag.className = 'lipid-card-tag';
      tailTag.textContent = l.chain_label || 'See full molecular structure';
      const scope = (l.validation_scopes || {})[selectedLipidParameterSource()];
      if (scope) {
        tailTag.textContent += ' · ' + scope.label;
        tailTag.title = scope.description;
      }
      tailTag.title = (scope ? scope.description + ' ' : '') + 'Double-bond positions count from chain C1; Z/E specify geometry. Chains are listed without assigning sn positions.';
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

  // Requests for new lipids go through the administrator for every FF family.
  const customOpt = document.createElement('button');
  customOpt.type = 'button';
  customOpt.className = 'lipid-card lipid-option-custom';
  customOpt.setAttribute('role', 'option');
  customOpt.setAttribute('aria-selected', 'false');
  customOpt.innerHTML = '<div class="lipid-card-info" style="text-align:center;padding:8px;">' +
    'Contact administrator<div class="hint">Request a new lipid by email. Include its SMILES.</div></div>';
  customOpt.addEventListener('click', () => {
    const returnFocus = _lipidPickerReturnFocus;
    closeLipidDropdown();
    openCustomLipidModal(returnFocus);
  });
  container.appendChild(customOpt);
}

// ===================================================================
// Administrator contact and read-only access to legacy task lipids
// ===================================================================

function modalFocusableElements(modal) {
  return Array.from(modal.querySelectorAll(
    'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )).filter(function(element) {
    return !element.disabled && !element.closest('.hidden') && element.getAttribute('aria-hidden') !== 'true';
  });
}

function ensureAccessibleModal(modal) {
  if (!modal || modal._accessibilityReady) return;
  modal._accessibilityReady = true;
  modal.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
      if (modal.id === 'custom-lipid-modal') {
        event.preventDefault();
        closeCustomLipidModal();

      }
      return;
    }
    if (event.key !== 'Tab') return;
    var focusable = modalFocusableElements(modal);
    if (!focusable.length) {
      event.preventDefault();
      modal.querySelector('.modal-panel')?.focus();
      return;
    }
    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (focusable.indexOf(document.activeElement) === -1) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    } else if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
}

function openAccessibleModal(modal, initialFocus, returnFocus) {
  if (!modal) return;
  ensureAccessibleModal(modal);
  if (modal.classList.contains('hidden')) modal._returnFocus = returnFocus || document.activeElement;
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
  window.requestAnimationFrame(function() {
    var target = initialFocus && !initialFocus.disabled ? initialFocus : modalFocusableElements(modal)[0];
    (target || modal.querySelector('.modal-panel'))?.focus();
  });
}

function closeAccessibleModal(modal) {
  if (!modal) return;
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
  var returnFocus = modal._returnFocus;
  modal._returnFocus = null;
  if (returnFocus && returnFocus.isConnected && typeof returnFocus.focus === 'function') {
    returnFocus.focus();
  }
}

function openCustomLipidModal(returnFocus) {
  const modal = document.getElementById('custom-lipid-modal');
  openAccessibleModal(modal, document.getElementById('custom-lipid-contact-link'), returnFocus);
}

function closeCustomLipidModal() {
  closeAccessibleModal(document.getElementById('custom-lipid-modal'));
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

async function loadTaskCustomLipids() {
  if (!state.taskId) return;
  const taskId = state.taskId;
  try {
    const response = await fetch('/api/task/' + taskId + '/custom-lipids');
    if (!response.ok) return;
    const payload = await response.json();
    if (state.taskId !== taskId) return;
    const records = payload.lipids || [];
    records.filter(record => record.state === 'ready').forEach(addReadyTaskLipid);
    const incomplete = records.filter(record => record.state !== 'ready');
    state.customLipidBusy = incomplete.length > 0;
    const notice = document.getElementById('custom-lipid-legacy-status');
    if (notice) {
      notice.classList.toggle('hidden', !state.customLipidBusy);
      notice.textContent = state.customLipidBusy
        ? 'An earlier custom lipid calculation is incomplete: ' +
          incomplete.map(record => record.name + ' (' + record.state + ')').join(', ') +
          '. Online calculation and retry are no longer available. Contact the administrator ' +
          'for help with this task, or start a new task using installed lipids.'
        : '';
    }
    updateNextButtonState();
    updateStepNavHighlight();
    if (state.customLipidBusy) openCustomLipidModal();
  } catch (error) {
    console.warn('Could not restore task custom lipids', error);
  }
}

function initCustomLipidModal() {
  if (initCustomLipidModal._done) return;
  initCustomLipidModal._done = true;
  const cancel = document.getElementById('custom-lipid-cancel-btn');
  if (cancel) cancel.addEventListener('click', closeCustomLipidModal);
  const modal = document.getElementById('custom-lipid-modal');
  if (modal) {
    modal.addEventListener('click', event => {
      if (event.target === modal) closeCustomLipidModal();
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
    svg += `<text x="45" y="48" text-anchor="middle" font-size="7" fill="#64748b">${escapeHtml(String(lipid.name))}</text>`;
  } else if (isSingleTail) {
    // Sphingolipids / lyso lipids: single tail from backbone
    svg += `<circle cx="20" cy="24" r="8" fill="${hc}" opacity="0.35"/>`;
    svg += `<text x="20" y="27" text-anchor="middle" font-size="6" font-weight="bold" fill="${hc}">${escapeHtml(String(lipid.category))}</text>`;
    svg += `<line x1="28" y1="20" x2="42" y2="10" stroke="#475569" stroke-width="1.5"/>`;
    svg += `<line x1="42" y1="10" x2="${42 + Math.max(t1Len*0.6, 8)}" y2="8" stroke="#475569" stroke-width="1.2"/>`;
    if (t1Len > 0) svg += `<text x="45" y="46" text-anchor="middle" font-size="6.5" fill="#94a3b8">C${t1Len}:${t1Unsat}</text>`;
  } else if (isGlycolipid) {
    // Glycolipids: sugar headgroup + two tails
    svg += `<circle cx="16" cy="20" r="6" fill="${hc}" opacity="0.25"/>`;
    svg += `<circle cx="22" cy="18" r="4" fill="${hc}" opacity="0.15"/>`;
    svg += `<text x="19" y="35" text-anchor="middle" font-size="6" fill="${hc}">${escapeHtml(String(lipid.category))}</text>`;
    svg += `<line x1="24" y1="24" x2="36" y2="16" stroke="#94a3b8" stroke-width="1"/>`;
    svg += `<line x1="24" y1="24" x2="36" y2="28" stroke="#94a3b8" stroke-width="1"/>`;
    const t1w = Math.max(6, Math.min(t1Len*1.0, 30));
    const t2w = Math.max(6, Math.min(t2Len*1.0, 30));
    svg += `<line x1="36" y1="14" x2="${36+t1w}" y2="11" stroke="#475569" stroke-width="1.3"/>`;
    svg += `<line x1="36" y1="28" x2="${36+t2w}" y2="30" stroke="#475569" stroke-width="1.3"/>`;
    svg += `<text x="40" y="46" text-anchor="start" font-size="6" fill="#94a3b8">${escapeHtml(String(lipid.formula).substring(0,15))}...</text>`;
  } else {
    // Glycerophospholipid: headgroup circle + glycerol + two tails
    svg += `<circle cx="18" cy="22" r="8" fill="${hc}" opacity="0.35"/>`;
    svg += `<text x="18" y="25" text-anchor="middle" font-size="7" font-weight="bold" fill="${hc}">${escapeHtml(String(lipid.category))}</text>`;

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
  svg += `<text x="45" y="57" text-anchor="middle" font-size="6" fill="#cbd5e1">${escapeHtml(String(lipid.formula))}</text>`;
  svg += `</svg>`;
  return svg;
}

// Load-order manifest: records that this file ran to completion.
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push("app_parts/custom_lipids.js");

/**
 * Ion Step — multi-species selection with per-ion concentration
 */
(function() {
  'use strict';

  var _ionCations = [{ name: "NA", charge: 1, conc: 0.15 }];
  var _ionAnions = [{ name: "CL", charge: -1, conc: 0.15 }];
  var _ionsChecked = false;
  var _systemConfirmed = false;
  var _ionViewer = null;
  var WATER_RESIDUES = ['SOL', 'HOH', 'WAT', 'TIP', 'TIP3', 'SPC', 'SPCE'];
  var ION_RESIDUES = ['NA', 'CL', 'K', 'LI', 'CS', 'CA', 'MG', 'ZN', 'BR', 'I'];

  var ION_POOL = {
    cations: [
      { name: "NA", label: "Na⁺", charge: 1, mass: 22.99 },
      { name: "K", label: "K⁺", charge: 1, mass: 39.10 },
      { name: "LI", label: "Li⁺", charge: 1, mass: 6.94 },
      { name: "CS", label: "Cs⁺", charge: 1, mass: 132.91 },
      { name: "CA", label: "Ca²⁺", charge: 2, mass: 40.08 },
      { name: "MG", label: "Mg²⁺", charge: 2, mass: 24.31 },
      { name: "ZN", label: "Zn²⁺", charge: 2, mass: 65.38 }
    ],
    anions: [
      { name: "CL", label: "Cl⁻", charge: -1, mass: 35.45 },
      { name: "BR", label: "Br⁻", charge: -1, mass: 79.90 },
      { name: "I", label: "I⁻", charge: -1, mass: 126.90 }
    ]
  };

  function getIonConc(ion) {
    return (typeof ion.conc === 'number') ? ion.conc : 0.15;
  }

  function invalidateIonCheck() {
    _ionsChecked = false;
    _systemConfirmed = false;
    if (typeof _checkedSteps !== 'undefined') _checkedSteps.delete('ions');
    if (typeof _checkedConfig !== 'undefined' && _checkedConfig) delete _checkedConfig.ions;
    var status = document.getElementById('ion-check-status');
    if (status) { status.textContent = 'Changed — Check again'; status.style.color = '#d97706'; }
    var confirm = document.getElementById('ion-confirm-system-btn');
    if (confirm) confirm.disabled = true;
    var confirmStatus = document.getElementById('ion-confirm-system-status');
    if (confirmStatus) {
      confirmStatus.textContent = 'Ion settings changed. Run Check again before confirming.';
      confirmStatus.style.color = '#d97706';
    }
    updateNextButtonState();
  }

  function updateIonMethodWarning() {
    var method = document.getElementById('ion-method')?.value || 'random';
    var warning = document.getElementById('ion-method-warning');
    if (warning) warning.classList.toggle('hidden', method === 'random');
  }

  function renderIonCards() {
    var catC = document.getElementById("ion-cation-cards");
    var aniC = document.getElementById("ion-anion-cards");
    if (catC) catC.innerHTML = _ionCations.map(function(ion, i) { return cardHTML(i, ion, "cation"); }).join("");
    if (aniC) aniC.innerHTML = _ionAnions.map(function(ion, i) { return cardHTML(i, ion, "anion"); }).join("");
    wireEvents();
  }

  function cardHTML(idx, ion, type) {
    var pool = type === "cation" ? ION_POOL.cations : ION_POOL.anions;
    var opts = pool.map(function(p, pi) {
      var sel = p.name === ion.name ? " selected" : "";
      return '<option value="' + pi + '"' + sel + '>' + p.label + '</option>';
    }).join("");
    var c = getIonConc(ion);
    return '<div class="ion-card">' +
      '<select class="ion-species-sel" data-type="' + type + '" data-idx="' + idx + '">' + opts + '</select>' +
      '<span class="ion-name">' + ion.name + '</span>' +
      '<span class="ion-charge">' + (ion.charge >= 0 ? '+' : '') + ion.charge + '</span>' +
      '<input class="ion-conc-input" type="number" data-type="' + type + '" data-idx="' + idx +
        '" value="' + c + '" step="0.01" min="0" max="2.0" style="width:70px;" title="Concentration (M)">' +
      '<span class="hint" style="font-size:11px;">M</span>' +
      '<button class="ion-remove" data-type="' + type + '" data-idx="' + idx + '" title="Remove">×</button>' +
      '</div>';
  }

  function wireEvents() {
    // Species change
    document.querySelectorAll(".ion-species-sel").forEach(function(sel) {
      sel.onchange = function() {
        var type = sel.dataset.type;
        var idx = parseInt(sel.dataset.idx);
        var pool = type === "cation" ? ION_POOL.cations : ION_POOL.anions;
        var chosen = pool[parseInt(sel.value)];
        if (chosen) {
          var arr = type === "cation" ? _ionCations : _ionAnions;
          var oldConc = arr[idx].conc;
          arr[idx] = { name: chosen.name, charge: chosen.charge, conc: oldConc };
          renderIonCards();
          invalidateIonCheck();
        }
      };
    });

    // Concentration change
    document.querySelectorAll(".ion-conc-input").forEach(function(inp) {
      inp.onchange = function() {
        var type = inp.dataset.type;
        var idx = parseInt(inp.dataset.idx);
        var v = parseFloat(inp.value);
        if (isNaN(v) || v < 0) v = 0;
        if (v > 2.0) v = 2.0;
        inp.value = v.toFixed(2);
        var arr = type === "cation" ? _ionCations : _ionAnions;
        if (arr[idx]) arr[idx].conc = v;
        invalidateIonCheck();
      };
    });

    // Remove
    document.querySelectorAll(".ion-remove").forEach(function(btn) {
      btn.onclick = function() {
        var type = btn.dataset.type;
        var idx = parseInt(btn.dataset.idx);
        var arr = type === "cation" ? _ionCations : _ionAnions;
        if (arr.length > 1) {
          arr.splice(idx, 1);
          renderIonCards();
          invalidateIonCheck();
        }
      };
    });
  }

  function addIonSpecies(type) {
    var arr = type === "cation" ? _ionCations : _ionAnions;
    var pool = type === "cation" ? ION_POOL.cations : ION_POOL.anions;
    var used = {};
    arr.forEach(function(i) { used[i.name] = true; });
    var avail = pool[0];
    for (var k = 0; k < pool.length; k++) {
      if (!used[pool[k].name]) { avail = pool[k]; break; }
    }
    arr.push({ name: avail.name, charge: avail.charge, conc: 0.15 });
    renderIonCards();
    invalidateIonCheck();
  }

  function renderIonSummary(metrics) {
    var salt = metrics.salt_counts || {};
    var neutralizing = metrics.neutralizing_counts || {};
    var totals = metrics.total_counts || {};
    var concentrations = metrics.concentrations_m || {};
    var tbody = document.getElementById("ion-summary-body");
    var resultDiv = document.getElementById("ion-result");
    var footer = document.getElementById("ion-summary-footer");
    if (!tbody) return;
    var rows = "";
    Object.keys(totals).forEach(function(name) {
      var pool = ION_POOL.cations.concat(ION_POOL.anions);
      var ion = pool.find(function(item) { return item.name === name; }) || { charge: 0 };
      rows += '<tr><td>' + name + '</td><td>' + (ion.charge > 0 ? '+' : '') + ion.charge +
        '</td><td>' + (salt[name] || 0) + '</td><td>' + (neutralizing[name] || 0) +
        '</td><td><b>' + totals[name] + '</b></td><td>' + Number(concentrations[name] || 0).toFixed(2) + '</td></tr>';
    });
    tbody.innerHTML = rows;
    resultDiv.classList.remove("hidden");
    footer.textContent = '✓ Backend-validated: solute ' + Number(metrics.solute_charge_e || 0).toFixed(0) +
      ' e, ions ' + Number(metrics.ion_charge_e || 0).toFixed(0) + ' e, final ' +
      Number(metrics.final_charge_e || 0).toFixed(0) + ' e; replaced ' +
      (metrics.waters_replaced || 0) + ' complete water molecules.';
    footer.className = "ion-summary-footer ok";
  }

  function countWaterOxygens(pdb) {
    var count = 0;
    pdb.split('\n').forEach(function(line) {
      if (line.indexOf('ATOM  ') !== 0 && line.indexOf('HETATM') !== 0) return;
      var residue = line.substring(17, 20).trim();
      var element = line.length >= 78 ? line.substring(76, 78).trim().toUpperCase() : '';
      var atomName = line.substring(12, 16).trim().toUpperCase();
      if (WATER_RESIDUES.indexOf(residue) >= 0 &&
          (element === 'O' || atomName === 'O' || atomName === 'OW' || atomName === 'OH2')) {
        count += 1;
      }
    });
    return count;
  }

  function applyIonSystemStyles(viewer) {
    // Keep complete waters visible without drawing three opaque spheres per
    // molecule. A blue line model plus one translucent oxygen sphere makes
    // the solvent envelope legible even for 30k-50k waters.
    viewer.setStyle(
      {resn: WATER_RESIDUES},
      {line: {opacity: 0.55, color: '0x60a5fa', linewidth: 1.0}}
    );
    viewer.addStyle(
      {resn: WATER_RESIDUES, elem: 'O'},
      {sphere: {radius: 0.11, opacity: 0.38, color: '0x3b82f6'}}
    );
    viewer.setStyle(
      {resn: ION_RESIDUES},
      {sphere: {radius: 0.45, opacity: 0.90, colorscheme: 'Jmol'}}
    );
  }

  async function renderIonViewer() {
    var host = document.getElementById('ion-viewer');
    if (!host || !state.taskId) return false;
    // 3Dmol positions its canvas absolutely. Keep it anchored to the
    // specified review panel instead of the page/viewport origin.
    host.style.position = 'relative';
    host.style.overflow = 'hidden';
    if (typeof $3Dmol === 'undefined') {
      var unavailable = document.getElementById('ion-confirm-system-status');
      if (unavailable) {
        unavailable.textContent = '3D viewer library is unavailable; reload the page before confirmation.';
        unavailable.style.color = '#dc2626';
      }
      return false;
    }
    var pdb = await _loadStepViewerPdb('ions');
    if (!pdb) return false;
    if (_ionViewer) {
      try { _ionViewer.clear(); } catch (error) {}
      _ionViewer = null;
    }
    while (host.firstChild) host.removeChild(host.firstChild);
    _ionViewer = $3Dmol.createViewer(host, {backgroundColor: '0xffffff', antialias: true});
    var viewer = _ionViewer;
    viewer.setBackgroundColor('0xffffff');
    viewer.setSlab(-100000, 100000);
    viewer.addModel(pdb, 'pdb');
    if (typeof _applyUnifiedStyle === 'function') {
      _applyUnifiedStyle(viewer, pdb);
    } else {
      viewer.setStyle({}, {stick: {radius: 0.12, colorscheme: 'Jmol'}});
    }
    applyIonSystemStyles(viewer);
    var waterCount = countWaterOxygens(pdb);
    var viewerLabel = document.getElementById('ion-viewer-label');
    if (viewerLabel) {
      viewerLabel.textContent = waterCount > 0
        ? waterCount.toLocaleString() + ' waters shown in blue; ions shown as large spheres.'
        : 'No water molecules are present in this checked system; ions are shown as large spheres.';
      viewerLabel.style.color = waterCount > 0 ? '#64748b' : '#d97706';
    }

    var boxA = 0, boxB = 0, boxC = 0;
    pdb.split('\n').some(function(line) {
      if (line.indexOf('CRYST1') !== 0) return false;
      boxA = Number(line.substring(6, 15));
      boxB = Number(line.substring(15, 24));
      boxC = Number(line.substring(24, 33));
      return true;
    });
    if (boxA > 0 && boxB > 0 && boxC > 0 && typeof drawOrthogonalBox === 'function') {
      drawOrthogonalBox(viewer, boxA, boxB, boxC, {x: 0, y: 0, z: 0});
    }
    viewer.zoomTo();
    viewer.render();
    viewer.setSlab(-100000, 100000);
    return true;
  }

  function confirmSimulationSystem() {
    if (!_ionsChecked || !_ionViewer) return;
    _systemConfirmed = true;
    var status = document.getElementById('ion-confirm-system-status');
    if (status) {
      status.textContent = '✓ Simulation system confirmed. Next is now enabled.';
      status.style.color = '#059669';
    }
    updateNextButtonState();
    updateStepNavHighlight();
  }

  async function runIonCheck() {
    var statusEl = document.getElementById("ion-check-status");
    var button = document.getElementById('ion-check-btn');
    if (!state.taskId) {
      statusEl.textContent = '✗ No task loaded';
      statusEl.style.color = '#dc2626';
      return;
    }
    if (_stepRunning) {
      statusEl.textContent = 'Please wait — another step is running';
      statusEl.style.color = '#d97706';
      return;
    }
    var cfg = buildModuleConfig().ions;
    try {
      _stepRunning = true;
      button.disabled = true;
      statusEl.textContent = 'Running backend validation and placement...';
      statusEl.style.color = '#d97706';
      var result = await _apiFetch('/api/step/' + state.taskId + '/ions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: cfg })
      });
      if (result.status !== 'ok') throw new Error(result.error || 'Ion placement failed');
      var metrics = (result.metrics || {}).ions || {};
      renderIonSummary(metrics);
      _ionsChecked = true;
      _checkedSteps.add('ions');
      _checkedConfig = _checkedConfig || {};
      _checkedConfig.ions = cfg;
      (result.invalidated_steps || []).forEach(function(step) { _checkedSteps.delete(step); });
      statusEl.textContent = '✓ Checked and saved (' + (result.elapsed_s == null ? '?' : result.elapsed_s) + 's)';
      statusEl.style.color = '#059669';
      _systemConfirmed = false;
      var viewerReady = await renderIonViewer();
      var confirm = document.getElementById('ion-confirm-system-btn');
      if (confirm) confirm.disabled = !viewerReady;
      var confirmStatus = document.getElementById('ion-confirm-system-status');
      if (confirmStatus && viewerReady) {
        confirmStatus.textContent = 'Inspect the exact checked coordinates and periodic box, then confirm.';
        confirmStatus.style.color = '#475569';
      }
      updateStepNavHighlight();
    } catch (error) {
      invalidateIonCheck();
      statusEl.textContent = '✗ ' + (error.message || 'Ion check failed');
      statusEl.style.color = '#dc2626';
    } finally {
      _stepRunning = false;
      button.disabled = false;
      updateNextButtonState();
    }
  }

  // ---- init ----
  document.addEventListener("DOMContentLoaded", function() {
    var addCat = document.getElementById("ion-add-cation");
    var addAni = document.getElementById("ion-add-anion");
    var checkBtn = document.getElementById("ion-check-btn");
    var neutCb = document.getElementById("ion-neutralize");
    var neutPair = document.getElementById("ion-neutralize-pair");
    var confirmSystem = document.getElementById('ion-confirm-system-btn');

    if (neutCb && neutPair) {
      neutCb.onchange = function() {
        if (neutCb.checked) neutPair.classList.remove("hidden");
        else neutPair.classList.add("hidden");
        invalidateIonCheck();
      };
      if (neutCb.checked) neutPair.classList.remove("hidden");
      else neutPair.classList.add("hidden");
    }

    if (addCat) addCat.onclick = function() { addIonSpecies("cation"); };
    if (addAni) addAni.onclick = function() { addIonSpecies("anion"); };
    if (checkBtn) checkBtn.onclick = function() { runIonCheck(); };
    if (confirmSystem) confirmSystem.onclick = confirmSimulationSystem;
    ["ion-method", "ion-exclusion", "ion-neutralize-cation", "ion-neutralize-anion"].forEach(function(id) {
      var element = document.getElementById(id);
      if (element) element.addEventListener('change', function() {
        invalidateIonCheck();
        if (id === 'ion-method') updateIonMethodWarning();
      });
    });

    renderIonCards();
    updateIonMethodWarning();

    window._getIonCations = function() { return _ionCations.map(function(i) { return i.name; }); };
    window._getIonAnions = function() { return _ionAnions.map(function(i) { return i.name; }); };
    window._getIonConcs = function() {
      var result = {};
      _ionCations.forEach(function(i) { result[i.name] = i.conc; });
      _ionAnions.forEach(function(i) { result[i.name] = i.conc; });
      return result;
    };
    window._isIonsChecked = function() { return _ionsChecked; };
    window._setIonsChecked = function(v) {
      _ionsChecked = Boolean(v);
      _systemConfirmed = false;
      if (confirmSystem) confirmSystem.disabled = !_ionsChecked;
    };
    window._setSystemConfirmed = function(v) {
      _systemConfirmed = Boolean(v) && _ionsChecked;
      if (confirmSystem) confirmSystem.disabled = !_ionsChecked;
    };
    window._isSystemConfirmed = function() { return _systemConfirmed; };
    window._renderIonViewer = renderIonViewer;
    window._applyIonSystemStyles = applyIonSystemStyles;
    window._countIonViewerWaters = countWaterOxygens;
  });

})();

// Loaded as a classic script after simulation.js.
// Public globals are retained for membrane checks and workflow navigation.

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
      document.getElementById("verify-status").textContent = "✓ System review confirmed. You may now proceed to Force Field.";
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

async function renderSystemViewer() {
  var el = document.getElementById("verify-viewer");
  if (!el) return;
  if (!state.pdbInfo || !state.pdbInfo.pdb_content) {
    setTimeout(renderSystemViewer, 300);
    return;
  }
  try { await GMXAssets.viewer(); } catch (error) { el.textContent = error.message; return; }
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

  window._verifyViewer = $3Dmol.createViewer(el, { backgroundColor: window.gmxViewerBackground(), antialias: true });
  window._verifyViewer.setBackgroundColor(window.gmxViewerBackground());
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
    var ionColors = GMX.ION_COLORS;
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
  if (window.invalidateFinalReview) window.invalidateFinalReview();
  _compositionChecked = false;
  _membraneCheckpointPdb = null;
  _membraneActualBox = null;
  _membraneActualCounts = null;
  renderMembraneCompositionWarnings([]);
  var countsNotice = document.getElementById("membrane-actual-counts");
  if (countsNotice) countsNotice.textContent = "";
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
    nLipidsEl.addEventListener('input', () => { _invalidateMembraneBuild(); updateCompositionStatus(); updateLipidCounts(); renderMembraneViewer(); });
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
var _membraneActualCounts = null;
var _membraneActualBox = null;       // [box_x, box_y, box_z] in nm from checkpoint

function weightedLeafletAPL(mix, lipids) {
  var weighted = 0;
  var totalRatio = 0;
  mix.forEach(function(m) {
    var lipid = lipids.find(function(candidate) { return candidate.name === m.name; });
    if (lipid && m.ratio > 0) {
      weighted += lipid.area_per_lipid * m.ratio;
      totalRatio += m.ratio;
    }
  });
  return totalRatio > 0 ? weighted / totalRatio : 0.65;
}

function previewBilayerAPL(lipids) {
  var upperAPL = weightedLeafletAPL(_mixUpper, lipids);
  if (!_asymmetric) return upperAPL;
  return Math.max(upperAPL, weightedLeafletAPL(_mixLower, lipids));
}

// Mirror MembraneBuilder._assign_lipids for the pre-Check count preview.
function allocatePreviewLipidCounts(nLipids, mix) {
  var requested = mix.filter(function(m) { return m.ratio > 0; }).map(function(m) {
    return {name: m.name, ratio: m.ratio, count: Math.max(1, Math.round(nLipids * m.ratio / 100))};
  });
  var remaining = nLipids;
  requested.forEach(function(item) {
    item.count = Math.min(item.count, remaining);
    remaining -= item.count;
  });
  if (remaining > 0 && requested.length) {
    var dominant = requested[0];
    requested.forEach(function(item) {
      if (item.ratio > dominant.ratio) dominant = item;
    });
    dominant.count += remaining;
  }
  return requested;
}

async function renderMembraneViewer() {
  var el = document.getElementById('membrane-3d-viewer');
  if (!el) return;
  var status = document.getElementById('membrane-viewer-status');
  if (el.offsetWidth === 0 || el.offsetHeight === 0) return;
  try { await GMXAssets.viewer(); } catch (error) { if (status) status.textContent = error.message; return; }

  // Use the membrane checkpoint PDB if available (set by checkComposition
  // after a successful build), otherwise the orient reference model.
  var hasCheckpoint = _checkedSteps.has('membrane');
  if (hasCheckpoint && window.GMXViewer) {
    const data = await GMXViewer.render('membrane-3d-viewer', 'membrane');
    if (data) {
      if (status) status.textContent = 'Assembled coordinates. Equilibration of the complete system is not established by this view.';
      document.getElementById('membrane-viewer-label').textContent = 'Checked box: ' + data.box_nm.map(row=>Math.hypot(...row).toFixed(2)).join(' × ') + ' nm';
    }
    return;
  }
  var pdbContent = _membraneCheckpointPdb || _orientedPdbContent || (state.pdbInfo && state.pdbInfo.pdb_content);
  var isPureMembrane = state.taskType && state.taskType.pipeline === 'pure_membrane';
  if (!pdbContent && isPureMembrane) {
    pdbContent = 'CRYST1   40.000   40.000   70.000  90.00  90.00  90.00 P 1           1\nEND\n';
  }
  if (!pdbContent) {
    if (status) status.textContent = 'Prepare a structure to display the membrane preview.';
    return;
  }
  if (status) status.textContent = 'Rendering membrane…';
  try {

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
    var lipids = _lipidPickerData.lipids || [];
    var avgAPL = previewBilayerAPL(lipids);
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
    // Mirror the backend construction fill factor (1.00).
    var lipidArea = nLipids * avgAPL;
    // Geometric preview only; the server measures each leaflet's footprint.
    boxXY = Math.max(Math.sqrt(lipidArea), protXY + (protXY > 0 ? 4.0 : 0), 4.0);
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

  _membraneViewer = $3Dmol.createViewer(el, {backgroundColor: window.gmxViewerBackground(), antialias: true});
  _membraneViewer.setBackgroundColor(window.gmxViewerBackground());
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

  v.zoomTo();
  v.render();
  if (status) status.textContent = hasCheckpoint
    ? 'Assembled coordinates. This view does not establish equilibration of the complete system.'
    : 'Geometric preview. Lipid placement and final box dimensions are determined when the membrane is built.';
  var label = document.getElementById('membrane-viewer-label');
  if (label) {
    var nLipidsLabel = 150;
    var nLipidsEl2 = document.getElementById('n-lipids-per-leaflet');
    if (nLipidsEl2) { var nv2 = parseInt(nLipidsEl2.value); if (!isNaN(nv2)) nLipidsLabel = nv2; }
    label.textContent = 'Box: ' + boxXY.toFixed(1) + '×' + boxXY.toFixed(1) + '×' + boxZ.toFixed(1)
      + ' nm  (n=' + (hasCheckpoint ? 'built' : nLipidsLabel) + '/leaflet)';
  }
  } catch (error) {
    if (status) status.textContent = 'The membrane view could not render. Retry the build or reload the page.';
    console.warn('Membrane viewer rendering failed', error);
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
    trigger.innerHTML = `<span class="mix-lipid-trigger-name">${escapeHtml(String(entry.name))}</span><span class="mix-lipid-trigger-cat">${escapeHtml(String(lip ? lip.category : ''))}</span><span class="mix-lipid-trigger-arrow">&#9662;</span>`;
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
  if (_stepRunning) return;
  revokeStepPass('membrane');
  await refreshV4LipidAvailability();
  var errors = v4CompositionErrors();
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
  } else if (totalLipids > 5000) {
    errors.push('At most 5000 lipids per leaflet are allowed by the build resource budget.');
  }

  if (errors.length > 0) {
    clearStepProgress('membrane');
    _invalidateMembraneBuild(); _compositionErrors = errors;
  } else {
    // Run membrane step on server
    var statusEl = document.getElementById('composition-status');
    if (statusEl) { statusEl.textContent = 'Running...'; statusEl.style.color = '#d97706'; }
    var button = document.getElementById('check-composition-btn');
    _stepRunning = true;
    if (button) button.disabled = true;
    var progress = startStepProgress('membrane', 'check-composition-btn', 'composition-status');
    try {
      var cfg = buildModuleConfig().membrane || {};
      var resp = await fetch('/api/step/' + state.taskId + '/membrane', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: cfg }),
      });
      var result = await resp.json();
      if (result.status === 'ok') {
        finishStepProgress(progress, true);
        _compositionChecked = true; _compositionErrors = [];
        _checkedSteps.add('membrane');
        if (statusEl) statusEl.textContent = '';
        // Capture actual box dimensions from the server — the builder may
        // shrink the box after lipid placement and relaxation (WYSIWYG).
        if (result.metrics && result.metrics.box_dimensions_nm) {
          _membraneActualBox = result.metrics.box_dimensions_nm;
        }
        _membraneActualCounts = result.metrics && result.metrics.membrane || null;
        renderMembraneCompositionWarnings((result.metrics || {}).membrane_composition_warnings || []);
        // Load the membrane checkpoint PDB so the viewer shows the actual
        // built membrane (WYSIWYG).  _membraneCheckpointPdb is used by
        // renderMembraneViewer — if set, it takes precedence over the
        // orient reference model.
        _membraneCheckpointPdb = null;
        if (typeof renderMembraneViewer === 'function') { renderMembraneViewer(); }
      } else {
        finishStepProgress(progress, false, result.error || 'Failed');
        _invalidateMembraneBuild();
        _compositionErrors = [result.error || 'Server error'];

      }
    } catch(e) {
      revokeStepPass('membrane');
      finishStepProgress(progress, false, e.message || 'Network error');
      _invalidateMembraneBuild();
      _compositionErrors = [e.message || 'Network error'];
    } finally {
      _stepRunning = false;
      if (button) button.disabled = false;
      finishUnfinishedStepProgress(progress);
    }
  }
  updateLipidCounts();
  updateCompositionStatus();
}

function membraneCountWarning() {
  if (!_compositionChecked || !_membraneActualCounts) return '';
  var actual = _membraneActualCounts;
  var requested = Number(actual.requested_lipids_per_leaflet ||
    document.getElementById('n-lipids-per-leaflet').value);
  var upper = Number(actual.n_lipids_upper), lower = Number(actual.n_lipids_lower);
  if (![requested, upper, lower].every(Number.isFinite) ||
      (upper === requested && lower === requested)) return '';
  return 'Requested ' + requested + ' lipids per leaflet; constructed upper ' + upper +
    ' and lower ' + lower + '. Check the actual lipid ratios in both leaflets to confirm they meet your requirements.';
}

function updateCompositionStatus() {
  if (typeof updateV4CompositionAvailability === 'function') updateV4CompositionAvailability();
  const el = document.getElementById('composition-status');
  if (!el) return;
  var warning = membraneCountWarning();
  var warningEl = document.getElementById('membrane-count-warning');
  if (warningEl) {
    warningEl.textContent = warning ? '⚠ ' + warning : '';
    warningEl.classList.toggle('hidden', !warning);
  }
  if (_compositionChecked && _compositionErrors.length === 0) {
    var progressEl = document.getElementById('step-progress-check-composition-btn');
    if (progressEl) {
      progressEl.dataset.state = warning ? 'warning' : 'done';
      progressEl.querySelector('.step-progress-phase').textContent = warning
        ? 'Complete with warning — review lipid ratios' : 'Complete';
      el.textContent = '';
    } else {
      el.textContent = warning ? '⚠ Check complete — review lipid ratios' : '✓ Composition valid';
      el.style.color = warning ? '#d97706' : 'var(--success)';
    }
  } else if (_compositionErrors && _compositionErrors.length > 0) {
    var feedback = stepFeedbackElement({stepName:'membrane', buttonId:'check-composition-btn',
      element:document.getElementById('step-progress-check-composition-btn') || el});
    feedback.replaceChildren();
    feedback.classList.remove('hidden');
    feedback.classList.add('error');
    appendFeedbackMessages(feedback, _compositionErrors);
    el.textContent = '';
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

  const check = currentCheckNotice();
  const fulfilled = !(check && !check.handle.settled) && isCurrentStepFulfilled();
  nextBtn.disabled = !fulfilled;
  nextBtn.style.opacity = fulfilled ? '' : '0.45';
  nextBtn.style.cursor = fulfilled ? '' : 'not-allowed';
  nextBtn.title = fulfilled ? '' : 'Complete the current step before proceeding';
  syncCheckNotice();
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
  var isPureMembrane = state.taskType && state.taskType.pipeline === 'pure_membrane';
  var protXY=isFinite(xMin)?Math.max(xMax-xMin,yMax-yMin):(isPureMembrane?0.0:3.0);
  var protArea=protXY*protXY;

  var nLipidsEl=document.getElementById('n-lipids-per-leaflet');
  var nLipids=nLipidsEl?parseInt(nLipidsEl.value):150;
  if(isNaN(nLipids)||nLipids<64)nLipids=150;

  var lipids=_lipidPickerData.lipids||[];

  var actual = _compositionChecked && _membraneActualCounts;
  var notice = document.getElementById('membrane-actual-counts');
  if (notice) notice.textContent = actual
    ? 'Actual lipids: upper ' + actual.n_lipids_upper + ', lower ' + actual.n_lipids_lower
      + '. Starting count: ' + (actual.requested_lipids_per_leaflet || nLipids)
      + '. Each leaflet fills the same box using its own available area and composition.'
    : 'Initial count estimate. Check computes the box and actual count for each leaflet.';
  function computeCounts(mix, label){
    var side = label.toLowerCase();
    var count = actual ? Number(actual['n_lipids_' + side]) : nLipids;
    var species = actual && actual['lipid_counts_' + side];
    var rows = allocatePreviewLipidCounts(count, mix);
    if (actual && !species) rows = [];
    if (species) rows = Object.keys(species).map(function(name) {
      return {name: name, count: species[name], ratio: (100 * species[name] / count).toFixed(1)};
    });
    var targetAPL = actual && actual.target_area_per_lipid_nm2 && Number(actual.target_area_per_lipid_nm2[side]);
    var hasTargetAPL = Number.isFinite(targetAPL) && targetAPL > 0;
    return {count:count, avgAPL:hasTargetAPL ? targetAPL : weightedLeafletAPL(mix,lipids),
      aplLabel:hasTargetAPL ? 'Target APL' : 'Estimated APL', counts:rows};
  }

  var avgAPL2=previewBilayerAPL(lipids);
  var lipidArea=nLipids*avgAPL2;
  var boxXY=actual && _membraneActualBox ? _membraneActualBox[0]
    : Math.max(Math.sqrt(lipidArea), Math.sqrt(protArea) + (protArea > 0 ? 4.0 : 0), 4.0);
  var memArea=boxXY*boxXY;

  function renderTable(el,mix,label){
    var r=computeCounts(mix,label);
    var sumOK=mix.reduce(function(s,m){return s+m.ratio;},0)===100;
    var html='<table class="count-table">';
    html+='<tr><td colspan="3" class="count-summary">';
    html+=(actual?'Actual ':'Estimated ')+label+' lipids: <b>'+r.count+'</b> &nbsp;|&nbsp; Box XY: <b>'+boxXY.toFixed(1)+' nm</b> &nbsp;|&nbsp; Area: <b>'+memArea.toFixed(1)+' nm²</b> &nbsp;|&nbsp; '+r.aplLabel+': <b>'+r.avgAPL.toFixed(3)+' nm²</b>';
    if(!sumOK)html+=' <span class="error-text">⚠ Ratios sum to '+mix.reduce(function(s,m){return s+m.ratio;},0)+'%, not 100%</span>';
    if(nLipids<64)html+=' <span class="error-text">⚠ Min 64 required</span>';
    html+='</td></tr>';
    html+='<tr><th>Lipid</th><th>Ratio</th><th>Count</th></tr>';
    if (actual && !r.counts.length) html+='<tr><td colspan="3">Species counts unavailable for this older checkpoint.</td></tr>';
    r.counts.forEach(function(c){html+='<tr><td>'+escapeHtml(c.name)+'</td><td>'+c.ratio+'%</td><td><b>'+c.count+'</b></td></tr>';});
    html+='</table>';
    el.innerHTML=html;
    el.classList.remove('hidden');
  }

  renderTable(upperCountsEl,_mixUpper,'Upper');
  var totalBilayer = actual ? actual.n_lipids_upper + actual.n_lipids_lower : nLipids * 2;
  if((_asymmetric||actual)&&lowerCountsEl){renderTable(lowerCountsEl,_asymmetric?_mixLower:_mixUpper,'Lower');lowerCountsEl.classList.remove('hidden');}
  else if(lowerCountsEl){lowerCountsEl.classList.add('hidden');}
  var totalEl = document.getElementById('bilayer-count-total');
  if (totalEl) {
    totalEl.textContent = 'Bilayer total: ' + totalBilayer + ' lipids';
    totalEl.classList.remove('hidden');
  }
}

// Load-order manifest: records that this file ran to completion.
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push("app_parts/system_verification.js");

// This file is loaded as a classic script immediately after app.js.
// Keep these globals public: app.js and browser integrations call them directly.

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
  enabled: true, nstlist: 20, comm_mode: "linear", comm_grps: "SOLU_MEMB SOLV",
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
  { bb:400, sc:40, lipid:0, dih:0, dt:1.0, nsteps:250000, ensemble:"nvt",
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

function restoreSimulationParams(saved, execution) {
  if (!saved || typeof saved !== "object") {
    renderSimStages();
    return;
  }
  if (execution && typeof execution === "object") {
    _simHardware = Object.assign({}, _simHardware, execution);
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
  html += '<div class="sim-stage-header open"><button type="button" class="sim-stage-toggle" aria-expanded="true" aria-controls="sim-stage-hardware-body">';
  html += '<span class="sim-stage-icon" style="color:#2563eb;">&#9889;</span>';
  html += '<span class="sim-stage-title" style="color:#2563eb;">Execution Hardware</span>';
  html += '<span class="sim-stage-summary">Defaults written to run_md.sh; MDP physics is unchanged</span></button></div>';
  html += '<div class="sim-stage-body open" id="sim-stage-hardware-body">';
  html += '<div class="param-row">';
  html += paramSelect("sim-hw-mode", "MPI mode", [["thread-mpi","Thread-MPI (single node)"],["external-mpi","External MPI / scheduler"]], _simHardware.mode);
  html += paramNumber("sim-hw-cpu", "Total CPU threads", _simHardware.cpu_threads, 1, 4096, 1, "wide");
  html += paramNumber("sim-hw-mpi", "MPI ranks", _simHardware.mpi_ranks, 1, 4096, 1, "wide");
  html += '<span class="param-item"><label for="sim-hw-omp">OpenMP threads/rank</label><output id="sim-hw-omp">' + ompThreads + '</output></span>';
  html += paramSelect("sim-hw-pin", "Thread pinning", [["auto","Automatic"],["on","On"],["off","Off"]], _simHardware.pin);
  html += '</div><div class="param-row">';
  html += '<span class="param-item"><span class="param-label">GPU execution</span><label class="hint" for="sim-hw-use-gpu"><input type="checkbox" id="sim-hw-use-gpu"' + (_simHardware.use_gpu ? ' checked' : '') + '> Enable GPU IDs</label></span>';
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
  html += '<div class="sim-stage-header" data-stage="em"><button type="button" class="sim-stage-toggle" aria-expanded="false" aria-controls="sim-stage-em-body">';
  html += '<span class="sim-stage-icon" style="color:#f59e0b;">&#9673;</span>';
  html += '<span class="sim-stage-title" style="color:#f59e0b;">Energy Minimization</span>';
  html += '<span class="sim-stage-summary">' + escapeHtml(_DEFAULT_EM.nsteps.toLocaleString()) + ' steps &nbsp;|&nbsp; emtol=' + escapeHtml(_DEFAULT_EM.emtol.toFixed(0)) + '</span>';
  html += '</button></div>';
  html += '<div class="sim-stage-body" data-stage="em" id="sim-stage-em-body">';
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
    html += '<div class="sim-stage-header" data-stage="eq' + i + '"><button type="button" class="sim-stage-toggle" aria-expanded="false" aria-controls="sim-stage-eq-' + i + '-body">';
    html += '<span class="sim-stage-icon">' + (i === 0 ? '&#9678;' : '&#9674;') + '</span>';
    html += '<span class="sim-stage-title">Equilibration ' + (i+1) + '</span>';
    // Disabled stages and legacy saved tasks can retain arbitrary draft values.
    // Keep them as text before innerHTML parses the stage card, not only during
    // the later textContent refresh, which cannot remove injected siblings.
    html += '<span class="sim-stage-summary">' + escapeHtml(st.nsteps.toLocaleString()) + ' steps × ' + escapeHtml(dtDisplay) + ' = ' + escapeHtml(timeNs.toFixed(1)) + ' ns &nbsp;|&nbsp; BB=' + escapeHtml(st.bb) + ' SC=' + escapeHtml(st.sc) + ' Lipid=' + escapeHtml(st.lipid) + '</span></button>';
    html += stageEnabledControl("eq-enabled-" + i, stageEnabled);
    html += '</div>';
    html += '<div class="sim-stage-body" data-stage="eq' + i + '" id="sim-stage-eq-' + i + '-body">';

    // Row 1: Basic
    html += '<div class="param-row">';
    html += paramNumber("eq-dt-" + i, "Timestep (fs)", st.dt, 0.5, st.constraints === "none" ? 1.0 : 2.0, 0.5, "narrow");
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
    html += '<div class="sim-stage-header open" data-stage="prod' + pi + '"><button type="button" class="sim-stage-toggle" aria-expanded="true" aria-controls="sim-stage-prod-' + pi + '-body">';
    html += '<span class="sim-stage-icon" style="color:#6366f1;">&#9679;</span>';
    html += '<span class="sim-stage-title" style="color:#6366f1;">Production' + (_prodIters.length > 1 ? ' #' + (pi+1) : '') + '</span>';
    html += '<span class="sim-stage-summary">' + repeats + ' segment' + (repeats === 1 ? '' : 's') + ' × ' + segmentNs.toFixed(1) + ' ns = ' + timeNs.toFixed(1) + ' ns &nbsp;|&nbsp; ' + frames.toLocaleString() + ' frames</span></button>';
    html += stageEnabledControl("prod-enabled-" + pi, productionEnabled);
    html += '</div>';
    html += '<div class="sim-stage-body open" data-stage="prod' + pi + '" id="sim-stage-prod-' + pi + '-body">';

    html += '<div class="param-row">';
    html += paramNumber("prod-dt-" + pi, "Timestep (fs)", pr.dt, 0.5, pr.constraints === "none" ? 1.0 : 2.0, 0.5, "narrow");
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
  var limitsId = id + '-limits';
  return '<span class="param-item"><label for="' + id + '">' + escapeHtml(label) + '</label>' +
    '<input type="number" id="' + id + '" value="' + escapeHtml(value) + '" min="' + min + '" max="' + max + '" step="' + step + '" class="' + (cls||'') + '" aria-describedby="' + limitsId + '">' +
    '<span class="sr-only" id="' + limitsId + '">Allowed range ' + min + ' to ' + max + '; step ' + step + '.</span></span>';
}

function paramText(id, label, value, cls) {
  return '<span class="param-item"><label for="' + id + '">' + escapeHtml(label) + '</label><input type="text" id="' + id + '" value="' + escapeHtml(value) + '" class="' + (cls||'') + '"></span>';
}

function paramSelect(id, label, options, selected) {
  var opts = options.map(function(o) {
    return '<option value="' + o[0] + '"' + (o[0] === selected ? ' selected' : '') + '>' + o[1] + '</option>';
  }).join("");
  return '<span class="param-item"><label for="' + id + '">' + escapeHtml(label) + '</label><select id="' + id + '">' + opts + '</select></span>';
}

function stageEnabledControl(id, enabled) {
  return '<label class="hint stage-enabled" for="' + id + '">' +
    '<input type="checkbox" id="' + id + '"' + (enabled ? ' checked' : '') + '> Run stage</label>';
}

function commGroupOptions() {
  var options = [["System", "System — all atoms"]];
  var modules = (state.taskType && state.taskType.visible_modules) || [];
  if (modules.indexOf("membrane") >= 0) {
    options.push(["SOLU_MEMB SOLV", "SOLU_MEMB + SOLV"]);
    options.push(["SOLU MEMB SOLV", "SOLU + MEMB + SOLV"]);
  } else {
    options.push(["SOLU SOLV", "SOLU + SOLV"]);
  }
  return options;
}

function paramTextarea(id, label, value, hint) {
  var hintId = id + '-hint';
  return '<div class="param-textarea"><label for="' + id + '">' + label + '</label>' +
    '<textarea id="' + id + '" rows="3" spellcheck="false" aria-describedby="' + hintId + '" placeholder="example: nstlist = 20">' + escapeHtml(value) + '</textarea>' +
    '<span class="hint" id="' + hintId + '">' + hint + '</span></div>';
}

function toggleStageCard(toggle) {
  var header = toggle.closest('.sim-stage-header');
  if (!header) return;
  var body = header.nextElementSibling;
  if (!body) return;
  var open = body.classList.toggle("open");
  header.classList.toggle("open", open);
  toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}

document.addEventListener('click', function(event) {
  var toggle = event.target.closest('.sim-stage-toggle');
  if (toggle) toggleStageCard(toggle);
});

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
      if (el.id.indexOf("eq-constraints-") === 0 ||
          el.id.indexOf("prod-constraints-") === 0) syncAtomisticTimestepBounds();
    });
  });
  syncAtomisticTimestepBounds();
}

function syncAtomisticTimestepBounds() {
  function apply(prefix, index) {
    var constraints = document.getElementById(prefix + "-constraints-" + index);
    var timestep = document.getElementById(prefix + "-dt-" + index);
    if (!constraints || !timestep) return;
    var maximum = constraints.value === "none" ? 1.0 : 2.0;
    timestep.max = String(maximum);
    var value = Number(timestep.value);
    timestep.setCustomValidity(
      Number.isFinite(value) && value <= maximum
        ? ""
        : "The selected constraints support at most " + maximum.toFixed(1) + " fs."
    );
    var limits = document.getElementById(timestep.id + "-limits");
    if (limits) limits.textContent = "Allowed range 0.5 to " + maximum + "; step 0.5.";
  }
  _simStages.forEach(function(_stage, index) { apply("eq", index); });
  _prodIters.forEach(function(_stage, index) { apply("prod", index); });
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
  eq.concat(prod).forEach(function(stage) {
    if (stage.enabled === false) return;
    var maximum = stage.constraints === "none" ? 1.0 : 2.0;
    if (!Number.isFinite(stage.dt) || stage.dt < 0.5 || stage.dt > maximum) {
      throw new Error(
        "Atomistic timestep must be 0.5–" + maximum.toFixed(1) +
        " fs for constraints=" + stage.constraints + ". HMR/virtual sites are not implemented."
      );
    }
  });
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
    prod_iters: prod
  };
}

function collectExecutionHardware() {
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
  return {
    mode: _simHardware.mode,
    cpu_threads: _simHardware.cpu_threads,
    mpi_ranks: _simHardware.mpi_ranks,
    use_gpu: _simHardware.use_gpu,
    gpu_count: _simHardware.use_gpu ? _simHardware.gpu_count : 0,
    gpu_ids: _simHardware.use_gpu ? String(_simHardware.gpu_ids).trim() : "",
    gmx_command: String(_simHardware.gmx_command).trim(),
    mpi_launcher: _simHardware.mpi_launcher,
    pin: _simHardware.pin
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

// Load-order manifest: records that this file ran to completion.
window.__gmxbuilderLoaded = window.__gmxbuilderLoaded || [];
window.__gmxbuilderLoaded.push("app_parts/simulation.js");

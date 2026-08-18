"""Frontend contracts for the final bilayer assembly workflow."""

from pathlib import Path

from gmxbuilder.web.task_types import get_all_task_types, get_task_type


ROOT = Path(__file__).parents[1]


def test_enabled_workflows_finish_ion_review_then_simulation_parameters():
    for summary in get_all_task_types():
        if not summary["enabled"]:
            continue
        task = get_task_type(summary["id"])
        assert task is not None
        visible = task.visible_modules
        assert "verify" not in visible
        assert "topology" not in visible
        if "ions" in visible:
            assert visible[-2:] == ["ions", "simparams"]


def test_ion_check_viewer_is_exact_and_requires_confirmation():
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    ions = (ROOT / "src/gmxbuilder/web/static/ions.js").read_text()
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()

    assert 'id="ion-confirm-system-btn" disabled' in template
    assert 'id="panel-verify"' not in template
    assert "exact coordinates saved by Ion Check" in template
    assert "_loadStepViewerPdb('ions')" in ions
    assert "$3Dmol.createViewer" in ions
    assert "CRYST1" in ions
    assert "host.style.position = 'relative'" in ions
    assert "applyIonSystemStyles(viewer)" in ions
    assert "{resn: WATER_RESIDUES, elem: 'O'}" in ions
    assert "opacity: 0.55" in ions
    assert "countWaterOxygens(pdb)" in ions
    assert 'id="ion-viewer-label"' in template
    assert 'style="position:relative;width:100%;height:520px;overflow:hidden' in template
    assert "Random Water Replacement (GROMACS-style)" in template
    assert "Experimental: formal-charge-ranked water replacement" in template
    assert "not equilibrium ion sampling" in template
    assert "Experimental: dimensionless Metropolis site optimization" in template
    assert "NGL.Stage" not in ions
    assert "window._isSystemConfirmed()" in app
    assert "Confirm Simulation System before proceeding" in app


def test_protonation_recalculation_is_visible_and_bound_to_current_ph():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()

    assert "phInput.addEventListener('input'" in app
    assert "runProtonation();" in app
    assert "Number(data.pH) !== pH" in app
    assert "Recalculated at pH" in app
    assert "no predicted pKa threshold was crossed" in app


def test_modification_payload_excludes_forcefield_derived_display_metadata():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()

    serializer = app.split("function serializeStructureModifications()", 1)[1].split(
        "// -------------------------------------------------------------------", 1
    )[0]
    structure_config = app.split("config.structure = {", 1)[1].split("// Orientation", 1)[0]

    assert "return { index: mod.index, patch_id: mod.patch_id };" in serializer
    assert "charge_shift" not in serializer
    assert "product_name" not in serializer
    assert "modifications: serializeStructureModifications()" in structure_config
    assert "modifications: _procModifications" not in structure_config
    assert "hydrateModificationMetadata();" in app
    restore = app.split("async function resumeTask", 1)[1].split("async function loadOptions", 1)[0]
    assert "taskState.step_forcefield_config || taskState.forcefield" in restore
    assert restore.index("resumedProteinForceField.value") < restore.index("showUploadInfo(info)")


def test_uploaded_modifications_are_auto_selected_and_require_review():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    server = (ROOT / "src/gmxbuilder/web/server.py").read_text()

    assert 'id="proc-upload-modification-notice"' in template
    assert "function setInputModificationReport(report)" in app
    assert "function applyDetectedInputModifications()" in app
    assert "source: 'input-detection'" in app
    assert "patch.supported === false" in app
    assert "Open the Modifications tab and verify" in app
    assert "cannot be restored with" in app
    assert "reloadModificationCatalog();" in app
    assert "input_modifications" in app
    assert "input_sequences" in app
    assert "state.pdbInfo.sequences = standardizedSequences" in app
    assert 'id="modification-geometry-report"' in template
    assert "function renderModificationGeometryReport(reports)" in app
    assert "equilibrium bond lengths and angles" in app
    assert "modification_geometry" in app
    assert "taskState.modification_geometry || []" in app
    assert 'state_update["modification_geometry"]' in server


def test_disulfide_crosslinks_use_a_dedicated_paired_residue_contract():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    server = (ROOT / "src/gmxbuilder/web/server.py").read_text()

    assert 'id="proc-disulfide-first"' in template
    assert 'id="proc-disulfide-second"' in template
    assert 'id="proc-disulfide-add"' in template
    assert "whose SG atoms already form a bridge" in template
    assert "function serializeStructureCrosslinks()" in app
    assert "crosslinks: serializeStructureCrosslinks()" in app
    assert "reloadCrosslinkCapabilities();" in app
    assert "SG–SG distance is validated" in app
    assert "hasSupportedPatch && !crosslinkedIdx.has(i)" in app
    assert '@app.get("/api/crosslink-capabilities")' in server


def test_simulation_parameter_editor_covers_common_and_expert_controls():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()

    for control in (
        "em-constraints",
        "eq-enabled-",
        "eq-tau-t-",
        "eq-temperature-",
        "eq-constraints-",
        "eq-pcoupl-type-",
        "eq-comm-mode-",
        "eq-comm-grps-",
        "eq-nstxout-compressed-",
        "eq-nstvout-",
        "eq-nstfout-",
        "eq-nstenergy-",
        "eq-nstlog-",
        "prod-enabled-",
        "prod-temperature-",
        "prod-constraints-",
        "prod-repeat-",
        "prod-pcoupl-type-",
        "prod-tau-p-",
        "prod-comm-mode-",
        "prod-comm-grps-",
        "prod-nstxout-compressed-",
        "prod-nstvout-",
        "prod-nstfout-",
        "prod-nstenergy-",
        "em-mdp-overrides",
        "parseMdpOverrides",
        "sim-hw-gpu-count",
    ):
        assert control in app
    assert "System — all atoms" in app
    assert '"SOLU_MEMB SOLV"' in app
    assert 'state.taskType.pipeline !== "liquid"' not in app
    assert 'id.indexOf("eq-ensemble-") === 0' in app
    assert app.count('copy.dt_unit = "fs"') == 2
    assert "Enable at least one equilibration stage" in app
    assert "var _SOLUTION_SCHEDULE" in app
    assert "bb:400, sc:40, lipid:0, dih:0" in app
    assert "repeat: solution ? 10 : 5" in app
    assert 'renderNonbondFields("em", _DEFAULT_EM, false)' in app
    assert 'renderNonbondFields("eq-" + i, st, true)' in app
    assert 'renderNonbondFields("prod-" + pi, pr, true)' in app
    assert "syncMdpNonbondDefaults()" in app
    assert 'dispcorr: isCharmm ? "no" : "EnerPres"' in app
    assert "Global MDP Settings" not in app
    assert "_simGlobals" not in app
    assert "schema_version: 2" in app
    assert "gpu_count:" in app
    assert 'paramSelect("sim-constraints"' not in app
    assert 'paramNumber("sim-temperature"' not in app
    assert "collectSimulationParams()" in app


def test_resume_restores_force_field_before_mdp_defaults():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    resume_start = app.index("async function resumeTask")
    resume_end = app.index("async function loadOptions", resume_start)
    resume_source = app[resume_start:resume_end]

    restore_force_field = resume_source.index(
        "resumedProteinForceField.value = savedForceFieldConfig.name"
    )
    initialize_mdp = resume_source.index("initSimParams()")
    assert restore_force_field < initialize_mdp


def test_membrane_preview_uses_the_checked_explicit_count_contract():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()

    # The browser preview must follow the explicit-count backend path:
    # A_box = N_leaflet * max(APL_upper, APL_lower) + A_protein.
    assert "function weightedLeafletAPL" in app
    assert "function previewBilayerAPL" in app
    assert "return Math.max(upperAPL, weightedLeafletAPL(_mixLower, lipids));" in app
    assert "var lipidArea = nLipids * avgAPL;" in app
    assert "var lipidArea=nLipids*avgAPL2;" in app
    assert "nLipids * avgAPL / 1.30" not in app
    assert "nLipids*avgAPL2/1.30" not in app

    # Pre-Check composition counts mirror MembraneBuilder._assign_lipids:
    # round each positive fraction, trim in input order, then pad the dominant
    # component so that the displayed total is exactly N_leaflet.
    assert "function allocatePreviewLipidCounts" in app
    assert "Math.max(1, Math.round(nLipids * m.ratio / 100))" in app
    assert "dominant.count += remaining;" in app
    assert "Math.floor(nLipids*m.ratio/100)" not in app

    assert "Minimum supported construction size: 64 per leaflet." in template
    assert "stability must be assessed by equilibration" in template
    assert "Li et al., JCIM 2025" not in template


def test_task_scoped_custom_lipid_and_history_route_contracts():
    app = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()

    assert "/api/task/' + state.taskId + '/custom-lipids" in app
    assert "function switchTaskToCustomLipidBackend()" in app
    assert "lipidFF.value = 'gaff2'" in app
    assert "gaffOption && gaffOption.enabled" in app
    assert "Run Force Field Check again" in app
    assert "state.customLipidBusy" in app
    assert "Custom Lipids" in app
    assert "record.state === 'ready'" in app
    assert "record.state === 'failed'" in app
    assert "history.pushState" in app
    assert "history.replaceState" in app
    assert "restoreRouteFromLocation" in app
    assert "const taskPart" not in app
    assert "Task identifiers are deliberately kept out" in app
    assert "if (window.location.pathname !== '/') history.replaceState({}, '', '/')" in app
    assert "async function copyTaskIdToClipboard()" in app
    assert "navigator.clipboard.writeText(taskId)" in app
    assert 'id="copy-task-id"' in template
    assert "BilayerBuilder" in app
    assert "PureBilayerSystem" in app
    assert "Solvator" in app
    assert 'id="custom-lipid-build-status"' in template
    assert "remain private to the current task" in template

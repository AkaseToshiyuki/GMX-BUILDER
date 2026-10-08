"""Frontend contracts for the final bilayer assembly workflow."""

from pathlib import Path

from gmxbuilder.web.task_types import get_all_task_types, get_task_type
from tests.frontend_bundle import frontend_source


def test_task_categories_follow_the_public_landing_page_order() -> None:
    categories = list(dict.fromkeys(item["category"] for item in get_all_task_types()))
    assert categories == ["Membrane", "Solution", "Coarse Grained"]


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
            assert visible[-3:] == ["ions", "final_review", "simparams"]


def test_protonation_recalculation_is_visible_and_bound_to_current_ph():
    app = frontend_source()

    assert "phInput.addEventListener('input'" in app
    assert "runProtonation();" in app
    assert "Number(data.pH) !== pH" in app
    assert "Recalculated at pH" in app
    assert "no predicted pKa threshold was crossed" in app


def test_uploaded_modifications_are_auto_selected_and_require_review():
    app = frontend_source()
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
    app = frontend_source()
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
    app = frontend_source()

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
    app = frontend_source()
    resume_start = app.index("async function resumeTask")
    resume_end = app.index("function loadOptions", resume_start)
    resume_source = app[resume_start:resume_end]

    restore_force_field = resume_source.index(
        "resumedProteinForceField.value = savedForceFieldConfig.name"
    )
    initialize_mdp = resume_source.index("initSimParams()")
    show_upload = resume_source.index("showUploadInfo(info)")
    assert restore_force_field < min(initialize_mdp, show_upload)


def test_membrane_preview_uses_the_checked_explicit_count_contract():
    app = frontend_source()
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

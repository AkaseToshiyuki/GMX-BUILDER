"""Exercise the real Check handler and navigation after a failed recheck."""

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_failed_recheck_revokes_navigation_and_renders_reason_safely(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 10).until(
        lambda d: d.execute_script("return typeof _doCheckStep === 'function';")
    )
    result = page.execute_async_script("""
        const done = arguments[0];
        state.taskId = 'input-readiness-browser';
        state.taskType = {pipeline:'solvator', visible_modules:['input','forcefield']};
        state.wizardSteps = ['input','forcefield'];
        state.currentStepIdx = 0;
        state.pdbInfo = {small_molecules:[]};
        initSimParams();
        document.querySelectorAll('.panel.active').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-input').classList.add('active');
        _checkedSteps.add('input'); _checkedSteps.add('forcefield');
        state.completedSteps = new Set([0,1]);
        updateNextButtonState();
        const next = document.querySelector('#panel-input .next-btn');
        const before = next.disabled;
        const reason = 'Chain R: missing SER 129 <img src=x onerror=alert(1)>';
        window.fetch = async (url) => ({ok:true, json:async () =>
          String(url).includes('/api/step/') ? {
            status:'error', error:reason,
            input_validation:{can_proceed:false,errors:[{message:reason}]}
          } : {status:'ok', fraction:0.1}});
        _doCheckStep('input','input-check-status','input-check-btn').then(() => {
          const report = document.getElementById('input-readiness-report');
          done({before, after:next.disabled, passes:Array.from(_checkedSteps),
            completed:Array.from(state.completedSteps), text:report.textContent,
            status:document.getElementById('input-check-status').textContent,
            images:report.querySelectorAll('img').length,
            hidden:report.classList.contains('hidden'), advance:canGoToStep(1)});
        }).catch(e => done({error:String(e)}));
    """)
    assert result["before"] is False
    assert result["after"] is True
    assert result["passes"] == result["completed"] == []
    assert result["advance"] is False
    assert "missing SER 129" in result["text"], result["status"]
    assert result["images"] == 0 and result["hidden"] is False


def test_selection_change_removes_obsolete_assessment_and_requires_check(page):
    page.execute_script("""
        state.wizardSteps = ['input','forcefield']; state.currentStepIdx = 0;
        _checkedSteps.add('input'); _checkedSteps.add('forcefield');
        state.completedSteps = new Set([0,1]);
        renderInputReadiness({errors:[{message:'Discarded chain R has a break.'}]});
        invalidateInputCheckpoint();
    """)
    assert page.execute_script("return Array.from(_checkedSteps);") == []
    assert page.execute_script("return canGoToStep(1);") is False
    assert page.execute_script("""
        return document.getElementById('input-readiness-report').classList.contains('hidden');
    """)


def test_legacy_resume_cannot_restore_stale_flags_or_jump_to_results(page):
    result = page.execute_async_script("""
        const done = arguments[0];
        const type = {id:'solvator',pipeline:'solvator',
          visible_modules:['input','forcefield','simparams']};
        state.taskType = type; state.wizardSteps = type.visible_modules;
        state.completedSteps = new Set([0,1,2]);
        _checkedSteps.add('input'); _checkedSteps.add('forcefield');
        const originalFetch = window.fetch;
        window.alert = message => {window.resumeAlert = message;};
        window.fetch = async (url, options) => {
          if (String(url).endsWith('/resume')) return {ok:true,json:async () => ({
            task_type:type, task_type_id:'solvator', pdb_info:{num_atoms:0},
            steps_completed:['input','forcefield'], input_check_required:true,
            resume_step:'input', current_step:'simparams',
            build_status:{status:'completed',download_available:true,result:{}}
          })};
          if (String(url).includes('/api/steps/')) return {ok:true,json:async () => ({
            steps:[{name:'input',has_checkpoint:false},{name:'forcefield',has_checkpoint:false}]
          })};
          return originalFetch(url,options);
        };
        resumeTask('legacy-input-gate',2).then(() => done({
          step:state.wizardSteps[state.currentStepIdx], checked:Array.from(_checkedSteps),
          completed:Array.from(state.completedSteps), alert:window.resumeAlert,
          next:document.querySelector('#panel-input .next-btn').disabled
        })).catch(e => done({error:String(e)}));
    """)
    assert result.get("alert") is None, result
    assert result["step"] == "input", result
    assert result["checked"] == result["completed"] == []
    assert result["next"] is True


def test_retained_metal_is_not_silently_excluded_by_check(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 15).until(
        lambda driver: driver.execute_script("return initComputeQueueStatus._done === true")
    )
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    result = page.execute_async_script("""
        const done = arguments[0];
        state.taskId = 'a'.repeat(32);
        state.pdbInfo = {small_molecules:[{resname:'ZN',chain:'M',resid:1}]};
        _chainState = {A:{included:true}};
        _smallMolState = {ZN:{included:true,name:'Zinc'}};
        let selection = null;
        window.fetch = async (url, options) => {
          if (String(url).includes('/api/filter-pdb/')) {
            selection = JSON.parse(options.body);
            return {ok:false,json:async()=>({error:'End isolated selection test'})};
          }
          return {ok:true,json:async()=>({})};
        };
        _doCheckStep('input','input-check-status','input-check-btn')
          .then(()=>done(selection)).catch(error=>done({error:String(error)}));
    """)
    assert "ZN" not in result["exclude_resnames"], result
    assert "M" in result["include_chains"], result


def test_reconstruction_options_and_real_chain_rename_enter_input_config(page):
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(page, 10).until(
        lambda d: d.execute_script("return typeof buildModuleConfig === 'function'")
    )
    result = page.execute_script("""
      state.taskId = 'reconstruction-browser';
      state.taskType = {pipeline:'solvator', visible_modules:['input','forcefield']};
      state.wizardSteps = ['input','forcefield']; state.currentStepIdx = 0;
      showUploadInfo({filename:'example.pdb',num_atoms:8,chains:['A'],box_nm:[3,3,3],
        sequences:[{chain_id:'A',length:2,residues:[{resname:'GLY',resid:54},{resname:'GLY',resid:128}]}],
        pdb_content:'',small_molecules:[],chain_mapping:{'R~1':'A'}});
      document.querySelector('.chain-rename').dispatchEvent(
        new MouseEvent('dblclick', {bubbles:true}));
      var nameInput = document.querySelector('.chain-rename-input');
      nameInput.value = 'Z';
      nameInput.dispatchEvent(new Event('blur'));
      _checkedSteps.add('input'); state.completedSteps = new Set([0]);
      document.getElementById('input-allow-incomplete').checked = true;
      document.getElementById('input-allow-incomplete').dispatchEvent(new Event('change'));
      document.getElementById('input-renumber-residues').checked = true;
      return {config:buildModuleConfig('input').input,passed:_checkedSteps.has('input'),
        title:document.getElementById('chain-sequences').textContent};
    """)
    assert result["config"]["allow_incomplete_protein"] is True
    assert result["config"]["renumber_residues"] is True
    assert result["config"]["chain_names"] == {"A": "Z"}
    assert result["passed"] is False
    assert "Source chain: R~1" in result["title"]


@pytest.mark.parametrize("workflow", ["solvator", "martini3-solvent", "martini3-bilayer"])
def test_check_filters_actual_chain_selection_in_every_protein_workflow(page, workflow):
    result = page.execute_async_script(
        """
      const workflow = arguments[0], done = arguments[1];
      state.taskId = 'a'.repeat(32);
      state.taskType = {id:workflow,pipeline:workflow,visible_modules:['input','forcefield']};
      state.wizardSteps = ['input','forcefield']; state.currentStepIdx = 0;
      document.getElementById('cg-include-protein').checked = true;
      state.pdbInfo = {filename:'two.pdb',num_atoms:8,chains:['A','B'],box_nm:[3,3,3],
        sequences:['A','B'].map(c=>({chain_id:c,length:1,
          residues:[{resname:'GLY',resid:1}]})),pdb_content:'',small_molecules:[]};
      showUploadInfo(state.pdbInfo);
      const selectB = document.querySelector('.chain-check input[data-chain="B"]');
      selectB.checked = false; selectB.dispatchEvent(new Event('change'));
      let selection = null;
      window.fetch = async (url, options) => {
        if (String(url).includes('/api/filter-pdb/')) {
          selection = JSON.parse(options.body);
          return {ok:false,json:async()=>({error:'End isolated selection test'})};
        }
        return {ok:true,json:async()=>({})};
      };
      _doCheckStep('input','input-check-status','input-check-btn')
        .then(()=>done(selection)).catch(error=>done({error:String(error)}));
    """,
        workflow,
    )
    assert result is not None
    assert result["include_chains"] == ["A"]


def test_successful_check_updates_summary_cards_and_edit_restores_source(page):
    result = page.execute_async_script("""
      const done = arguments[0];
      state.taskId = 'a'.repeat(32);
      state.taskType = {id:'solvator',pipeline:'solvator',visible_modules:['input','forcefield']};
      state.wizardSteps = ['input','forcefield']; state.currentStepIdx = 0;
      state.pdbInfo = {filename:'fragment.pdb',num_atoms:8,chains:['A'],box_nm:[3,3,3],
        sequences:[{chain_id:'A',length:2,residues:[{resname:'GLY',resid:54},
          {resname:'GLY',resid:128}]}],pdb_content:'',small_molecules:[]};
      showUploadInfo(state.pdbInfo);
      const summary = {num_atoms:10,chains:['A','C'],box_nm:[4,5,6],small_molecules:[],
        sequences:['A','C'].map(chain_id=>({chain_id,length:1,
          residues:[{resname:'GLY',resid:1,is_protein:true}]}))};
      window.fetch = async url => ({ok:true,text:async()=>'',json:async()=>
        String(url).includes('/api/step/') ? {status:'ok',metrics:{input_summary:summary,
          input_sequences:summary.sequences}} : {status:'ok'}});
      _doCheckStep('input','input-check-status','input-check-btn').then(()=>{
        const checked = {
          atoms:document.getElementById('info-atoms').textContent,
          chains:document.getElementById('info-chains').textContent,
          box:document.getElementById('info-box').textContent,
          cards:document.querySelectorAll('#checked-chain-sequences .chain-card').length,
          numbers:Array.from(document.querySelectorAll('#checked-chain-sequences .seq-group-num'))
            .map(el=>el.textContent),
          sourceHidden:document.getElementById('input-selection-details').classList.contains('hidden')
        };
        document.getElementById('edit-input-selection').click();
        done({checked,restoredAtoms:document.getElementById('info-atoms').textContent,
          restoredNumber:document.querySelector('#chain-sequences .seq-group-num').textContent,
          restoredChains:document.getElementById('info-chains').textContent,
          passed:_checkedSteps.has('input'),config:buildModuleConfig('input').input});
      }).catch(error=>done({error:String(error)}));
    """)
    assert result["checked"] == {
        "atoms": "10",
        "chains": "A, C",
        "box": "4.0 nm × 5.0 nm × 6.0 nm",
        "cards": 2,
        "numbers": ["1", "1"],
        "sourceHidden": True,
    }
    assert result["restoredAtoms"] == "8"
    assert result["restoredChains"] == "A"
    assert result["restoredNumber"] == "54"
    assert result["passed"] is False
    assert result["config"]["chain_names"] == {}


@pytest.mark.parametrize("width", [1400, 390])
def test_experimental_warning_is_yellow_and_checkbox_is_left_centered(page, width):
    page.set_window_size(width, 1000)
    result = page.execute_script("""
      state.taskId = null;
      document.querySelectorAll('.panel.active').forEach(p=>p.classList.remove('active'));
      document.getElementById('panel-forcefield').classList.add('active');
      const select = document.getElementById('ff-ligand');
      select.add(new Option('CHARMM local','charmm_compat')); select.value='charmm_compat';
      window._ffCompatibility={ligand_names:[]};
      renderLigandChargeInputs();
      const control=document.getElementById('ff-charmm-research'), box=control.parentElement;
      const label=box.querySelector('span'), a=control.getBoundingClientRect();
      const b=label.getBoundingClientRect(), frame=box.getBoundingClientRect();
      return {background:getComputedStyle(box).backgroundColor,left:a.right<=b.left,
        centered:Math.abs((a.top+a.bottom-b.top-b.bottom)/2)<2,
        fits:frame.right<=document.documentElement.clientWidth,
        text:label.textContent};
    """)
    assert result["background"] == "rgb(255, 240, 179)"
    assert result["left"] and result["centered"] and result["fits"]
    assert "Physical accuracy has not been validated" in result["text"]


def test_checked_display_preserves_restored_source_selection(page):
    result = page.execute_script("""
      state.taskType={id:'solvator',pipeline:'solvator',visible_modules:['input','forcefield']};
      state.wizardSteps=['input','forcefield'];state.currentStepIdx=0;
      state.pdbInfo={filename:'two.pdb',num_atoms:9,chains:['A','B'],box_nm:[3,3,3],
        sequences:['A','B'].map(chain_id=>({chain_id,length:1,
          residues:[{resname:'GLY',resid:54}]})),pdb_content:'',
        small_molecules:[{resname:'ZN',chain:'M',resid:1,atom_count:1}]};
      showUploadInfo(state.pdbInfo);
      restoreInputSelection({include_chains:['A'],exclude_resnames:['ZN']});
      showCheckedInputSummary({num_atoms:4,chains:['Z'],box_nm:[3,3,3],small_molecules:[],
        sequences:[{chain_id:'Z',length:1,residues:[{resname:'GLY',resid:1}]}]});
      document.getElementById('edit-input-selection').click();
      return {b:document.querySelector('#chain-sequences input[data-chain="B"]').checked,
        zinc:document.querySelector('#small-molecules input[data-smres="ZN"]').checked,
        atoms:document.getElementById('info-atoms').textContent};
    """)
    assert result == {"b": False, "zinc": False, "atoms": "9"}


def test_checked_fragments_share_source_group_and_mark_internal_ends(page):
    result = page.execute_script("""
      state.pdbInfo={};
      const sequences=['A','C'].map((chain_id,index)=>({chain_id,source_chain:'A',
        author_chain:'R',fragment_count:2,fragment_index:index+1,length:1,
        internal_n_terminus:index===1,internal_c_terminus:index===0,
        residues:[{resname:index?'ASN':'GLU',resid:1,author_chain:'R',
          author_resid:index?128:54,is_protein:true}]}));
      showCheckedInputSummary({num_atoms:8,chains:['A','C'],box_nm:[3,3,3],sequences,small_molecules:[]});
      state.pdbInfo.sequences=sequences;
      window.fetch=async()=>({ok:true,json:async()=>({})});
      loadProcResidues();
      return {groups:document.querySelectorAll('.source-protein-group').length,
        cards:document.querySelectorAll('.source-protein-group .chain-card').length,
        text:document.getElementById('checked-chain-sequences').textContent,
        title:document.querySelector('#checked-chain-sequences .seq-tag').title,
        summary:document.getElementById('info-chains').textContent,
        termini:document.getElementById('proc-termini-chains').textContent,
        defaults:Array.from(document.querySelectorAll('#proc-termini-chains select'))
          .map(x=>x.value)};
    """)
    assert result["groups"] == 1 and result["cards"] == 2
    assert "Source protein A — 2 coordinate fragments" in result["text"]
    assert "GLU 54 → ASN 128" in result["text"]
    assert "Fragment A" in result["text"] and "Fragment C" in result["text"]
    assert "Deposited: R:54" in result["title"]
    assert "1 source polymer(s), 2 coordinate fragments" in result["summary"]
    assert "C terminus created at an internal coordinate gap" in result["termini"]
    assert "N terminus created at an internal coordinate gap" in result["termini"]
    assert result["defaults"] == ["", "", "", ""]


def test_checked_cards_remain_editable_and_recheck_carries_component_choices(page):
    result = page.execute_async_script("""
      const done = arguments[0];
      state.taskId = 'a'.repeat(32);
      state.taskType = {id:'solvator',pipeline:'solvator',visible_modules:['input','forcefield']};
      state.wizardSteps=['input','forcefield'];state.currentStepIdx=0;
      state.pdbInfo={filename:'fragment.pdb',num_atoms:9,chains:['A'],box_nm:[3,3,3],
        sequences:[{chain_id:'A',length:2,residues:[{resname:'GLY',resid:54},{resname:'GLY',resid:128}]}],
        pdb_content:'',small_molecules:[{resname:'AMP',chain:'B',resid:1,atom_count:1}]};
      showUploadInfo(state.pdbInfo);
      const choices=['A','C'].map((chain_id,i)=>({chain_id,fragment_key:`A:${i+1}`,
        source_chain:'A',fragment_count:2,fragment_index:i+1,length:1,
        residues:[{resname:'GLY',resid:1,author_resid:i?128:54}]}));
      const summary={num_atoms:9,chains:['A','C'],box_nm:[3,3,3],sequences:choices,
        fragment_choices:choices,small_molecules:state.pdbInfo.small_molecules};
      showCheckedInputSummary(summary);
      _checkedSteps.add('input'); _checkedSteps.add('forcefield');
      state.completedSteps=new Set([0,1]);
      document.querySelector('#checked-chain-sequences button[data-chain="C"]')
        .dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
      let field=document.querySelector('#checked-chain-sequences .chain-rename-input');
      field.value='Z';field.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
      document.querySelector('#checked-small-molecules .smallmol-name')
        .dispatchEvent(new MouseEvent('dblclick',{bubbles:true}));
      field=document.querySelector('#checked-small-molecules .smallmol-rename-input');
      field.value='Adenosine';
      field.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
      for (const selector of ['#checked-chain-sequences input[data-chain="A"]',
                               '#checked-small-molecules input[data-smres="AMP"]']) {
        const cb=document.querySelector(selector);cb.checked=false;
        cb.dispatchEvent(new Event('change'));
      }
      const invalidated={checked:[..._checkedSteps],completed:[...state.completedSteps],
        advance:canGoToStep(1),visible:!document.getElementById('checked-input-details').classList.contains('hidden')};
      let filter,config;
      window.fetch=async(url,options)=> {
        if(String(url).includes('/api/filter-pdb/')) filter=JSON.parse(options.body);
        if(String(url).endsWith('/input') && options?.body) {
          config=JSON.parse(options.body).config;
          return {ok:true,json:async()=>({status:'ok',metrics:{input_summary:{...summary,
            chains:['Z'],sequences:[{...choices[1],chain_id:'Z'}],small_molecules:[]}}})};
        }
        return {ok:true,text:async()=>'',json:async()=>({status:'ok'})};
      };
      _doCheckStep('input','input-check-status','input-check-btn').then(()=>done({invalidated,filter,config,
        label:document.querySelector('#checked-chain-sequences button[data-chain="C"]').textContent,
        fragmentChecked:document.querySelector(
          '#checked-chain-sequences input[data-chain="A"]').checked,
        moleculeChecked:document.querySelector('#checked-small-molecules input').checked,
        moleculeLabel:document.querySelector('#checked-small-molecules .smallmol-name').textContent,
        passed:_checkedSteps.has('input')})).catch(e=>done({error:String(e)}));
    """)
    assert result["invalidated"] == {
        "checked": [],
        "completed": [],
        "advance": False,
        "visible": True,
    }
    assert result["config"]["fragment_names"] == {"A:1": "A", "A:2": "Z"}
    assert result["config"]["exclude_fragments"] == ["A:1"]
    assert result["filter"]["small_molecule_labels"] == {"AMP": "Adenosine"}
    assert "AMP" in result["filter"]["exclude_resnames"]
    assert result["label"] == "Z" and result["moleculeLabel"] == "Adenosine"
    assert not result["fragmentChecked"] and not result["moleculeChecked"]
    assert result["passed"]


def test_input_edit_during_check_cannot_restore_stale_pass(page):
    result = page.execute_async_script("""
      const done=arguments[0];
      state.taskId='a'.repeat(32);
      state.taskType={id:'solvator',pipeline:'solvator',visible_modules:['input','forcefield']};
      state.wizardSteps=['input','forcefield'];state.currentStepIdx=0;
      state.pdbInfo={num_atoms:4,chains:['A'],box_nm:[3,3,3],pdb_content:'',
        sequences:[{chain_id:'A',length:1,residues:[{resname:'GLY',resid:1}]}],
        small_molecules:[]};
      showUploadInfo(state.pdbInfo);
      let release;
      window.fetch=async (url, options)=>{
        if(String(url).endsWith('/input') && options?.body) {
          await new Promise(resolve=>{release=resolve});
        }
        return {ok:true,json:async()=>({status:'ok',metrics:{}})};
      };
      const pending=_doCheckStep('input','input-check-status','input-check-btn');
      const timer=setInterval(()=>{
        if(!release)return;
        clearInterval(timer);
        const cb=document.querySelector('#chain-sequences input');
        cb.checked=false;cb.dispatchEvent(new Event('change'));release();
      },5);
      pending.then(()=>done({passed:_checkedSteps.has('input'),advance:canGoToStep(1),
        text:document.getElementById('input-readiness-report').textContent}));
    """)
    assert not result["passed"] and not result["advance"]
    assert "Input selection changed while checking" in result["text"]


def test_source_tooltips_keep_insertion_codes_in_upload_view(page):
    result = page.execute_script("""
        renderChainSequences([{chain_id:'A',length:2,residues:[
          {resname:'ALA',resid:1,is_protein:true,author_chain:'R',author_resid:27,insertion_code:'A'},
          {resname:'GLY',resid:2,is_protein:true,author_chain:'R',author_resid:27,insertion_code:'B'}
        ]}], ['A'], {R:'A'});
        return [...document.querySelectorAll('#chain-sequences .seq-tag')]
          .filter(x=>x.textContent).map(x=>x.title);
    """)
    assert result == ["ALA 1 · Deposited: R:27A", "GLY 2 · Deposited: R:27B"]

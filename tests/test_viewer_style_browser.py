"""The same atoms retain representations and colors across actual viewer paths."""

import gzip
import json

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.web.server_parts.viewer_data import build_viewer
from tests.test_viewer_style import lipid_pair

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_input_and_checkpoint_views_share_styles_and_component_order_does_not_recolor(
    page, tmp_path
):
    page.execute_async_script("selectTaskType('solvator').then(arguments[0])")
    page.execute_async_script("GMXAssets.viewer().then(arguments[0])")
    lipids, _ = lipid_pair()
    protein = Structure(
        np.array([[0, 0, 0], [0.3, 0, 0], [0, 1, 0], [0.3, 1, 0]]),
        np.eye(3) * 8,
        atom_names=["CA"] * 4,
        resnames=["ALA"] * 4,
        chain_ids=["A", "A", "C", "C"],
        resids=[1, 2, 1, 2],
        elements=["C"] * 4,
    )
    system = System(protein.append(lipids.structure))
    system.add_component(Component("Protein", ComponentKind.PROTEIN, np.arange(4)))
    system.add_component(
        Component("Membrane", ComponentKind.MEMBRANE, np.arange(4, system.num_atoms))
    )
    water_start = system.num_atoms
    water = Structure(
        np.array([[0, 0, 2], [0.1, 0, 2], [0, 0.1, 2]]),
        np.eye(3) * 8,
        atom_names=["OW", "HW1", "HW2"],
        elements=["O", "H", "H"],
        resnames=["SOL"] * 3,
        chain_ids=["W"] * 3,
        resids=[1] * 3,
    )
    system.structure = system.structure.append(water)
    system.add_component(
        Component("Water", ComponentKind.SOLVENT, np.arange(water_start, system.num_atoms))
    )
    payloads = {}
    for step in ["membrane", "solvation", "ions"]:
        if step == "solvation":
            system.components.reverse()
        directory = tmp_path / step
        system.save_checkpoint(directory)
        payloads[step] = json.loads(gzip.decompress(build_viewer(directory).read_bytes()))
    result = page.execute_async_script(
        """
      const payloads=arguments[0],done=arguments[1];
      state.taskId='unified-viewer';state.pdbInfo={sequences:[{chain_id:'A'},{chain_id:'C'}]};
      _smallMolState={};
      document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-input'));
      // Exercise real 3Dmol models/selection/styles without requiring multiple GPU contexts.
      const library=$3Dmol;
      window.$3Dmol={...library,createViewer:host=>{
        host.append(document.createElement('canvas'));
        let models=[];
        return {addModel(){const m=new library.GLModel(models.length);models.push(m);return m;},
          selectedAtoms(sel){return models.flatMap(m=>m.selectedAtoms(sel));},
          setStyle(sel,style){models.forEach(m=>m.setStyle(sel,style));},
          clear(){models=[];},addLine(){},resize(){},render(){},zoomTo(){},setSlab(){},rotate(){}};
      }};
      const input=$3Dmol.createViewer(document.getElementById('pdb-viewer'));
      input.addModel().addAtoms(['A','C'].map((chain,i)=>({chain,atom:'CA',resn:'ALA',elem:'C',
        x:0,y:i,z:0,resi:1,bonds:[],bondOrder:[],hetflag:false})));
      _applyUnifiedStyle(input,'');
      const initial=Object.fromEntries(input.selectedAtoms({}).map(a=>[a.chain,a.style.cartoon]));
      _applyUnifiedStyle(input,'',new Set(['C']));
      const filtered=input.selectedAtoms({chain:'C'})[0].style.cartoon;
      const realFetch=window.fetch;
      window.fetch=(url,...args)=>{
        const step=String(url).split('/').at(-2);
        if(String(url).endsWith('/viewer.json')&&payloads[step])
          return Promise.resolve(new Response(JSON.stringify(payloads[step])));
        return realFetch(url,...args);
      };
      (async()=>{
        const results=[];
        for(const [step,host,panel] of [
          ['membrane','membrane-3d-viewer','panel-membrane'],
          ['solvation','solvation-3d-viewer','panel-solvation'],
          ['ions','ion-viewer','panel-final_review']]) {
          document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===panel));
          const payload=await GMXViewer.render(host,step);
          if(!payload) throw new Error(document.getElementById(host+'-loading').textContent);
          const atoms=GMXViewer.scenes.get(host).viewer.selectedAtoms({});
          const prot=Object.fromEntries(atoms.filter(a=>a.resn==='ALA')
            .map(a=>[a.chain,a.style.cartoon]));
          const lipids=atoms.filter(a=>a.resn==='POPC');
          results.push({step,prot,style:lipids[0].style,
            water:atoms.filter(a=>a.resn==='SOL').map(a=>({elem:a.elem,style:a.style})),
            disconnected:lipids.filter(a=>!a.bonds.length).length,
            spheres:lipids.filter(a=>a.style.sphere).length});
        }
        // An interrupted rendering reuses its canvas; context loss must still revoke readiness.
        GMXViewer.scenes.get('ion-viewer').ready=false;
        await GMXViewer.render('ion-viewer','ions');
        let lost=0;const realInvalidate=window.invalidateFinalReview;
        window.invalidateFinalReview=()=>{lost++;};
        document.querySelector('#ion-viewer canvas').dispatchEvent(new Event('webglcontextlost'));
        window.invalidateFinalReview=realInvalidate;
        const contextRecovered=!GMXViewer.scenes.has('ion-viewer') && lost===1;
        GMXViewer.invalidate();
        // Cancel while the optional viewer library is still loading.
        const realLoader=GMXAssets.viewer;let release;
        GMXAssets.viewer=()=>new Promise(resolve=>release=resolve);
        const pending=GMXViewer.render('ion-viewer','ions');
        while(!release)await new Promise(resolve=>setTimeout(resolve,10));
        document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
        release();const cancelled=await pending;
        GMXAssets.viewer=realLoader;
        done({initial,filtered,results,contextRecovered,
          cancelledWithoutScene:cancelled===null && GMXViewer.scenes.size===0});
      })().catch(e=>done({error:String(e),}));
    """,
        payloads,
    )
    assert "error" not in result, result
    assert result["contextRecovered"], result
    assert result["cancelledWithoutScene"], result
    assert result["initial"]["A"]["color"] != result["initial"]["C"]["color"]
    assert result["filtered"] == result["initial"]["C"]
    for stage in result["results"]:
        assert stage["prot"] == result["initial"]
        assert stage["disconnected"] == stage["spheres"] == 0
        assert stage["style"] == result["results"][0]["style"]
        assert stage["style"]["stick"]["radius"] == 0.12
        assert len(stage["water"]) == 1
        assert stage["water"][0]["elem"] == "O"
        assert stage["water"] == result["results"][0]["water"]
        # Water must remain visible at whole-box zoom while ions stay larger.
        sphere = stage["water"][0]["style"]["sphere"]
        assert 0.2 <= sphere["radius"] < 0.3
        assert sphere["opacity"] >= 0.5


def test_shared_style_keeps_molecule_ion_water_and_cg_colors_stable(page):
    result = page.execute_script("""
      state.taskId='all-component-colors';
      state.pdbInfo={sequences:[{chain_id:'A'},{chain_id:'C'}],
        small_molecules:[{resname:'AMP'},{resname:'ATP'}]};
      const atoms=[{chain:'A',resn:'ALA'},{chain:'C',resn:'ALA'},
        {resn:'AMP'},{resn:'ATP'},{resn:'SOL'},{resn:'NA'},{resn:'POPC'}];
      const apply=(rows,options)=>{
        const calls=[];
        GMXStyle.apply({setStyle:(selection,style)=>calls.push([selection,style])},rows,options);
        return Object.fromEntries(calls.slice(1)
          .map(([sel,style])=>[sel.chain||sel.resn[0],style]));
      };
      const original=apply(atoms,{}), reversed=apply(atoms.slice().reverse(),{});
      const cg=apply(atoms,{coarse:true});
      const hidden=apply(atoms,{moleculeState:{AMP:{included:false}}});
      return {original,reversed,cg,hidden};
    """)
    assert result["original"] == result["reversed"]
    assert "AMP" not in result["hidden"]
    assert result["hidden"]["ATP"] == result["original"]["ATP"]
    assert result["cg"]["C"]["sphere"]["color"] == result["original"]["C"]["cartoon"]["color"]
    assert result["cg"]["POPC"]["stick"]["color"] == result["original"]["POPC"]["stick"]["color"]
    assert "sphere" not in result["original"]["POPC"]
    assert result["original"]["NA"]["sphere"]["color"] == "0x3b82f6"


def test_large_component_batches_preserve_cross_batch_bonds_and_yield(page):
    import base64

    n = 5000

    def packed(array, dtype):
        return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode()

    def dictionary(label):
        return {"labels": [label], "codes": packed(np.zeros(n), "<u4")}

    payload = {
        "schema": 3,
        "revision": "large-independent-reference",
        "source_step": "ions",
        "resolution": "all-atom",
        "atom_count": n,
        "display_count": n,
        "box_nm": np.eye(3).tolist(),
        "bond_sources": {"topology": n - 1},
        "coordinates_A": packed(np.column_stack([np.arange(n), np.zeros(n), np.ones(n)]), "<f4"),
        "original_indices": packed(np.arange(n), "<u4"),
        "component_indices": packed(np.zeros(n), "<u4"),
        "components": [{"name": "Polymer", "kind": "PROTEIN", "atoms": n, "display_atoms": n}],
        "names": dictionary("CA"),
        "resnames": dictionary("ALA"),
        "chains": dictionary("A"),
        "elements": dictionary("C"),
        "resids": {"labels": [str(i + 1) for i in range(n)], "codes": packed(np.arange(n), "<u4")},
        "bonds": [[i, i + 1] for i in range(n - 1)],
    }
    page.execute_async_script("GMXAssets.viewer().then(arguments[0])")
    result = page.execute_async_script(
        r"""
      const payload=arguments[0],done=arguments[1],lib=$3Dmol;
      state.taskId='large-viewer-reference';state.pdbInfo={};
      document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-final_review'));
      let batches=[],frames=0,stop=false,model;
      function tick(){frames++;if(!stop)requestAnimationFrame(tick);}requestAnimationFrame(tick);
      window.$3Dmol={...lib,createViewer:host=>{
        host.append(document.createElement('canvas'));return {
          clear(){},addModel(){model=new lib.GLModel(0);const add=model.addAtoms.bind(model);
            model.addAtoms=atoms=>{batches.push(atoms.length);add(atoms);};return model;},
          addLine(){},resize(){},render(){},zoomTo(){},setSlab(){},rotate(){}
        };
      }};
      const fetchOriginal=window.fetch;
      window.fetch=(url,...args)=>String(url).endsWith('/viewer.json')
        ?Promise.resolve(new Response(JSON.stringify(payload))):fetchOriginal(url,...args);
      GMXViewer.render('ion-viewer','ions').then(data=>{
        stop=true;const atoms=model.selectedAtoms({});
        const valid=atoms.every((a,i)=>a.x===i&&a.serial===i+1&&
          JSON.stringify(a.bonds.slice().sort((a,b)=>a-b))===JSON.stringify([i-1,i+1].filter(x=>x>=0&&x<atoms.length)));
        done({count:atoms.length,valid,frames,batches,ready:!!data,metrics:GMXViewer.metrics.get('ion-viewer')});
      }).catch(e=>done({error:String(e)}));
    """,
        payload,
    )
    assert "error" not in result, result
    assert result["ready"] and result["valid"] and result["count"] == n
    assert result["frames"] >= 3
    assert max(result["batches"]) <= 2048 and sum(result["batches"]) == n
    assert result["metrics"]["decodeMs"] >= 0

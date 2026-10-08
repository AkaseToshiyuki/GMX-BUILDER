"""Only constructed checkpoints supply a periodic box to the solvation viewer."""

import gzip
import json

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.solvation.solvate import SolvationBuilder
from gmxbuilder.web.server_parts.viewer_data import build_viewer

pytestmark = [pytest.mark.browser, pytest.mark.slow]


def test_solute_then_solvated_checkpoint_keeps_coordinates_and_box_consistent(page, tmp_path):
    # Negative input coordinates reproduce the reported corner-offset display.
    original = np.array([[-1.0, -0.5, -0.2], [0.6, 0.4, 0.7]])
    system = System(
        Structure(
            coordinates=original.copy(),
            box_vectors=np.eye(3) * 6,
            atom_names=["CA", "CA"],
            resnames=["ALA", "ALA"],
            resids=[1, 2],
            chain_ids=["A", "A"],
            elements=["C", "C"],
        )
    )
    system.add_component(Component("Protein", ComponentKind.PROTEIN, np.array([0, 1])))
    system.save_checkpoint(tmp_path / "structure")
    solvated = (
        SolvationBuilder()
        .run(system, {"box_padding": 1.5, "use_prebuilt_water": False, "seed": 7})
        .system
    )
    solvated.save_checkpoint(tmp_path / "solvation")
    payloads = {
        step: json.loads(gzip.decompress(build_viewer(tmp_path / step).read_bytes()))
        for step in ("structure", "solvation")
    }
    page.execute_script(
        """
        state.taskId='viewer-box-regression';
        state.taskType={id:'solvator',pipeline:'solvator'};
        document.querySelectorAll('.panel').forEach(p=>
          p.classList.toggle('active',p.id==='panel-solvation'));
        const payloads=arguments[0], realFetch=window.fetch;
        window.fetch=(url,...args)=>{
          const step=String(url).split('/').at(-2);
          if(String(url).endsWith('/viewer.json') && payloads[step])
            return Promise.resolve(new Response(JSON.stringify(payloads[step])));
          return realFetch(url,...args);
        };
        // Capture actual renderer calls; parsing and checkpoint selection remain real.
        window.drawn={atoms:[],lines:[]};
        window.$3Dmol={createViewer:host=>{
          host.append(document.createElement('canvas'));
          return {
            clear(){drawn.atoms=[];drawn.lines=[];},
            addModel(){const own=[];return {
              addAtoms(atoms){own.push(...atoms);drawn.atoms.push(...atoms);},
              selectedAtoms(){return own;},setStyle(){}};},
            addLine(line){drawn.lines.push(line);},
            resize(){},render(){},zoomTo(){},setSlab(){},rotate(){}
          };
        }};
        """,
        payloads,
    )
    # Include backward navigation and cached redraws, which must remove old box lines.
    for checked in (False, True, True, False, False):
        result = page.execute_async_script(
            """
            const checked=arguments[0],done=arguments[1];
            checked ? _checkedSteps.add('solvation') : _checkedSteps.delete('solvation');
            renderSolvationViewer().then(()=>done({
              atoms:drawn.atoms.filter(a=>a.resn==='ALA').map(a=>[a.x,a.y,a.z]),
              lines:drawn.lines,
              label:document.getElementById('solvation-viewer-label').textContent,
              status:document.getElementById('solvation-3d-viewer-loading').textContent
            })).catch(e=>done({error:String(e)}));
            """,
            checked,
        )
        assert "error" not in result
        assert "Structure ready" in result["status"]
        if not checked:
            np.testing.assert_allclose(result["atoms"], original * 10, atol=1e-5)
            assert result["lines"] == []
            assert "compute solvent to create the box" in result["label"]
            assert "6.00" not in result["label"]
            continue
        assert "Checked solvent box:" in result["label"]
        assert len(result["lines"]) == 12
        corners = np.array(
            [
                [edge[end][axis] for axis in ("x", "y", "z")]
                for edge in result["lines"]
                for end in ("start", "end")
            ]
        )
        atoms = np.array(result["atoms"])
        np.testing.assert_allclose(corners.min(0), 0, atol=1e-5)
        # Six-face padding is an independent geometric oracle, in angstroms.
        np.testing.assert_allclose(atoms.min(0) - corners.min(0), 15, atol=1e-5)
        np.testing.assert_allclose(corners.max(0) - atoms.max(0), 15, atol=1e-5)
        np.testing.assert_allclose(atoms[1] - atoms[0], (original[1] - original[0]) * 10)

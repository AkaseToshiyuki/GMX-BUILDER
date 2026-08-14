import subprocess
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from gmxbuilder.modules.export.exporter import ExportModule
from gmxbuilder.modules.solvation.solvate import SolvationBuilder
from gmxbuilder.modules.solvation.water_models import (
    WaterRegistry,
    supported_force_fields,
    water_model_supported,
)
from gmxbuilder.geometry.overlap import find_overlapping_atoms
from tests.test_gromacs_smoke import _find_gmx
from tests.test_membrane_gromacs_smoke import (
    _two_residue_system,
    _write_smoke_mdp,
)
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.web.task_types import get_task_type


WATER_MODELS = ("tip3p", "spc", "spce", "tip4p")


def test_solvation_viewer_uses_checkpoint_specific_box_origin():
    app_js = (
        Path(__file__).parents[1]
        / "src"
        / "gmxbuilder"
        / "web"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    renderer = app_js.split("async function renderSolvationViewer()", 1)[1].split(
        "// ===================================================================\n"
        "// Simulation Parameters",
        1,
    )[0]

    assert "if (_solvChecked)" in renderer
    assert "checkpointStep = 'solvation'" in renderer
    assert (
        "boxOrigin.x = (bounds.minX + bounds.maxX) / 2.0 - boxA / 2.0"
        in renderer
    )
    assert "var membraneBounds = _pdbMembraneZBoundsAngstrom(pdbContent)" in renderer
    assert "boxC = membraneBounds.maxZ - membraneBounds.minZ + 2.0 * paddingA" in renderer
    assert "boxOrigin.z = membraneMidZ - boxC / 2.0" in renderer
    assert "drawOrthogonalBox(v, boxA, boxB, boxC, boxOrigin)" in renderer
    assert "drawOrthogonalBox(v, boxA, boxB, boxC, boxOrigin);\n\n  // Checked" in renderer
    assert "v.zoomTo();\n  v.render();" in renderer


def _empty_system(box=(1.5, 1.5, 1.5)) -> System:
    return System(Structure(
        coordinates=np.empty((0, 3)),
        box_vectors=np.diag(box),
        atom_names=[], resnames=[], resids=[], elements=[],
    ))


def _one_atom_system() -> System:
    structure = Structure(
        coordinates=np.array([[-1.0, 2.0, 3.0]]),
        box_vectors=np.eye(3), atom_names=["CA"], resnames=["ALA"],
        resids=[1], elements=["C"], chain_ids=["A"],
    )
    system = System(structure)
    system.add_component(Component(
        name="PROTEIN", kind=ComponentKind.PROTEIN,
        atom_indices=np.array([0]), metadata={},
    ))
    return system


def _asymmetric_membrane_system(protein_z=(-2.5, 3.0)) -> System:
    coordinates = np.array([
        [0.0, 0.0, protein_z[0]],
        [0.0, 0.0, protein_z[1]],
        [0.5, 0.5, -2.0],
        [0.5, 0.5, 2.0],
    ])
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.diag([4.0, 4.0, 7.0]),
        atom_names=["CA", "CA", "P", "P"],
        resnames=["ALA", "ALA", "POPC", "POPC"],
        resids=[1, 2, 3, 4],
        chain_ids=["A", "A", "", ""],
        elements=["C", "C", "P", "P"],
    )
    system = System(structure, metadata={"water_model": "tip3p"})
    system.add_component(Component(
        name="PROTEIN",
        kind=ComponentKind.PROTEIN,
        atom_indices=np.array([0, 1]),
        metadata={},
    ))
    system.add_component(Component(
        name="MEMBRANE_POPC(100%)",
        kind=ComponentKind.MEMBRANE,
        atom_indices=np.array([2, 3]),
        metadata={"bilayer_thickness": 4.0},
    ))
    return system


def test_membrane_padding_is_equal_from_both_lipid_water_interfaces():
    solvated = SolvationBuilder().run(_asymmetric_membrane_system(), {
        "box_padding": 2.0,
        "remove_overlap": False,
        "use_prebuilt_water": False,
    }).system

    assert solvated.structure.dimensions()[2] == pytest.approx(8.0)
    lower_interface, upper_interface = solvated.metadata["solvation"][
        "membrane_interface_z_nm"
    ]
    assert lower_interface == pytest.approx(2.0)
    assert solvated.structure.dimensions()[2] - upper_interface == pytest.approx(2.0)
    membrane = solvated.component_by_kind(ComponentKind.MEMBRANE)[0]
    membrane_midpoint = np.mean(solvated.coordinates[membrane.atom_indices, 2])
    assert membrane_midpoint == pytest.approx(4.0)


def test_membrane_padding_rejects_protein_that_would_cross_box_boundary():
    with pytest.raises(ModuleConfigError, match="Increase Z Padding to at least 1.00 nm"):
        SolvationBuilder().run(_asymmetric_membrane_system(), {
            "box_padding": 0.5,
            "remove_overlap": False,
        })


def test_overlap_detection_uses_periodic_minimum_image_when_box_is_supplied():
    mobile = np.array([[0.99, 0.5, 0.5]])
    fixed = np.array([[0.01, 0.5, 0.5]])
    assert not find_overlapping_atoms(mobile, fixed, scale=0.8).any()
    assert find_overlapping_atoms(
        mobile, fixed, scale=0.8, box_dimensions=np.ones(3),
    ).all()


def test_overlap_detection_handles_large_cutoff_in_a_small_periodic_box():
    fixed = np.array([[0.05, 0.1, 0.1]])
    mobile = np.array([[0.45, 0.1, 0.1]])
    assert find_overlapping_atoms(
        mobile,
        fixed,
        vdw_radii_mobile=np.float64(0.2),
        vdw_radii_fixed=np.float64(0.2),
        scale=0.8,
        box_dimensions=np.array([0.5, 0.5, 0.5]),
    ).all()


def test_topology_preserves_noncontiguous_water_runs(tmp_path):
    water = WaterRegistry.get("tip3p")
    atom_names = list(water.atom_names) + ["NA"] + list(water.atom_names)
    structure = Structure(
        coordinates=np.zeros((7, 3), dtype=float),
        box_vectors=np.diag([3.0, 3.0, 3.0]),
        atom_names=atom_names,
        resnames=["SOL", "SOL", "SOL", "NA", "HOH", "HOH", "HOH"],
        resids=[1, 1, 1, 2, 3, 3, 3],
        chain_ids=[""] * 7,
        elements=["O", "H", "H", "Na", "O", "H", "H"],
    )
    topology = tmp_path / "topol.top"
    TopologyWriter("amber99sb-ildn", {"water_model": "tip3p"}).write_top(
        structure, topology
    )
    molecules = topology.read_text().split("[ molecules ]", 1)[1]
    records = [
        line.split()
        for line in molecules.splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ]
    assert records[:3] == [["SOL", "1"], ["NA", "1"], ["SOL", "1"]]


@pytest.mark.parametrize("model_name", WATER_MODELS)
def test_grid_fallback_honors_water_site_count_and_solution_frame(model_name):
    system = _one_atom_system()
    system.metadata["water_model"] = model_name
    config = {
        "water_model": model_name, "box_padding": 0.5,
        "use_prebuilt_water": False, "remove_overlap": False, "seed": 7,
    }
    SolvationBuilder().validate_config(config)
    result = SolvationBuilder().run(system, config)
    assert result.success
    solvated = result.system
    model = WaterRegistry.get(model_name)
    solvent = solvated.component_by_kind(ComponentKind.SOLVENT)[0]
    assert len(solvent.atom_indices) == solvent.metadata["n_molecules"] * model.n_atoms
    assert np.allclose(solvated.coordinates[0], [0.5, 0.5, 0.5])
    assert np.allclose(solvated.structure.dimensions(), [1.0, 1.0, 1.0])
    assert np.isfinite(solvated.coordinates).all()
    assert solvated.metadata["water_model"] == model_name
    if model_name == "tip4p":
        assert "MW" in solvated.structure.atom_names


@pytest.mark.parametrize("model_name", WATER_MODELS)
def test_prebuilt_water_and_matching_topology_pass_grompp(tmp_path, model_name):
    gmx = _find_gmx()
    config = {
        "water_model": model_name, "box_size": [1.5, 1.5, 1.5],
        "box_padding": 0.0, "remove_overlap": False,
    }
    result = SolvationBuilder().run(_empty_system(), config)
    structure = result.system.structure
    model = WaterRegistry.get(model_name)
    assert structure.num_atoms > 0
    assert structure.num_atoms % model.n_atoms == 0
    assert np.all(structure.coordinates >= 0.0)
    assert np.all(structure.coordinates < 1.5)

    case_dir = tmp_path / model_name
    case_dir.mkdir()
    GROWriter.write(structure, case_dir / "input.gro")
    TopologyWriter(
        "charmm36m", ff_config={"water_model": model_name}
    ).write_top(structure, case_dir / "topol.top")
    top_text = (case_dir / "topol.top").read_text()
    assert f'#include "{model_name}.itp"' in top_text
    (case_dir / "smoke.mdp").write_text(
        "integrator = steep\nnsteps = 1\ncutoff-scheme = Verlet\n"
        "rlist = 0.5\nrcoulomb = 0.5\nrvdw = 0.5\n"
        "coulombtype = Cut-off\nvdwtype = Cut-off\nconstraints = none\npbc = xyz\n"
    )
    proc = subprocess.run(
        [gmx, "grompp", "-f", "smoke.mdp", "-c", "input.gro",
         "-p", "topol.top", "-o", "smoke.tpr"],
        cwd=case_dir, text=True, capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def test_water_model_is_selected_with_force_field_and_locked_for_solvation():
    system = _empty_system()
    selected = ForceFieldSelector().run(
        system, {"name": "charmm36m", "water_model": "spce"}
    ).system
    assert selected.metadata["water_model"] == "spce"
    assert water_model_supported("charmm36m", "spce")
    assert "charmm36m" in supported_force_fields("spce")
    with pytest.raises(ModuleConfigError, match="locked by Step 2"):
        SolvationBuilder().run(selected, {
            "water_model": "tip3p", "box_size": [1.0, 1.0, 1.0]
        })


def test_exporter_uses_water_model_locked_in_system_metadata(tmp_path):
    selected = ForceFieldSelector().run(
        _empty_system(), {"name": "charmm36m", "water_model": "spce"}
    ).system
    solvated = SolvationBuilder().run(selected, {
        "box_size": [1.0, 1.0, 1.0], "box_padding": 0.0,
        "remove_overlap": False,
    }).system
    output = tmp_path / "export"
    result = ExportModule().run(solvated, {
        "output_dir": output, "system_name": "water", "write_mdp": False,
    })
    assert result.success
    assert '#include "spce.itp"' in (output / "topol.top").read_text()


def test_complete_solvated_membrane_passes_grompp_and_mdrun(tmp_path):
    gmx = _find_gmx()
    protein = StructureProcessor().run(
        _two_residue_system("charmm36m"), {"skip_protonation": True}
    ).system
    protein.metadata.update({
        "_oriented": True, "force_field": "charmm36m",
        "water_model": "spce",
    })
    membrane = MembraneBuilder().run(protein, {
        "lipid_composition": {
            "upper": [{"name": "POPC", "ratio": 100}], "lower": None,
        },
        "n_lipids_per_leaflet": 64, "seed": 20260712,
    }).system
    solvated = SolvationBuilder().run(membrane, {
        "box_padding": 0.6, "overlap_scale": 0.8,
    }).system
    assert solvated.component_by_kind(ComponentKind.SOLVENT)[0].metadata["n_molecules"] > 0

    GROWriter.write(solvated.structure, tmp_path / "input.gro")
    TopologyWriter(
        "charmm36m", ff_config={"water_model": "spce"}
    ).write_top(solvated.structure, tmp_path / "topol.top")
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    grompp = subprocess.run(
        [gmx, "grompp", "-f", "smoke.mdp", "-c", "input.gro",
         "-p", "topol.top", "-o", "smoke.tpr", "-po", "processed.mdp"],
        cwd=tmp_path, text=True, capture_output=True, timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr
    mdrun = subprocess.run(
        [gmx, "mdrun", "-s", "smoke.tpr", "-deffnm", "smoke-em", "-ntmpi", "1"],
        cwd=tmp_path, text=True, capture_output=True, timeout=120,
    )
    output = mdrun.stdout + "\n" + mdrun.stderr
    assert mdrun.returncode == 0, output
    assert "force on at least one atom is not finite" not in output


def test_default_solution_protocol_exports_and_passes_grompp(tmp_path):
    gmx = _find_gmx()
    system = StructureProcessor().run(
        _two_residue_system("amber14sb"), {"skip_protonation": True}
    ).system
    system.metadata.update({
        "force_field": "amber14sb",
        "water_model": "tip3p",
    })
    solvated = SolvationBuilder().run(system, {
        "box_padding": 1.5,
        "overlap_scale": 0.8,
    }).system
    output_dir = tmp_path / "solution-export"
    exported = ExportModule().run(solvated, {
        "output_dir": output_dir,
        "system_name": "solution",
        "write_mdp": True,
    })

    assert exported.success, "\n".join(exported.log)
    assert len(list((output_dir / "mdp").glob("*.mdp"))) == 12
    assert len(list((output_dir / "mdp").glob("production_*.mdp"))) == 10
    assert "vdw-modifier            = Potential-shift" in (
        output_dir / "mdp" / "equili_1.mdp"
    ).read_text()
    assert "DispCorr                = EnerPres" in (
        output_dir / "mdp" / "equili_1.mdp"
    ).read_text()

    for mdp_name in ("mini.mdp", "equili_1.mdp"):
        grompp = subprocess.run(
            [
                gmx, "grompp",
                "-f", f"mdp/{mdp_name}",
                "-c", "input.gro",
                "-r", "input.gro",
                "-p", "topol.top",
                "-n", "index.ndx",
                "-o", f"{mdp_name}.tpr",
            ],
            cwd=output_dir,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr


@pytest.mark.parametrize(
    "config",
    [
        {"water_model": "unknown"},
        {"box_padding": -0.1},
        {"overlap_scale": 1.1},
        {"box_size": [1.0, 2.0]},
        {"box_size": [1.0, float("nan"), 1.0]},
    ],
)
def test_solvation_config_rejects_invalid_values(config):
    with pytest.raises(ModuleConfigError):
        SolvationBuilder().validate_config(config)


def test_enabled_workflows_define_water_model_in_force_field_step():
    for task_id in ("membrane-bilayer", "pure-membrane", "solvator"):
        defaults = get_task_type(task_id).default_config
        assert defaults["forcefield"]["water_model"] == "tip3p"
        assert "water_model" not in defaults["solvation"]

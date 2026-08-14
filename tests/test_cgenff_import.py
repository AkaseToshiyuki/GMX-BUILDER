from pathlib import Path
import math
import subprocess

import numpy as np
from fastapi.testclient import TestClient

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.cgenff_import import prepare_cgenff_molecule
from gmxbuilder.modules.forcefield.compatibility import compatibility_report
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from tests.test_gromacs_smoke import _find_gmx
from tests.test_membrane_gromacs_smoke import _write_smoke_mdp
from gmxbuilder.web.server import app, task_manager


MOL2 = """@<TRIPOS>MOLECULE
LIG
 2 1 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 C1 0.0000 0.0000 0.0000 C.3 1 LIG 0.0
2 H1 1.0900 0.0000 0.0000 H   1 LIG 0.0
@<TRIPOS>BOND
1 1 2 1
"""

STREAM = """* CGenFF program version 4.6
read rtf card append
36 1
MASS -1 CGX1 12.011
MASS -1 HGX1 1.008
RESI LIG 0.000000
GROUP
ATOM C1 CGX1 -0.100000 ! penalty = 12.0
ATOM H1 HGX1  0.100000
BOND C1 H1
END
read para card flex append
ATOMS
MASS -1 CGX1 12.011
MASS -1 HGX1 1.008
BONDS
CGX1 HGX1 340.0 1.090
ANGLES
DIHEDRALS
IMPROPERS
NONBONDED nbxmod 5 atom cdiel
CGX1 0.0 -0.1100 2.0000
HGX1 0.0 -0.0200 1.2000
END
"""


def _write_package(tmp_path):
    mol2 = tmp_path / "LIG.mol2"
    stream = tmp_path / "LIG.str"
    mol2.write_text(MOL2)
    stream.write_text(STREAM)
    return mol2, stream


def _ligand_system():
    structure = Structure(
        coordinates=np.array([[0.5, 0.5, 0.5]]),
        box_vectors=np.diag([3.0, 3.0, 3.0]),
        atom_names=["C1"], resnames=["LIG"], resids=[1],
        chain_ids=["L"], elements=["C"],
    )
    return System(structure, components=[Component(
        "UNKNOWN", ComponentKind.UNKNOWN, np.array([0]), {},
    )])


def test_charmm_ligands_offer_external_cgenff_import():
    report = compatibility_report(_ligand_system(), "charmm36m", [])
    option = next(item for item in report["ligand_options"] if item["value"] == "cgenff")
    assert option["enabled"] is True
    assert "MOL2" in option["reason"]


def test_cgenff_package_is_imported_and_hydrogens_are_added(tmp_path):
    mol2, stream = _write_package(tmp_path)
    template = prepare_cgenff_molecule(
        "LIG", mol2, stream, "charmm36m", tmp_path / "generated",
    )
    assert template.atom_names == ("C1", "H1")
    assert template.net_charge == 0
    assert template.cgenff_version == "4.6"
    assert template.maximum_penalty == 12.0

    result = ForceFieldSelector().run(_ligand_system(), {
        "name": "charmm36m", "lipid_names": [], "lipid_ff": "none",
        "ligand_ff": "cgenff", "water_model": "tip3p",
        "cgenff_parameters": {
            "LIG": {"mol2_path": str(mol2), "str_path": str(stream)},
        },
    }).system
    assert result.structure.atom_names == ["C1", "H1"]
    assert result.metadata["ligand_parameters"]["LIG"]["source"] == "cgenff"
    assert result.component_by_kind(ComponentKind.LIGAND)[0].metadata["molecule_charges"] == {"LIG": 0}


def test_cgenff_nbfix_uses_pair_rmin_not_atom_rmin_half(tmp_path):
    mol2, stream = _write_package(tmp_path)
    stream.write_text(
        STREAM.replace(
            "HGX1 0.0 -0.0200 1.2000\nEND\n",
            "HGX1 0.0 -0.0200 1.2000\n"
            "NBFIX\nCGX1 HGX1 -0.0500 3.5000\nEND\n",
        )
    )
    template = prepare_cgenff_molecule(
        "LIG", mol2, stream, "charmm36m", tmp_path / "nbfix"
    )
    record = next(
        line.split()
        for line in template.atomtypes_path.read_text().splitlines()
        if line.startswith("CGX1") and "HGX1" in line
    )
    expected_sigma_nm = 0.35 / (2.0 ** (1.0 / 6.0))
    assert math.isclose(float(record[3]), expected_sigma_nm, rel_tol=1e-9)
    assert math.isclose(float(record[4]), 0.05 * 4.184, rel_tol=1e-9)


def test_imported_cgenff_topology_passes_grompp(tmp_path):
    mol2, stream = _write_package(tmp_path)
    system = ForceFieldSelector().run(_ligand_system(), {
        "name": "charmm36m", "lipid_names": [], "lipid_ff": "none",
        "ligand_ff": "cgenff", "water_model": "tip3p",
        "cgenff_parameters": {
            "LIG": {"mol2_path": str(mol2), "str_path": str(stream)},
        },
    }).system
    ff_config = {
        "water_model": "tip3p",
        "ligand_parameters": system.metadata["ligand_parameters"],
    }
    GROWriter.write(system.structure, tmp_path / "input.gro")
    TopologyWriter("charmm36m", ff_config).write_top(system.structure, tmp_path / "topol.top")
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    proc = subprocess.run(
        [_find_gmx(), "grompp", "-f", "smoke.mdp", "-c", "input.gro", "-p", "topol.top", "-o", "smoke.tpr"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def test_cgenff_frontend_requires_web_output_upload():
    root = Path(__file__).parents[1]
    source = (root / "src/gmxbuilder/web/static/app.js").read_text()
    assert "https://cgenff.com/" in source
    assert "MOL2 + STR required" in source
    assert "Choose submitted MOL2" in source
    assert "Choose returned STR" in source
    assert "No file selected" in source
    assert "'/api/cgenff-upload/' + state.taskId" in source
    assert (
        "cgenff_parameters: isPureMembrane ? {} : collectCGenFFParameters()"
        in source
    )


def test_cgenff_upload_validates_and_persists_package(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "root", tmp_path)
    task = task_manager.create_task(filename="ligand.pdb")
    task_id = task["task_id"]
    _ligand_system().save_checkpoint(tmp_path / task_id / "steps" / "input")

    with TestClient(app) as client:
        response = client.post(
            f"/api/cgenff-upload/{task_id}",
            data={"ligand_name": "LIG", "force_field": "charmm36m"},
            files={
                "mol2_file": ("LIG.mol2", MOL2, "chemical/x-mol2"),
                "str_file": ("LIG.str", STREAM, "text/plain"),
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["cgenff_version"] == "4.6"
    assert "review" in payload["warning"]
    assert payload["ready"] is True
    assert "mol2_path" not in payload
    assert "str_path" not in payload
    saved = task_manager.get_state(task_id)["cgenff_uploads"]["LIG"]
    package_dir = task_manager.get_task_dir(task_id) / "cgenff" / "LIG"
    assert (package_dir / saved["mol2_file"]).is_file()
    assert (package_dir / saved["str_file"]).is_file()
    assert saved["force_field"] == "charmm36m"

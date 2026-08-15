"""External GROMACS validation of GMXBUILDER coordinates and topology."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.runtime.hardware import find_gromacs_executable


def _find_gmx() -> str:
    executable = find_gromacs_executable()
    if executable:
        return executable
    pytest.skip("GROMACS executable not available; set GMX_BIN to enable this test")


def _two_residue_system(force_field: str) -> System:
    names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    coordinates = np.array(
        [
            [2.000, 2.000, 2.000],
            [2.145, 2.000, 2.000],
            [2.245, 2.100, 2.000],
            [2.225, 2.220, 2.000],
            [2.160, 1.850, 2.000],
            [2.370, 2.080, 2.000],
            [2.470, 2.160, 2.000],
            [2.590, 2.100, 2.000],
            [2.690, 2.160, 2.000],
        ]
    )
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 5.0,
        atom_names=names,
        resnames=["ALA"] * 5 + ["GLY"] * 4,
        resids=[1] * 5 + [2] * 4,
        chain_ids=["A"] * len(names),
        elements=["N", "C", "C", "O", "C", "N", "C", "C", "O"],
    )
    return System(
        structure=structure,
        components=[Component("PROTEIN_A", ComponentKind.PROTEIN, np.arange(len(names)))],
        metadata={"force_field": force_field},
    )


def _three_residue_modification_system(force_field: str, residue: str) -> System:
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    template = load_force_field_rtp(force_field).get_residue(residue)
    assert template is not None
    sidechains = [
        atom[0]
        for atom in template["atoms"]
        if atom[0] not in {"N", "CA", "C", "O", "OXT", "OT1", "OT2"}
        and not atom[0].startswith("H")
        and not (len(atom[0]) > 1 and atom[0][0].isdigit() and atom[0][1] == "H")
    ]
    names = ["N", "CA", "C", "O", "CB"]
    names += ["N", "CA", "C", "O"] + sidechains
    names += ["N", "CA", "C", "O", "CB"]
    target_count = 4 + len(sidechains)
    resnames = ["ALA"] * 5 + [residue] * target_count + ["ALA"] * 5
    resids = [1] * 5 + [2] * target_count + [3] * 5

    first = np.array(
        [
            [1.800, 2.000, 2.000],
            [1.945, 2.000, 2.000],
            [2.045, 2.100, 2.000],
            [2.025, 2.220, 2.000],
            [1.960, 1.850, 2.000],
        ]
    )
    target_backbone = np.array(
        [
            [2.170, 2.080, 2.000],
            [2.270, 2.160, 2.000],
            [2.390, 2.100, 2.000],
            [2.490, 2.160, 2.000],
        ]
    )
    sidechain_coords = np.array(
        [
            [2.280 + 0.025 * i, 2.310 + 0.055 * i, 2.000 + 0.035 * (i % 2)]
            for i in range(len(sidechains))
        ]
    )
    last = np.array(
        [
            [2.515, 2.080, 2.000],
            [2.615, 2.160, 2.000],
            [2.735, 2.100, 2.000],
            [2.835, 2.160, 2.000],
            [2.620, 2.310, 2.000],
        ]
    )
    coordinates = np.vstack([first, target_backbone, sidechain_coords, last])
    elements = [next(char for char in name if char.isalpha()) for name in names]
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 5.0,
        atom_names=names,
        resnames=resnames,
        resids=resids,
        chain_ids=["A"] * len(names),
        elements=elements,
    )
    return System(
        structure=structure,
        components=[Component("PROTEIN_A", ComponentKind.PROTEIN, np.arange(len(names)))],
        metadata={"force_field": force_field},
    )


def _build_gromacs_input(tmp_path: Path, force_field: str) -> tuple[str, Path]:
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(_two_residue_system(force_field), {"skip_protonation": True})
        .system
    )
    gro_path = tmp_path / "input.gro"
    top_path = tmp_path / "topol.top"
    mdp_path = tmp_path / "smoke.mdp"
    GROWriter.write(system.structure, gro_path, title="GMXBUILDER GROMACS smoke test")
    TopologyWriter(force_field).write_top(
        system.structure, top_path, system_name="GMXBUILDER smoke"
    )
    protein_itp = tmp_path / "topol_Protein_chain_A.itp"
    protein_text = protein_itp.read_text()
    for section in ("[ angles ]", "[ dihedrals ]", "[ pairs ]"):
        assert section in protein_text
    mdp_path.write_text(
        "integrator = steep\n"
        "nsteps = 1\n"
        "emtol = 1000\n"
        "cutoff-scheme = Verlet\n"
        "nstlist = 10\n"
        "rlist = 1.0\n"
        "coulombtype = Cut-off\n"
        "rcoulomb = 1.0\n"
        "vdwtype = Cut-off\n"
        "rvdw = 1.0\n"
        "constraints = none\n"
        "pbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            str(mdp_path),
            "-c",
            str(gro_path),
            "-p",
            str(top_path),
            "-o",
            "smoke.tpr",
            "-po",
            "processed.mdp",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    tpr_path = tmp_path / "smoke.tpr"
    assert tpr_path.is_file() and tpr_path.stat().st_size > 0
    dump = subprocess.run(
        [gmx, "dump", "-s", str(tpr_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert dump.returncode == 0, dump.stdout + "\n" + dump.stderr
    assert any(token in dump.stdout for token in ("ANGLES", "UREY_BRADLEY"))
    assert any(token in dump.stdout for token in ("PDIHS", "RBDIHS"))
    assert "LJ14" in dump.stdout
    return gmx, tpr_path


@pytest.mark.parametrize(
    "force_field",
    ["charmm36", "charmm36m", "amber14sb", "amber99sb", "amber99sb-ildn", "oplsaa"],
)
def test_grompp_accepts_generated_protein_system(tmp_path, force_field):
    _build_gromacs_input(tmp_path, force_field)


@pytest.mark.parametrize("force_field", ["charmm36m", "amber14sb"])
def test_grompp_accepts_explicit_ace_nme_caps(tmp_path, force_field):
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _two_residue_system(force_field),
            {
                "skip_protonation": True,
                "termini": {"A": {"nter": "ACE", "cter": "NME"}},
            },
        )
        .system
    )
    gro_path = tmp_path / "capped.gro"
    top_path = tmp_path / "capped.top"
    mdp_path = tmp_path / "capped.mdp"
    GROWriter.write(system.structure, gro_path, title="ACE/NME cap smoke test")
    TopologyWriter(force_field).write_top(system.structure, top_path, system_name="capped")
    mdp_path.write_text(
        "integrator = steep\nnsteps = 1\nemtol = 1000\ncutoff-scheme = Verlet\n"
        "nstlist = 10\nrlist = 1.0\ncoulombtype = Cut-off\nrcoulomb = 1.0\n"
        "vdwtype = Cut-off\nrvdw = 1.0\nconstraints = none\npbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            str(mdp_path),
            "-c",
            str(gro_path),
            "-p",
            str(top_path),
            "-o",
            "capped.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize(
    "residue,patch_id,product,expected_charge",
    [
        ("SER", "PHOS_SER", "SEP", -2),
        ("SER", "PHOS1_SER", "S1P", -1),
        ("THR", "PHOS_THR", "TPO", -2),
        ("THR", "PHOS1_THR", "T1P", -1),
        ("TYR", "PHOS_TYR", "PTR", -2),
        ("TYR", "PHOS1_TYR", "Y1P", -1),
    ],
)
def test_amber14sb_phosphorylation_states_pass_grompp(
    tmp_path, residue, patch_id, product, expected_charge
):
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _three_residue_modification_system("amber14sb", residue),
            {
                "skip_protonation": True,
                "modifications": [{"index": 1, "patch_id": patch_id}],
            },
        )
        .system
    )
    target_indices = [
        index for index, name in enumerate(system.structure.resnames) if name == product
    ]
    assert target_indices
    assert all(np.isfinite(system.structure.coordinates[index]).all() for index in target_indices)

    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    template = load_force_field_rtp("amber14sb").get_residue(product)
    assert {system.structure.atom_names[index] for index in target_indices} == {
        atom[0] for atom in template["atoms"]
    }
    assert sum(atom[2] for atom in template["atoms"]) == pytest.approx(expected_charge)
    geometry = system.metadata["modification_geometry"]
    assert len(geometry) == 1
    assert geometry[0]["status"] == "passed"
    assert geometry[0]["max_bond_error_nm"] < 0.005
    # Multi-term harmonic angle targets are not always exactly co-realizable;
    # the local least-squares compromise must nevertheless stay close to all
    # force-field equilibrium values and far from the former 90/180° geometry.
    assert geometry[0]["max_angle_error_deg"] < 10.0
    assert geometry[0]["min_nonbonded_distance_nm"] >= 0.08

    gro_path = tmp_path / "phospho.gro"
    top_path = tmp_path / "phospho.top"
    mdp_path = tmp_path / "phospho.mdp"
    GROWriter.write(system.structure, gro_path, title="Amber14SB phosphorylation")
    TopologyWriter("amber14sb").write_top(system.structure, top_path)
    mdp_path.write_text(
        "integrator = steep\nnsteps = 1\ncutoff-scheme = Verlet\n"
        "nstlist = 10\nrlist = 1.0\ncoulombtype = Cut-off\nrcoulomb = 1.0\n"
        "vdwtype = Cut-off\nrvdw = 1.0\nconstraints = none\npbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            str(mdp_path),
            "-c",
            str(gro_path),
            "-p",
            str(top_path),
            "-o",
            "phospho.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize(
    "residue,patch_id",
    [
        ("SER", "PHOS_SER"),
        ("THR", "PHOS_THR"),
        ("TYR", "PHOS_TYR"),
        ("LYS", "ACET_LYS"),
        ("ARG", "CIT_ARG"),
        ("CYS", "CSO_CYS"),
        ("CYS", "CSX_CYS"),
        ("TYR", "TYS_TYR"),
        ("LYS", "CARBOXY_LYS"),
        ("LYS", "KME_LYS"),
        ("LYS", "KME2_LYS"),
        ("LYS", "KME3_LYS"),
        ("ARG", "RME2_ARG"),
        ("ARG", "RME2A_ARG"),
        ("CYS", "CSN_CYS"),
        ("CYS", "SMC_CYS"),
        ("CYS", "OCS_CYS"),
        ("SER", "SAC_SER"),
        ("TYR", "NIY_TYR"),
        ("MET", "MSO_R_MET"),
        ("PRO", "HYP_PRO"),
        ("LYS", "HYL_LYS"),
    ],
)
def test_every_enabled_charmm36m_native_ptm_passes_grompp(tmp_path, residue, patch_id):
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _three_residue_modification_system("charmm36m", residue),
            {
                "skip_protonation": True,
                "modifications": [{"index": 1, "patch_id": patch_id}],
            },
        )
        .system
    )
    geometry = system.metadata["modification_geometry"]
    assert len(geometry) == 1
    assert geometry[0]["status"] == "passed"
    assert geometry[0]["max_bond_error_nm"] < 0.005
    assert geometry[0]["max_angle_error_deg"] < 10.0
    assert geometry[0]["min_nonbonded_distance_nm"] >= 0.08
    GROWriter.write(system.structure, tmp_path / "ptm.gro")
    TopologyWriter("charmm36m").write_top(system.structure, tmp_path / "ptm.top")
    (tmp_path / "ptm.mdp").write_text(
        "integrator = steep\nnsteps = 1\ncutoff-scheme = Verlet\n"
        "nstlist = 10\nrlist = 1.0\ncoulombtype = Cut-off\nrcoulomb = 1.0\n"
        "vdwtype = Cut-off\nrvdw = 1.0\nconstraints = none\npbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "ptm.mdp",
            "-c",
            "ptm.gro",
            "-p",
            "ptm.top",
            "-o",
            "ptm.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize(
    "force_field",
    ["amber14sb", "amber99sb", "amber99sb-ildn"],
)
def test_amber_hydroxyproline_passes_grompp(tmp_path, force_field):
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _three_residue_modification_system(force_field, "PRO"),
            {
                "skip_protonation": True,
                "modifications": [{"index": 1, "patch_id": "HYP_PRO"}],
            },
        )
        .system
    )
    geometry = system.metadata["modification_geometry"][0]
    assert geometry["stereo_centres"] == ["4R carbon (HYP)"]
    GROWriter.write(system.structure, tmp_path / "hyp.gro")
    TopologyWriter(force_field).write_top(
        system.structure, tmp_path / "hyp.top", topology=system.topology
    )
    (tmp_path / "hyp.mdp").write_text(
        "integrator = steep\nnsteps = 1\ncutoff-scheme = Verlet\n"
        "nstlist = 10\nrlist = 1.0\ncoulombtype = Cut-off\nrcoulomb = 1.0\n"
        "vdwtype = Cut-off\nrvdw = 1.0\nconstraints = none\npbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "hyp.mdp",
            "-c",
            "hyp.gro",
            "-p",
            "hyp.top",
            "-o",
            "hyp.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize(
    "force_field",
    ["amber14sb", "amber99sb", "amber99sb-ildn"],
)
def test_amber_disulfide_pair_passes_grompp(tmp_path, force_field):
    from tests.test_structure_processor import _disulfide_system

    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _disulfide_system(force_field),
            {
                "skip_protonation": True,
                "prepare_standard_termini": False,
                "crosslinks": [{"type": "disulfide", "first_index": 0, "second_index": 1}],
            },
        )
        .system
    )
    system = ForceFieldAssigner().run(system, {}).system
    GROWriter.write(system.structure, tmp_path / "disulfide.gro")
    TopologyWriter(force_field).write_top(
        system.structure, tmp_path / "disulfide.top", topology=system.topology
    )
    (tmp_path / "disulfide.mdp").write_text(
        "integrator = steep\nnsteps = 1\ncutoff-scheme = Verlet\n"
        "nstlist = 10\nrlist = 1.0\ncoulombtype = Cut-off\nrcoulomb = 1.0\n"
        "vdwtype = Cut-off\nrvdw = 1.0\nconstraints = none\npbc = xyz\n"
    )
    result = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "disulfide.mdp",
            "-c",
            "disulfide.gro",
            "-p",
            "disulfide.top",
            "-o",
            "disulfide.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_representative_new_charmm36m_ptm_minimizes_and_roundtrips(tmp_path):
    """A newly enabled, charged PTM must survive persistence and real dynamics."""
    gmx = _find_gmx()
    system = (
        StructureProcessor()
        .run(
            _three_residue_modification_system("charmm36m", "LYS"),
            {
                "skip_protonation": True,
                "modifications": [{"index": 1, "patch_id": "KME3_LYS"}],
            },
        )
        .system
    )
    checkpoint = tmp_path / "checkpoint"
    system.save_checkpoint(checkpoint)
    restored = System.load_checkpoint(checkpoint)
    assert restored.structure.atom_names == system.structure.atom_names
    assert restored.structure.resnames == system.structure.resnames
    assert np.array_equal(restored.structure.coordinates, system.structure.coordinates)
    assert restored.metadata["modification_geometry"] == system.metadata["modification_geometry"]

    GROWriter.write(restored.structure, tmp_path / "ptm.gro")
    TopologyWriter("charmm36m").write_top(restored.structure, tmp_path / "ptm.top")
    (tmp_path / "ptm.mdp").write_text(
        "integrator = steep\nnsteps = 200\nemtol = 10000\nemstep = 0.001\n"
        "cutoff-scheme = Verlet\nnstlist = 20\nrlist = 1.2\n"
        # This deliberately non-periodic scientific smoke fixture has no
        # solvent/ions.  Cut-off electrostatics avoids accepting an Ewald net-
        # charge warning while the production workflow remains PME + neutralized.
        "coulombtype = Cut-off\nrcoulomb = 1.2\n"
        "vdwtype = Cut-off\nvdw-modifier = Force-switch\n"
        "rvdw-switch = 1.0\nrvdw = 1.2\nDispCorr = no\n"
        "constraints = none\npbc = xyz\n"
    )
    grompp = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "ptm.mdp",
            "-c",
            "ptm.gro",
            "-p",
            "ptm.top",
            "-o",
            "ptm.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr
    mdrun = subprocess.run(
        [
            gmx,
            "mdrun",
            "-s",
            "ptm.tpr",
            "-deffnm",
            "ptm-em",
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    output = mdrun.stdout + "\n" + mdrun.stderr
    assert mdrun.returncode == 0, output
    assert "nan" not in output.lower()
    assert (tmp_path / "ptm-em.gro").is_file()


def test_amber14sb_uses_water_specific_ion_parameters(tmp_path):
    system = (
        StructureProcessor()
        .run(_two_residue_system("amber14sb"), {"skip_protonation": True})
        .system
    )
    top_path = tmp_path / "topol.top"
    TopologyWriter("amber14sb").write_top(system.structure, top_path)
    topology = top_path.read_text()
    assert '#include "ions_tip3p.itp"' in topology
    assert '#include "ions.itp"' not in topology


def test_one_step_mdrun_can_use_gpu(tmp_path):
    if os.environ.get("GMXBUILDER_TEST_GPU") != "1":
        pytest.skip("set GMXBUILDER_TEST_GPU=1 to exercise CUDA mdrun")
    gmx, tpr_path = _build_gromacs_input(tmp_path, "charmm36m")
    result = subprocess.run(
        [
            gmx,
            "mdrun",
            "-s",
            str(tpr_path),
            "-deffnm",
            "gpu-smoke",
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
            "-nb",
            "gpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-noconfout",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert "GPU" in output or "CUDA" in output
    assert (tmp_path / "gpu-smoke.log").is_file()

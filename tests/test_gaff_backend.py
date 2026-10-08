import subprocess
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.gaff_backend import (
    _itp_charges,
    estimate_gaff_net_charge,
    gaff_available,
    prepare_gaff_lipid,
    prepare_gaff_molecule,
)
from gmxbuilder.modules.forcefield.lipid_policy import (
    membrane_lipid_names,
    resolve_lipid_force_field,
)
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from gmxbuilder.modules.membrane.lipids import CATEGORY_NAMES, LipidRegistry
from gmxbuilder.pipeline.config import PipelineConfig
from gmxbuilder.pipeline.pipeline import Pipeline
from tests.test_gromacs_smoke import _find_gmx
from tests.test_membrane_gromacs_smoke import _write_smoke_mdp

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not gaff_available(), reason="isolated AmberTools/ACPYPE environment unavailable"
    ),
]


def _small_molecule_system():
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [0.145, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C01", "N02"],
        resnames=["LIG", "LIG"],
        resids=[1, 1],
        chain_ids=["L", "L"],
        elements=["C", "N"],
        source_ids=["methylamine:C", "methylamine:N"],
        source_info={"atoms": {"methylamine:C": {"name": "C01"}, "methylamine:N": {"name": "N02"}}},
    )
    system = System(structure)
    system.add_component(
        Component(
            name="UNKNOWN",
            kind=ComponentKind.UNKNOWN,
            atom_indices=np.array([0, 1]),
        )
    )
    return system


def test_gaff2_coordinate_molecule_preserves_heavy_atoms_and_adds_hydrogens(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    system = _small_molecule_system()
    template = prepare_gaff_molecule("LIG", system.structure, [0, 1], 1)
    assert template.atom_names[:2] == ("C01", "N02")
    assert len(template.atom_names) > 2

    result = ForceFieldSelector().run(
        system,
        {
            "name": "amber99sb-ildn",
            "lipid_names": [],
            "lipid_ff": "none",
            "ligand_ff": "gaff2",
            "ligand_charges": {"LIG": 1},
        },
    )
    assert result.system.num_atoms == len(template.atom_names)
    from tests.test_atom_provenance import assert_surviving_heavy_sources

    assert_surviving_heavy_sources(system.structure, result.system.structure)
    assert result.system.component_by_kind(ComponentKind.LIGAND)
    assert result.system.total_charge() == 1
    assert result.system.metadata["ligand_parameters"]["LIG"]["charge_method"] == "gas"

    output = tmp_path / "topology"
    output.mkdir()
    TopologyWriter(
        "amber99sb-ildn",
        ff_config={
            "water_model": "tip3p",
            "ligand_parameters": result.system.metadata["ligand_parameters"],
        },
    ).write_top(result.system.structure, output / "topol.top")
    assert '#include "LIG.itp"' in (output / "topol.top").read_text()
    GROWriter.write(result.system.structure, output / "input.gro")
    _write_smoke_mdp(output / "smoke.mdp")
    process = subprocess.run(
        [
            _find_gmx(),
            "grompp",
            "-f",
            "smoke.mdp",
            "-c",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "smoke.tpr",
        ],
        cwd=output,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr


def test_gaff_charge_suggestion_is_ph_dependent_for_primary_amine():
    system = _small_molecule_system()

    physiological = estimate_gaff_net_charge(
        "LIG",
        system.structure,
        [0, 1],
        pH=7.0,
    )
    basic = estimate_gaff_net_charge(
        "LIG",
        system.structure,
        [0, 1],
        pH=13.0,
    )

    assert physiological.net_charge == 1
    assert physiological.formula.endswith("+")
    assert basic.net_charge == 0


def test_gaff2_ph_protonation_preserves_uploaded_heavy_atom_names(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    system = _small_molecule_system()

    template = prepare_gaff_molecule(
        "LIG",
        system.structure,
        [0, 1],
        1,
        charge_method="gas",
        target_pH=7.0,
    )

    assert template.atom_names[:2] == ("C01", "N02")
    assert sum(_itp_charges(template.itp_path)) == pytest.approx(1.0, abs=0.02)


def test_gaff_template_is_cached_and_namespaced(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    lipid = LipidRegistry.get("CAMP")
    first = prepare_gaff_lipid("CAMP", lipid.smiles, lipid.charge, charge_method="gas")
    second = prepare_gaff_lipid("CAMP", lipid.smiles, lipid.charge, charge_method="gas")

    assert first.itp_path == second.itp_path
    assert first.coordinates.shape == (len(first.atom_names), 3)
    assert np.isfinite(first.coordinates).all()
    assert "[ atomtypes ]" in first.atomtypes_path.read_text()
    assert "g_camp_" in first.atomtypes_path.read_text()
    assert "[ atomtypes ]" not in first.itp_path.read_text()
    assert sum(_itp_charges(first.itp_path)) == pytest.approx(lipid.charge, abs=0.02)
    assert set(Path(tmp_path).glob("CAMP-*"))


@pytest.mark.parametrize("lipid_name", ["POPC", "CHOL"])
def test_explicit_gaff_lipid_geometry_uses_cached_topology_order(lipid_name):
    lipid = LipidRegistry.get(lipid_name)
    template = prepare_gaff_lipid(lipid_name, lipid.smiles, lipid.charge)

    coordinates, atom_names = build_rdkit_lipid_geometry(
        lipid_name,
        lipid.smiles,
        force_field="amber14sb",
        lipid_ff="gaff2",
        net_charge=lipid.charge,
    )

    assert coordinates.shape == template.coordinates.shape
    assert tuple(atom_names) == template.atom_names


def test_force_field_policy_preserves_rtp_and_uses_one_gaff_family():
    assert {LipidRegistry.get(name).category for name in LipidRegistry.list()} <= set(
        CATEGORY_NAMES
    )
    assert {
        name: (
            Chem.GetFormalCharge(Chem.MolFromSmiles(LipidRegistry.get(name).smiles)),
            LipidRegistry.get(name).charge,
        )
        for name in LipidRegistry.list()
        if Chem.GetFormalCharge(Chem.MolFromSmiles(LipidRegistry.get(name).smiles))
        != LipidRegistry.get(name).charge
    } == {}
    rtp = resolve_lipid_force_field("charmm36m", ["POPC", "CHOL"])
    assert rtp.protein_force_field == "charmm36m"
    assert rtp.lipid_force_field == "charmm36m"
    assert not rtp.gaff_lipids

    camp = resolve_lipid_force_field("charmm36m", ["POPC", "CAMP"])
    assert camp.protein_force_field == "charmm36m"
    assert camp.lipid_force_field == "charmm36m"

    gaff = resolve_lipid_force_field("charmm36m", ["POPC", "20AHC"])
    assert gaff.protein_force_field == "amber14sb"
    assert gaff.lipid_force_field == "gaff2"
    assert gaff.gaff_lipids == ("20AHC", "POPC")

    assert membrane_lipid_names({"lipid_type": "popc"}) == ("POPC",)
    assert membrane_lipid_names(
        {
            "lipid_composition": {
                "upper": [{"name": "POPC", "ratio": 50}, {"name": "camp", "ratio": 50}],
                "lower": [{"name": "POPC", "ratio": 100}],
            }
        }
    ) == ("CAMP", "POPC")


def test_numeric_lipid_name_uses_gromacs_safe_molecule_type(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    lipid = LipidRegistry.get("20AHC")
    template = prepare_gaff_lipid("20AHC", lipid.smiles, lipid.charge)
    structure = Structure(
        coordinates=template.coordinates,
        box_vectors=np.eye(3) * 12.0,
        atom_names=list(template.atom_names),
        resnames=["20AHC"] * len(template.atom_names),
        resids=[1] * len(template.atom_names),
    )
    top_path = tmp_path / "topol.top"
    TopologyWriter("amber99sb-ildn").write_top(structure, top_path)
    assert "L_20AHC" in top_path.read_text()


def test_pipeline_derives_policy_input_from_membrane_config():
    system = System(
        Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3),
            atom_names=[],
            resnames=[],
            resids=[],
        )
    )
    pipeline = Pipeline().add_module(ForceFieldSelector())
    config = PipelineConfig(
        modules={
            "forcefield": {
                "name": "amber99sb-ildn",
                "lipid_ff": "lipid21",
                "ligand_ff": "none",
            },
            "membrane": {
                "lipid_composition": {
                    "upper": [{"name": "POPC", "ratio": 50}, {"name": "DAPC", "ratio": 50}],
                }
            },
        }
    )
    result = pipeline.run(system, config)
    assert result.system.metadata["force_field"] == "amber99sb-ildn"
    assert result.system.metadata["lipid_ff"] == "lipid21"
    assert result.system.metadata["lipid21_lipids"] == ["DAPC", "POPC"]
    assert result.system.metadata["ligand_ff"] == "none"


def test_charge_fitting_threads_are_bounded_by_default():
    """Unbounded, OpenMP takes every core and AM1-BCC does not repay it.

    Measured on this host: sqm held 96 threads open on 96 logical cores and drew
    114% CPU -- one core of work and ninety-five threads contending for it. The
    charge fit is the long pole in building a lipid entry, so the default is
    bounded rather than left to OpenMP.
    """
    import os

    from gmxbuilder.modules.forcefield.gaff_backend import (
        DEFAULT_GAFF_THREADS,
        _gaff_tool_environment,
    )

    environment = _gaff_tool_environment()
    assert environment["OMP_NUM_THREADS"] == str(
        min(DEFAULT_GAFF_THREADS, os.cpu_count() or DEFAULT_GAFF_THREADS)
    )
    assert environment["OMP_THREAD_LIMIT"] == environment["OMP_NUM_THREADS"]


def test_a_task_budget_still_wins_over_the_default(monkeypatch):
    from gmxbuilder.modules.forcefield.gaff_backend import _gaff_tool_environment
    from gmxbuilder.runtime import hardware

    token = hardware._task_threads.set(4)
    try:
        assert _gaff_tool_environment()["OMP_NUM_THREADS"] == "4"
    finally:
        hardware._task_threads.reset(token)


def test_an_explicit_setting_still_wins(monkeypatch):
    from gmxbuilder.modules.forcefield.gaff_backend import _gaff_tool_environment

    monkeypatch.setenv("GMXBUILDER_GAFF_THREADS", "6")
    assert _gaff_tool_environment()["OMP_NUM_THREADS"] == "6"


def test_outside_a_task_the_budget_does_not_collapse_to_one():
    """`current_task_threads` reports 1 where no deployment configures it.

    Clamping against that would have run every charge fit on a single thread --
    worse than the unbounded default it replaced.
    """
    from gmxbuilder.modules.forcefield.gaff_backend import _gaff_tool_environment
    from gmxbuilder.runtime.hardware import current_task_threads, scoped_task_threads

    assert scoped_task_threads() is None
    assert current_task_threads() >= 1
    assert int(_gaff_tool_environment()["OMP_NUM_THREADS"]) > 1


def test_acpype_duplicate_atom_names_are_made_unique(tmp_path):
    """ACPYPE's element-first naming can repeat a name inside one molecule.

    CER16 came back with two atoms called C, two called O and two called H out
    of 105, and the cache validator rejected the whole template, leaving the
    lipid unbuildable under GAFF2. GROMACS does not mind; this project does,
    because those names are the handles that tie a conformer to a topology.
    """
    from gmxbuilder.modules.forcefield.gaff_backend import (
        _make_atom_names_unique,
        _read_gro,
    )

    gro = tmp_path / "lipid.gro"
    gro.write_text(
        "molecule\n 4\n"
        "    1  UNK    C    1   0.000   0.000   0.000\n"
        "    1  UNK   C1    2   0.100   0.000   0.000\n"
        "    1  UNK    C    3   0.200   0.000   0.000\n"
        "    1  UNK    O    4   0.300   0.000   0.000\n"
        "   1.00000   1.00000   1.00000\n"
    )
    itp = tmp_path / "lipid.itp"
    itp.write_text(
        "[ moleculetype ]\nUNK 3\n[ atoms ]\n"
        "1 g_unk_c3 1 UNK C 1 -0.1 12.011\n"
        "2 g_unk_c3 1 UNK C1 2 -0.1 12.011\n"
        "3 g_unk_c3 1 UNK C 3 -0.1 12.011\n"
        "4 g_unk_oh 1 UNK O 4 -0.4 15.999\n"
    )

    _make_atom_names_unique(itp, gro)

    _coordinates, names = _read_gro(gro)
    assert len(set(names)) == len(names) == 4, names
    topology_names = [
        line.split()[4] for line in itp.read_text().splitlines() if line and line[0].isdigit()
    ]
    assert topology_names == list(names), "the topology and the coordinates must agree"


def test_a_charge_fit_is_exclusive_across_processes(tmp_path):
    """Two replicas of one library entry are two processes, not two threads.

    The queue starts both replicas at once, one per GPU. Guarded only by a
    thread lock they each ran the same half-hour AM1-BCC fit and then each
    removed the other's cache directory before renaming its own into place.
    """
    import fcntl
    import subprocess
    import sys

    from gmxbuilder.modules.forcefield.gaff_backend import _exclusive_fit

    directory = tmp_path / "MOL-0123456789abcdef0123"
    probe = (
        "import fcntl, sys\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "try:\n"
        "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    print('acquired')\n"
        "except BlockingIOError:\n"
        "    print('blocked')\n"
    )

    def other_process(path):
        return subprocess.run(
            [sys.executable, "-c", probe, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()

    with _exclusive_fit("0123456789abcdef0123", directory):
        lock_path = directory.parent / ".locks" / f"{directory.name}.lock"
        assert lock_path.is_file(), "the fit must take a lock another process can see"
        assert other_process(lock_path) == "blocked"

    assert other_process(lock_path) == "acquired", "the lock must be released"
    # And the lock file lives beside the entry, never inside it: the entry is
    # what gets replaced when the fit finishes.
    assert not directory.exists()
    with (lock_path).open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

"""External GROMACS smoke tests for complete protein-membrane systems."""

import subprocess
from pathlib import Path

import pytest

from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from gmxbuilder.modules.modifications.processor import StructureProcessor
from tests.prerequisites import require_v4_entries, requires_lfs_assets
from tests.test_gromacs_smoke import _find_gmx, _two_residue_system

# Every test here drives a real external tool, so the whole module is slow;
# `--fast` deselects it. See the note at the top of tests/conftest.py.
# Every case here builds a real membrane, so it needs the equilibrated
# conformer library as well as GROMACS. Declaring only GROMACS left a latent
# failure: CI skips these for want of GROMACS, so the missing asset gate would
# have surfaced the day CI gained one. Found by scripts/reproduce_ci.sh, which
# strips the two independently.
pytestmark = [pytest.mark.slow, requires_lfs_assets]


def _write_smoke_mdp(path: Path) -> None:
    path.write_text(
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


def _expanded_lipid_sequence(top_path: Path) -> list[str]:
    sequence = []
    section = ""
    for raw in top_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
            continue
        if section != "molecules" or not line:
            continue
        name, count = line.split()[:2]
        if not name.startswith("Protein_chain"):
            sequence.extend([name] * int(count))
    return sequence


def _coordinate_lipid_sequence(system) -> list[str]:
    lipid_names = {"POPC", "POPE", "POPG", "CHOL"}
    blocks = TopologyWriter._ordered_residue_runs(system.structure, lipid_names)
    return [name for name, count in blocks for _ in range(count)]


@pytest.mark.parametrize(
    "upper,lower",
    [
        ([{"name": "POPC", "ratio": 100}], None),
        ([{"name": "POPC", "ratio": 50}, {"name": "CHOL", "ratio": 50}], None),
        (
            [{"name": "POPC", "ratio": 50}, {"name": "CHOL", "ratio": 50}],
            [{"name": "POPE", "ratio": 50}, {"name": "POPG", "ratio": 50}],
        ),
    ],
    ids=["single", "mixed", "asymmetric"],
)
def test_complete_membrane_system_passes_grompp(tmp_path, upper, lower, require_simulation):
    require_v4_entries(sorted({e["name"] for e in upper + (lower or [])}), "charmm36m")
    gmx = _find_gmx()
    from gmxbuilder.io.pdb import PDBWriter
    from gmxbuilder.modules.input.pdb_input import PDBInputModule
    from tests.structure_fixtures import peptide_structure
    from tests.test_atom_provenance import assert_surviving_heavy_sources

    input_path = tmp_path / "protein.pdb"
    PDBWriter.write(peptide_structure("AG"), input_path)
    parsed = PDBInputModule().run(System(peptide_structure("AG")), {"pdb": str(input_path)})
    assert parsed.success, parsed.log
    original = parsed.system.structure.copy()
    parsed.system.metadata["force_field"] = "charmm36m"
    processed = StructureProcessor().run(parsed.system, {"skip_protonation": True})
    assert processed.success, processed.log
    protein = processed.system
    assert_surviving_heavy_sources(original, protein.structure)
    # `_oriented` is a scientific contract: membrane-normal placement has
    # already centered the transmembrane span around Z=0.  Keep this generated
    # smoke fixture consistent with the real OrientModule output.
    protein.structure.coordinates[:, 2] -= protein.structure.coordinates[:, 2].mean()
    protein.metadata["_oriented"] = True
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {"upper": upper, "lower": lower},
            "n_lipids_per_leaflet": 64,
            "seed": 20260711,
        },
    )
    assert result.success, "\n".join(result.log)
    system = result.system
    assert_surviving_heavy_sources(original, system.structure)
    checkpoint = tmp_path / "checkpoint"
    system.save_checkpoint(checkpoint)
    assert_surviving_heavy_sources(original, System.load_checkpoint(checkpoint).structure)

    gro_path = tmp_path / "input.gro"
    top_path = tmp_path / "topol.top"
    mdp_path = tmp_path / "smoke.mdp"
    GROWriter.write(system.structure, gro_path)
    TopologyWriter("charmm36m").write_top(
        system.structure, top_path, system_name="GMXBUILDER membrane smoke"
    )
    protein_itp = (tmp_path / "topol_Protein_chain_A.itp").read_text()
    popc_itp = (tmp_path / "POPC.itp").read_text()
    assert "#ifdef POSRES" in protein_itp
    assert "POSRES_FC_BB" in protein_itp
    assert "POSRES_FC_SC" in protein_itp
    assert "POSRES_FC_LIPID" in popc_itp
    assert "[ dihedral_restraints ]" in popc_itp
    MDPWriter().generate_all(tmp_path / "mdp", {"em_nsteps": 1})
    restrained_grompp = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "mdp/mini.mdp",
            "-c",
            "input.gro",
            "-r",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "restrained.tpr",
            "-maxwarn",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert restrained_grompp.returncode == 0, (
        restrained_grompp.stdout + "\n" + restrained_grompp.stderr
    )
    assert _expanded_lipid_sequence(top_path) == _coordinate_lipid_sequence(system)
    _write_smoke_mdp(mdp_path)

    grompp = subprocess.run(
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
        timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr
    assert (tmp_path / "smoke.tpr").stat().st_size > 0

    require_simulation()
    mdrun = subprocess.run(
        [gmx, "mdrun", "-s", "smoke.tpr", "-deffnm", "smoke-em", "-ntmpi", "1"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    output = mdrun.stdout + "\n" + mdrun.stderr
    assert mdrun.returncode == 0, output
    assert "force on at least one atom is not finite" not in output


def test_modular_charmm_lipid_mixture_passes_grompp(tmp_path):
    """Curated head/tail compositions must link to real CHARMM parameters."""
    require_v4_entries(
        [
            "PAPC",
            "PAPI",
            "LPC16",
            "LYSPG",
            "DOPGD",
            "PPCPL",
            "PAPE",
            "SOPI",
            "SMPC",
            "LPE16",
            "DPPGD",
            "PPEPL",
        ],
        "charmm36m",
    )
    gmx = _find_gmx()
    protein = (
        StructureProcessor()
        .run(_two_residue_system("charmm36m"), {"skip_protonation": True})
        .system
    )
    protein.metadata["_oriented"] = True
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {
                "upper": [
                    {"name": "PAPC", "ratio": 17},
                    {"name": "PAPI", "ratio": 17},
                    {"name": "LPC16", "ratio": 17},
                    {"name": "LYSPG", "ratio": 17},
                    {"name": "DOPGD", "ratio": 16},
                    {"name": "PPCPL", "ratio": 16},
                ],
                "lower": [
                    {"name": "PAPE", "ratio": 17},
                    {"name": "SOPI", "ratio": 17},
                    {"name": "SMPC", "ratio": 17},
                    {"name": "LPE16", "ratio": 17},
                    {"name": "DPPGD", "ratio": 16},
                    {"name": "PPEPL", "ratio": 16},
                ],
            },
            "n_lipids_per_leaflet": 64,
            "seed": 20260718,
        },
    )
    assert result.success, "\n".join(result.log)
    checkpoint = tmp_path / "membrane-checkpoint"
    result.system.save_checkpoint(checkpoint)
    checked = System.load_checkpoint(checkpoint)
    assert {"LPC16", "LYSPG", "DOPGD", "PPCPL", "LPE16", "DPPGD", "PPEPL"} <= set(
        checked.structure.resnames
    )
    GROWriter.write(checked.structure, tmp_path / "input.gro")
    TopologyWriter("charmm36m").write_top(
        checked.structure,
        tmp_path / "topol.top",
        system_name="GMXBUILDER modular CHARMM lipid smoke",
    )
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    grompp = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "smoke.mdp",
            "-c",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "smoke.tpr",
            "-maxwarn",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr


def test_plasmalogen_bilayer_passes_old_charmm_grompp(tmp_path):
    require_v4_entries(["PPCPL", "PPEPL"], "charmm36")
    """The West vinyl-ether additions must also work with CHARMM36 protein terms."""
    gmx = _find_gmx()
    protein = (
        StructureProcessor().run(_two_residue_system("charmm36"), {"skip_protonation": True}).system
    )
    protein.metadata["_oriented"] = True
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {
                "upper": [{"name": "PPCPL", "ratio": 100}],
                "lower": [{"name": "PPEPL", "ratio": 100}],
            },
            "n_lipids_per_leaflet": 64,
            "seed": 20260718,
        },
    )
    assert result.success, "\n".join(result.log)
    GROWriter.write(result.system.structure, tmp_path / "input.gro")
    TopologyWriter("charmm36").write_top(
        result.system.structure,
        tmp_path / "topol.top",
        system_name="GMXBUILDER CHARMM36 plasmalogen smoke",
    )
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    grompp = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "smoke.mdp",
            "-c",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "smoke.tpr",
            "-maxwarn",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr


def test_current_lipid_stream_passes_classic_charmm_grompp(tmp_path):
    """Newer CHARMM36 lipids must remain usable with classic protein terms."""
    require_v4_entries(["DLIPA", "DLIPC", "DLIPE", "DLIPG", "DLIPS", "ERG", "PUPC"], "charmm36")
    gmx = _find_gmx()
    protein = (
        StructureProcessor().run(_two_residue_system("charmm36"), {"skip_protonation": True}).system
    )
    protein.metadata["_oriented"] = True
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {
                "upper": [
                    {"name": "DLIPA", "ratio": 15},
                    {"name": "DLIPC", "ratio": 15},
                    {"name": "DLIPE", "ratio": 14},
                    {"name": "DLIPG", "ratio": 14},
                    {"name": "DLIPS", "ratio": 14},
                    {"name": "ERG", "ratio": 14},
                    {"name": "PUPC", "ratio": 14},
                ],
                "lower": [
                    {"name": "LPC16", "ratio": 17},
                    {"name": "LPC18", "ratio": 17},
                    {"name": "LPE16", "ratio": 17},
                    {"name": "LYSPG", "ratio": 17},
                    {"name": "SITO", "ratio": 16},
                    {"name": "STIG", "ratio": 16},
                ],
            },
            "n_lipids_per_leaflet": 64,
            "seed": 20260719,
        },
    )
    assert result.success, "\n".join(result.log)
    GROWriter.write(result.system.structure, tmp_path / "input.gro")
    TopologyWriter("charmm36").write_top(
        result.system.structure,
        tmp_path / "topol.top",
        system_name="GMXBUILDER modern lipids with classic CHARMM",
    )
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    grompp = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "smoke.mdp",
            "-c",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "smoke.tpr",
            "-maxwarn",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr


def test_generated_glyco_and_plant_sterols_pass_both_charmm_releases(tmp_path, require_simulation):
    """Generated glycolipids/plant sterols must resolve in both releases."""
    require_v4_entries(["MGDG", "DGDG", "CAMP", "GM1"], "charmm36m")
    gmx = _find_gmx()
    protein = (
        StructureProcessor()
        .run(_two_residue_system("charmm36m"), {"skip_protonation": True})
        .system
    )
    protein.structure.coordinates[:, 2] -= protein.structure.coordinates[:, 2].mean()
    protein.metadata["_oriented"] = True
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {
                "upper": [
                    {"name": "MGDG", "ratio": 73},
                    {"name": "CAMP", "ratio": 25},
                    {"name": "GM1", "ratio": 2},
                ],
                "lower": [
                    {"name": "DGDG", "ratio": 73},
                    {"name": "CAMP", "ratio": 25},
                    {"name": "GM1", "ratio": 2},
                ],
            },
            "n_lipids_per_leaflet": 64,
            "seed": 20260720,
        },
    )
    assert result.success, "\n".join(result.log)
    for force_field in ("charmm36m", "charmm36"):
        ff_dir = tmp_path / force_field
        ff_dir.mkdir()
        GROWriter.write(result.system.structure, ff_dir / "input.gro")
        TopologyWriter(force_field).write_top(
            result.system.structure,
            ff_dir / "topol.top",
            system_name=f"GMXBUILDER {force_field} generated-lipid smoke",
        )
        _write_smoke_mdp(ff_dir / "smoke.mdp")
        grompp = subprocess.run(
            [
                gmx,
                "grompp",
                "-f",
                "smoke.mdp",
                "-c",
                "input.gro",
                "-p",
                "topol.top",
                "-o",
                "smoke.tpr",
                "-maxwarn",
                "1",
            ],
            cwd=ff_dir,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr
    require_simulation()
    for force_field in ("charmm36m", "charmm36"):
        ff_dir = tmp_path / force_field
        mdrun = subprocess.run(
            [
                gmx,
                "mdrun",
                "-s",
                "smoke.tpr",
                "-deffnm",
                "smoke-em",
                "-ntmpi",
                "1",
                "-ntomp",
                "2",
            ],
            cwd=ff_dir,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert mdrun.returncode == 0, mdrun.stdout + "\n" + mdrun.stderr


def test_mixed_gaff2_membrane_passes_grompp_and_mdrun(tmp_path, monkeypatch, require_simulation):
    """An explicit whole-GAFF2 comparison retains that model for all species.

    Automatic routing now prefers Lipid21 per species; this probe deliberately
    selects the independently supported whole-GAFF2 alternative.
    """
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    require_v4_entries(["DPPC", "CER16"], "amber99sb-ildn", "gaff2")
    gmx = _find_gmx()
    protein = (
        StructureProcessor()
        .run(_two_residue_system("amber99sb-ildn"), {"skip_protonation": True})
        .system
    )
    protein.metadata["_oriented"] = True
    protein.metadata["force_field"] = "amber99sb-ildn"
    protein.metadata["lipid_ff"] = "gaff2"
    result = MembraneBuilder().run(
        protein,
        {
            "lipid_composition": {
                "upper": [{"name": "DPPC", "ratio": 75}, {"name": "CER16", "ratio": 25}],
                "lower": [{"name": "DPPC", "ratio": 50}, {"name": "CER16", "ratio": 50}],
            },
            "n_lipids_per_leaflet": 64,
            "seed": 20260712,
        },
    )
    assert result.success, "\n".join(result.log)

    gro_path = tmp_path / "input.gro"
    top_path = tmp_path / "topol.top"
    mdp_path = tmp_path / "smoke.mdp"
    GROWriter.write(result.system.structure, gro_path)
    TopologyWriter("amber99sb-ildn").write_top(
        result.system.structure, top_path, system_name="GAFF2 membrane smoke"
    )
    _write_smoke_mdp(mdp_path)
    grompp = subprocess.run(
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
        timeout=120,
    )
    assert grompp.returncode == 0, grompp.stdout + "\n" + grompp.stderr
    require_simulation()
    mdrun = subprocess.run(
        [gmx, "mdrun", "-s", "smoke.tpr", "-deffnm", "smoke-em", "-ntmpi", "1"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    output = mdrun.stdout + "\n" + mdrun.stderr
    assert mdrun.returncode == 0, output
    assert "force on at least one atom is not finite" not in output

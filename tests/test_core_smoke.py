"""Fast, offline regression checks for core file-generation paths."""

import numpy as np
import pytest
import subprocess

from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.io.mdp import MDPWriter
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.modules.export.exporter import ExportModule
from gmxbuilder.runtime.citations import atomistic_citations
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Bond, Topology


def test_atomistic_citations_follow_selected_parameter_families():
    manifest = atomistic_citations(
        {
            "force_field": "charmm36m",
            "lipid_ff": "charmm36",
            "ligand_ff": "cgenff",
            "water_model": "tip3p",
        }
    )

    identifiers = {reference["id"] for reference in manifest["references"]}
    assert {
        "gmxbuilder",
        "gromacs",
        "charmm36m",
        "charmm36-lipid",
        "charmm-gromacs",
        "cgenff",
        "tip3p",
    } <= identifiers


def test_atomistic_citations_include_optional_metropolis_method():
    manifest = atomistic_citations(
        {
            "force_field": "amber14sb",
            "water_model": "tip3p",
            "ions": {"placement_method": "mc"},
        }
    )
    identifiers = {reference["id"] for reference in manifest["references"]}
    assert "metropolis" in identifiers


def test_structure_rejects_mismatched_per_atom_arrays():
    with pytest.raises(ValueError, match="atom_names"):
        Structure(
            coordinates=np.zeros((2, 3)),
            box_vectors=np.eye(3),
            atom_names=["CA"],
        )


def test_structure_append_uses_contiguous_residue_numbers():
    first = Structure(coordinates=np.zeros((1, 3)), box_vectors=np.eye(3), resids=[7])
    second = Structure(coordinates=np.ones((2, 3)), box_vectors=np.eye(3), resids=[1, 2])

    assert first.append(second).resids == [7, 8, 9]


def test_topology_merge_infers_offset_when_atom_types_are_absent():
    first = Topology(bonds=[Bond(0, 1)], atom_count=2)
    second = Topology(bonds=[Bond(0, 1)], atom_count=2)

    first.merge(second)

    assert first.num_atoms() == 4
    assert (first.bonds[-1].i, first.bonds[-1].j) == (2, 3)


def test_pdb_parser_converts_coordinates_and_box(small_pdb_file):
    structure = PDBParser().parse(small_pdb_file)

    assert structure.num_atoms == 5
    assert np.allclose(structure.coordinates[1], [0.1458, 0.0, 0.0])
    assert np.allclose(structure.dimensions(), [5.0, 5.0, 5.0])
    assert structure.resnames == ["ALA"] * 5


def test_mdp_writer_generates_a_complete_default_protocol(tmp_path):
    paths = MDPWriter().generate_all(tmp_path, {"gen_seed": 42})

    assert len(paths) == 12
    assert {path.name for path in paths} == {
        "mini.mdp",
        *(f"equili_{stage}.mdp" for stage in range(1, 7)),
        *(f"production_{stage}.mdp" for stage in range(1, 6)),
    }
    assert all(path.stat().st_size > 0 for path in paths)
    assert "comm-grps               = SOLU_MEMB SOLV" in (tmp_path / "equili_1.mdp").read_text()
    assert "comm-grps               = SOLU_MEMB SOLV" in (tmp_path / "production_1.mdp").read_text()


def test_simulation_config_keeps_workflow_controls_out_of_mdp_context():
    normalized = MDPWriter.normalize_simulation_config(
        {
            "eq_stages": [_short_stage()],
            "prod_iters": [_short_stage(ensemble="npt")],
            "hardware": {"cpu_threads": 1, "mpi_ranks": 1},
            "mdp_overrides_text": "",
            "system_name": "reported_failure_shape",
        },
        {"force_field_family": "amber", "has_membrane": True},
    )

    assert set(normalized) == {
        "schema_version",
        "minimization",
        "eq_stages",
        "prod_iters",
        "hardware",
    }
    assert normalized["schema_version"] == 2
    assert "system_name" not in normalized
    assert "hardware" not in normalized["eq_stages"][0]


def test_legacy_global_mdp_values_are_migrated_to_each_stage():
    normalized = MDPWriter.normalize_simulation_config(
        {
            "temperature": 303.15,
            "rlist": 1.1,
            "mdp_overrides": {"lincs-order": "6"},
            "em_nsteps": 7000,
            "eq_stages": [_short_stage()],
            "prod_iters": [_short_stage(ensemble="npt")],
        },
        {"force_field_family": "amber", "has_membrane": True},
    )

    assert normalized["minimization"]["nsteps"] == 7000
    assert normalized["minimization"]["rlist"] == 1.1
    for stage in normalized["eq_stages"] + normalized["prod_iters"]:
        assert stage["temperature"] == 303.15
        assert stage["rlist"] == 1.1
        assert stage["mdp_overrides"]["lincs-order"] == "6"


def test_schema_two_rejects_misplaced_global_mdp_values():
    with pytest.raises(ValueError, match="requires MDP values to belong"):
        MDPWriter.normalize_simulation_config(
            {
                "schema_version": 2,
                "temperature": 300.0,
            },
            {"force_field_family": "amber", "has_membrane": True},
        )


@pytest.mark.parametrize(
    ("family", "rlist", "modifier", "switch", "dispcorr"),
    [
        ("amber", 1.0, "Potential-shift", None, "EnerPres"),
        ("charmm", 1.2, "Force-switch", 1.0, "no"),
    ],
)
def test_stage_owned_force_field_defaults_are_consistent_across_protocol(
    family, rlist, modifier, switch, dispcorr
):
    normalized = MDPWriter.normalize_simulation_config(
        {}, {"force_field_family": family, "has_membrane": True}
    )

    for stage in [normalized["minimization"], *normalized["eq_stages"], *normalized["prod_iters"]]:
        assert stage["rlist"] == rlist
        assert stage["vdw_modifier"] == modifier
        assert stage["rvdw_switch"] == switch
        assert stage["dispcorr"] == dispcorr


def test_minimization_nonbond_settings_are_independent_of_equilibration(tmp_path):
    normalized = MDPWriter.normalize_simulation_config(
        {}, {"force_field_family": "amber", "has_membrane": True}
    )
    normalized["minimization"]["rlist"] = 1.05
    normalized["eq_stages"][0]["rlist"] = 1.15

    MDPWriter().generate_all(
        tmp_path,
        {"force_field_family": "amber", "has_membrane": True},
        eq_stages=normalized["eq_stages"],
        prod_iters=normalized["prod_iters"],
        minimization=normalized["minimization"],
    )

    assert "rlist                   = 1.05" in (tmp_path / "mini.mdp").read_text()
    assert "rlist                   = 1.15" in (tmp_path / "equili_1.mdp").read_text()


def test_mdp_writer_uses_compact_solution_protocol_by_default(tmp_path):
    paths = MDPWriter().generate_all(
        tmp_path,
        {
            "has_membrane": False,
            "n_tc_groups": 2,
            "protein_position_restraints": True,
        },
    )

    assert len(paths) == 12
    assert {path.name for path in paths} == {
        "mini.mdp",
        "equili_1.mdp",
        *(f"production_{stage}.mdp" for stage in range(1, 11)),
    }
    equil = (tmp_path / "equili_1.mdp").read_text()
    production = (tmp_path / "production_1.mdp").read_text()
    minim = (tmp_path / "mini.mdp").read_text()
    assert "nsteps                  = 5000" in minim
    assert "constraints             = h-bonds" in minim
    assert "POSRES_FC_BB=400.0" in equil
    assert "POSRES_FC_SC=40.0" in equil
    assert "POSRES_FC_LIPID" not in equil
    assert "comm-grps               = SOLU SOLV" in equil
    assert "comm-grps               = SOLU SOLV" in production
    assert "nsteps                  = 500000" in production
    assert "pcoupltype              = Isotropic" in production
    assert "nstxout-compressed      = 50000" in production
    assert "ref-t                   = 310.15 310.15" in production


@pytest.mark.parametrize(
    "family,expected_modifier,expected_cutoff,expected_dispcorr",
    [
        ("charmm", "Force-switch", "1.2", "no"),
        ("amber", "Potential-shift", "1.0", "EnerPres"),
    ],
)
def test_nonbond_defaults_follow_force_field_family(
    tmp_path, family, expected_modifier, expected_cutoff, expected_dispcorr
):
    MDPWriter().generate_all(
        tmp_path,
        {"force_field_family": family},
        eq_stages=[_short_stage()],
        prod_iters=[_short_stage(ensemble="npt")],
    )
    content = (tmp_path / "equili_1.mdp").read_text()

    assert f"vdw-modifier            = {expected_modifier}" in content
    assert f"rvdw                    = {expected_cutoff}" in content
    assert f"rcoulomb                = {expected_cutoff}" in content
    assert f"DispCorr                = {expected_dispcorr}" in content
    assert ("rvdw_switch             = 1.0" in content) is (family == "charmm")


def test_production_repeat_generates_restart_friendly_segments(tmp_path):
    paths = MDPWriter().generate_all(
        tmp_path,
        {},
        eq_stages=[_short_stage()],
        prod_iters=[_short_stage(ensemble="npt", repeat=3, nsteps=25)],
    )

    assert {path.name for path in paths if path.name.startswith("production")} == {
        "production_1.mdp",
        "production_2.mdp",
        "production_3.mdp",
    }
    assert all(
        "nsteps                  = 25" in path.read_text()
        for path in paths
        if path.name.startswith("production")
    )


def _short_stage(**updates):
    stage = {
        "enabled": True,
        "bb": 0,
        "sc": 0,
        "lipid": 0,
        "dih": 0,
        "dt": 1.0,
        "dt_fs": True,
        "nsteps": 10,
        "ensemble": "nvt",
        "comm_grps": "System",
    }
    stage.update(updates)
    return stage


def test_mdp_writer_omits_user_disabled_equilibration_stages(tmp_path):
    schedule = [
        _short_stage(ensemble="nvt"),
        _short_stage(ensemble="nvt"),
        _short_stage(ensemble="npt"),
        _short_stage(enabled=False),
        _short_stage(enabled=False),
        _short_stage(enabled=False),
    ]

    paths = MDPWriter().generate_all(tmp_path, {}, eq_stages=schedule)

    assert {path.name for path in paths} == {
        "mini.mdp",
        "equili_1.mdp",
        "equili_2.mdp",
        "equili_3.mdp",
        *(f"production_{stage}.mdp" for stage in range(1, 6)),
    }
    assert not (tmp_path / "equili_4.mdp").exists()
    assert "dt                      = 0.001" in (tmp_path / "equili_1.mdp").read_text()


def test_mdp_writer_requires_an_explicit_timestep_unit(tmp_path):
    ambiguous = _short_stage()
    ambiguous.pop("dt_fs")

    with pytest.raises(ValueError, match="explicit dt_unit"):
        MDPWriter().generate_all(
            tmp_path,
            {},
            eq_stages=[ambiguous],
            prod_iters=[ambiguous],
        )


def test_mdp_writer_converts_the_same_explicit_unit_for_every_stage(tmp_path):
    stage = _short_stage(dt=0.5, dt_unit="fs")
    stage.pop("dt_fs")

    MDPWriter().generate_all(
        tmp_path,
        {},
        eq_stages=[stage],
        prod_iters=[stage],
    )

    assert "dt                      = 0.0005" in (tmp_path / "equili_1.mdp").read_text()
    assert "dt                      = 0.0005" in (tmp_path / "production.mdp").read_text()


def test_gro_writer_accepts_exactly_five_character_names(tmp_path):
    structure = Structure(
        coordinates=np.array([[0.1, 0.2, 0.3]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C1234"],
        resnames=["MYLIP"],
        resids=[1],
    )
    path = tmp_path / "five.gro"

    GROWriter.write(structure, path)
    loaded = GROReader().read(path)

    assert loaded.atom_names == ["C1234"]
    assert loaded.resnames == ["MYLIP"]


def test_gro_triclinic_box_uses_official_field_order(tmp_path):
    structure = Structure(
        coordinates=np.array([[0.1, 0.2, 0.3]]),
        box_vectors=np.array(
            [
                [5.0, 0.2, 0.3],
                [0.4, 6.0, 0.5],
                [0.6, 0.7, 7.0],
            ]
        ),
        atom_names=["BB"],
        resnames=["ALA"],
        resids=[1],
    )
    path = tmp_path / "triclinic.gro"
    GROWriter.write(structure, path)
    fields = [float(value) for value in path.read_text().splitlines()[-1].split()]
    assert fields == [5.0, 6.0, 7.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    loaded = GROReader().read(path)
    assert np.allclose(loaded.box_vectors, structure.box_vectors, atol=1e-5)


@pytest.mark.parametrize("box_line", ["bad box", "0 0 0", "1 2"])
def test_gro_reader_rejects_malformed_or_degenerate_box(tmp_path, box_line):
    path = tmp_path / "invalid-box.gro"
    path.write_text(f"invalid\n1\n    1ALA     CA    1   0.000   0.000   0.000\n{box_line}\n")

    with pytest.raises(ParseError, match="GRO box|Malformed GRO box"):
        GROReader().read(path)


def test_gro_writer_validates_before_creating_partial_file(tmp_path):
    structure = Structure(
        coordinates=np.asarray([[np.nan, 0.0, 0.0]]),
        box_vectors=np.eye(3),
        atom_names=["CA"],
        resnames=["ALA"],
        resids=[1],
    )
    path = tmp_path / "partial.gro"

    with pytest.raises(ValueError, match="coordinates"):
        GROWriter.write(structure, path)
    assert not path.exists()


def test_gro_writer_normalizes_nonpositive_residue_id(tmp_path):
    structure = Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3),
        atom_names=["CA"],
        resnames=["ALA"],
        resids=[0],
    )
    path = tmp_path / "resid-zero.gro"

    GROWriter.write(structure, path)

    assert path.read_text().splitlines()[2][:5] == "    1"


@pytest.mark.parametrize(
    ("atom_name", "resname", "match"),
    [
        ("C12345", "POPC", "atom name"),
        ("C1", "CUSTOM", "residue name"),
        ("C 1", "POPC", "atom name"),
    ],
)
def test_gro_writer_rejects_names_that_cannot_fit_fixed_columns(
    tmp_path, atom_name, resname, match
):
    structure = Structure(
        coordinates=np.array([[0.1, 0.2, 0.3]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=[atom_name],
        resnames=[resname],
        resids=[1],
    )

    with pytest.raises(ValueError, match=match):
        GROWriter.write(structure, tmp_path / "invalid.gro")


def test_first_enabled_npt_stage_initializes_velocities(tmp_path):
    schedule = [
        _short_stage(enabled=False),
        _short_stage(ensemble="npt", temperature=305),
    ]

    MDPWriter().generate_all(tmp_path, {"gen_seed": 17}, eq_stages=schedule)
    first = (tmp_path / "equili_2.mdp").read_text()

    assert "gen-vel                 = yes" in first
    assert "gen-temp                = 305" in first
    assert "gen-seed                = 17" in first
    assert "continuation            = no" in first


def test_mdp_writer_rejects_silently_empty_stage_selection(tmp_path):
    with pytest.raises(ValueError, match="at least one equilibration stage"):
        MDPWriter().generate_all(
            tmp_path,
            {},
            eq_stages=[_short_stage(enabled=False)],
        )
    with pytest.raises(ValueError, match="at least one production stage"):
        MDPWriter().generate_all(
            tmp_path,
            {},
            prod_iters=[_short_stage(enabled=False)],
        )


@pytest.mark.parametrize("repeat", [0, 101, 1.5, True])
def test_mdp_writer_rejects_invalid_production_repeat(tmp_path, repeat):
    with pytest.raises(ValueError, match="repeat must be an integer from 1 to 100"):
        MDPWriter().generate_all(
            tmp_path,
            {},
            eq_stages=[_short_stage()],
            prod_iters=[_short_stage(ensemble="npt", repeat=repeat)],
        )


def test_energy_minimization_constraints_are_explicit(tmp_path):
    MDPWriter().generate_all(tmp_path, {"em_constraints": "all-bonds"})
    assert "constraints             = all-bonds" in (tmp_path / "mini.mdp").read_text()


def test_mdp_writer_applies_per_stage_thermostat_and_barostat(tmp_path):
    schedule = [
        {
            "bb": 1000,
            "sc": 500,
            "lipid": 400,
            "dih": 200,
            "dt": 0.001,
            "dt_unit": "ps",
            "nsteps": 10,
            "ensemble": "npt",
            "tcoupl": "nose-hoover",
            "tau_t": "2.5",
            "pcoupl": "berendsen",
            "tau_p": "3.0",
            "ref_p": "1.2",
            "compress": "3.0e-5",
        }
    ]

    paths = MDPWriter().generate_all(
        tmp_path,
        {"has_membrane": True, "n_tc_groups": 2},
        eq_stages=schedule,
    )
    equil = (tmp_path / "equili_1.mdp").read_text()

    assert paths
    assert "tcoupl                  = nose-hoover" in equil
    assert "tau-t                   = 2.5 2.5" in equil
    assert "pcoupl                  = berendsen" in equil
    assert "tau-p                   = 3.0" in equil
    assert "compressibility         = 3.0e-5  3.0e-5" in equil


def test_mdp_writer_honours_output_com_motion_and_advanced_overrides(tmp_path):
    schedule = [
        {
            "bb": 0,
            "sc": 0,
            "lipid": 0,
            "dih": 0,
            "dt": 2.0,
            "dt_unit": "fs",
            "nsteps": 1234,
            "ensemble": "npt",
            "nstxout_compressed": 37,
            "nstxout": 41,
            "nstvout": 43,
            "nstfout": 47,
            "nstcalcenergy": 11,
            "nstenergy": 13,
            "nstlog": 17,
            "comm_mode": "none",
            "nstcomm": 0,
            "comm_grps": "SOLU_MEMB SOLV",
            "mdp_overrides": {"nstlist": "29"},
        }
    ]
    production = [
        {
            "dt": 2.0,
            "dt_unit": "fs",
            "nsteps": 4321,
            "nstxout_compressed": 53,
            "nstxout": 59,
            "nstvout": 61,
            "nstfout": 67,
            "nstcalcenergy": 19,
            "nstenergy": 23,
            "nstlog": 31,
            "comm_mode": "linear",
            "nstcomm": 73,
            "comm_grps": "System",
        }
    ]

    MDPWriter().generate_all(
        tmp_path,
        {"mdp_overrides": {"fourierspacing": "0.10"}},
        eq_stages=schedule,
        prod_iters=production,
    )
    equil = (tmp_path / "equili_1.mdp").read_text()
    prod = (tmp_path / "production.mdp").read_text()

    for key, value in {
        "nstxout-compressed": 37,
        "nstxout": 41,
        "nstvout": 43,
        "nstfout": 47,
        "nstcalcenergy": 11,
        "nstenergy": 13,
        "nstlog": 17,
        "nstcomm": 0,
    }.items():
        assert f"{key}" in equil and f"= {value}" in equil
    assert "comm-mode               = none" in equil
    assert "comm-grps               = SOLU_MEMB SOLV" in equil
    assert equil.count("nstlist") == 1
    assert "nstlist                 = 29" in equil
    assert equil.count("fourierspacing") == 1
    assert "fourierspacing          = 0.10" in equil
    assert "nsteps                  = 4321" in prod
    assert "nstxout-compressed      = 53" in prod
    assert "nstcomm                 = 73" in prod
    assert "comm-grps               = System" in prod


@pytest.mark.parametrize(
    "params, match",
    [
        ({"has_membrane": False, "comm_grps": "MEMB SOLV"}, "unavailable"),
        ({"comm_grps": "System SOLV"}, "cannot combine System"),
        ({"comm_grps": "SOLU_MEMB MEMB SOLV"}, "cannot overlap"),
        ({"mdp_overrides": {"comm-grps": "TYPO"}}, "unavailable"),
    ],
)
def test_mdp_writer_rejects_invalid_com_index_groups(tmp_path, params, match):
    with pytest.raises(ValueError, match=match):
        MDPWriter().generate_all(tmp_path, params)


def test_export_index_groups_keep_ligands_with_solute():
    structure = Structure(
        coordinates=np.zeros((5, 3)),
        box_vectors=np.eye(3),
        atom_names=["CA", "C1", "P", "OW", "NA"],
        resnames=["ALA", "LIG", "POPC", "SOL", "NA"],
        resids=[1, 2, 3, 4, 5],
        elements=["C", "C", "P", "O", "Na"],
    )
    system = System(
        structure=structure,
        components=[
            Component("PROTEIN", ComponentKind.PROTEIN, np.array([0]), {}),
            Component("LIGAND", ComponentKind.LIGAND, np.array([1]), {}),
            Component("MEMBRANE", ComponentKind.MEMBRANE, np.array([2]), {}),
            Component("SOLVENT", ComponentKind.SOLVENT, np.array([3]), {}),
            Component("IONS", ComponentKind.IONS, np.array([4]), {}),
        ],
    )

    assert ExportModule._index_groups(system) == {
        "System": [1, 2, 3, 4, 5],
        "SOLU": [1, 2],
        "SOLV": [4, 5],
        "MEMB": [3],
        "SOLU_MEMB": [1, 2, 3],
    }


def test_mdp_writer_rejects_invalid_or_duplicate_line_injection(tmp_path):
    with pytest.raises(ValueError, match="Invalid MDP key"):
        MDPWriter().generate_all(tmp_path, {"mdp_overrides": {"bad key": "20"}})
    with pytest.raises(ValueError, match="Invalid value"):
        MDPWriter().generate_all(tmp_path, {"mdp_overrides": {"nstlist": "20\ninclude"}})


@pytest.mark.parametrize(
    "params,pattern",
    [
        ({"temperature": float("nan")}, "finite"),
        ({"em_step": float("inf")}, "finite"),
        ({"rvdw_switch": 1.2, "rvdw": 1.2}, "below rvdw"),
        ({"unknown_global": 1}, "unknown global"),
        ({"gen_seed": 1.5}, "integer"),
    ],
)
def test_mdp_global_settings_fail_closed(params, pattern):
    with pytest.raises(ValueError, match=pattern):
        MDPWriter.validate_protocol(params)


@pytest.mark.parametrize(
    "stage,pattern",
    [
        ({"nsteps": 1000, "dt": float("nan")}, "finite"),
        ({"nsteps": 1000, "temperature": float("inf")}, "finite"),
        ({"nsteps": 1000, "ignored": 1}, "unknown equilibration"),
        ({"nsteps": 1000, "compress": 0}, "positive"),
    ],
)
def test_mdp_stage_settings_fail_closed(stage, pattern):
    with pytest.raises(ValueError, match=pattern):
        MDPWriter.validate_protocol({}, [stage], [{"nsteps": 1000}])


def test_mdp_writer_hydrates_minimal_user_stages(tmp_path):
    paths = MDPWriter().generate_all(
        tmp_path,
        {"has_membrane": True, "force_field": "amber99sb-ildn"},
        eq_stages=[
            {
                "enabled": True,
                "ensemble": "nvt",
                "nsteps": 1000,
                "dt": 1.0,
                "dt_unit": "fs",
            }
        ],
        prod_iters=[
            {
                "enabled": True,
                "nsteps": 1000,
                "dt": 2.0,
                "dt_unit": "fs",
            }
        ],
    )
    assert {path.name for path in paths} == {
        "mini.mdp",
        "equili_1.mdp",
        "production.mdp",
    }
    assert "ref-t" in (tmp_path / "equili_1.mdp").read_text()
    assert "pcoupl" in (tmp_path / "production.mdp").read_text()


def test_run_script_discovers_the_exact_generated_stage_set(tmp_path):
    script = tmp_path / "run_md.sh"
    mdps = [tmp_path / "mini.mdp", tmp_path / "equili_1.mdp", tmp_path / "production.mdp"]
    ExportModule._write_run_script(script, "custom", 42, mdps)
    content = script.read_text()

    assert "EQ_MDPS=(mdp/equili_*.mdp)" in content
    assert "PROD_MDPS=(mdp/production*.mdp)" in content
    assert 'grompp -f "$MDP"' in content
    assert '-r "$GRO"' in content
    assert "mdp/equili_6.mdp" not in content
    assert "GMX_PREC" not in content
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_run_script_separates_external_mpi_from_thread_mpi(tmp_path):
    script = tmp_path / "run_md.sh"
    ExportModule._write_run_script(
        script,
        "parallel",
        42,
        [tmp_path / "mini.mdp"],
        {
            "mode": "external-mpi",
            "cpu_threads": 24,
            "mpi_ranks": 4,
            "use_gpu": True,
            "gpu_count": 2,
            "gpu_ids": [0, 1],
            "gmx_command": "gmx_mpi",
            "mpi_launcher": "srun",
            "pin": "on",
        },
    )
    content = script.read_text()

    assert 'MDRUN=(srun -n "$MPI_RANKS"' in content
    assert 'MDRUN+=(-ntmpi "$MPI_RANKS")' in content
    assert 'MDRUN+=(-gpu_id "$GPU_IDS")' in content
    assert "DEFAULT_GPU_COUNT=2" in content
    assert "GPU_COUNT must equal the number of entries in GPU_IDS" in content
    invalid = subprocess.run(
        ["bash", str(script)],
        env={
            "PATH": "/usr/bin:/bin",
            "CPU_THREADS": "10",
            "MPI_RANKS": "3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "must divide" in invalid.stderr


def test_package_readme_describes_the_actual_flat_parameter_layout(tmp_path):
    (tmp_path / "input.gro").write_text("coordinates")
    (tmp_path / "topol.top").write_text('#include "forcefield.itp"\n')
    (tmp_path / "index.ndx").write_text("[ System ]\n")
    (tmp_path / "forcefield.itp").write_text("[ defaults ]\n")
    (tmp_path / "POPC.itp").write_text("[ moleculetype ]\n")
    readme = tmp_path / "README.txt"

    ExportModule._write_readme(
        readme,
        "documented",
        42,
        "amber14sb",
        "tip3p",
        [tmp_path / "mdp" / "mini.mdp", tmp_path / "mdp" / "production.mdp"],
    )
    content = readme.read_text()

    assert "stored in the package root" in content
    assert "forcefield.itp" in content
    assert "POPC.itp" in content
    assert "toppar/" not in content
    assert "input.pdb" not in content


def test_mdp_macros_match_the_available_restraint_sections(tmp_path):
    MDPWriter().generate_all(
        tmp_path,
        {
            "protein_position_restraints": True,
            "lipid_position_restraints": True,
            "lipid_dihedral_restraints": False,
        },
    )
    content = (tmp_path / "mini.mdp").read_text()

    assert "-DPOSRES_FC_BB=" in content
    assert "-DPOSRES_FC_LIPID=" in content
    assert "-DDIHRES" not in content


def test_large_step_viewer_keeps_fixed_pdb_columns(tmp_path):
    n_atoms = 100001
    structure = Structure(
        coordinates=np.zeros((n_atoms, 3)),
        box_vectors=np.eye(3) * 10,
        atom_names=["O"] * n_atoms,
        resnames=["SOL"] * n_atoms,
        resids=list(range(1, n_atoms + 1)),
        chain_ids=["W"] * n_atoms,
        elements=["O"] * n_atoms,
    )
    system = System(structure=structure)
    viewer = tmp_path / "viewer.pdb"

    system.write_viewer_pdb(viewer)

    atom_lines = [line for line in viewer.read_text().splitlines() if line.startswith("HETATM")]
    assert len(atom_lines) == n_atoms
    assert all(line[17:20] == "SOL" for line in atom_lines[-3:])
    assert atom_lines[-1][6:11].strip() == "2"

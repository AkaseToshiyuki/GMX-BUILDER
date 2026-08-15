import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import AtomType, Bond, Topology
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.ions.add_ions import IonBuilder, _WaterSite
from gmxbuilder.modules.solvation.water_models import WaterRegistry
from gmxbuilder.pipeline.step_executor import StepRunner
from tests.test_gromacs_smoke import _find_gmx
from tests.test_membrane_gromacs_smoke import _write_smoke_mdp


def _water_system(model_name="tip3p", n_water=100, protein_resname=None):
    model = WaterRegistry.get(model_name)
    coordinates = []
    atom_names = []
    resnames = []
    resids = []
    chain_ids = []
    elements = []
    if protein_resname:
        coordinates.append([0.2, 0.2, 0.2])
        atom_names.append("CA")
        resnames.append(protein_resname)
        resids.append(1)
        chain_ids.append("A")
        elements.append("C")
    solvent_start = len(coordinates)
    for mol in range(n_water):
        x = 1.0 + (mol % 10) * 0.45
        y = 1.0 + ((mol // 10) % 10) * 0.45
        z = 1.0 + (mol // 100) * 0.45
        origin = np.array([x, y, z])
        offsets = [np.zeros(3), [0.09572, 0, 0], [-0.03, 0.09, 0]]
        if model.n_atoms == 4:
            offsets.append([0.01546, 0, 0])
        for atom_index, offset in enumerate(offsets):
            coordinates.append((origin + offset).tolist())
            atom_names.append(model.atom_names[atom_index])
            resnames.append("SOL")
            resids.append(mol + 2)
            chain_ids.append("W")
            elements.append("O" if atom_index == 0 else "H" if atom_index < 3 else "")
    structure = Structure(
        coordinates=np.asarray(coordinates, dtype=float),
        box_vectors=np.diag([6.0, 6.0, 6.0]),
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=chain_ids,
        elements=elements,
    )
    system = System(
        structure,
        metadata={
            "force_field": "amber14sb",
            "water_model": model_name,
            "seed": 7,
        },
    )
    if protein_resname:
        system.add_component(
            Component(
                "PROTEIN",
                ComponentKind.PROTEIN,
                np.array([0]),
                {},
            )
        )
    system.add_component(
        Component(
            "SOLVENT",
            ComponentKind.SOLVENT,
            np.arange(solvent_start, structure.num_atoms),
            {"water_model": model_name, "n_molecules": n_water},
        )
    )
    return system


def _config(**updates):
    config = {
        "cations": ["NA"],
        "anions": ["CL"],
        "concentration": {"NA": 0.0, "CL": 0.0},
        "neutralize": True,
        "neutralize_cation": "NA",
        "neutralize_anion": "CL",
        "ion_method": "replace",
        "exclusion_radius": 0.35,
    }
    config.update(updates)
    return config


@pytest.mark.parametrize(
    "bad",
    [
        {"cations": ["CL"]},
        {"anions": ["NA"]},
        {"anions": ["XX"]},
        {"ion_method": "unknown"},
        {"exclusion_radius": -1},
        {"concentration": {"NA": float("nan"), "CL": 0.0}},
    ],
)
def test_invalid_ion_config_is_rejected(bad):
    config = _config()
    config.update(bad)
    with pytest.raises(ModuleConfigError):
        IonBuilder().validate_config(config)


def test_multivalent_salt_requires_charge_balanced_concentrations():
    with pytest.raises(ModuleConfigError, match="0.15 M CaCl2"):
        IonBuilder().validate_config(
            _config(
                cations=["CA"],
                anions=["CL"],
                concentration={"CA": 0.15, "CL": 0.15},
            )
        )
    assert IonBuilder().validate_config(
        _config(
            cations=["CA"],
            anions=["CL"],
            concentration={"CA": 0.15, "CL": 0.30},
        )
    )


def _solvated_disulfide_stub() -> System:
    protein = System(
        Structure(
            coordinates=np.array([[0.2, 0.2, 0.2], [0.403, 0.2, 0.2]]),
            box_vectors=np.diag([6.0, 6.0, 6.0]),
            atom_names=["SG", "SG"],
            resnames=["CYX", "CYX"],
            resids=[10, 20],
            chain_ids=["A", "A"],
            elements=["S", "S"],
        ),
        topology=Topology(force_field="amber14sb", bonds=[Bond(0, 1)]),
        components=[Component("PROTEIN", ComponentKind.PROTEIN, np.array([0, 1]), {})],
        metadata={
            "force_field": "amber14sb",
            "water_model": "tip3p",
            "crosslinks": [
                {
                    "type": "disulfide",
                    "first": {"chain": "A", "resid": 10},
                    "second": {"chain": "A", "resid": 20},
                    "status": "passed",
                }
            ],
        },
    )
    return protein.merge(_water_system(n_water=100))


def test_ions_accept_validated_structure_step_disulfide_stub():
    system = _solvated_disulfide_stub()
    result = IonBuilder().run(system, _config(neutralize=False))

    assert result.success
    assert result.system.topology is None
    assert result.system.metadata["crosslinks"] == system.metadata["crosslinks"]
    assert "Validated disulfide topology stub" in "\n".join(result.log)


def test_ions_reject_complete_or_unrecognised_preexisting_topology():
    complete = _solvated_disulfide_stub()
    complete.topology.atom_types.append(AtomType("S", 32.0, 0.0, 0.3, 0.1))
    with pytest.raises(ModuleConfigError, match="before final topology"):
        IonBuilder().run(complete, _config(neutralize=False))

    unrecognised = _solvated_disulfide_stub()
    unrecognised.metadata.pop("crosslinks")
    with pytest.raises(ModuleConfigError, match="before final topology"):
        IonBuilder().run(unrecognised, _config(neutralize=False))


@pytest.mark.parametrize("model_name", ["tip3p", "tip4p"])
def test_ions_replace_complete_water_and_remap_components(model_name):
    system = _water_system(model_name, protein_resname="LYS")
    if model_name == "tip4p":
        system.metadata["force_field"] = "charmm36m"
    result = IonBuilder().run(system, _config())
    output = result.system
    model = WaterRegistry.get(model_name)
    metrics = output.metadata["ions"]
    assert metrics["total_counts"] == {"NA": 0, "CL": 1}
    assert metrics["final_charge_e"] == 0
    assert output.num_atoms == system.num_atoms - model.n_atoms + 1
    solvent = output.component_by_kind(ComponentKind.SOLVENT)[0]
    ions = output.component_by_kind(ComponentKind.IONS)[0]
    assert solvent.metadata["n_molecules"] == 99
    assert len(solvent.atom_indices) == 99 * model.n_atoms
    assert max(max(comp.atom_indices, default=-1) for comp in output.components) < output.num_atoms
    nearest = cKDTree(output.coordinates[solvent.atom_indices]).query(
        output.coordinates[ions.atom_indices]
    )[0]
    assert nearest.min() > 0.0


def test_divalent_counterion_cannot_neutralize_odd_charge():
    system = _water_system(protein_resname="ASP")
    with pytest.raises(ModuleConfigError, match="monovalent"):
        IonBuilder().run(system, _config(neutralize_cation="CA"))


def test_distinct_neutralizing_species_keeps_cation_anion_site_order():
    result = (
        IonBuilder()
        .run(
            _water_system(protein_resname="ASP"),
            _config(
                concentration={"NA": 0.60, "CL": 0.60},
                neutralize_cation="K",
                ion_method="replace",
            ),
        )
        .system
    )
    ions = result.component_by_kind(ComponentKind.IONS)[0]
    names = np.asarray(result.structure.resnames)[ions.atom_indices].tolist()
    assert names == ["NA", "K", "CL"]


@pytest.mark.parametrize("method", ["replace", "random", "mc"])
def test_every_placement_method_replaces_complete_waters(method):
    system = _water_system(n_water=100)
    result = (
        IonBuilder()
        .run(
            system,
            _config(
                concentration={"NA": 0.60, "CL": 0.60},
                neutralize=False,
                ion_method=method,
            ),
        )
        .system
    )
    metrics = result.metadata["ions"]
    assert metrics["waters_replaced"] == sum(metrics["total_counts"].values())
    assert result.num_atoms == system.num_atoms - 2 * metrics["waters_replaced"]
    assert (
        metrics["placement_strategy"]
        == {
            "random": "uniform_random_water_replacement",
            "replace": "periodic_electrostatic_water_replacement",
            "mc": "metropolis_water_site_sampling",
        }[method]
    )


def test_ion_check_materializes_canonical_index_before_simparams(tmp_path):
    runner = StepRunner(tmp_path, "liquid-builder")
    source = _water_system(n_water=100)
    source.save_checkpoint(runner.step_dir("solvation"))

    result = runner.run_step(
        "ions",
        _config(
            concentration={"NA": 0.60, "CL": 0.60},
            neutralize=False,
            ion_method="random",
        ),
    )

    index_path = runner.step_dir("ions") / "index.ndx"
    assert result["status"] == "ok"
    assert result["index_path"] == str(index_path)
    content = index_path.read_text()
    assert "[ System ]" in content
    assert "[ SOLU ]" in content
    assert "[ SOLV ]" in content


def test_recommended_random_method_does_not_use_electrostatic_extrema(monkeypatch):
    def unexpected_potential_call(*_args, **_kwargs):
        raise AssertionError("recommended random replacement must not score electrostatic extrema")

    monkeypatch.setattr(IonBuilder, "_site_potentials", unexpected_potential_call)
    result = (
        IonBuilder()
        .run(
            _water_system(n_water=100),
            _config(
                concentration={"NA": 0.60, "CL": 0.60},
                neutralize=False,
                ion_method="random",
            ),
        )
        .system
    )
    assert result.metadata["ions"]["placement_strategy"] == ("uniform_random_water_replacement")


def test_replace_and_monte_carlo_are_distinct_electrostatic_algorithms(monkeypatch):
    system = _water_system(n_water=100)
    sites, _model = IonBuilder._water_sites(system, "tip3p")
    calls = []

    def tracked_potentials(_self, _system, candidate_sites):
        calls.append(len(candidate_sites))
        return np.linspace(-1.0, 1.0, len(candidate_sites))

    monkeypatch.setattr(IonBuilder, "_site_potentials", tracked_potentials)
    IonBuilder()._select_sites(
        system,
        sites,
        {"NA": 1, "CL": 1},
        "replace",
        np.random.default_rng(11),
        0.35,
    )

    class TrackingRng:
        def __init__(self):
            self.generator = np.random.default_rng(12)
            self.acceptance_draws = 0

        def __getattr__(self, name):
            return getattr(self.generator, name)

        def random(self):
            self.acceptance_draws += 1
            return self.generator.random()

    mc_rng = TrackingRng()
    IonBuilder()._select_sites(
        system,
        sites,
        {"NA": 1, "CL": 1},
        "mc",
        mc_rng,
        0.35,
    )

    assert calls == [100, 100]
    assert mc_rng.acceptance_draws > 0


def test_eligible_water_exclusion_uses_periodic_minimum_image():
    system = _water_system(n_water=1, protein_resname="LYS")
    protein = system.component_by_kind(ComponentKind.PROTEIN)[0]
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    system.coordinates[protein.atom_indices[0]] = [0.1, 0.1, 0.1]
    system.coordinates[water.atom_indices[:3]] = np.array(
        [
            [5.9, 0.1, 0.1],
            [5.99572, 0.1, 0.1],
            [5.87, 0.19, 0.1],
        ]
    )
    sites, _model = IonBuilder._water_sites(system, "tip3p")

    eligible = IonBuilder._eligible_sites(system, sites, [(0.0, 6.0)], 0.35)

    assert eligible == []


def test_selected_ions_respect_exclusion_across_periodic_faces():
    system = _water_system(n_water=1)
    sites = [
        _WaterSite(0, (0, 1, 2), np.array([0.1, 0.1, 0.1])),
        _WaterSite(3, (3, 4, 5), np.array([5.9, 0.1, 0.1])),
        _WaterSite(6, (6, 7, 8), np.array([3.0, 3.0, 3.0])),
    ]

    class OrderedRng:
        def __init__(self):
            self.calls = 0

        def permutation(self, _size):
            self.calls += 1
            if self.calls == 1:
                return np.array([0, 1, 2])
            return np.array([1, 0, 2])

    selected = IonBuilder()._select_sites(
        system,
        sites,
        {"NA": 1, "CL": 1},
        "replace",
        OrderedRng(),
        0.35,
    )

    assert selected[0] is sites[0]
    assert selected[1] is sites[2]


def test_protein_charge_counts_residue_ids_independently_per_chain():
    system = _water_system(protein_resname=None)
    protein = Structure(
        coordinates=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
        box_vectors=system.structure.box_vectors.copy(),
        atom_names=["CA", "CA"],
        resnames=["LYS", "LYS"],
        resids=[1, 1],
        chain_ids=["A", "B"],
        elements=["C", "C"],
    )
    merged = System(protein).merge(system)
    merged.add_component(Component("PROTEIN", ComponentKind.PROTEIN, np.array([0, 1]), {}))
    assert merged.total_charge() == 2.0


def test_topology_uses_force_field_ions_without_redefinition(tmp_path):
    gmx = _find_gmx()
    model = WaterRegistry.get("tip3p")
    structure = Structure(
        coordinates=np.array(
            [
                [0.5, 0.5, 0.5],
                [0.59572, 0.5, 0.5],
                [0.47, 0.59, 0.5],
                [1.2, 1.2, 1.2],
                [1.8, 1.8, 1.8],
            ]
        ),
        box_vectors=np.diag([2.5, 2.5, 2.5]),
        atom_names=model.atom_names + ["NA", "CL"],
        resnames=["SOL", "SOL", "SOL", "NA", "CL"],
        resids=[1, 1, 1, 2, 3],
        elements=["O", "H", "H", "Na", "Cl"],
    )
    GROWriter.write(structure, tmp_path / "input.gro", title="ion smoke")
    TopologyWriter("amber14sb", {"water_model": "tip3p"}).write_top(
        structure,
        tmp_path / "topol.top",
    )
    text = (tmp_path / "topol.top").read_text()
    assert '#include "ions_tip3p.itp"' in text
    assert '#include "NA.itp"' not in text
    assert '#include "CL.itp"' not in text
    _write_smoke_mdp(tmp_path / "smoke.mdp")
    proc = subprocess.run(
        [gmx, "grompp", "-f", "smoke.mdp", "-c", "input.gro", "-p", "topol.top", "-o", "smoke.tpr"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def test_ion_check_uses_backend_checkpoint_and_viewer():
    root = Path(__file__).parents[1]
    source = (root / "src/gmxbuilder/web/static/ions.js").read_text()
    assert "'/api/step/' + state.taskId + '/ions'" in source
    assert "_checkedSteps.add('ions')" in source
    assert "_loadStepViewerPdb('ions')" in source
    assert "Math.round(getIonConc" not in source

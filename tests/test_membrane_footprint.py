"""What a solute actually denies the bilayer, and what follows from measuring it."""

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from tests.prerequisites import require_v4_entries, requires_lfs_assets

EXCLUSION = MembraneBuilder._PROTEIN_EXCLUSION_XY
SPACING = MembraneBuilder._DENSE_GRID_SPACING
POPC_HALF_THICKNESS = 1.9  # dh/2 for POPC


def _cylinder(radius, z_low, z_high, count, seed):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(count))
    angle = rng.random(count) * 2.0 * np.pi
    return np.stack(
        [r * np.cos(angle), r * np.sin(angle), rng.uniform(z_low, z_high, count)], axis=1
    )


def _mushroom(seed=7):
    """A narrow transmembrane stalk under a wide extracellular cap."""
    return np.vstack(
        [
            _cylinder(1.5, -2.2, 2.2, 1200, seed),
            _cylinder(5.5, 2.6, 6.2, 2400, seed + 1),
        ]
    )


def _solute_system(coords):
    count = len(coords)
    system = System(
        structure=Structure(
            coordinates=coords,
            box_vectors=np.diag([20.0, 20.0, 20.0]),
            atom_names=["CA"] * count,
            resnames=["ALA"] * count,
            resids=list(range(1, count + 1)),
            chain_ids=["A"] * count,
            elements=["C"] * count,
        )
    )
    system.add_component(
        Component(name="probe", kind=ComponentKind.PROTEIN, atom_indices=np.arange(count))
    )
    system.metadata["_oriented"] = True
    system.metadata["force_field"] = "charmm36m"
    return system


@pytest.mark.parametrize("radius", [1.0, 2.0, 3.5])
def test_the_denied_area_is_the_area_it_claims_to_be(radius):
    """Measured against a shape whose answer is known in closed form."""
    atoms = _cylinder(radius, -2.5, 2.5, 8000, seed=3)
    measured = MembraneBuilder._slab_footprint_area(
        MembraneBuilder._slab_atoms(atoms, POPC_HALF_THICKNESS), EXCLUSION, SPACING
    )
    analytic = np.pi * (radius + EXCLUSION) ** 2
    assert measured == pytest.approx(analytic, rel=0.05)


def test_a_domain_above_the_bilayer_denies_the_bilayer_nothing():
    """The whole point: a cap that sits over the membrane is not in it.

    Sized by the old rule -- the square of the solute's larger XY span over
    every height -- this shape reserved thirteen times the area it occupies.
    The box grew by the difference and the lipid count did not, so the
    difference became water lying in the hydrophobic core.
    """
    atoms = _mushroom()
    slab = MembraneBuilder._slab_atoms(atoms, POPC_HALF_THICKNESS)
    measured = MembraneBuilder._slab_footprint_area(slab, EXCLUSION, SPACING)
    stalk_area = np.pi * (1.5 + EXCLUSION) ** 2
    assert measured == pytest.approx(stalk_area, rel=0.05)

    bounding_square = float(np.max(atoms[:, :2].max(axis=0) - atoms[:, :2].min(axis=0))) ** 2
    assert bounding_square > 10.0 * measured


def test_nothing_in_the_slab_denies_nothing():
    """A solute entirely clear of the bilayer, and an empty one."""
    above = _cylinder(5.0, 4.0, 8.0, 2000, seed=5)
    assert (
        MembraneBuilder._slab_footprint_area(
            MembraneBuilder._slab_atoms(above, POPC_HALF_THICKNESS), EXCLUSION, SPACING
        )
        == 0.0
    )
    empty = np.zeros((0, 3))
    assert (
        MembraneBuilder._slab_footprint_area(
            MembraneBuilder._slab_atoms(empty, POPC_HALF_THICKNESS), EXCLUSION, SPACING
        )
        == 0.0
    )


@pytest.fixture(scope="module")
def rebuilt():
    require_v4_entries(["POPC"], "charmm36m", "charmm36m")
    system = _solute_system(_mushroom())
    n_solute = system.num_atoms
    return (
        150,
        MembraneBuilder().run(
            system,
            {
                "lipid_type": "POPC",
                "n_lipids_per_leaflet": 150,
            },
        ),
        n_solute,
    )


@requires_lfs_assets
def test_periodic_clearance_automatically_adds_lipids_instead_of_leaving_holes(rebuilt):
    baseline, result, _ = rebuilt
    assert result.success
    component = result.system.component_by_kind(ComponentKind.MEMBRANE)[0]
    metadata = component.metadata
    assert metadata["requested_lipids_per_leaflet"] == baseline
    assert metadata["n_lipids_upper"] > baseline
    assert metadata["n_lipids_lower"] > baseline
    assert metadata["leaflet_count_policy"] == "area-balanced"
    assert any("Area-balanced lipid counts" in line for line in result.log)
    assert not [line for line in result.log if line.startswith("⚠")], result.log


def test_wedge_footprints_produce_different_counts_at_one_target_area():
    # Independent analytic budget: 100 lipids * 0.6 + 20 nm² -> 80 nm².
    # The other leaflet has 10 nm² more free area, requiring ceil(70/0.6)=117.
    box, counts = MembraneBuilder._area_balanced_counts(100, [0.6, 0.6], [20, 10], 4)
    assert box * box == pytest.approx(80)
    assert counts == [100, 117]


def test_different_leaflet_apls_are_balanced_without_diluting_the_smaller_lipid():
    box, counts = MembraneBuilder._area_balanced_counts(100, [0.6, 0.8], [0, 0], 4)
    assert box * box == pytest.approx(80)
    assert counts == [134, 100]


@requires_lfs_assets
def test_the_solute_sits_at_the_centre_of_its_own_box(rebuilt):
    """Its distance to its periodic image is not a lipid tail's business.

    Centring on the union of solute and lipids handed that distance to
    whichever conformer happened to reach furthest across a periodic edge --
    a rigid translation that means nothing to a periodic membrane and
    everything to the solute.
    """
    _recommended, result, n_solute = rebuilt
    coords = result.system.structure.coordinates[:n_solute, :2]
    centre = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    assert np.allclose(centre, 0.0, atol=0.05)


@requires_lfs_assets
def test_checked_counts_and_species_survive_resume(rebuilt, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from gmxbuilder.pipeline.step_executor import _compute_step_metrics
    from gmxbuilder.web import server
    from gmxbuilder.web.task_manager import TaskManager

    _, result, _ = rebuilt
    manager = TaskManager(tmp_path / "tasks")
    monkeypatch.setattr(server, "task_manager", manager)
    monkeypatch.setattr(server, "_step_runners", {})
    task = manager.create_task()
    task_id = task["task_id"]
    manager.update_state(task_id, {"task_type": {"id": "pure-membrane"}})
    runner = server._get_step_runner(task_id, "pure-membrane")
    result.system.save_checkpoint(runner.step_dir("membrane"))
    expected = _compute_step_metrics(result.system, "membrane")
    counts = expected["membrane"]
    membrane = result.system.component_by_kind(ComponentKind.MEMBRANE)[0]
    # Independent count from the produced atom/residue records (POPC is 134 atoms).
    assert len(membrane.atom_indices) == 134 * (counts["n_lipids_upper"] + counts["n_lipids_lower"])
    assert counts["lipid_counts_upper"] == {"POPC": counts["n_lipids_upper"]}
    assert counts["lipid_counts_lower"] == {"POPC": counts["n_lipids_lower"]}
    with TestClient(server.app) as client:
        response = client.get(f"/api/steps/{task_id}")
    assert response.status_code == 200
    actual = next(row for row in response.json()["steps"] if row["name"] == "membrane")
    assert actual["membrane_metrics"] == expected


def test_oversized_footprint_is_rejected_before_grid_allocation(monkeypatch):
    from gmxbuilder.core.exceptions import ModuleConfigError
    from gmxbuilder.modules.membrane.builder import MembraneBuilder

    monkeypatch.setattr(np, "meshgrid", lambda *a, **k: pytest.fail("Allocated oversized grid"))
    with pytest.raises(ModuleConfigError, match="100 nm"):
        MembraneBuilder._slab_footprint_area(np.array([[0.0, 0.0], [1e20, 1.0]]), 0.3, 0.1)
    with pytest.raises(ModuleConfigError, match="cell budget"):
        MembraneBuilder._slab_footprint_area(np.array([[0.0, 0.0], [10.0, 10.0]]), 0.3, 1e-6)

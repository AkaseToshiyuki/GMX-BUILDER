"""The solute the lipids are built around, and the count the user asked for.

Two defects are pinned here, and they are the same defect seen twice.

**L1.** The builder chose its obstacles by ``ComponentKind.PROTEIN``. A ligand,
a cofactor or a nucleic acid was therefore invisible to the placement grid, the
clash filter and the box, and lipids were built straight through it. Obstacles
are now chosen by *presence*: the membrane step runs before solvation and ions,
so everything already in the system is solute.

**M1.** ``n_lipids_per_leaflet`` was enforced before the last filter that could
remove a lipid, and nothing put the removed ones back. Measured on the real
validation protein: 128 requested, 127 delivered, identically in two force
fields and two independent builds. The contract test that should have caught
it ran on an empty system -- the one configuration where the contract held.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.builder import MembraneBuilder, _seat_lipids_against_solute
from tests.prerequisites import requires_lfs_assets

MIN_CONTACT = MembraneBuilder._LIPID_PROTEIN_MIN_DIST


def _cylinder(radius: float = 1.0, half_z: float = 2.2, spacing: float = 0.35) -> np.ndarray:
    """A solid transmembrane plug: an obstacle with no grooves to argue about."""
    axis = np.arange(-radius, radius + 1e-9, spacing)
    zs = np.arange(-half_z, half_z + 1e-9, spacing)
    return np.asarray(
        [(x, y, z) for x in axis for y in axis for z in zs if x * x + y * y <= radius * radius]
    )


def _system_with(solute: np.ndarray, kind: ComponentKind) -> System:
    n = len(solute)
    system = System(
        structure=Structure(
            coordinates=solute.copy(),
            box_vectors=np.eye(3) * 10.0,
            atom_names=["C"] * n,
            resnames=["UNK"] * n,
            resids=list(range(1, n + 1)),
            elements=["C"] * n,
        ),
        metadata={
            "seed": 42,
            "force_field": "charmm36m",
            "lipid_ff": "charmm36m",
            "_oriented": True,
        },
    )
    system.add_component(Component(name="S", kind=kind, atom_indices=np.arange(n)))
    return system


@cache
def _build(kind: ComponentKind, requested: int = 64):
    """Build once per (kind, count) and share it.

    Every case here needs a whole membrane, and the same three builds were
    otherwise done twice over. Cached because the builder is deterministic in
    its seed and the tests only read the result.
    """
    from tests.prerequisites import require_v4_entries

    require_v4_entries(["DPPC"], "charmm36m")
    system = _system_with(_cylinder(), kind)
    n_solute = system.num_atoms
    result = MembraneBuilder().run(
        system, {"lipid_type": "DPPC", "n_lipids_per_leaflet": requested}
    )
    membrane = next(c for c in result.system.components if c.kind == ComponentKind.MEMBRANE)
    return result, membrane, n_solute


# --------------------------------------------------------------------------
# L1 -- obstacles are chosen by presence, not by kind


@requires_lfs_assets
@pytest.mark.slow
@pytest.mark.parametrize(
    "kind",
    [ComponentKind.PROTEIN, ComponentKind.LIGAND, ComponentKind.NUCLEIC_ACID],
)
def test_every_kind_of_solute_is_an_obstacle(kind):
    """A ligand must displace lipids exactly as a protein does."""
    result, membrane, n_solute = _build(kind)

    lipids = result.system.coordinates[membrane.atom_indices]
    closest = float(cKDTree(result.system.coordinates[:n_solute]).query(lipids, k=1)[0].min())
    assert closest >= MIN_CONTACT, f"a lipid was built into the {kind.name}"


@requires_lfs_assets
@pytest.mark.slow
def test_the_kinds_are_not_merely_all_accepted_but_all_identical():
    """The obstacle rule must not depend on the kind at all."""
    counts = set()
    for kind in (ComponentKind.PROTEIN, ComponentKind.LIGAND, ComponentKind.NUCLEIC_ACID):
        _, membrane, _ = _build(kind)
        counts.add((membrane.metadata["n_lipids_upper"], membrane.metadata["n_lipids_lower"]))
    assert len(counts) == 1, f"the same geometry gave different membranes per kind: {counts}"


def test_orientation_and_embedding_stay_protein_specific():
    """Generalising the obstacle set must not make a ligand define a normal.

    A membrane normal is computed from a protein. Broadening *that* would be a
    different and wrong change, so the source still asks for PROTEIN there.
    """
    import inspect

    source = inspect.getsource(MembraneBuilder.run)
    orientation = source[source.index("has_protein = bool") :]
    assert "orient_protein(" in orientation
    assert "embed_protein(" in orientation
    # ... and the geometry no longer does.
    geometry = source[: source.index("has_protein = bool")]
    assert "component_by_kind(ComponentKind.PROTEIN)" not in geometry


# --------------------------------------------------------------------------
# M1 -- the count is the contract, in the configuration where it used to break


@requires_lfs_assets
@pytest.mark.slow
def test_the_explicit_count_holds_with_a_solute_present():
    """The coverage hole: this contract was only ever tested on an empty system."""
    _, membrane, _ = _build(ComponentKind.PROTEIN, requested=64)
    assert membrane.metadata["n_lipids_upper"] == 64
    assert membrane.metadata["n_lipids_lower"] == 64


@requires_lfs_assets
@pytest.mark.slow
def test_the_count_is_settled_after_the_last_removal_not_before_it():
    """Enforcing it earlier is what let a later filter violate it silently."""
    result, _, _ = _build(ComponentKind.PROTEIN, requested=64)

    order = [
        index
        for index, line in enumerate(result.log)
        if "clash filter" in line or "Explicit count" in line
    ]
    labels = [result.log[i] for i in order]
    trims = [i for i, line in zip(order, labels, strict=True) if "Explicit count" in line]
    filters = [i for i, line in zip(order, labels, strict=True) if "clash filter" in line]
    if trims and filters:
        assert min(trims) > max(filters), (
            "a filter that can remove lipids runs after the count was enforced"
        )


@requires_lfs_assets
def test_an_unmeetable_count_is_refused_with_the_number_that_would_work(monkeypatch):
    """D1: a shortfall is the user's decision, not a quiet difference.

    The removal is forced rather than constructed from geometry. The box is
    *sized from* the requested count plus the solute footprint, so a geometry
    that genuinely cannot hold its own count is delicate to build and would
    make this test a coin toss. What needs pinning is the contract: if a lipid
    is gone by the time the count is settled, the build stops and says so.
    """
    original = MembraneBuilder._filter_protein_clashes

    def strip_one(leaflet_sys, tree, label, min_dist, log, *, label_prefix="Protein clash filter"):
        original(leaflet_sys, tree, label, min_dist, log, label_prefix=label_prefix)
        keep = np.ones(int(leaflet_sys.metadata["n_lipids"]), dtype=bool)
        keep[0] = False
        MembraneBuilder._retain_lipids(leaflet_sys, keep)
        return 1

    monkeypatch.setattr(MembraneBuilder, "_filter_protein_clashes", staticmethod(strip_one))
    monkeypatch.setattr(MembraneBuilder, "_MIN_LIPIDS_PER_LEAFLET", 8)

    from tests.prerequisites import require_v4_entries

    require_v4_entries(["DPPC"], "charmm36m")
    system = _system_with(_cylinder(), ComponentKind.PROTEIN)
    with pytest.raises(ModuleConfigError) as excinfo:
        # Ask for exactly what the oversampled pool can supply, then take
        # lipids away at every filter so the pool cannot cover the request.
        MembraneBuilder().run(system, {"lipid_type": "DPPC", "n_lipids_per_leaflet": 70})

    message = str(excinfo.value)
    assert "leaflet requires" in message
    assert "survive the solute clash checks" in message
    assert "Review protein orientation" in message, (
        "an error the user cannot act on is not much better than a silent shortfall"
    )


# --------------------------------------------------------------------------
# Seating: the mechanism that replaced deletion


def _rod(x0: float, n: int = 20) -> np.ndarray:
    return np.column_stack([np.full(n, x0), np.zeros(n), np.linspace(0.0, 1.8, n)])


def _wall() -> np.ndarray:
    gy, gz = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-2, 2, 41))
    return np.column_stack([np.zeros(gy.size), gy.ravel(), gz.ravel()])


def _one_lipid_system(coords: np.ndarray) -> System:
    return System(
        structure=Structure(
            coordinates=coords.copy(),
            box_vectors=np.eye(3) * 10.0,
            atom_names=["C"] * len(coords),
            resnames=["DPPC"] * len(coords),
            resids=[1] * len(coords),
            elements=["C"] * len(coords),
        ),
        metadata={"n_lipids": 1, "lipid_sizes": [len(coords)]},
    )


@pytest.mark.parametrize("deficit_nm", [0.01, 0.03, 0.044, 0.08])
def test_a_lipid_inside_the_solute_is_moved_out_rather_than_deleted(deficit_nm):
    wall = _wall()
    system = _one_lipid_system(_rod(MIN_CONTACT - deficit_nm))

    unseated = _seat_lipids_against_solute(
        system,
        wall,
        target_contact=MIN_CONTACT + 0.15,
        max_shift=0.05,
        log=[],
        leaflet_label="probe",
        min_distance=MIN_CONTACT,
    )

    after = float(cKDTree(wall).query(system.coordinates, k=1)[0].min())
    assert unseated == 0
    assert after >= MIN_CONTACT
    assert system.num_atoms == 20, "seating moves whole lipids; it never removes atoms"


def test_seating_still_draws_a_distant_interface_lipid_in():
    """The half that already existed must survive the generalisation.

    Water-sized cavities at the solute interface nucleate pores, which is why
    drawing lipids in was added in the first place.
    """
    wall = _wall()
    system = _one_lipid_system(_rod(0.45))
    before = float(cKDTree(wall).query(system.coordinates, k=1)[0].min())

    _seat_lipids_against_solute(
        system,
        wall,
        target_contact=MIN_CONTACT + 0.15,
        max_shift=0.05,
        log=[],
        leaflet_label="probe",
        min_distance=MIN_CONTACT,
    )

    after = float(cKDTree(wall).query(system.coordinates, k=1)[0].min())
    assert after < before, "a lipid in a cavity should have been drawn in"
    assert after >= MIN_CONTACT, "and not drawn into the solute"


def test_a_bulk_lipid_far_from_the_solute_is_left_alone():
    wall = _wall()
    system = _one_lipid_system(_rod(2.0))
    before = system.coordinates.copy()

    _seat_lipids_against_solute(
        system,
        wall,
        target_contact=MIN_CONTACT + 0.15,
        max_shift=0.05,
        log=[],
        leaflet_label="probe",
        min_distance=MIN_CONTACT,
    )

    assert np.array_equal(system.coordinates, before)


def test_seating_reports_what_it_could_not_resolve():
    """A lipid with no radial direction to move along must be counted, not hidden."""
    # A rod whose closest approach is straight down the Z axis of a single
    # solute atom: dx and dy are both zero, so there is nowhere to push it.
    solute = np.asarray([[0.0, 0.0, 0.0]])
    system = _one_lipid_system(
        np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0.02, 0.5, 5)])
    )

    unseated = _seat_lipids_against_solute(
        system,
        solute,
        target_contact=MIN_CONTACT + 0.15,
        max_shift=0.05,
        log=[],
        leaflet_label="probe",
        min_distance=MIN_CONTACT,
    )
    assert unseated == 1

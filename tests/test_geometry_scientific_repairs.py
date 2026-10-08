import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.overlap import find_overlapping_atoms
from gmxbuilder.modules.solvation.solvate import SolvationBuilder


def test_append_only_neighbors_match_independent_periodic_distances():
    from gmxbuilder.geometry.neighbor_index import AppendOnlyNeighbors

    rng = np.random.default_rng(414)
    index = AppendOnlyNeighbors(boxsize=[5, 5, 0])
    fixed = []
    for _ in range(19):
        block = rng.uniform(0, 5, (7, 3))
        index.append(block)
        fixed.extend(block)
        query = rng.uniform(0, 5, (13, 3))
        query[0] = fixed[0]
        delta = query[:, None, :] - np.asarray(fixed)[None, :, :]
        delta[:, :, :2] -= 5 * np.round(delta[:, :, :2] / 5)
        expected = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)
        np.testing.assert_allclose(index.distances(query), expected, atol=1e-14)


@pytest.mark.parametrize("periodic", [False, True])
def test_exact_coincidences_are_overlaps(periodic):
    kwargs = {"box_dimensions": np.full(3, 3.0)} if periodic else {}
    mobile = np.array([[3.0, 0.0, 0.0]]) if periodic else np.zeros((1, 3))
    assert find_overlapping_atoms(mobile, np.zeros((1, 3)), **kwargs).tolist() == [True]


def test_construction_solvent_checks_remove_pore_and_colliding_whole_water():
    coords = np.array([[1, 1, 0], [1, 1, 4], [5, 5, 0], [5, 5, 4]], dtype=float)
    system = System(
        Structure(
            coords,
            np.diag([6.0, 6.0, 8.0]),
            atom_names=["C"] * 4,
            resnames=["POPC"] * 4,
            resids=[1, 1, 2, 2],
            elements=["C"] * 4,
        ),
        components=[Component("membrane", ComponentKind.MEMBRANE, np.arange(4))],
    )

    class KnownWaters(SolvationBuilder):
        def _fill_from_prebuilt(self, box_dims, water_model_name, water_model, seed):
            oxygen = np.array([[3, 3, 4], [3, 3, 1], [3, 3, 7], [1, 1, 2]], dtype=float)
            molecule = np.array([[0, 0, 0], [0.0757, 0, 0.0586], [-0.0757, 0, 0.0586]])
            return (oxygen[:, None, :] + molecule[None, :, :]).reshape(-1, 3), 4

    result = KnownWaters().run(system, {"box_padding": 2.0})
    solvent = result.system.component_by_kind(ComponentKind.SOLVENT)[0]
    assert solvent.metadata["n_molecules"] == 2
    waters = result.system.structure.coordinates[solvent.atom_indices].reshape(-1, 3, 3)
    assert not any(np.allclose(m[0], [3, 3, 4]) for m in waters)
    assert {tuple(m[0]) for m in waters} == {(3, 3, 1), (3, 3, 7)}
    assert not any(np.allclose(m[0], [1, 1, 2]) for m in waters)


def test_aa_and_cg_scores_use_the_interface_column_and_correct_ionization_states():
    from gmxbuilder.modules.martini3_bilayer.orientation import _TRANSFER
    from gmxbuilder.modules.membrane.orient import _WW_TRANSFER

    # Independent transcription of White laboratory Table 1, kcal/mol.
    expected = {
        "GLU": 2.02,
        "GLH": -0.01,
        "GLUP": -0.01,
        "VAL": 0.07,
        "MET": -0.23,
        "ASH": -0.07,
        "ASPP": -0.07,
        "HID": 0.17,
        "HIE": 0.17,
        "HSD": 0.17,
        "HSE": 0.17,
        "HIP": 0.96,
        "HSP": 0.96,
    }
    for table in (_WW_TRANSFER, _TRANSFER):
        for state, energy in expected.items():
            assert table[state] == pytest.approx(energy)

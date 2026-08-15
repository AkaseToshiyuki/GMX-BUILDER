from __future__ import annotations

import numpy as np

from gmxbuilder.modules.forcefield.hdb import HDBHydrogenAdder, _compute_h_positions


def test_hdb_resolves_previous_residue_control_atom(tmp_path):
    hdb = tmp_path / "aminoacids.hdb"
    hdb.write_text("ALA 1\n1 1 H N -C CA\n", encoding="utf-8")
    adder = HDBHydrogenAdder(hdb)

    names, coords, resnames, resids, chains = adder.add_hydrogens(
        ["C", "N", "CA"],
        np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        ["GLY", "ALA", "ALA"],
        [1, 2, 2],
        ["A", "A", "A"],
    )

    assert names[-1] == "H"
    assert resnames[-1] == "ALA"
    assert resids[-1] == 2
    assert chains[-1] == "A"
    direction = coords[-1] - coords[1]
    assert direction[0] < 0.0
    assert direction[1] < 0.0
    assert np.isclose(np.linalg.norm(direction), 0.101, atol=1e-8)


def test_hdb_does_not_cross_chain_boundaries(tmp_path):
    hdb = tmp_path / "aminoacids.hdb"
    hdb.write_text("ALA 1\n1 1 H N -C CA\n", encoding="utf-8")
    adder = HDBHydrogenAdder(hdb)

    names, coords, *_ = adder.add_hydrogens(
        ["C", "N", "CA"],
        np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        ["GLY", "ALA", "ALA"],
        [1, 1, 1],
        ["A", "B", "B"],
    )

    assert names[-1] == "H"
    direction = coords[-1] - coords[1]
    assert np.isclose(direction[1], 0.0, atol=1e-8)
    assert direction[0] < 0.0


def test_method_four_generates_tetrahedral_methyl_geometry():
    control = np.zeros(3)
    positions = _compute_h_positions(
        control,
        [np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.1, 0.0])],
        3,
        method=4,
    )

    directions = [
        (position - control) / np.linalg.norm(position - control) for position in positions
    ]
    axis = np.array([1.0, 0.0, 0.0])
    assert len(directions) == 3
    for direction in directions:
        assert np.isclose(np.dot(direction, axis), -1.0 / 3.0, atol=2e-3)
    for first in range(3):
        for second in range(first + 1, 3):
            assert np.isclose(np.dot(directions[first], directions[second]), -1.0 / 3.0, atol=2e-3)


def test_method_six_generates_distinct_tetrahedral_pair():
    positions = _compute_h_positions(
        np.zeros(3),
        [np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0])],
        2,
        method=6,
    )
    directions = [position / np.linalg.norm(position) for position in positions]

    assert len(directions) == 2
    assert np.isclose(np.dot(directions[0], directions[1]), -1.0 / 3.0, atol=2e-3)
    assert not np.allclose(positions[0], positions[1])

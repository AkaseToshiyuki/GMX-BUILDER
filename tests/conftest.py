"""Shared test fixtures."""

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System


@pytest.fixture
def empty_system():
    """An empty System with a 10 nm cubic box."""
    return System(
        structure=Structure(
            coordinates=np.empty((0, 3)),
            box_vectors=np.eye(3) * 10.0,
        ),
    )


@pytest.fixture
def simple_protein_structure():
    """A minimal 3-atom 'protein' structure."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.38, 0.0, 0.0],
            [0.76, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return Structure(
        coordinates=coords,
        box_vectors=np.eye(3) * 5.0,
        atom_names=["N", "CA", "C"],
        resnames=["ALA", "ALA", "ALA"],
        resids=[1, 1, 1],
        chain_ids=["A", "A", "A"],
        elements=["N", "C", "C"],
    )


@pytest.fixture
def small_pdb_file(tmp_path):
    """Create a small PDB file for testing."""
    content = """\
HEADER    TEST PROTEIN
CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.209   2.354   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.752   1.247  1.00  0.00           C
TER
END
"""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(content)
    return pdb_path

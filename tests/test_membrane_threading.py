"""Where SciPy's KD-tree threading helps, and where it actively hurts.

`cKDTree.query(workers=…)` parallelises inside SciPy's C extension, so it costs
nothing in Python and cannot change a result. It is still not free: each call
spawns a pool. That pays for a whole-leaflet query evaluated a few hundred
times, and it is a heavy loss for a query of one lipid's ~130 atoms run once
per lipid.

Measured on the validation system: asking for 48 workers on the per-lipid
queries took the protein-packing phase from 1.3 s to 7.2 s and made the whole
membrane step slower than single-threaded.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.geometry.relax import (
    relax_interleaflet_clashes_xy,
    rotate_lipids_away_from_external_clashes,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "src" / "gmxbuilder" / "modules" / "membrane" / "builder.py"


def _leaflets(n_lipids: int = 12, per_lipid: int = 8):
    rng = np.random.default_rng(7)
    box = 4.0
    upper = np.column_stack(
        [
            rng.random(n_lipids * per_lipid) * box,
            rng.random(n_lipids * per_lipid) * box,
            rng.random(n_lipids * per_lipid) * 0.8 + 1.5,
        ]
    )
    lower = np.column_stack(
        [
            rng.random(n_lipids * per_lipid) * box,
            rng.random(n_lipids * per_lipid) * box,
            rng.random(n_lipids * per_lipid) * 0.8 - 1.5,
        ]
    )
    sizes = [per_lipid] * n_lipids
    return upper, lower, sizes, box


# --------------------------------------------------------------------------
# Threading must not change an answer


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_interleaflet_relax_is_independent_of_worker_count(workers):
    upper, lower, sizes, box = _leaflets()
    reference = relax_interleaflet_clashes_xy(
        upper.copy(), lower.copy(), sizes, sizes, box_xy=box, workers=1
    )
    threaded = relax_interleaflet_clashes_xy(
        upper.copy(), lower.copy(), sizes, sizes, box_xy=box, workers=workers
    )
    assert np.array_equal(reference[0], threaded[0])
    assert np.array_equal(reference[1], threaded[1])


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_external_rotation_is_independent_of_worker_count(workers):
    upper, lower, sizes, box = _leaflets()
    reference, reference_clearance = rotate_lipids_away_from_external_clashes(
        upper.copy(), sizes, lower, box_xy=box, workers=1
    )
    threaded, threaded_clearance = rotate_lipids_away_from_external_clashes(
        upper.copy(), sizes, lower, box_xy=box, workers=workers
    )
    assert np.array_equal(reference, threaded)
    assert reference_clearance == threaded_clearance


def test_the_geometry_layer_stays_single_threaded_by_default():
    """A caller outside a task budget must behave exactly as before."""
    for function in (relax_interleaflet_clashes_xy, rotate_lipids_away_from_external_clashes):
        assert inspect.signature(function).parameters["workers"].default == 1


# --------------------------------------------------------------------------
# The regression guard: no thread pool inside a per-molecule loop


def _threaded_query_lines() -> list[tuple[int, bool]]:
    """Every cKDTree.query with workers=, and whether it sits inside a for loop."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    loops: list[tuple[int, int]] = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
    ]
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "query"):
            continue
        if not any(kw.arg == "workers" for kw in node.keywords):
            continue
        inside = any(start <= node.lineno <= end for start, end in loops)
        found.append((node.lineno, inside))
    return found


def test_no_threaded_query_sits_inside_a_loop():
    """One thread pool per lipid is the loss this guards against.

    A query of a single lipid's atoms cannot amortise spawning a pool, and the
    loop runs once per lipid per pass. Measured at 48 workers, this alone cost
    six seconds of a twelve-second step.
    """
    inside = [line for line, in_loop in _threaded_query_lines() if in_loop]
    assert inside == [], (
        f"builder.py lines {inside} ask for worker threads inside a loop; "
        "thread them only on whole-system queries evaluated once"
    )


def test_the_whole_system_queries_are_threaded_on_the_task_budget():
    """Otherwise the guard above could be satisfied by removing all of them.

    ``configured_task_threads`` is the deployment ceiling; a build must ask
    for ``current_task_threads``, which is this task's share of it.
    """
    assert len(_threaded_query_lines()) >= 3
    source = BUILDER.read_text(encoding="utf-8")
    assert "workers=current_task_threads()" in source
    assert "workers=configured_task_threads()" not in source


# --------------------------------------------------------------------------
# Occupancy binning


def test_the_occupancy_histogram_matches_the_loop_it_replaced():
    rng = np.random.default_rng(0)
    box, cells = 8.4, 8
    edges = np.linspace(-box / 2, box / 2, cells + 1)
    xy = rng.uniform(-box / 2, box / 2, (4000, 2))

    expected = np.zeros((cells, cells), dtype=int)
    for i in range(cells):
        for j in range(cells):
            expected[i, j] = (
                (xy[:, 0] >= edges[i])
                & (xy[:, 0] < edges[i + 1])
                & (xy[:, 1] >= edges[j])
                & (xy[:, 1] < edges[j + 1])
            ).sum()

    actual = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])[0].astype(int)
    assert np.array_equal(expected, actual)


def test_the_quality_check_no_longer_scans_every_atom_per_cell():
    from gmxbuilder.modules.membrane import builder

    source = inspect.getsource(builder._validate_membrane_quality)
    assert "np.histogram2d" in source
    assert "for j in range(n_cells)" not in source

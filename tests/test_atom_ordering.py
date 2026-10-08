"""Reordering atoms must renumber everything that refers to them.

Protein hydrogens are put beside the heavy atom they are bonded to. This is
a pure permutation -- no atom moves in space. Other components retain the
atom order required by their external parameter files.

The danger is not the permutation but everything that addresses atoms by
index: components, bonds, angles, exclusions, per-atom types. The first version
of this left topology bonds behind and produced a silently wrong disulfide.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.ordering import (
    AtomOrderingError,
    apply_permutation,
    order_hydrogens_after_parents,
)
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Bond, Topology


def _system(resnames, resids, names, coords, elements=None, kind=ComponentKind.PROTEIN):
    structure = Structure(
        coordinates=np.asarray(coords, dtype=float),
        box_vectors=np.eye(3) * 10.0,
        atom_names=list(names),
        resnames=list(resnames),
        resids=list(resids),
        chain_ids=[""] * len(names),
        elements=list(elements) if elements else [n[0] for n in names],
    )
    return System(
        structure=structure,
        components=[Component("TEST", kind, np.arange(len(names)))],
        metadata={},
    )


# --------------------------------------------------------------------------
# Hydrogens beside their parents


def _residue(resid, resname, heavy, hydrogens):
    """One residue written the wrong way round: heavy atoms, then hydrogens."""
    names, coords = [], []
    for name, position in heavy:
        names.append(name)
        coords.append(position)
    # Hydrogens go perpendicular to the heavy-atom chain. Placing them along
    # it puts one nearer a neighbouring heavy atom than its own parent, which
    # is a property of the fixture rather than of the code under test.
    for offset, (name, parent_index) in enumerate(hydrogens):
        names.append(name)
        base = heavy[parent_index][1]
        coords.append((base[0], base[1] + 0.10, base[2] + 0.002 * offset))
    return [resid] * len(names), [resname] * len(names), names, coords


def test_hydrogens_are_moved_next_to_the_atom_they_are_bonded_to():
    resids, resnames, names, coords = _residue(
        1,
        "ALA",
        [("N", (0, 0, 0)), ("CA", (0.15, 0, 0)), ("CB", (0.30, 0, 0))],
        [("H", 0), ("HA", 1), ("HB1", 2), ("HB2", 2)],
    )
    system = _system(resnames, resids, names, coords)
    summary = order_hydrogens_after_parents(system)

    assert summary["reordered"] is True
    assert summary["unassigned_hydrogens"] == 0
    assert system.structure.atom_names == ["N", "H", "CA", "HA", "CB", "HB1", "HB2"]


def test_a_residue_already_in_the_right_order_is_left_alone():
    system = _system(
        ["ALA"] * 4,
        [1] * 4,
        ["N", "H", "CA", "HA"],
        [(0, 0, 0), (0, 0.10, 0), (0.15, 0, 0), (0.15, 0.10, 0)],
    )
    assert order_hydrogens_after_parents(system)["reordered"] is False


def test_water_is_left_alone():
    """One heavy atom is already a contiguous group."""
    system = _system(
        ["SOL"] * 3,
        [1] * 3,
        ["OW", "HW1", "HW2"],
        [(0, 0, 0), (0.10, 0, 0), (0, 0.10, 0)],
        kind=ComponentKind.SOLVENT,
    )
    assert order_hydrogens_after_parents(system)["reordered"] is False


def test_a_hydrogen_with_no_heavy_atom_nearby_is_left_in_place():
    """A distant hydrogen is not attached to a guess."""
    resids, resnames, names, coords = _residue(
        1,
        "ALA",
        [("N", (0, 0, 0)), ("CA", (0.15, 0, 0))],
        [("H", 0)],
    )
    coords[-1] = (5.0, 5.0, 5.0)
    system = _system(resnames, resids, names, coords)
    summary = order_hydrogens_after_parents(system)
    assert summary.get("unassigned_hydrogens") == 1
    assert system.structure.atom_names[-1] == "H"


def test_reordering_moves_no_atom_in_space():
    resids, resnames, names, coords = _residue(
        1,
        "ALA",
        [("N", (0, 0, 0)), ("CA", (0.15, 0, 0)), ("CB", (0.30, 0, 0))],
        [("H", 0), ("HA", 1), ("HB1", 2)],
    )
    system = _system(resnames, resids, names, coords)
    before = {n: tuple(c) for n, c in zip(names, system.structure.coordinates, strict=True)}
    order_hydrogens_after_parents(system)
    after = {
        n: tuple(c)
        for n, c in zip(system.structure.atom_names, system.structure.coordinates, strict=True)
    }
    assert before == after


@pytest.mark.parametrize(
    "kind",
    [
        ComponentKind.LIGAND,
        ComponentKind.NUCLEIC_ACID,
        ComponentKind.MEMBRANE,
        ComponentKind.UNKNOWN,
    ],
)
def test_only_protein_atoms_are_reordered_in_a_mixed_system(kind):
    # Both residues would be reordered by the old implementation. Even a
    # protein-like residue name must not override the component's ownership.
    names = ["N", "CA", "H", "HA"]
    coords = [(0, 0, 0), (0.15, 0, 0), (0, 0.10, 0), (0.15, 0.10, 0)]
    system = _system(["ALA"] * 8, [1] * 4 + [2] * 4, names * 2, coords * 2)
    system.components = [
        Component("PROTEIN", ComponentKind.PROTEIN, np.arange(4)),
        Component("EXTERNAL", kind, np.arange(4, 8)),
    ]
    original = system.copy()

    summary = order_hydrogens_after_parents(system)

    assert summary["residues_reordered"] == 1
    assert system.structure.atom_names[:4] == ["N", "H", "CA", "HA"]
    assert system.structure.atom_names[4:] == names
    np.testing.assert_array_equal(system.coordinates[4:], original.coordinates[4:])
    assert order_hydrogens_after_parents(system)["reordered"] is False


# --------------------------------------------------------------------------
# Everything that addresses an atom by index


def test_a_permutation_renumbers_topology_bonds():
    """The defect: bonds kept pointing at whatever now occupied the index."""
    system = _system(
        ["CYS"] * 3,
        [1] * 3,
        ["SG", "CB", "CA"],
        [(0, 0, 0), (0.15, 0, 0), (0.30, 0, 0)],
    )
    system.topology = Topology(bonds=[Bond(i=0, j=2)], atom_count=3)
    apply_permutation(system, np.array([2, 1, 0]))

    assert system.structure.atom_names == ["CA", "CB", "SG"]
    bond = system.topology.bonds[0]
    assert {bond.i, bond.j} == {0, 2}
    # And they still join the same two atoms.
    joined = {system.structure.atom_names[bond.i], system.structure.atom_names[bond.j]}
    assert joined == {"SG", "CA"}


def test_a_permutation_renumbers_component_indices():
    system = _system(
        ["ALA"] * 3, [1] * 3, ["N", "CA", "CB"], [(0, 0, 0), (0.15, 0, 0), (0.30, 0, 0)]
    )
    system.components = [
        Component(name="P", kind=ComponentKind.PROTEIN, atom_indices=np.array([0, 2]))
    ]
    apply_permutation(system, np.array([2, 0, 1]))
    names = system.structure.atom_names
    assert {names[i] for i in system.components[0].atom_indices} == {"N", "CB"}


@pytest.mark.parametrize("bad", [np.array([0, 0, 1]), np.array([0, 1]), np.array([0, 1, 5])])
def test_a_permutation_that_is_not_a_bijection_is_refused(bad):
    system = _system(
        ["ALA"] * 3, [1] * 3, ["N", "CA", "CB"], [(0, 0, 0), (0.15, 0, 0), (0.30, 0, 0)]
    )
    with pytest.raises(AtomOrderingError):
        apply_permutation(system, bad)


def test_every_per_atom_structure_field_is_permuted():
    """A field added later and not permuted pairs wrong values with atoms."""
    from dataclasses import fields

    from gmxbuilder.core.ordering import _PER_ATOM_FIELDS

    declared = {
        f.name
        for f in fields(Structure)
        if f.name not in {"coordinates", "box_vectors", "source_info"}
    }
    assert declared == set(_PER_ATOM_FIELDS), (
        f"Structure fields not permuted by ordering.py: {declared - set(_PER_ATOM_FIELDS)}"
    )


def test_remap_covers_every_field_reindex_shifts():
    """`reindex` is the existing inventory of index-bearing topology fields."""
    source = Path(inspect.getfile(Topology)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Topology"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }

    def touched(name):
        return {
            node.attr
            for node in ast.walk(methods[name])
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }

    missing = touched("reindex") - touched("remap")
    assert missing == set(), f"remap does not renumber: {missing}"
    # atom_types is per-atom and needs permuting, which reindex has no reason to do.
    assert "atom_types" in touched("remap")

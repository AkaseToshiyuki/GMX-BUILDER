"""Explicit synthetic evidence for admission contracts, not physical validation."""

from functools import lru_cache

import numpy as np
from rdkit import Chem

from gmxbuilder.modules.membrane.local_conformations import (
    descriptor_definition,
    population_descriptors,
)
from gmxbuilder.modules.membrane.local_relaxation import evidence
from gmxbuilder.modules.membrane.v4_atom_selection import chemical_selection


@lru_cache(maxsize=32)
def molecule_definition(smiles="OCCCCC", name="toy"):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    elements = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
    names = tuple(f"{element}{index}" for index, element in enumerate(elements))
    bonds = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds())
    selection = chemical_selection(name, names, elements, bonds)
    definition = descriptor_definition(
        smiles,
        names,
        elements,
        bonds,
        tuple(map(int, selection["polar"])),
        tuple(map(int, selection["tails"])),
        selection["anchor"],
    )
    return selection["record"], definition


def local_fixture(smiles="OCCCCC", name="toy", seed=101, frames=2001):
    _, definition = molecule_definition(smiles, name)
    keys = population_descriptors(
        np.zeros((2, len(definition["atom_names"]), 3)),
        definition,
        np.array([True, False]),
        np.zeros(2),
    )
    rng = np.random.default_rng(seed)
    series = {}
    for key in keys:
        if key.startswith("shape:"):
            mean = {"q10": 0.8, "q50": 1.0, "q90": 1.2}[key.rsplit(":", 1)[1]]
            series[key] = mean + rng.normal(0, 0.002, frames)
        elif key.startswith("hydration:"):
            series[key] = 2 + rng.normal(0, 0.005, frames)
        elif key.startswith("bond:"):
            series[key] = 0.153 + rng.normal(0, 0.00005, frames)
        elif key.startswith("angle:"):
            series[key] = 110 + rng.normal(0, 0.05, frames)
        elif ":minus:" in key:
            series[key] = 0.2 + rng.normal(0, 0.002, frames)
        elif ":plus:" in key:
            series[key] = 0.3 + rng.normal(0, 0.002, frames)
    for key in keys:
        if ":trans:" in key:
            series[key] = (
                1
                - series[key.replace(":trans:", ":minus:")]
                - series[key.replace(":trans:", ":plus:")]
            )
    return evidence(np.arange(frames) * 100.0, series, definition)

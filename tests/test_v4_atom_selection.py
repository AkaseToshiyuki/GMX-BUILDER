"""Chemical selection invariance and hand-checkable graph-distance boundaries."""

import copy

import numpy as np
import pytest
from mdtraj.formats import XTCTrajectoryFile

from gmxbuilder.core.structure import Structure
from gmxbuilder.modules.membrane.v4_atom_selection import (
    chemical_selection,
    selections_by_lipid,
    validate_selections,
)
from gmxbuilder.modules.membrane.v4_trajectory import TrajectorySampler


def synthetic_selection(name="POPC"):
    """A six-heavy-atom toy graph for evidence-contract tests, not lipid physics."""
    return chemical_selection(
        name,
        ["O", "C1", "C2", "C3", "C4", "C5"],
        ["O", "C", "C", "C", "C", "C"],
        [(i, i + 1) for i in range(5)],
    )["record"]


def test_tail_excludes_two_bond_shells_and_anchor_is_polar():
    record = synthetic_selection()
    assert record["polar_atom_names"] == ["O"]
    assert record["tail_atom_names"] == ["C3", "C4", "C5"]
    assert record["anchor_atom_name"] == "O"


def test_missing_changed_and_mixed_selection_definitions_fail():
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    protocol = resolve_protocol("POPC", "charmm36m-lipid")
    original = {"atom_selections": [synthetic_selection()]}
    validate_selections(original, protocol)
    for field, value in [
        ("method", "unknown"),
        ("anchor_atom_name", "C5"),
        ("tail_atom_names", ["C1"]),
        ("sha256", "0" * 64),
    ]:
        modified = copy.deepcopy(original)
        modified["atom_selections"][0][field] = value
        with pytest.raises(ValueError):
            validate_selections(modified, protocol)
    with pytest.raises(ValueError):
        validate_selections({}, protocol)


def test_multi_species_selection_order_is_irrelevant_but_changes_fail():
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    protocol = resolve_protocol("POP3", "charmm36m-lipid")
    records = [synthetic_selection("POPC"), synthetic_selection("POP3")]
    first = selections_by_lipid({"atom_selections": records}, protocol)
    assert first == selections_by_lipid({"atom_selections": records[::-1]}, protocol)
    alternative = chemical_selection(
        "POP3",
        ["O", "C1", "C2", "C3", "C4", "C6"],
        ["O", "C", "C", "C", "C", "C"],
        [(i, i + 1) for i in range(5)],
    )["record"]
    assert first != selections_by_lipid({"atom_selections": [records[0], alternative]}, protocol)
    changed = copy.deepcopy(records)
    changed[1]["tail_atom_names"] = ["C4", "C5"]
    with pytest.raises(ValueError):
        selections_by_lipid({"atom_selections": changed}, protocol)
    with pytest.raises(ValueError):
        selections_by_lipid({"atom_selections": [records[0], records[0]]}, protocol)


def test_extended_trajectory_prefix_is_independent_of_endpoint_exemplar(tmp_path, monkeypatch):
    from gmxbuilder.modules.membrane import v4_trajectory

    record = synthetic_selection("CER16")
    names, elements, bonds = record["atom_names"], record["elements"], record["bonds"]
    monkeypatch.setattr(
        v4_trajectory, "ordered_graph", lambda *args: (elements, tuple(map(tuple, bonds)))
    )
    upper = np.array([[0, 0, 0.8 - i * 0.15] for i in range(6)])
    lower = upper.copy()
    lower[:, 2] *= -1
    xyz = np.vstack([upper, lower])
    box = np.eye(3) * 6
    structure = Structure(
        xyz.copy(),
        box,
        atom_names=names * 2,
        resnames=["CER16"] * 12,
        resids=[1] * 6 + [2] * 6,
        elements=elements * 2,
    )
    protocol = {"lipid": "CER16", "lipids_per_leaflet": 1, "composition": {"CER16": 100}}
    records = [(list(range(6)), True), (list(range(6, 12)), False)]
    first = TrajectorySampler(structure, records, {}, protocol, "charmm36m", "charmm36m")
    # An unrelated final conformer must not redefine measurements at old times.
    structure.coordinates[:] = np.random.default_rng(5).normal(size=(12, 3))
    later = TrajectorySampler(structure, records, {}, protocol, "charmm36m", "charmm36m")
    assert first.groups[0]["selection"] == later.groups[0]["selection"]
    coordinates = np.repeat(xyz[None, :, :], 31, axis=0).astype("float32")
    coordinates[:, :, 0] += np.arange(31)[:, None] * 0.0001
    for name, count in [("short", 11), ("long", 31)]:
        with XTCTrajectoryFile(str(tmp_path / f"{name}.xtc"), "w") as stream:
            stream.write(
                coordinates[:count],
                time=np.arange(count, dtype="float32") * 10,
                step=np.arange(count, dtype="int32"),
                box=np.repeat(box[None], count, axis=0),
            )
    old = [
        (time, first.metrics(box, blocks))
        for time, box, blocks in first.frames(tmp_path / "short.xtc")
    ]
    extended = [
        (time, later.metrics(box, blocks))
        for time, box, blocks in later.frames(tmp_path / "long.xtc")
    ]
    assert old == extended[: len(old)]


def test_supported_registry_graphs_have_coordinate_independent_definitions():
    import json
    from pathlib import Path

    from rdkit import Chem

    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    records = json.loads(
        (Path(__file__).parents[1] / "src/gmxbuilder/data/v4_protocols.json").read_text()
    )["records"]
    names = sorted({record["lipid"] for record in records if record["status"] == "candidate"})
    assert len(names) >= 80
    for name in names:
        molecule = Chem.AddHs(Chem.MolFromSmiles(LipidRegistry.get(name).smiles))
        elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atom_names = [f"{element}{i}" for i, element in enumerate(elements)]
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()]
        selected = chemical_selection(name, atom_names, elements, bonds)
        assert len(selected["tails"]) >= 3
        assert elements[selected["anchor"]] in {"O", "N", "P", "S"}

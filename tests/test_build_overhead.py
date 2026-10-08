"""Work a build does that is not science, and should not be paid for twice.

A profile of a 176048-atom membrane build found 28% of its wall time in the
step runner rather than in any module: serialising per-atom force-field types
as JSON, and rendering a full-system viewer PDB after every step whether or
not anyone would look at it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import AtomType, Topology


def _system(n_atoms: int = 64, atom_classes: list[str | None] | None = None) -> System:
    structure = Structure(
        coordinates=np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) / 10.0,
        box_vectors=np.eye(3) * 8.0,
        atom_names=[f"C{index}" for index in range(n_atoms)],
        resnames=["POPC"] * n_atoms,
        resids=list(range(1, n_atoms + 1)),
        chain_ids=[""] * n_atoms,
        elements=["C"] * n_atoms,
    )
    system = System(structure=structure, metadata={})
    classes = atom_classes or [
        None if index % 3 == 0 else f"K{index % 4}" for index in range(n_atoms)
    ]
    system.topology = Topology(atom_count=n_atoms, force_field="amber14sb")
    for index in range(n_atoms):
        system.topology.atom_types.append(
            AtomType(
                name=f"T{index % 5}",
                mass=12.011 + index % 3,
                charge=-0.1 * (index % 7),
                sigma=0.34,
                epsilon=0.36,
                atom_class=classes[index],
            )
        )
    return system


# --------------------------------------------------------------------------
# Per-atom types live in the compressed array file


def test_atom_types_are_not_written_as_json(tmp_path):
    """18.85 MB of the 35.9 MB written after the topology step was this list."""
    _system(128).save_checkpoint(tmp_path)
    stored = json.loads((tmp_path / "system.json").read_text())
    assert stored["topology"]["atom_types"] == []
    assert "atom_type_name" in np.load(tmp_path / "system.npz", allow_pickle=False).files


def test_atom_types_survive_a_round_trip_exactly(tmp_path):
    original = _system(128)
    original.save_checkpoint(tmp_path)
    restored = System.load_checkpoint(tmp_path)

    before = original.topology.atom_types
    after = restored.topology.atom_types
    assert len(after) == len(before)
    for one, other in zip(before, after, strict=True):
        assert (one.name, one.mass, one.charge, one.sigma, one.epsilon) == (
            other.name,
            other.mass,
            other.charge,
            other.sigma,
            other.epsilon,
        )


def test_an_absent_atom_class_stays_absent(tmp_path):
    """None and "" are different answers and the storage must not merge them."""
    original = _system(6, atom_classes=[None, "", "CT", None, "HC", ""])
    original.save_checkpoint(tmp_path)
    restored = System.load_checkpoint(tmp_path)
    assert [t.atom_class for t in restored.topology.atom_types] == [
        None,
        "",
        "CT",
        None,
        "HC",
        "",
    ]


def test_a_checkpoint_written_before_the_move_still_loads(tmp_path):
    """Old checkpoints carry the JSON list and must keep working."""
    original = _system(16)
    original.save_checkpoint(tmp_path)

    # Rewrite it the old way: types in JSON, absent from the npz.
    arrays = dict(np.load(tmp_path / "system.npz", allow_pickle=False))
    legacy = {key: value for key, value in arrays.items() if not key.startswith("atom_type_")}
    np.savez_compressed(tmp_path / "system.npz", **legacy)
    stored = json.loads((tmp_path / "system.json").read_text())
    stored["topology"]["atom_types"] = [
        {
            "name": t.name,
            "mass": t.mass,
            "charge": t.charge,
            "sigma": t.sigma,
            "epsilon": t.epsilon,
            "atom_class": t.atom_class,
        }
        for t in original.topology.atom_types
    ]
    (tmp_path / "system.json").write_text(json.dumps(stored))

    restored = System.load_checkpoint(tmp_path)
    assert len(restored.topology.atom_types) == 16
    assert restored.topology.atom_types[0].name == original.topology.atom_types[0].name


def test_a_system_without_a_topology_writes_no_type_arrays(tmp_path):
    structure = Structure(
        coordinates=np.zeros((3, 3)),
        box_vectors=np.eye(3) * 5.0,
        atom_names=["N", "CA", "C"],
        resnames=["ALA"] * 3,
        resids=[1, 1, 1],
    )
    System(structure=structure, metadata={}).save_checkpoint(tmp_path)
    assert "atom_type_name" not in np.load(tmp_path / "system.npz", allow_pickle=False).files
    assert System.load_checkpoint(tmp_path).num_atoms == 3


# --------------------------------------------------------------------------
# The viewer is rendered when it is asked for


def test_a_step_no_longer_writes_a_viewer_eagerly():
    """The interface never shows one for topology or export."""
    import inspect

    from gmxbuilder.pipeline.step_executor import StepRunner

    source = inspect.getsource(StepRunner.run_step)
    body = source[source.index("# ---- 5.") :]
    body = body[: body.index("return {")]
    assert "write_viewer_pdb" not in body
    assert "write_cg_viewer_pdb" not in body


def test_the_step_result_still_offers_a_viewer_url(tmp_path):
    """Rendering later must not make the viewer look unavailable."""
    from gmxbuilder.web import server

    public = server._public_step_result(
        "a" * 32,
        {"status": "ok", "step": "ions", "viewer_pdb_path": str(tmp_path / "viewer.pdb")},
    )
    assert public["viewer_pdb_url"] == f"/api/step/{'a' * 32}/ions/viewer.pdb"


def test_the_endpoint_renders_a_missing_viewer_from_the_checkpoint(tmp_path):
    from gmxbuilder.pipeline.step_executor import StepRunner
    from gmxbuilder.web.server import _render_step_viewer

    runner = StepRunner(str(tmp_path), pipeline_type="membrane-bilayer")
    step_dir = runner.step_dir("membrane")
    step_dir.mkdir(parents=True, exist_ok=True)
    _system(32).save_checkpoint(step_dir)

    target = step_dir / "viewer.pdb"
    assert not target.exists()
    assert _render_step_viewer(runner, "membrane", "membrane-bilayer") is True
    assert target.exists()
    written = target.read_text()
    assert sum(1 for line in written.splitlines() if line.startswith(("ATOM", "HETATM"))) == 32


def test_rendering_a_step_that_never_ran_reports_failure(tmp_path):
    from gmxbuilder.pipeline.step_executor import StepRunner
    from gmxbuilder.web.server import _render_step_viewer

    runner = StepRunner(str(tmp_path), pipeline_type="membrane-bilayer")
    assert _render_step_viewer(runner, "membrane", "membrane-bilayer") is False


# --------------------------------------------------------------------------
# Element derivation is a property of the names alone


def test_elements_are_derived_once_per_distinct_name_set():
    """A profile found 14.2 million str.upper() calls in one membrane build."""
    from gmxbuilder.modules.membrane.builder import _elements_for_atom_names

    _elements_for_atom_names.cache_clear()
    names = tuple(f"C{index}X" for index in range(40))
    for _ in range(500):
        _elements_for_atom_names(names)
    assert _elements_for_atom_names.cache_info().hits == 499


@pytest.mark.parametrize(
    ("name", "element"),
    [
        ("C16X", "C"),
        ("H16X", "H"),
        ("P31", "P"),
        ("O11", "O"),
        ("N31", "N"),
        ("SG", "S"),
        ("CL", "Cl"),
        ("NA", "Na"),
        ("MG", "Mg"),
        ("BR", "Br"),
        ("ZN", "Zn"),
        ("FE", "Fe"),
        # A lipid CA is an alpha carbon, never calcium.
        ("CA", "C"),
    ],
)
def test_element_symbols_are_unchanged(name, element):
    from gmxbuilder.modules.membrane.builder import _elements_for_atom_names

    assert _elements_for_atom_names((name,))[0] == element

from pathlib import Path

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield import compatibility
from gmxbuilder.modules.forcefield.lipid21_backend import (
    lipid21_capability,
    lipid21_lipids,
    load_lipid21_geometry,
)
from gmxbuilder.modules.forcefield.lipid_policy import (
    amber_lipid_backend,
    amber_lipid_backend_candidates,
)
from gmxbuilder.modules.membrane.equilibrated_library import lipid_parameter_family
from gmxbuilder.modules.membrane.lipid_orientation import infer_lipid_orientation


def test_exact_lipid21_inventory_and_nonester_exclusion():
    assert len(lipid21_lipids()) == 39
    assert lipid21_capability("POPC")[0]
    assert lipid21_capability("CHOL")[0]
    assert not lipid21_capability("PPCPL")[0]


def test_amber_backend_priority_and_coherent_fallback():
    assert amber_lipid_backend(["POPC"])[0] == "lipid21"
    assert amber_lipid_backend(["POPC", "CHOL"])[0] == "lipid21"
    backend, reason = amber_lipid_backend(["POPC", "DPPE"])
    assert backend == "gaff2"
    assert "Lipid21 NPT library is unavailable" in reason
    assert amber_lipid_backend(["POPC", "POPI"])[0] == "gaff2"
    assert amber_lipid_backend(["POPC", "GM1"])[0] is None


def test_amber_candidates_include_exact_and_gaff_backends():
    assert amber_lipid_backend_candidates(["POPC", "CHOL"]) == ("lipid21",)


def test_exact_lipid21_membrane_can_explicitly_switch_to_gaff2(monkeypatch):
    monkeypatch.setattr(compatibility, "gaff_available", lambda: True)
    system = System(Structure(
        coordinates=np.empty((0, 3)),
        box_vectors=np.eye(3),
    ))

    report = compatibility.compatibility_report(
        system, "amber14sb", ["POPC"],
    )
    enabled = {
        option["value"] for option in report["lipid_options"]
        if option["enabled"]
    }

    assert enabled == {"lipid21", "gaff2"}


def test_lipid21_geometry_is_unique_and_amphiphilic():
    for name in lipid21_lipids():
        coordinates, atom_names = load_lipid21_geometry(name)
        assert coordinates.shape == (len(atom_names), 3)
        assert len(atom_names) == len(set(atom_names))
        assert infer_lipid_orientation(coordinates, atom_names).separation >= 0.15


def test_geometry_dispatches_to_exact_lipid21():
    expected, expected_names = load_lipid21_geometry("POPC")
    coordinates, atom_names = build_rdkit_lipid_geometry(
        "POPC", "unused", force_field="amber14sb", lipid_ff="lipid21"
    )
    assert atom_names == expected_names
    assert coordinates.shape == expected.shape


def test_lipid21_topology_writer_preserves_coordinate_order(tmp_path: Path):
    coordinates, atom_names = load_lipid21_geometry("POPC")
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 8.0,
        atom_names=atom_names,
        resnames=["POPC"] * len(atom_names),
        resids=[1] * len(atom_names),
        elements=[next(char for char in name if char.isalpha()) for name in atom_names],
    )
    topology = tmp_path / "topol.top"
    TopologyWriter(
        "amber14sb",
        {"protein": "amber14sb", "lipid_ff": "lipid21", "water_model": "tip3p"},
    ).write_top(structure, topology)
    assert '#include "lipid21_atomtypes.itp"' in topology.read_text()
    lipid_itp = (tmp_path / "POPC.itp").read_text()
    assert "Explicit [pairs] preserve Lipid21-specific 1-4 scaling" in lipid_itp
    assert "#ifdef POSRES" in lipid_itp
    assert "POSRES_FC_LIPID" in lipid_itp


def test_lipid21_uses_separate_strict_library_namespace():
    assert lipid_parameter_family("amber14sb", "lipid21") == "amber-lipid21"
    assert lipid_parameter_family("amber14sb", "gaff2") == "amber-gaff2"

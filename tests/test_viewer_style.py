"""Display connectivity is bounded by molecular identity, not spatial contacts."""

import base64
import gzip
import json

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Bond, Topology
from gmxbuilder.geometry.molecular_identity import itp_graph
from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_itp_path, load_lipid21_geometry
from gmxbuilder.web.server_parts.viewer_data import build_viewer, cached_viewer, checkpoint_revision


def lipid_pair():
    coords, names = load_lipid21_geometry("POPC")
    _, elements, reference = itp_graph(lipid21_itp_path("POPC"))
    n = len(names)
    structure = Structure(
        np.vstack([coords, coords + [0.05, 0, 0]]),
        np.eye(3) * 8,
        atom_names=list(names) * 2,
        elements=list(elements) * 2,
        resnames=["POPC"] * (2 * n),
        resids=[1] * n + [2] * n,
        chain_ids=["L"] * (2 * n),
    )
    system = System(structure)
    system.add_component(Component("Lipids", ComponentKind.MEMBRANE, np.arange(2 * n)))
    return system, reference


def test_geometry_sticks_match_independent_lipid_itp_without_intermolecular_bonds(tmp_path):
    system, reference = lipid_pair()
    system.save_checkpoint(tmp_path)
    original = checkpoint_revision(tmp_path)
    data = json.loads(gzip.decompress(build_viewer(tmp_path).read_bytes()))
    selected = np.frombuffer(base64.b64decode(data["original_indices"]), dtype="<u4")
    actual = {tuple(sorted((int(selected[a]), int(selected[b])))) for a, b in data["bonds"]}
    n = system.num_atoms // 2
    expected = {
        (a + offset, b + offset)
        for offset in [0, n]
        for a, b in reference
        if system.structure.elements[a] != "H" and system.structure.elements[b] != "H"
    }
    assert actual == expected
    assert data["bond_sources"] == {"topology": 0, "residue_geometry": len(expected)}
    assert checkpoint_revision(tmp_path) == original
    assert System.load_checkpoint(tmp_path).topology is None


def test_components_and_chains_remain_separate_even_when_residue_numbers_repeat(tmp_path):
    system, reference = lipid_pair()
    n = system.num_atoms // 2
    system.structure.resids = [1] * system.num_atoms
    system.structure.chain_ids = ["A"] * n + ["B"] * n
    system.save_checkpoint(tmp_path)
    data = json.loads(gzip.decompress(build_viewer(tmp_path).read_bytes()))
    selected = np.frombuffer(base64.b64decode(data["original_indices"]), dtype="<u4")
    assert all((selected[a] < n) == (selected[b] < n) for a, b in data["bonds"])
    system.structure.chain_ids = ["A"] * system.num_atoms
    system.components = [
        Component(str(i), ComponentKind.MEMBRANE, np.arange(i * n, (i + 1) * n)) for i in range(2)
    ]
    system.save_checkpoint(tmp_path)
    data = json.loads(gzip.decompress(build_viewer(tmp_path).read_bytes()))
    assert all((selected[a] < n) == (selected[b] < n) for a, b in data["bonds"])


def test_explicit_graph_and_cg_beads_never_gain_geometry_bonds(tmp_path):
    system, _ = lipid_pair()
    system.metadata["resolution"] = "coarse-grained"
    system.topology = Topology(bonds=[Bond(0, 1)])
    # Select two heavy beads regardless of the template's atom order.
    system.structure.elements = ["C"] * system.num_atoms
    system.save_checkpoint(tmp_path)
    data = json.loads(gzip.decompress(build_viewer(tmp_path).read_bytes()))
    assert data["bonds"] == [[0, 1]]
    assert data["bond_sources"] == {"topology": 1, "residue_geometry": 0}
    assert data["resolution"] == "coarse-grained"


def test_old_viewer_schema_is_regenerated_without_changing_checkpoint(tmp_path):
    system, _ = lipid_pair()
    system.save_checkpoint(tmp_path)
    build_viewer(tmp_path)
    index = tmp_path / "display-index.json"
    record = json.loads(index.read_text())
    revision = record["revision"]
    record["schema"] = 2
    index.write_text(json.dumps(record))
    assert cached_viewer(tmp_path) is None
    data = json.loads(gzip.decompress(build_viewer(tmp_path).read_bytes()))
    assert data["schema"] == 3 and data["revision"] == revision

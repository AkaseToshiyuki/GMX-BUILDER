"""Checkpoint round-trip tests for complete per-atom state."""

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System


def test_checkpoint_preserves_optional_per_atom_fields(tmp_path):
    structure = Structure(
        coordinates=np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]),
        box_vectors=np.diag([4.0, 5.0, 6.0]),
        atom_names=["CA", "ZN"],
        resnames=["ALA", "ZN"],
        resids=[1, 2],
        chain_ids=["A", "B"],
        segids=["PROA", "IONB"],
        elements=["C", "ZN"],
        occupancies=[0.55, 0.85],
        tempfactors=[12.5, 31.25],
    )
    system = System(
        structure=structure,
        components=[
            Component("PROTEIN", ComponentKind.PROTEIN, np.array([0])),
            Component("IONS", ComponentKind.IONS, np.array([1])),
        ],
        metadata={"seed": 17, "source": "roundtrip-test"},
    )

    checkpoint = tmp_path / "input"
    system.save_checkpoint(checkpoint)
    loaded = System.load_checkpoint(checkpoint)

    assert np.array_equal(loaded.structure.coordinates, structure.coordinates)
    assert np.array_equal(loaded.structure.box_vectors, structure.box_vectors)
    assert loaded.structure.segids == structure.segids
    assert loaded.structure.occupancies == structure.occupancies
    assert loaded.structure.tempfactors == structure.tempfactors
    assert loaded.metadata == system.metadata
    assert [component.name for component in loaded.components] == ["PROTEIN", "IONS"]

    # Checkpoints created before v0.6.6 do not contain the optional fields.
    npz_path = checkpoint / "system.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        legacy_arrays = {
            key: arrays[key]
            for key in (
                "coordinates", "box_vectors", "atom_names", "resnames",
                "resids", "chain_ids", "elements",
            )
        }
    np.savez_compressed(npz_path, **legacy_arrays)
    legacy_loaded = System.load_checkpoint(checkpoint)
    assert legacy_loaded.structure.segids == ["", ""]
    assert legacy_loaded.structure.occupancies == [1.0, 1.0]
    assert legacy_loaded.structure.tempfactors == [0.0, 0.0]


def test_checkpoint_preserves_full_gromacs_names_without_truncation(tmp_path):
    structure = Structure(
        coordinates=np.array([[0.1, 0.2, 0.3]]),
        box_vectors=np.diag([4.0, 5.0, 6.0]),
        atom_names=["C1234"],
        resnames=["20AHC"],
        resids=[1],
        chain_ids=["CHAIN_A"],
        segids=["MEMBRANE_A"],
        elements=["C"],
    )
    checkpoint = tmp_path / "membrane"

    System(
        structure=structure,
        metadata={"selected_lipid_names": ["20AHC"]},
    ).save_checkpoint(checkpoint)
    loaded = System.load_checkpoint(checkpoint)

    assert loaded.structure.atom_names == ["C1234"]
    assert loaded.structure.resnames == ["20AHC"]
    assert loaded.structure.chain_ids == ["CHAIN_A"]
    assert loaded.structure.segids == ["MEMBRANE_A"]


def test_legacy_checkpoint_rejects_detectably_truncated_lipid_names(tmp_path):
    import json

    system = System(
        structure=Structure(
            coordinates=np.array([[0.1, 0.2, 0.3]]),
            box_vectors=np.eye(3) * 4.0,
            atom_names=["C1"],
            resnames=["20AH"],
            resids=[1],
        ),
        metadata={"selected_lipid_names": ["20AHC"]},
    )
    checkpoint = tmp_path / "legacy"
    system.save_checkpoint(checkpoint)

    json_path = checkpoint / "system.json"
    metadata = json.loads(json_path.read_text())
    metadata.pop("checkpoint_schema_version", None)
    json_path.write_text(json.dumps(metadata))
    npz_path = checkpoint / "system.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        legacy = {key: arrays[key] for key in arrays.files}
    legacy["resnames"] = np.asarray(["20AH"], dtype="U4")
    np.savez_compressed(npz_path, **legacy)

    with np.testing.assert_raises_regex(
        ValueError, "legacy checkpoint.*20AHC.*re-run"
    ):
        System.load_checkpoint(checkpoint)


def test_every_builtin_lipid_identifier_survives_checkpoint_roundtrip(tmp_path):
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    names = LipidRegistry.list_builtin()
    structure = Structure(
        coordinates=np.zeros((len(names), 3)),
        box_vectors=np.eye(3) * 10.0,
        atom_names=["C1"] * len(names),
        resnames=names,
        resids=list(range(1, len(names) + 1)),
    )
    checkpoint = tmp_path / "all-lipids"

    System(
        structure=structure,
        metadata={"selected_lipid_names": names},
    ).save_checkpoint(checkpoint)

    assert System.load_checkpoint(checkpoint).structure.resnames == names

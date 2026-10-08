"""The deposited identity must survive whole-pipeline atom transformations."""

import copy

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.ordering import AtomOrderingError, apply_permutation
from gmxbuilder.core.structure import PER_ATOM_FIELDS, Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.ions.add_ions import IonBuilder
from gmxbuilder.modules.modifications.processor import _remap_system_atoms
from gmxbuilder.modules.nucleic_acid.native import _make_polymer_molecules_contiguous, _subset


def tagged_structure():
    return Structure(
        np.array([[0.0, 0.0, 0.0], [0.13, 0.0, 0.0], [0.26, 0.0, 0.0], [0.39, 0.0, 0.0]]),
        np.eye(3) * 5,
        atom_names=["CA", "P", "O3'", "H"],
        elements=["C", "P", "O", "H"],
        resnames=["ALA", "DA", "DA", "ALA"],
        resids=[1, 2, 2, 1],
        chain_ids=["A", "D", "D", "A"],
        source_ids=[f"original:{i}" for i in range(4)],
        source_info={
            "sha256": "test-source",
            "atoms": {f"original:{i}": {"row": i} for i in range(4)},
        },
    )


def assert_mapping(original, changed, order):
    for name in PER_ATOM_FIELDS:
        assert len(getattr(changed, name)) == changed.num_atoms
        assert getattr(changed, name) == [getattr(original, name)[i] for i in order]
    np.testing.assert_array_equal(changed.coordinates, original.coordinates[order])
    assert changed.source_info == original.source_info


def assert_surviving_heavy_sources(original, changed, *, tolerance=0.0):
    """Check identity, element and internal geometry against the original input.

    Allow a rigid placement of the whole solute, and optional GRO rounding.
    Newly constructed atoms have no deposited identity.
    """
    from scipy.spatial.distance import pdist

    original.validate_atom_fields()
    changed.validate_atom_fields()
    before = {uid: i for i, uid in enumerate(original.source_ids) if uid}
    after = {uid: i for i, uid in enumerate(changed.source_ids) if uid}
    assert before and before.keys() == after.keys()
    assert sum(bool(uid) for uid in changed.source_ids) == len(before)
    for uid, i in before.items():
        assert original.elements[i] == changed.elements[after[uid]]
        assert changed.source_info["atoms"][uid] == original.source_info["atoms"][uid]
    np.testing.assert_allclose(
        pdist(original.coordinates[list(before.values())]),
        pdist(changed.coordinates[[after[uid] for uid in before]]),
        atol=max(tolerance, 1e-12),
        rtol=0,
    )


@pytest.mark.parametrize("seed", range(5))
def test_permutations_preserve_values_and_are_reversible(seed):
    original = tagged_structure()
    system = System(original.copy())
    permutation = np.random.default_rng(seed).permutation(original.num_atoms)
    apply_permutation(system, permutation)
    assert_mapping(original, system.structure, permutation)
    apply_permutation(system, np.argsort(permutation))
    assert_mapping(original, system.structure, np.arange(original.num_atoms))


def test_invalid_field_fails_before_permutation_or_checkpoint_write(tmp_path):
    system = System(tagged_structure())
    system.structure.source_ids.pop()
    before = copy.deepcopy(system.structure.__dict__)
    with pytest.raises(AtomOrderingError, match="source_ids"):
        apply_permutation(system, np.array([3, 2, 1, 0]))
    np.testing.assert_array_equal(system.coordinates, before["coordinates"])
    assert system.structure.atom_names == before["atom_names"]
    with pytest.raises(ValueError, match="source_ids"):
        system.save_checkpoint(tmp_path / "checkpoint")
    assert not (tmp_path / "checkpoint").exists()
    from gmxbuilder.io.input_document import write_input

    with pytest.raises(ValueError, match="source_ids"):
        write_input(system.structure, tmp_path / "canonical.input.npz")
    assert not (tmp_path / "canonical.input.npz").exists()


def test_deletion_nucleic_subset_and_checkpoint_preserve_sources(tmp_path):
    original = tagged_structure()
    system = System(original.copy())
    _remap_system_atoms(system, [0, 2])
    assert_mapping(original, system.structure, [0, 2])
    assert_mapping(original, _subset(original, [1, 2]), [1, 2])
    system.save_checkpoint(tmp_path / "checkpoint")
    assert_mapping(original, System.load_checkpoint(tmp_path / "checkpoint").structure, [0, 2])


def test_polymer_regrouping_carries_sources_with_atoms():
    original = tagged_structure()
    system = System(
        original.copy(),
        components=[
            Component("protein", ComponentKind.PROTEIN, np.array([0, 3])),
            Component("DNA", ComponentKind.NUCLEIC_ACID, np.array([1, 2])),
        ],
    )
    _make_polymer_molecules_contiguous(system)
    assert_mapping(original, system.structure, [0, 3, 1, 2])


def test_ion_removal_keeps_solute_provenance():
    original = tagged_structure()
    system = System(original.copy())
    kept = IonBuilder._remove_atoms(system, [1, 2], waters_removed=0)
    assert_mapping(original, kept.structure, [0, 3])


def test_merge_preserves_both_documents_without_aliasing(tmp_path):
    left = tagged_structure()
    right = tagged_structure()
    right.source_ids = [uid.replace("original", "second") for uid in right.source_ids]
    right.source_info = {
        "sha256": "second-source",
        "atoms": {uid: {"row": i} for i, uid in enumerate(right.source_ids)},
    }
    merged = System(left).merge(System(right))
    merged.structure.validate_atom_fields()
    assert merged.structure.source_ids == left.source_ids + right.source_ids
    assert merged.structure.source_info["documents"] == [left.source_info, right.source_info]
    merged.save_checkpoint(tmp_path / "merged")
    loaded = System.load_checkpoint(tmp_path / "merged")
    assert loaded.structure.source_ids == merged.structure.source_ids
    assert loaded.structure.source_info == merged.structure.source_info
    merged.structure.source_info["documents"][0]["atoms"].clear()
    assert left.source_info["atoms"]


def test_native_mapping_accepts_format_rounding_but_rejects_moved_heavy_atoms():
    from gmxbuilder.core.exceptions import ModuleConfigError
    from gmxbuilder.modules.nucleic_acid.native import _restore_source_identity

    source = tagged_structure().take([0])
    processed = source.copy()
    processed.source_ids = [""]
    processed.coordinates += 0.00054
    _restore_source_identity(source, processed)
    assert processed.source_ids == source.source_ids
    processed.coordinates += 0.01
    with pytest.raises(ModuleConfigError, match="heavy atom"):
        _restore_source_identity(source, processed)

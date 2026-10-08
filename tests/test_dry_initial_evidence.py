"""Independent saved-coordinate checks and rejection of unverifiable starts."""

import copy
import hashlib

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.modules.membrane.initial_water import (
    initial_water_evidence,
    initial_water_evidence_valid,
)
from tests.dry_initial_fixture import dry_initial_fixture


def test_saved_initials_require_every_water_site_outside_full_envelope(tmp_path):
    structure = Structure(
        coordinates=np.array([[0, 0, 2], [1, 1, 6], [2, 2, 1.9], [2, 2, 1.95], [2, 2, 1.85]]),
        atom_names=["C", "C", "OW", "HW1", "HW2"],
        resnames=["POPC", "POPC", "SOL", "SOL", "SOL"],
        resids=[1, 2, 3, 3, 3],
        elements=["C", "C", "O", "H", "H"],
        box_vectors=np.eye(3) * 8,
    )
    for name in ("solvated.gro", "ionized.gro"):
        GROWriter.write(structure, tmp_path / name)
    evidence = initial_water_evidence(tmp_path)
    assert initial_water_evidence_valid(evidence)
    for name, record in evidence["files"].items():
        assert record["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert record["lipid_z_bounds_nm"] == [2, 6]
    # Oxygen stays outside, but a hydrogen enters the full slab.
    saved = GROReader().read(tmp_path / "ionized.gro")
    saved.coordinates[3, 2] = 2.02
    GROWriter.write(saved, tmp_path / "ionized.gro")
    with pytest.raises(ModuleConfigError, match="solvent sites"):
        initial_water_evidence(tmp_path)


@pytest.mark.parametrize("defect", ["missing", "wet", "hash", "nan", "count", "bool"])
def test_invalid_initial_evidence_is_rejected(defect):
    evidence = copy.deepcopy(dry_initial_fixture())
    record = evidence["files"]["solvated.gro"]
    if defect == "missing":
        evidence = None
    elif defect == "wet":
        record["water_sites_in_membrane"] = 1
    elif defect == "hash":
        record["sha256"] = "unknown"
    elif defect == "nan":
        record["lipid_z_bounds_nm"][0] = float("nan")
    elif defect == "count":
        record["lipid_atoms"] += 1
    else:
        record["water_sites_in_membrane"] = False
    assert not initial_water_evidence_valid(evidence)

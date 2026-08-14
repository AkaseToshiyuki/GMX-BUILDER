"""Regression tests for defensive final-system verification."""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.modules.verify.system_verify import SystemVerificationModule


def _system(atom_count: int) -> System:
    structure = Structure(
        coordinates=np.zeros((atom_count, 3), dtype=float),
        box_vectors=np.diag([3.0, 3.0, 3.0]),
        atom_names=["CA"] * atom_count,
        resnames=["ALA"] * atom_count,
        resids=list(range(1, atom_count + 1)),
        chain_ids=["A"] * atom_count,
        elements=["C"] * atom_count,
    )
    return System(
        structure=structure,
        components=[
            Component(
                "PROTEIN",
                ComponentKind.PROTEIN,
                np.arange(atom_count, dtype=int),
                {},
            )
        ],
    )


def test_verifier_creates_output_directory(tmp_path):
    output = tmp_path / "new" / "verification"
    result = SystemVerificationModule().run(_system(1), {"output_dir": output})
    assert result.success is True
    assert (output / "preview.pdb").is_file()
    assert (output / "verification_metrics.json").is_file()


def test_old_gro_with_different_atom_count_is_reported_not_crashed(tmp_path):
    checked = _system(2)
    output = tmp_path / "verify"
    output.mkdir()
    GROWriter.write(_system(1).structure, output / "input.gro")

    result = SystemVerificationModule().run(checked, {"output_dir": output})

    assert result.success is False
    assert any("GRO atom count differs" in line for line in result.log)

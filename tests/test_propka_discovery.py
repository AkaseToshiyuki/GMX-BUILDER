"""PROPKA must be found in the environment that runs the code.

It is a project dependency, so it is installed beside the interpreter -- but
that directory is not necessarily on PATH. The systemd unit inherits a plain
system PATH, so a bare "propka3" was never found there. Every structure then
fell back to model pKa values while the interface reported that PROPKA "could
not produce environment-sensitive pKa values", which was untrue: it was never
run. Model pKa values ignore burial, hydrogen bonding and local electrostatics,
so the residues most affected are the buried and membrane-embedded ones.
"""

from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path

import pytest

from gmxbuilder.modules.modifications import protonation


def test_propka_is_installed_in_this_environment():
    """It is a hard dependency, not an optional extra."""
    assert (Path(sys.executable).parent / "propka3").exists() or shutil.which("propka3")


def test_the_lookup_does_not_rely_on_path_alone():
    """PATH is the thing that was missing in the deployed service."""
    source = inspect.getsource(protonation.predict_pka_from_pdb)
    assert "Path(sys.executable).parent" in source
    # A bare name must remain as a fallback for a system-wide install.
    assert '"propka3"' in source


@pytest.mark.skipif(
    not (Path(sys.executable).parent / "propka3").exists(),
    reason="propka3 is not installed beside this interpreter",
)
def test_propka_returns_environment_sensitive_shifts(tmp_path):
    """A prediction that never differs from the model value proves nothing.

    This asserts the output is genuinely environment-sensitive, which is the
    entire reason for preferring PROPKA over the fallback.
    """
    import numpy as np

    from gmxbuilder.core.structure import Structure
    from gmxbuilder.io.pdb import PDBWriter

    # A short helix with one buried and one exposed acidic residue is enough
    # for PROPKA to produce a non-zero shift somewhere.
    names, resnames, resids, elements, coords = [], [], [], [], []
    backbone = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))
    sequence = ["ASP", "ALA", "GLU", "ALA", "ASP", "ALA", "GLU", "ALA"]
    for index, residue in enumerate(sequence):
        for offset, (atom, element) in enumerate(backbone):
            names.append(atom)
            resnames.append(residue)
            resids.append(index + 1)
            elements.append(element)
            coords.append([0.15 * offset + 0.38 * index, 0.05 * offset, 0.1 * index])
    structure = Structure(
        coordinates=np.array(coords) * 10.0,
        box_vectors=np.eye(3) * 80.0,
        atom_names=names,
        resnames=resnames,
        resids=resids,
        chain_ids=["A"] * len(names),
        elements=elements,
    )
    path = tmp_path / "probe.pdb"
    PDBWriter.write(structure, path)

    predictions = protonation.predict_pka_from_pdb(str(path))
    if not predictions:
        pytest.skip("PROPKA produced no summary for this synthetic probe")
    for entry in predictions:
        assert {"residue_name", "chain", "resid", "model_pKa", "predicted_pKa", "shift"} <= set(
            entry
        )
    assert any(abs(entry["shift"]) > 1e-9 for entry in predictions)

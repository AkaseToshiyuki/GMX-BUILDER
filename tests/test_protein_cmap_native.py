"""Compare generated protein corrections to native pdb2gmx, without running MD."""

import os
import re
import subprocess

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.rtp_parser import _force_field_path
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.runtime.hardware import find_gromacs_executable
from tests.test_terminal_chemistry import _two_residue_system


def section(text, name):
    current = ""
    result = []
    for raw in text.splitlines():
        line = raw.partition(";")[0].strip()
        if line.startswith("["):
            current = line.strip("[] ")
        elif current == name and line and not line.startswith("#"):
            result.append(tuple(line.split()))
    return result


@pytest.mark.slow
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
@pytest.mark.parametrize("sequence", ["AAA", "AGPA"])
def test_cmap_atoms_and_functions_match_native_pdb2gmx(tmp_path, force_field, sequence):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    gmx = find_gromacs_executable()
    if not gmx:
        pytest.skip("Native GROMACS is required for independent topology comparison")
    molecule = Chem.AddHs(Chem.MolFromFASTA(sequence))
    assert AllChem.EmbedMolecule(molecule, randomSeed=872) == 0
    Chem.MolToPDBFile(molecule, str(tmp_path / "input.pdb"))
    (tmp_path / f"{force_field}.ff").symlink_to(_force_field_path(force_field))
    native = subprocess.run(
        [
            gmx,
            "pdb2gmx",
            "-f",
            "input.pdb",
            "-o",
            "native.gro",
            "-p",
            "native.top",
            "-ff",
            force_field,
            "-water",
            "none",
            "-ignh",
        ],
        cwd=tmp_path,
        env={**os.environ, "GMXLIB": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert native.returncode == 0, native.stdout + native.stderr
    structure = GROReader().read(tmp_path / "native.gro")
    structure.chain_ids = ["A"] * structure.num_atoms
    path = tmp_path / "builder.itp"
    TopologyWriter(force_field)._write_protein_itp_for_indices(
        structure, path, list(range(structure.num_atoms))
    )
    original = (tmp_path / "native.top").read_text()
    expected = section(original, "cmap")
    assert len(expected) == len(sequence) - 2
    assert section(path.read_text(), "cmap") == expected
    # Native preprocessing independently resolves the emitted CMAP parameter type.
    head = re.split(r"\[\s*moleculetype\s*\]", original, maxsplit=1)[0]
    (tmp_path / "builder.top").write_text(
        head + '#include "builder.itp"\n[ system ]\nPeptide\n[ molecules ]\nProtein_chain 1\n'
    )
    structure.box_vectors = np.eye(3) * 8
    GROWriter.write(structure, tmp_path / "boxed.gro")
    (tmp_path / "check.mdp").write_text(
        "integrator=steep\nnsteps=0\ncutoff-scheme=Verlet\n"
        "coulombtype=PME\nrcoulomb=1.2\nrvdw=1.2\n"
        "vdw-modifier=force-switch\nrvdw-switch=1.0\nDispCorr=no\n"
        f"include=-I{_force_field_path(force_field)}\n"
    )
    checked = subprocess.run(
        [
            gmx,
            "grompp",
            "-f",
            "check.mdp",
            "-c",
            "boxed.gro",
            "-p",
            "builder.top",
            "-o",
            "builder.tpr",
            "-maxwarn",
            "0",
        ],
        cwd=tmp_path,
        env={**os.environ, "GMXLIB": str(_force_field_path(force_field))},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_structure_processing_rejects_chain_break_without_mutating_input():
    system = _two_residue_system("amber14sb")
    system.structure.coordinates[5:] += [1.5, 0, 0]
    original = system.structure.coordinates.copy()
    with pytest.raises(ModuleConfigError, match="is broken"):
        StructureProcessor().run(system, {"skip_protonation": True})
    np.testing.assert_array_equal(system.coordinates, original)

"""Construction-only regressions for microstates and CHARMM stream semantics."""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.modules.forcefield import gaff_backend as gaff
from gmxbuilder.modules.forcefield.cgenff_import import prepare_cgenff_molecule
from tests.prerequisites import requires_forcefield
from tests.test_cgenff_import import MOL2, STREAM


def test_coordinate_gaff_applies_ph_even_when_net_charge_does_not_change(tmp_path, monkeypatch):
    if not gaff.gaff_available():
        pytest.skip("Open Babel environment required")
    molecule = Chem.AddHs(Chem.MolFromSmiles("NCC(=O)O"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    path = tmp_path / "glycine.pdb"
    Chem.MolToPDBFile(Chem.RemoveHs(molecule), str(path))
    structure = PDBParser().parse(path)
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    original = gaff._run_external

    class Inspected(Exception):
        pass

    def inspect(command, **kwargs):
        if Path(command[0]).name == "acpype":
            mol2 = Path(command[command.index("-i") + 1])
            mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=False)
            assert mol is not None
            nitrogens = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
            oxygens = [a for a in mol.GetAtoms() if a.GetSymbol() == "O"]
            assert gaff._mol2_integer_charge(mol2, "LIG") == 0
            assert sum(a.GetAtomicNum() == 1 for a in nitrogens[0].GetNeighbors()) == 3
            assert sum(n.GetAtomicNum() == 1 for a in oxygens for n in a.GetNeighbors()) == 0
            raise Inspected
        return original(command, **kwargs)

    monkeypatch.setattr(gaff, "_run_external", inspect)
    with pytest.raises(Inspected):
        gaff.prepare_gaff_molecule(
            "LIG", structure, list(range(structure.num_atoms)), 0, target_pH=7
        )


def test_cgenff_program_version_is_not_a_force_field_version(tmp_path):
    mol2, stream = tmp_path / "lig.mol2", tmp_path / "lig.str"
    mol2.write_text(MOL2)
    stream.write_text(STREAM.replace("* For use with CGenFF version 4.6\n", ""))
    with pytest.raises(ModuleConfigError, match="force-field version"):
        prepare_cgenff_molecule("LIG", mol2, stream, "charmm36m", tmp_path / "out")


@requires_forcefield("charmm36m")
def test_incremental_stream_uses_installed_masses_and_bond_parameters(tmp_path):
    mol2, stream = tmp_path / "lig.mol2", tmp_path / "lig.str"
    mol2.write_text(MOL2)
    text = STREAM.replace("CGX1", "CG331").replace("HGX1", "HGA3")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("MASS"))
    text = text.replace("CG331 HGA3 340.0 1.090", "")
    # A standard stream contains no redundant native LJ or MASS data.
    text = text.replace("CG331 0.0 -0.1100 2.0000", "").replace("HGA3 0.0 -0.0200 1.2000", "")
    stream.write_text(text)
    result = prepare_cgenff_molecule("LIG", mol2, stream, "charmm36m", tmp_path / "out")
    itp = result.itp_path.read_text()
    assert "12.01100" in itp
    bond = itp.split("[ bonds ]")[1].splitlines()[-1].split()
    # Independently published bundled CG331-HGA3: 322 kcal/mol/A^2, 1.111 A.
    assert float(bond[3]) == pytest.approx(0.1111)
    assert float(bond[4]) == pytest.approx(322 * 836.8)


@requires_forcefield("charmm36m")
def test_harmonic_improper_preserves_nonzero_equilibrium_angle(tmp_path):
    mol2 = tmp_path / "lig.mol2"
    mol2.write_text(
        "@<TRIPOS>ATOM\n"
        + "\n".join(f"{i} H{i} {i} 0 0 H 1 LIG 0" for i in range(1, 5))
        + "\n@<TRIPOS>BOND\n"
    )
    stream = tmp_path / "lig.str"
    stream.write_text("""* For use with CGenFF version 4.6
read rtf card append
RESI LIG 0
ATOM H1 HGA3 0
ATOM H2 HGA3 0
ATOM H3 HGA3 0
ATOM H4 HGA3 0
IMPR H1 H2 H3 H4
END
read para card flex append
IMPROPERS
HGA3 HGA3 HGA3 HGA3 10.0 0 35.264
END
""")
    template = prepare_cgenff_molecule("LIG", mol2, stream, "charmm36m", tmp_path / "out")
    improper = template.itp_path.read_text().splitlines()[-1].split()
    assert improper[4] == "2"
    assert float(improper[5]) == pytest.approx(35.264)
    assert float(improper[6]) == pytest.approx(83.68)

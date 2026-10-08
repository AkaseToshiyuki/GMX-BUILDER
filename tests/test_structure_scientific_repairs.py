import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.core.exceptions import ModuleConfigError, ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.cif import CIFParser
from gmxbuilder.io.pdb import PDBParser
from gmxbuilder.modules.input.pdb_input import PDBInputModule
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.modules.modifications.protonation import assign_protonation


@pytest.mark.parametrize("explicit_override", [False, True])
def test_unavailable_tyrosinate_requires_explicit_override(tmp_path, explicit_override):
    molecule = Chem.AddHs(Chem.MolFromFASTA("AYA"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=872) == 0
    path = tmp_path / "aya.pdb"
    Chem.MolToPDBFile(molecule, str(path))
    system = (
        PDBInputModule()
        .run(System(Structure(np.empty((0, 3)), np.eye(3) * 5)), {"pdb": str(path)})
        .system
    )
    system.metadata["force_field"] = "amber14sb"
    config = {"pH": 11.0, "termini": {"A": {"nter": "ACE", "cter": "NME"}}}
    if explicit_override:
        assignment = assign_protonation("TYR", 11, force_field="amber14sb")
        config["protonation"] = [{**assignment, "index": 1, "state_override": True}]
        assert StructureProcessor().run(system, config).success
    else:
        with pytest.raises(ModuleConfigError, match="predicted protonation state"):
            StructureProcessor().run(system, config)


@pytest.mark.parametrize("format", ["pdb", "cif"])
def test_altloc_keeps_one_complete_conformer(tmp_path, format):
    atoms = [
        ("CB", "A", 0.0, 0.6),
        ("CG", "A", 1.5, 0.4),
        ("CB", "B", 5.0, 0.4),
        ("CG", "B", 6.5, 0.6),
    ]
    path = tmp_path / f"alt.{format}"
    if format == "pdb":
        text = (
            "\n".join(
                f"ATOM  {i:5d} {name:>4s}{alt}LEU A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"{occ:6.2f}{0.0:6.2f}          C "
                for i, (name, alt, x, occ) in enumerate(atoms, 1)
            )
            + "\nEND\n"
        )
        parser = PDBParser()
    else:
        text = (
            "data_alt\nloop_\n"
            + "\n".join(
                "_atom_site." + f
                for f in (
                    "label_atom_id",
                    "type_symbol",
                    "label_alt_id",
                    "label_comp_id",
                    "auth_asym_id",
                    "auth_seq_id",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                    "occupancy",
                )
            )
            + "\n"
            + "\n".join(f"{n} C {a} LEU A 1 {x} 0 0 {o}" for n, a, x, o in atoms)
            + "\n#\n"
        )
        parser = CIFParser()
    path.write_text(text)
    structure = parser.parse(path)
    assert structure.num_atoms == 2
    assert np.linalg.norm(np.diff(structure.coordinates, axis=0)) == pytest.approx(0.15)


def test_partial_altloc_is_not_completed_from_another_label():
    from gmxbuilder.io.altloc import select_residue_conformers

    with pytest.raises(ParseError, match="incomplete"):
        select_residue_conformers(
            [
                (0, ("A", 1), "LEU", "CB", "A", 0.8),
                (1, ("A", 1), "LEU", "CB", "B", 0.2),
                (2, ("A", 1), "LEU", "CG", "B", 0.2),
            ]
        )

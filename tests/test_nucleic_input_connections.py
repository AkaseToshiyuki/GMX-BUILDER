"""Independent DNA/RNA graphs exercise both deposited connection formats."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.input.pdb_input import PDBInputModule


@pytest.mark.parametrize("flavor", [2, 6], ids=["rna", "dna"])
@pytest.mark.parametrize("file_format", ["pdb", "cif"])
@pytest.mark.parametrize("connection", ["implicit", "canonical", "unsupported", "broken"])
def test_nucleic_connection_semantics_are_format_independent(
    tmp_path, flavor, file_format, connection
):
    mol = Chem.AddHs(Chem.MolFromFASTA("CG", flavor=flavor))
    assert AllChem.EmbedMolecule(mol, randomSeed=872) == 0
    mol = Chem.RemoveHs(mol)
    atoms = list(mol.GetAtoms())
    left = next(
        a.GetIdx()
        for a in atoms
        if a.GetPDBResidueInfo().GetResidueNumber() == 1
        and a.GetPDBResidueInfo().GetName().strip() == "O3'"
    )
    right = next(
        a.GetIdx()
        for a in atoms
        if a.GetPDBResidueInfo().GetResidueNumber() == 2
        and a.GetPDBResidueInfo().GetName().strip() == "P"
    )
    if connection == "broken":
        xyz = np.asarray(mol.GetConformer().GetAtomPosition(right)) + [10, 0, 0]
        mol.GetConformer().SetAtomPosition(right, xyz)
    if connection == "unsupported":
        left = next(
            a.GetIdx()
            for a in atoms
            if a.GetPDBResidueInfo().GetResidueNumber() == 1
            and a.GetPDBResidueInfo().GetName().strip() == "C1'"
        )
    path = tmp_path / ("polymer." + file_format)
    explicit = connection in {"canonical", "unsupported"}
    if file_format == "pdb":
        lines = [line for line in Chem.MolToPDBBlock(mol).splitlines() if line.startswith("ATOM")]
        if explicit:
            lines.append(f"CONECT{left + 1:5d}{right + 1:5d}")
        path.write_text("\n".join(lines) + "\nEND\n")
    else:
        import gemmi

        fields = [
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "auth_asym_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_atom_id",
            "pdbx_PDB_model_num",
        ]
        doc = gemmi.cif.Document()
        block = doc.add_new_block("polymer")
        loop = block.init_loop("_atom_site.", fields)
        for a in atoms:
            r = a.GetPDBResidueInfo()
            name, res, number = (
                r.GetName().strip(),
                r.GetResidueName().strip(),
                str(r.GetResidueNumber()),
            )
            xyz = mol.GetConformer().GetAtomPosition(a.GetIdx())
            loop.add_row(
                [
                    gemmi.cif.quote(v)
                    for v in [
                        "ATOM",
                        str(a.GetIdx() + 1),
                        a.GetSymbol(),
                        name,
                        res,
                        "A",
                        "1",
                        number,
                        *(str(v) for v in xyz),
                        "1",
                        "A",
                        number,
                        res,
                        name,
                        "1",
                    ]
                ]
            )
        if explicit:
            loop = block.init_loop(
                "_struct_conn.",
                [
                    "id",
                    "conn_type_id",
                    "ptnr1_label_asym_id",
                    "ptnr1_label_seq_id",
                    "ptnr1_label_atom_id",
                    "ptnr2_label_asym_id",
                    "ptnr2_label_seq_id",
                    "ptnr2_label_atom_id",
                ],
            )
            loop.add_row(
                [
                    gemmi.cif.quote(v)
                    for v in [
                        "bond1",
                        "covale",
                        "A",
                        "1",
                        atoms[left].GetPDBResidueInfo().GetName().strip(),
                        "A",
                        "2",
                        "P",
                    ]
                ]
            )
        doc.write_file(str(path))
    result = PDBInputModule().run(
        System(Structure(np.empty((0, 3)), np.eye(3) * 5)), {"pdb": str(path)}
    )
    report = result.system.metadata["input_validation"]
    if connection in {"implicit", "canonical"}:
        assert result.success, result.log
        assert report["can_proceed"]
    else:
        assert not result.success
        code = "nucleic_backbone" if connection == "broken" else "unsupported_connection"
        assert any(issue["code"] == code for issue in report["errors"]), report

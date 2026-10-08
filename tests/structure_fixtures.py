"""Independent, stereochemical peptide fixtures from RDKit's residue graph."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.core.structure import Structure


def peptide_structure(sequence: str) -> Structure:
    molecule = Chem.AddHs(Chem.MolFromFASTA(sequence))
    assert AllChem.EmbedMolecule(molecule, randomSeed=872) == 0
    # Only retained heavy atoms: added hydrogens and terminal OXT are generated
    # by the stage under test. Positions come from an independent embedding.
    indices = [
        a.GetIdx()
        for a in molecule.GetAtoms()
        if a.GetAtomicNum() != 1 and a.GetPDBResidueInfo().GetName().strip() != "OXT"
    ]
    records = [molecule.GetAtomWithIdx(i).GetPDBResidueInfo() for i in indices]
    return Structure(
        molecule.GetConformer().GetPositions()[indices] / 10,
        np.eye(3) * 5,
        atom_names=[r.GetName().strip() for r in records],
        resnames=[r.GetResidueName().strip() for r in records],
        resids=[r.GetResidueNumber() for r in records],
        chain_ids=[r.GetChainId().strip() for r in records],
        elements=[molecule.GetAtomWithIdx(i).GetSymbol() for i in indices],
    )

"""Identity-preserving input and output checks for SMILES-defined GAFF lipids."""

from pathlib import Path

from gmxbuilder.geometry.molecular_identity import itp_graph, validate_stereochemistry


def validate_template(smiles, template):
    """Check the fitted atom graph/order and coordinates against the requested identity."""
    names, elements, bonds = itp_graph(template.itp_path)
    if names != template.atom_names:
        raise ValueError("GAFF lipid coordinate/topology atom order differs")
    return validate_stereochemistry(smiles, elements, bonds, template.coordinates)


def write_lipid_sdf(smiles: str, output: Path) -> None:
    """Give ACPYPE a checked 3D molecule instead of asking it to interpret stereo.

    The explicit hydrogen graph is preserved. Seeds only change conformation;
    they may never change the specified stereoisomer or double-bond geometry.
    Fitted output must be checked again because minimization can still fail.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    reference = Chem.MolFromSmiles(smiles)
    if reference is None:
        raise ValueError("Invalid GAFF lipid SMILES")
    errors = []
    for seed in (42, 7811, 101, 202, 303):
        molecule = Chem.AddHs(reference)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        parameters.useRandomCoords = True
        parameters.numThreads = 1
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            errors.append(f"seed {seed}: embedding failed")
            continue
        elements = tuple(a.GetSymbol() for a in molecule.GetAtoms())
        bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())
        try:
            validate_stereochemistry(
                smiles, elements, bonds, molecule.GetConformer().GetPositions() / 10.0
            )
        except ValueError as exc:
            errors.append(f"seed {seed}: {exc}")
            continue
        with Chem.SDWriter(str(output)) as writer:
            writer.write(molecule)
        return
    raise ValueError("No identity-valid GAFF input: " + "; ".join(errors))

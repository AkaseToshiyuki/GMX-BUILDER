"""Stereochemical completeness without phosphate resonance artefacts."""


def stereo_reference(molecule):
    """Use an equivalent charge-separated phosphate form for stereo perception.

    P(=O)([O-]) has two equivalent non-protonated terminal oxygens. Treating
    its arbitrarily drawn double bond as distinct invents a phosphorus centre
    and, in symmetric cardiolipin, a dependent central carbon centre. Only a
    copy is changed; atom order, hydrogens and total charge are preserved.
    Protonated oxygens, isotopes and genuine carbon asymmetry remain distinct.
    """
    from rdkit import Chem

    result = Chem.Mol(molecule)
    for atom in result.GetAtoms():
        if atom.GetAtomicNum() != 15 or atom.GetFormalCharge() != 0:
            continue
        if not any(
            other.GetAtomicNum() == 8
            and other.GetDegree() == 1
            and other.GetFormalCharge() == -1
            and other.GetTotalNumHs() == 0
            for other in atom.GetNeighbors()
        ):
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                other.GetAtomicNum() == 8
                and other.GetDegree() == 1
                and other.GetFormalCharge() == 0
                and other.GetTotalNumHs() == 0
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                bond.SetBondType(Chem.BondType.SINGLE)
                atom.SetFormalCharge(1)
                other.SetFormalCharge(-1)
                break
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result

"""Local CHARMM assignment: chemistry, provenance and real topology preprocessing."""

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield.charmm_compat import (
    TEMPLATES,
    CharmmCompatError,
    _molecule_itp,
    inspect_smiles,
    prepare_local_molecule,
    template_database,
    validate_artifacts,
)
from gmxbuilder.modules.forcefield.charmm_research import (
    charge_matrix,
    check_domain,
    train_model,
)
from gmxbuilder.modules.forcefield.compatibility import compatibility_report
from gmxbuilder.modules.forcefield.rtp_parser import RTPParser
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from tests.prerequisites import requires_forcefield, requires_gromacs


def molecule_system(smiles, *, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    n = mol.GetNumAtoms()
    structure = Structure(
        coordinates=mol.GetConformer().GetPositions() / 10,
        box_vectors=np.eye(3) * 6,
        atom_names=[f"X{i + 1}" for i in range(n)],
        elements=[a.GetSymbol() for a in mol.GetAtoms()],
        resnames=["LIG"] * n,
        resids=[501] * n,
        chain_ids=["A"] * n,
    )
    return System(
        structure=structure,
        components=[Component(name="LIG", kind=ComponentKind.UNKNOWN, atom_indices=np.arange(n))],
    )


@pytest.mark.parametrize(
    "smiles,code",
    [
        ("", "INPUT_PARSE_ERROR"),
        ("invalid", "INPUT_PARSE_ERROR"),
        ("CC.O", "UNSUPPORTED_CHEMISTRY"),
        ("[Na+]", "UNSUPPORTED_CHEMISTRY"),
        ("[CH3]", "UNSUPPORTED_CHEMISTRY"),
        ("[13CH3]O", "UNSUPPORTED_CHEMISTRY"),
        ("C[C@H](O)CC", "CHARGE_MODEL_MISSING"),
        ("CC(N)Cc1ccccc1", "STEREOCHEMISTRY_UNSPECIFIED"),
        ("CCC(=O)N", "CHARGE_MODEL_MISSING"),
    ],
)
def test_explicit_chemistry_rejects_unsupported(smiles, code):
    with pytest.raises(CharmmCompatError) as caught:
        inspect_smiles(smiles)
    assert caught.value.code == code


def test_identity_is_independent_of_smiles_direction_and_aromatic_notation():
    assert inspect_smiles("OCC")[1].residue == "ETOH"
    assert inspect_smiles("C1=CC=CC=C1")[1].residue == "BENZ"
    assert inspect_smiles("C[NH3+]")[1].residue != inspect_smiles("CN")[1].residue


# Chemistry the installed release does not document, so the research path must
# refuse it rather than reach for an analogy. The domain is no longer a
# hand-written predicate -- it is whatever the release describes -- so these are
# molecules chosen because CGenFF Jul2022 genuinely lacks their environments:
# a quaternary carbon, a primary alkylamine, a carbamate, a cyclopropyl amide,
# and an element outside the supported set.
@pytest.mark.parametrize(
    "smiles",
    [
        "CC(C)(C)C",  # neopentane: quaternary carbon
        "CCN",  # ethylamine: primary alkylamine
        "CC(C)(C)OC(=O)N",  # Boc-amine: carbamate
        "C1CC1C(=O)NC2CC2",  # dicyclopropyl amide
        "FC(F)(F)S(=O)(=O)N",  # triflamide: environment typed three ways
    ],
)
@requires_forcefield("charmm36m")
def test_research_refuses_chemistry_the_release_does_not_document(smiles):
    from rdkit import Chem

    with pytest.raises(CharmmCompatError, match="ATOM_TYPE_"):
        check_domain(Chem.MolFromSmiles(smiles), "charmm36m")


# The counterpart property: acceptance is never a generic fallback. Every atom
# of an accepted molecule must have been typed from an environment the release
# actually documents, with its radius and witness count recorded.
@pytest.mark.parametrize("smiles", ["C1CCCCC1", "CC=O", "OCCO", "C[O-]"])
@requires_forcefield("charmm36m")
def test_accepted_research_chemistry_is_traceable_to_the_release(smiles):
    from rdkit import Chem

    from gmxbuilder.modules.forcefield.charmm_research import assign_research

    parser = RTPParser(template_database("charmm36m"))
    _molecule, _template, record, report = assign_research(
        Chem.MolFromSmiles(smiles), parser, "charmm36m"
    )
    assert report["research_model"] == "general"
    assert report["charge_model"]["distributed_parameters"] == "none"
    assert len(report["atom_assignments"]) == len(record["atoms"])
    for atom in report["atom_assignments"]:
        assert atom["typing_support"] >= 1
        assert 1 <= atom["typing_radius"] <= 4
        assert atom["charge_witnesses"] >= 1


def test_charge_transfers_conserve_charge_and_survive_permutation():
    from rdkit import Chem

    m = Chem.MolFromSmiles("CCO")
    types = ["A", "A", "B"]
    forward = charge_matrix(m, types, [("A", "B")])
    reverse = charge_matrix(Chem.RenumberAtoms(m, [2, 1, 0]), types[::-1], [("A", "B")])
    np.testing.assert_array_equal(forward, reverse[::-1])
    np.testing.assert_array_equal(forward.sum(axis=0), 0)
    with pytest.raises(CharmmCompatError, match="CHARGE_MODEL_MISSING"):
        charge_matrix(m, types, [])


def test_adapter_preserves_multiterm_dihedrals_and_ub():
    text = (
        "[ moleculetype ]\nOther 3\n[ atoms ]\n1 CT 1 TMP A 1 0 12\n"
        "[ angles ]\n1 2 3 5 110 100 0.2 50\n"
        "[ dihedrals ]\n1 2 3 4 9 0 2 1\n1 2 3 4 9 180 3 2\n[ system ]\nTest\n"
    )
    adapted = _molecule_itp(text, "LIG", {"A": "X1"})
    assert "1 2 3 5 110 100 0.2 50" in adapted
    assert "1 2 3 4 9 0 2 1\n1 2 3 4 9 180 3 2" in adapted
    assert "1 CT 1 LIG X1 1 0 12" in adapted


def test_local_default_replaces_rtp_and_keeps_import(monkeypatch):
    monkeypatch.setattr(
        "gmxbuilder.modules.forcefield.charmm_compat.availability", lambda ff: (True, "ready")
    )
    structure = Structure(
        coordinates=np.zeros((1, 3)),
        box_vectors=np.eye(3),
        atom_names=["X1"],
        resnames=["LIG"],
        resids=[1],
        elements=["C"],
    )
    system = System(
        structure=structure, components=[Component("LIG", ComponentKind.UNKNOWN, np.array([0]))]
    )
    options = compatibility_report(system, "charmm36m", [])["ligand_options"]
    assert [o["value"] for o in options] == ["charmm_compat", "cgenff"]
    assert all(o["enabled"] for o in options)
    with pytest.raises(Exception, match="explicit SMILES"):
        ForceFieldSelector().run(system, {"name": "charmm36m"})


@requires_forcefield("charmm36m")
def test_charge_model_has_full_rank_and_disjoint_holdout():
    from gmxbuilder.modules.forcefield.catalog import force_field_directory

    report = train_model(RTPParser(force_field_directory("charmm36m") / "cgenff.rtp"))[-1]
    assert report["rank"] == report["feature_count"]
    assert set(report["training_residues"]).isdisjoint(report["holdout_residues"])
    assert report["training_max_error_e"] < 1e-8
    assert all(h["charge_max_error_e"] < 1e-8 for h in report["holdout"])
    assert report["physical_validation"] == "not_evaluated"


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
@requires_forcefield("charmm36")
@pytest.mark.parametrize("ff", ["charmm36m", "charmm36"])
@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.residue)
def test_native_template_build_preserves_coordinates_and_provenance(template, ff, tmp_path):
    from rdkit import Chem

    system = molecule_system(template.smiles)
    molecule = Chem.MolFromSmiles(template.smiles)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    smiles = Chem.MolToSmiles(molecule, canonical=False)
    if template.residue == "MP_1":
        # P=O / P-O- resonance creates an artificial RDKit stereocenter.
        # This representation is blocked until that normalization is reviewed.
        with pytest.raises(CharmmCompatError, match="STEREOCHEMISTRY_UNSPECIFIED"):
            prepare_local_molecule(
                "LIG", system.structure, list(range(system.num_atoms)), smiles, ff, tmp_path
            )
        return
    if ff == "charmm36" and template.residue == "ACN":
        with pytest.raises(CharmmCompatError, match="PARAMETER_MISSING"):
            prepare_local_molecule(
                "LIG", system.structure, list(range(system.num_atoms)), smiles, ff, tmp_path
            )
        return
    if ff == "charmm36" and template.residue == "ACEM":
        # Legacy HDB cannot distinguish the amide hydrogens' unequal charges.
        with pytest.raises(CharmmCompatError, match="inequivalent hydrogen"):
            prepare_local_molecule(
                "LIG",
                system.structure,
                list(range(system.num_atoms)),
                smiles,
                ff,
                tmp_path,
            )
        return
    result = prepare_local_molecule(
        "LIG",
        system.structure,
        list(range(system.num_atoms)),
        smiles,
        ff,
        tmp_path,
    )
    for index, name in enumerate(system.structure.atom_names):
        np.testing.assert_array_equal(
            result.coordinates[result.atom_names.index(name)], system.coordinates[index]
        )
    report = validate_artifacts(result.itp_path, ff)
    assert report["gromacs_preprocessing"] == "passed_maxwarn_0"
    assert report["numerical_validation"]["dynamics_steps"] == 0
    assert report["template"] == template.residue


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
@requires_forcefield("charmm36")
@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
@pytest.mark.parametrize("smiles", ["CCCCO", "CCCCOC", "CCCC"])
def test_new_scaffolds_require_opt_in_and_build_complete_models(ff, smiles, tmp_path):
    system = molecule_system(smiles)
    config = {"name": ff, "charmm_compat_smiles": {"LIG": smiles}, "_task_dir": str(tmp_path)}
    with pytest.raises(CharmmCompatError, match="PHYSICAL_VALIDATION_REQUIRED"):
        ForceFieldSelector().run(system, config)
    result = ForceFieldSelector().run(system, {**config, "charmm_compat_allow_research": True})
    parameters = result.system.metadata["ligand_parameters"]["LIG"]
    from pathlib import Path

    report = validate_artifacts(Path(parameters["itp_path"]), ff)
    assert report["export_eligibility"] == "research_only"
    assert report["charge_method"] == "environment_lookup"
    assert result.system.num_atoms > system.num_atoms
    assert abs(sum(a["charge_e"] for a in report["atom_assignments"])) < 1e-10
    # Cached files cannot be silently edited or reused with a different FF.
    with pytest.raises(CharmmCompatError, match="FF_VERSION_MISMATCH"):
        validate_artifacts(
            Path(parameters["itp_path"]), "charmm36" if ff == "charmm36m" else "charmm36m"
        )
    Path(parameters["itp_path"]).write_text("modified")
    with pytest.raises(CharmmCompatError, match="NUMERICAL_VALIDATION_FAILED"):
        validate_artifacts(Path(parameters["itp_path"]), ff)


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
def test_invalid_geometry_or_duplicate_names_cannot_be_assigned(tmp_path):
    system = molecule_system("CCO")
    system.coordinates[1] += 2
    with pytest.raises(CharmmCompatError, match="ATOM_MAPPING_AMBIGUOUS"):
        prepare_local_molecule("LIG", system.structure, [0, 1, 2], "CCO", "charmm36m", tmp_path)
    system.structure.atom_names[1] = system.structure.atom_names[0]
    with pytest.raises(CharmmCompatError, match="ATOM_MAPPING_AMBIGUOUS"):
        prepare_local_molecule("LIG", system.structure, [0, 1, 2], "CCO", "charmm36m", tmp_path)


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
def test_exported_research_package_preprocesses_and_rejects_atom_order_corruption(tmp_path):
    import json
    import subprocess
    import zipfile
    from pathlib import Path

    from gmxbuilder.core.exceptions import TopologyError
    from gmxbuilder.io.top import TopologyWriter
    from gmxbuilder.modules.export.exporter import ExportModule
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    original = molecule_system("CCCCO")
    system = (
        ForceFieldSelector()
        .run(
            original,
            {
                "name": "charmm36m",
                "charmm_compat_smiles": {"LIG": "CCCCO"},
                "charmm_compat_allow_research": True,
                "_task_dir": str(tmp_path),
            },
        )
        .system
    )
    result = ExportModule().run(
        system, {"output_dir": str(tmp_path / "export"), "write_mdp": False}
    )
    assert result.success
    output = tmp_path / "export"
    assert "EXPERIMENTAL" in (output / "README.txt").read_text()
    report = json.loads((output / "topology/LIG_assignment.json").read_text())
    assert report["export_eligibility"] == "research_only"
    archive = next(output.glob("*.zip"))
    with zipfile.ZipFile(archive) as package:
        assert any(name.endswith("topology/LIG_assignment.json") for name in package.namelist())
    assignment_dir = Path(system.metadata["ligand_parameters"]["LIG"]["itp_path"]).parent
    (output / "check.mdp").write_text((assignment_dir / "check.mdp").read_text())
    proc = subprocess.run(
        [
            find_gromacs_executable(),
            "grompp",
            "-f",
            "check.mdp",
            "-c",
            "structure/input.gro",
            "-p",
            "topology/topol.top",
            "-o",
            "check.tpr",
            "-maxwarn",
            "0",
        ],
        cwd=output,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    system.structure.atom_names[0], system.structure.atom_names[1] = (
        system.structure.atom_names[1],
        system.structure.atom_names[0],
    )
    with pytest.raises(TopologyError, match="order|mismatch"):
        TopologyWriter(
            "charmm36m", {"ligand_parameters": system.metadata["ligand_parameters"]}
        ).write_top(system.structure, output / "topology/bad.top")


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
def test_input_reordering_and_multiple_conformers_preserve_atom_identity(tmp_path):
    first = molecule_system("CCCO", seed=41)
    second = molecule_system("CCCO", seed=43)
    second.coordinates[:] += 1
    second.structure.resids = [502] * second.num_atoms
    combined = first.structure.append(second.structure)
    original_coordinates = combined.coordinates.copy()
    original_names = list(combined.atom_names)
    original_resids = list(combined.resids)
    system = System(
        combined,
        components=[Component("LIG", ComponentKind.UNKNOWN, np.arange(combined.num_atoms))],
    )
    result = (
        ForceFieldSelector()
        .run(
            system,
            {
                "name": "charmm36m",
                "_task_dir": str(tmp_path),
                "charmm_compat_smiles": {"LIG": "CCCO"},
            },
        )
        .system
    )
    for name, resid, xyz in zip(original_names, original_resids, original_coordinates, strict=True):
        index = next(
            i
            for i in range(result.num_atoms)
            if result.structure.atom_names[i] == name and result.structure.resids[i] == resid
        )
        np.testing.assert_array_equal(result.coordinates[index], xyz)
    native = prepare_local_molecule(
        "LIG", first.structure, [0, 1, 2, 3], "CCCO", "charmm36m", tmp_path
    )
    permuted = prepare_local_molecule(
        "LIG", first.structure, [3, 1, 0, 2], "OCCC", "charmm36m", tmp_path
    )
    assert native.itp_path.read_text() == permuted.itp_path.read_text()
    np.testing.assert_array_equal(native.coordinates, permuted.coordinates)

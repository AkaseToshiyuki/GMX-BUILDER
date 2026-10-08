"""Complex research ligands: source-derived assignments, stereochemistry and export."""

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield.charmm_aryl_ammonium import (
    HOLDOUT,
    TRAINING,
    assign_research,
    check_domain,
    train_model,
    transfer_matrix,
)
from gmxbuilder.modules.forcefield.charmm_bonded_research import resolve_missing
from gmxbuilder.modules.forcefield.charmm_compat import (
    CharmmCompatError,
    _map_heavy_atoms,
    inspect_smiles,
    prepare_local_molecule,
    template_database,
    validate_artifacts,
)
from gmxbuilder.modules.forcefield.rtp_parser import RTPParser
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from tests.prerequisites import requires_forcefield, requires_gromacs
from tests.test_charmm_compat import molecule_system

AMP_S = "C[C@H]([NH3+])Cc1ccccc1"
AMP_R = "C[C@@H]([NH3+])Cc1ccccc1"


def amp_system():
    """Retained AMP heavy coordinates from the reported 1AMP_S input, in Angstrom."""
    coordinates = (
        np.array(
            [
                [100.823, 109.301, 101.636],
                [102.225, 108.664, 101.603],
                [103.349, 109.600, 101.092],
                [103.137, 111.137, 101.173],
                [103.368, 111.814, 102.361],
                [103.186, 113.186, 102.414],
                [102.780, 113.879, 101.275],
                [102.556, 113.206, 100.087],
                [102.744, 111.832, 100.034],
                [102.585, 108.117, 102.901],
            ]
        )
        / 10
    )
    structure = Structure(
        coordinates=coordinates,
        box_vectors=np.eye(3) * 15,
        atom_names=[f"C{i:02d}" for i in range(1, 10)] + ["N10"],
        elements=["C"] * 9 + ["N"],
        resnames=["AMP"] * 10,
        resids=[501] * 10,
        chain_ids=["A"] * 10,
    )
    return System(structure, components=[Component("AMP", ComponentKind.UNKNOWN, np.arange(10))])


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(N)Cc1ccccc1",
        "C[C@H](N)Cc1ccccc1",
        "C[NH2+]CC",
        "c1ccncc1",
        "[NH3+]c1ccccc1",
        "[NH3+]CC[NH3+]",
        "C1CCCCC1",
        "Cc1ccc(C)cc1",
        "C[C@H]([NH3+])CO",
        "CC(=O)N",
        "C1=CC2=CC=CC=C2C=C1",
    ],
)
def test_extended_domain_rejects_uncovered_functional_groups(smiles):
    with pytest.raises(CharmmCompatError):
        molecule, _ = inspect_smiles(smiles, allow_unknown=True)
        check_domain(molecule)


@requires_forcefield("charmm36")
@requires_forcefield("charmm36m")
@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
def test_charge_model_is_identifiable_and_reports_real_holdout_error(ff):
    parser = RTPParser(template_database(ff))
    _, features, coefficients, _, report = train_model(parser)
    assert set(TRAINING).isdisjoint(HOLDOUT)
    assert report["rank"] == report["feature_count"] == 6
    assert report["training_max_error_e"] < 1e-8
    benzyl = next(h for h in report["holdout"] if h["residue"] == "BZAM")
    assert benzyl["charge_max_error_e"] == pytest.approx(0.13 if ff == "charmm36" else 0.09)
    assert report["physical_validation"] == "not_evaluated"
    mol, _, record, assignment = assign_research(Chem.MolFromSmiles(AMP_S), parser)
    matrix = transfer_matrix(mol, features)
    np.testing.assert_array_equal(matrix.sum(axis=0), 0)
    reverse = list(reversed(range(mol.GetNumAtoms())))
    permuted = transfer_matrix(Chem.RenumberAtoms(mol, reverse), features)
    np.testing.assert_allclose(matrix @ coefficients, (permuted @ coefficients)[::-1])
    assert sum(a[2] for a in record["atoms"]) == pytest.approx(1.0)
    assert {
        a["type"] for a in assignment["atom_assignments"] if a["charge_environment_extrapolated"]
    } == {"CG314", "HGA1"}
    for atom in assignment["atom_assignments"]:
        assert atom["charge_e"] == pytest.approx(
            atom["formal_charge_e"] + sum(atom["charge_contributions_e"])
        )
    with pytest.raises(CharmmCompatError, match="CHARGE_MODEL_MISSING"):
        transfer_matrix(mol, [])


@requires_forcefield("charmm36m")
def test_real_amp_stereochemistry_and_reflected_coordinates_are_not_interchangeable():
    system = amp_system()
    parser = RTPParser(template_database("charmm36m"))
    _, template, record, _ = assign_research(Chem.MolFromSmiles(AMP_S), parser)
    mapping = _map_heavy_atoms(system.structure, list(range(10)), template, record)
    assert mapping["A2"] == 1  # C02 is S; residue name and filename never determine this.
    _, reverse, reverse_record, _ = assign_research(Chem.MolFromSmiles(AMP_R), parser)
    with pytest.raises(CharmmCompatError, match="STEREOCHEMISTRY_MISMATCH"):
        _map_heavy_atoms(system.structure, list(range(10)), reverse, reverse_record)
    system.coordinates[:, 0] *= -1
    with pytest.raises(CharmmCompatError, match="STEREOCHEMISTRY_MISMATCH"):
        _map_heavy_atoms(system.structure, list(range(10)), template, record)


def test_bonded_analogy_preserves_all_fourier_terms_and_native_matches(tmp_path):
    database = tmp_path / "ffbonded.itp"
    database.write_text(
        "[ dihedraltypes ]\nCG331 CG311 CG321 CG2R61 9 0 2 1\nCG331 CG311 CG321 CG2R61 9 180 3 2\n"
    )
    top = tmp_path / "native.top"
    original = "[ atoms ]\n1 CG331\n2 CG314\n3 CG321\n4 CG2R61\n[ dihedrals ]\n1 2 3 4 9\n"
    top.write_text(original)
    audit = resolve_missing(top, database)
    assert audit[0]["fourier_terms_retained"] == 2
    assert audit[0]["source_lines"] == [2, 3]
    assert "1 2 3 4 9 0 2 1" in top.read_text()
    assert "1 2 3 4 9 180 3 2" in top.read_text()
    # A native wildcard always takes precedence over the research substitution.
    database.write_text(database.read_text() + "X CG314 CG321 X 9 0 9 3\n")
    top.write_text(original)
    assert resolve_missing(top, database) == []
    assert top.read_text() == original
    database.write_text("[ dihedraltypes ]\n")
    with pytest.raises(CharmmCompatError, match="PARAMETER_MISSING"):
        resolve_missing(top, database)
    assert top.read_text() == original  # No partial model is published.


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
@requires_forcefield("charmm36")
@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
def test_actual_amp_build_is_complete_and_preserves_coordinates_and_charge(ff, tmp_path):
    original = amp_system()
    config = {
        "name": ff,
        "charmm_compat_smiles": {"AMP": AMP_S},
        "_task_dir": str(tmp_path),
        "ligand_pH": 7.4,
    }
    with pytest.raises(CharmmCompatError, match="PHYSICAL_VALIDATION_REQUIRED"):
        ForceFieldSelector().run(original, config)
    result = (
        ForceFieldSelector().run(original, {**config, "charmm_compat_allow_research": True}).system
    )
    assert result.num_atoms == 24
    assert result.total_charge() == pytest.approx(1.0)
    for name, xyz in zip(original.structure.atom_names, original.coordinates, strict=True):
        np.testing.assert_array_equal(
            result.coordinates[result.structure.atom_names.index(name)], xyz
        )
    report = validate_artifacts(Path(result.metadata["ligand_parameters"]["AMP"]["itp_path"]), ff)
    assert report["environment_pH"] == 7.4
    assert result.metadata["ligand_environment_pH"] == 7.4
    assert "ligand_protonation_pH" not in result.metadata
    assert report["export_eligibility"] == "research_only"
    assert len(report["bonded_parameter_analogies"]) == (7 if ff == "charmm36" else 1)
    assert report["gromacs_preprocessing"] == "passed_maxwarn_0"
    assert report["numerical_validation"]["dynamics_steps"] == 0
    # Equivalent SMILES and a scrambled input atom order must yield the same ITP.
    permuted = prepare_local_molecule(
        "AMP",
        original.structure,
        list(reversed(range(10))),
        "[NH3+][C@@H](C)Cc1ccccc1",
        ff,
        tmp_path,
        allow_research=True,
    )
    assert (
        permuted.itp_path.read_text()
        == Path(result.metadata["ligand_parameters"]["AMP"]["itp_path"]).read_text()
    )


@pytest.mark.slow
@requires_gromacs
@requires_forcefield("charmm36m")
@pytest.mark.parametrize(
    "smiles", ["CCc1ccccc1CC", "[NH3+]CCCc1ccccc1", "CCCc1ccccc1", "CCC[NH3+]"]
)
def test_extended_scaffolds_build_or_fail_at_declared_chemistry_boundary(smiles, tmp_path):
    system = molecule_system(smiles)
    if smiles == "CCc1ccccc1CC":
        # Still refused, but now by the general model, which names the three
        # types the release gives this ortho-disubstituted aromatic environment
        # rather than reporting which hand-written predicate rejected it last.
        with pytest.raises(CharmmCompatError, match="UNSUPPORTED_CHEMISTRY|ATOM_TYPE_"):
            prepare_local_molecule(
                "LIG",
                system.structure,
                list(range(system.num_atoms)),
                smiles,
                "charmm36m",
                tmp_path,
                allow_research=True,
            )
        return
    result = prepare_local_molecule(
        "LIG",
        system.structure,
        list(range(system.num_atoms)),
        smiles,
        "charmm36m",
        tmp_path,
        allow_research=True,
    )
    report = json.loads(result.itp_path.with_name("report.json").read_text())
    assert report["parameter_completeness"] == "complete"
    assert report["export_eligibility"] == "research_only"


def test_bonded_analogy_preserves_atom_index_association(tmp_path):
    """The analogy pass rewrites a validated topology as text; prove it stays aligned.

    ``resolve_missing`` parses the ITP by splitting strings and writes the file
    back. The numerical equivalence gate runs after it, but both sides of that
    comparison are derived from this same rewritten file, so a rewrite that is
    syntactically valid yet attaches a term to the wrong atoms would be
    inherited by both and pass. This asserts the property that gate cannot:
    every emitted term keeps the atom indices it was resolved for, and every
    line that already had parameters survives untouched.
    """
    parameters = tmp_path / "ffbonded.itp"
    parameters.write_text(
        "[ dihedraltypes ]\n"
        "CG331 CG321 CG321 CG331 9 0.0 0.15 2\n"
        "CG311 CG321 CG321 CG331 9 0.0 0.25 3\n"
        "[ bondtypes ]\n"
        "CG311 CG321 1 0.153 186188.0\n"
    )
    topology = tmp_path / "native.top"
    topology.write_text(
        "[ atoms ]\n"
        "1 CG314 1 LIG A1 1 -0.1 12.011\n"
        "2 CG321 1 LIG A2 1 -0.1 12.011\n"
        "3 CG321 1 LIG A3 1 -0.1 12.011\n"
        "4 CG331 1 LIG A4 1 -0.1 12.011\n"
        "[ bonds ]\n"
        "1 2 1\n"
        "[ dihedrals ]\n"
        "1 2 3 4 9\n"
    )
    audit = resolve_missing(topology, parameters)

    sections: dict[str, list[list[str]]] = {}
    current = ""
    for line in topology.read_text().splitlines():
        code = line.split(";")[0].strip()
        if code.startswith("["):
            current = code.strip("[] ").lower()
        elif code:
            sections.setdefault(current, []).append(code.split())

    # CG314 -> CG311 is a type substitution for the *lookup* only; every term
    # must still be written for the atoms it was resolved for, never renumbered.
    assert sections["dihedrals"], "the missing dihedral was not resolved at all"
    assert all(f[:4] == ["1", "2", "3", "4"] for f in sections["dihedrals"])
    assert all(len(f) > 5 for f in sections["dihedrals"]), "resolved term carries no parameters"
    assert all(f[:2] == ["1", "2"] for f in sections["bonds"])
    # The atom table is untouched: the pass may add parameters, never retype.
    assert [f[1] for f in sections["atoms"]] == ["CG314", "CG321", "CG321", "CG331"]

    # Both terms are resolved, and each records the indices it was resolved for
    # alongside the substitution used, so the rewrite is auditable line by line.
    audited = {entry["section"]: entry for entry in audit}
    assert set(audited) == {"bonds", "dihedrals"}
    assert audited["bonds"]["atom_indices"] == [1, 2]
    assert audited["dihedrals"]["atom_indices"] == [1, 2, 3, 4]
    for entry in audit:
        # Only CG314 -> CG311 changes; every other position is carried through.
        assert entry["target_types"][0] == "CG314"
        assert entry["source_types"][0] == "CG311"
        assert entry["target_types"][1:] == entry["source_types"][1:]
        assert entry["physical_validation"] == "not_evaluated"

"""Protein chemistry contracts across input, selection, assignment and export."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
from gmxbuilder.modules.input.modification_detection import normalize_detected_modifications
from gmxbuilder.modules.modifications.processor import StructureProcessor
from gmxbuilder.modules.modifications.selection import chemistry_fingerprint
from tests.prerequisites import requires_forcefield, skip_without_forcefield
from tests.structure_fixtures import peptide_structure


def peptide(sequence, force_field="amber14sb"):
    structure = peptide_structure(sequence)
    return System(
        structure,
        components=[Component("protein", ComponentKind.PROTEIN, np.arange(structure.num_atoms))],
        metadata={"force_field": force_field, "requested_force_field": force_field},
    )


def residue_names(system):
    return dict(zip(system.structure.resids, system.structure.resnames, strict=True))


@requires_forcefield("amber14sb")
def test_identity_overrides_frontend_index_in_mixed_system():
    protein = peptide("NNN")
    zinc = Structure(
        np.array([[4.0, 4.0, 4.0]]),
        np.eye(3) * 5,
        atom_names=["ZN"],
        resnames=["ZN"],
        resids=[1],
        chain_ids=["Z"],
        elements=["Zn"],
    )
    mixed = System(zinc, components=[Component("zinc", ComponentKind.IONS, np.array([0]))])
    mixed = mixed.merge(protein)
    mixed.metadata.update(protein.metadata)
    request = {"skip_protonation": True, "modifications": [{"index": 1, "patch_id": "DEA_ASN"}]}
    with pytest.raises(ModuleConfigError, match="ambiguous"):
        StructureProcessor().run(copy.deepcopy(mixed), request)
    request["modifications"][0]["target"] = {"chain": "A", "resid": 3, "resname": "ASN"}
    checked = StructureProcessor().run(copy.deepcopy(mixed), request).system
    assert residue_names(checked)[3] == "ASP"
    assert residue_names(checked)[2] == "ASN"
    request["modifications"][0]["target"]["resname"] = "GLN"
    with pytest.raises(ModuleConfigError, match="no longer matches"):
        StructureProcessor().run(mixed, request)


@pytest.mark.parametrize("ff", ["amber14sb", "charmm36m"])
@pytest.mark.parametrize(
    "source,reference,patch", [("ANA", "ADA", "DEA_ASN"), ("AQA", "AEA", "DEG_GLN")]
)
def test_deamidation_has_product_protonation(ff, source, reference, patch):
    skip_without_forcefield(ff)
    config = {"pH": 2.0, "termini": {"A": {"nter": "ACE", "cter": "NME"}}}
    expected = StructureProcessor().run(peptide(reference, ff), config).system
    observed = (
        StructureProcessor()
        .run(peptide(source, ff), {**config, "modifications": [{"index": 1, "patch_id": patch}]})
        .system
    )
    assert residue_names(observed) == residue_names(expected)
    observed = ForceFieldAssigner().run(observed, {}).system
    expected = ForceFieldAssigner().run(expected, {}).system
    assert sum(a.charge for a in observed.topology.atom_types) == pytest.approx(0, abs=1e-5)
    assert sum(a.charge for a in observed.topology.atom_types) == pytest.approx(
        sum(a.charge for a in expected.topology.atom_types), abs=1e-5
    )


@requires_forcefield("charmm36m")
@pytest.mark.parametrize("condition", ["intact", "missing", "invalid", "legacy"])
def test_uploaded_ptm_needs_decision_and_preserves_valid_geometry(condition):
    original = (
        StructureProcessor()
        .run(
            peptide("ASA", "charmm36m"),
            {
                "skip_protonation": True,
                "prepare_standard_termini": False,
                "modifications": [{"index": 1, "patch_id": "PHOS_SER"}],
            },
        )
        .system
    )
    normalized, report = normalize_detected_modifications(
        copy.deepcopy(original.structure), {"ALA", "SER"}
    )
    uploaded = System(
        normalized,
        components=[Component("protein", ComponentKind.PROTEIN, np.arange(normalized.num_atoms))],
        metadata={"force_field": "charmm36m", "input_modifications": report},
    )
    config = {"skip_protonation": True, "prepare_standard_termini": False}
    with pytest.raises(ModuleConfigError, match="explicitly accept removing"):
        StructureProcessor().run(copy.deepcopy(uploaded), config)
    removal = {"action": "remove", "target": {"chain": "A", "resid": 2, "resname": "SER"}}
    removed = (
        StructureProcessor()
        .run(
            copy.deepcopy(uploaded),
            {
                **config,
                "input_modification_decisions": [removal],
            },
        )
        .system
    )
    assert residue_names(removed)[2] == "SER"
    assert removed.metadata["input_modification_decisions"][0]["action"] == "remove"
    if condition == "invalid":
        for atom in report["records"][0]["original_atoms"]:
            if atom["name"] == "P":
                atom["coordinates_nm"] = [90, 90, 90]
    if condition == "invalid":
        with pytest.raises(ModuleConfigError, match="preserving deposited atoms"):
            StructureProcessor().run(
                uploaded,
                {
                    **config,
                    "modifications": [{"index": 1, "patch_id": "PHOS_SER"}],
                },
            )
        return
    if condition == "legacy":
        del report["records"][0]["original_atoms"]
        with pytest.raises(ModuleConfigError, match="Run Check Upload again"):
            StructureProcessor().run(
                uploaded,
                {
                    **config,
                    "modifications": [{"index": 1, "patch_id": "PHOS_SER"}],
                },
            )
        return
    if condition == "missing":
        report["records"][0]["original_atoms"] = [
            atom for atom in report["records"][0]["original_atoms"] if atom["name"] != "O1P"
        ]
    checked = (
        StructureProcessor()
        .run(
            uploaded,
            {
                **config,
                "modifications": [{"index": 1, "patch_id": "PHOS_SER"}],
            },
        )
        .system
    )
    disposition = checked.metadata["modification_geometry"][0]["deposited_geometry"]
    assert residue_names(checked)[2] == "SEP"
    assert "P" in disposition["retained_deposited_atoms"]
    if condition == "missing":
        assert disposition["rebuilt_atoms"] == ["O1P"]
    for name in disposition["retained_deposited_atoms"]:

        def coordinate(s):
            index = next(
                i
                for i, (rid, atom) in enumerate(zip(s.resids, s.atom_names))
                if rid == 2 and atom == name
            )
            return s.coordinates[index]

        np.testing.assert_array_equal(coordinate(checked.structure), coordinate(original.structure))


# These boundaries come from the independent grompp audit, not the runtime allowlist.
@pytest.mark.parametrize(
    "ff,sequence,patch",
    [
        ("amber14sb", "SAA", "PHOS_SER"),
        ("amber99sb", "AAP", "HYP_PRO"),
        ("charmm36m", "PAA", "HYP_PRO"),
        ("charmm36m", "AAC", "CSO_CYS"),
    ],
)
def test_unvalidated_free_terminal_modification_is_rejected_early(ff, sequence, patch):
    skip_without_forcefield(ff)
    position = 0 if sequence[0] != "A" else 2
    with pytest.raises(ModuleConfigError, match="terminal"):
        StructureProcessor().run(
            peptide(sequence, ff),
            {
                "skip_protonation": True,
                "modifications": [{"index": position, "patch_id": patch}],
            },
        )


@pytest.mark.parametrize(
    "ff,sequence,patch,net_charge",
    [
        ("amber14sb", "ASA", "PHOS_SER", -2),
        ("charmm36m", "ASA", "PHOS_SER", -1),
        ("oplsaa", "ANA", "DEA_ASN", -1),
    ],
)
def test_in_memory_atom_parameters_match_export_and_expected_charge(
    tmp_path, ff, sequence, patch, net_charge
):
    skip_without_forcefield(ff)
    system = (
        StructureProcessor()
        .run(
            peptide(sequence, ff),
            {
                "skip_protonation": True,
                "modifications": [{"index": 1, "patch_id": patch}],
            },
        )
        .system
    )
    system = ForceFieldAssigner().run(system, {}).system
    assert sum(a.charge for a in system.topology.atom_types) == pytest.approx(net_charge, abs=1e-5)
    path = tmp_path / "protein.itp"
    TopologyWriter(ff)._write_protein_itp_for_indices(
        system.structure,
        path,
        list(range(system.structure.num_atoms)),
        topology=system.topology,
    )
    section = ""
    exported = []
    for line in path.read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[] ")
        elif section == "atoms" and line:
            row = line.split()
            exported.append((row[4], row[1], float(row[6])))
    assert len(exported) == system.structure.num_atoms
    for i, (name, atom_type, charge) in enumerate(exported):
        assert name == system.structure.atom_names[i]
        assert atom_type == system.topology.atom_types[i].name
        assert charge == pytest.approx(system.topology.atom_types[i].charge, abs=1e-6)


def test_chemistry_fingerprint_ignores_only_runtime_injection():
    checked = {"pH": 7.0, "modifications": [{"index": 1, "patch_id": "DEA_ASN"}]}
    assert chemistry_fingerprint(checked) == chemistry_fingerprint(
        {**checked, "seed": 42, "_task_dir": "/tmp/x"}
    )
    assert chemistry_fingerprint(checked) != chemistry_fingerprint({**checked, "pH": 2.0})


def test_frontend_edit_invalidates_checks_and_serializes_identity():
    script = r"""
const fs = require('fs'), vm = require('vm');
const c = {console, window: {}, document: {getElementById:()=>null, querySelector:()=>null},
  state:{wizardSteps:['input','structure','solvation','ions'],completedSteps:new Set([0,1,2,3])},
  _checkedSteps:new Set(['input','structure','solvation','ions']),
  _checkedConfig:{structure:{modifications:[]},ions:{}},
  updateNextButtonState:()=>{},updateStepNavHighlight:()=>{}};
vm.createContext(c);
vm.runInContext(fs.readFileSync('src/gmxbuilder/web/static/app_parts/structure_processing.js','utf8'),c);
const result = vm.runInContext(`
  _procResidues = [{chain:'A',resid:2,resname:'ASN',index:0},
    {chain:'A',resid:3,resname:'ASN',index:1}];
  applyModification(1,'DEA_ASN','ASP',-1);
  JSON.stringify({mods:serializeStructureModifications(),checks:[..._checkedSteps],config:_checkedConfig,
     completed:[...state.completedSteps]});`,c);
console.log(result);
"""
    process = subprocess.run(
        ["node", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(process.stdout)
    assert result["checks"] == ["input"]
    assert result["completed"] == [0]
    assert result["config"] == {}
    assert result["mods"] == [
        {
            "index": 1,
            "target": {"chain": "A", "resid": 3, "resname": "ASN"},
            "patch_id": "DEA_ASN",
        }
    ]


@pytest.mark.parametrize("changed", [True, False])
def test_finalization_checks_chemistry_before_running_modules(tmp_path, monkeypatch, changed):
    from gmxbuilder.modules.modifications.selection import CHEMISTRY_POLICY_VERSION
    from gmxbuilder.pipeline.step_executor import StepRunner

    runner = StepRunner(tmp_path, pipeline_type="pure-membrane")
    system = peptide("AAA")
    config = {"skip_protonation": True, "modifications": []}
    system.metadata["structure_chemistry"] = {
        "policy_version": CHEMISTRY_POLICY_VERSION,
        "config_sha256": chemistry_fingerprint(config),
    }
    system.save_checkpoint(runner.step_dir("membrane"))
    called = []

    def get_module(*args):
        called.append(args)
        raise RuntimeError("Reached topology module")

    monkeypatch.setattr("gmxbuilder.pipeline.step_executor._get_module", get_module)
    if changed:
        config["modifications"] = [{"index": 1, "patch_id": "DEA_ASN"}]
        result = runner.finalize_from_checkpoint("membrane", structure_config=config)
        assert result["status"] == "error"
        assert "Chemistry does not match" in result["error"]
        assert not called
    else:
        with pytest.raises(RuntimeError, match="Reached topology"):
            runner.finalize_from_checkpoint("membrane", structure_config=config)
        assert called


@pytest.mark.parametrize("ff", ["amber14sb", "charmm36m", "oplsaa"])
def test_native_atomtypes_use_standard_hydrogen_mass(ff):
    skip_without_forcefield(ff)
    from gmxbuilder.modules.forcefield.protein_templates import _nonbonded_types
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    atomtypes = _nonbonded_types(ff)
    template = load_force_field_rtp(ff).get_residue("ALA")
    hydrogens = [atom for atom in template["atoms"] if atom[0].startswith("H")]
    assert hydrogens
    for atom in hydrogens:
        assert atomtypes[atom[1]][0] == pytest.approx(1.008, abs=0.001)

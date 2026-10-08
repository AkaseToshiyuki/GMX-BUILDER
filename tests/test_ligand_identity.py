"""Chemical identity recovery, ambiguity gates and task-owned MOL2 inputs."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.forcefield import ligand_identity as identity
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from gmxbuilder.web import server
from gmxbuilder.web.server_parts.ligand_chemistry import trusted_config
from tests.prerequisites import requires_forcefield, requires_gaff_runtime, requires_gromacs
from tests.test_charmm_complex import AMP_R, AMP_S, amp_system


def ethanol():
    structure = Structure(
        coordinates=np.array([[0, 0, 0], [1.51, 0, 0], [2.02, 1.33, 0]]) / 10,
        box_vectors=np.eye(3) * 3,
        atom_names=["C1", "C2", "O1"],
        elements=["C", "C", "O"],
        resnames=["LIG"] * 3,
        resids=[1] * 3,
    )
    return System(structure, components=[Component("LIG", ComponentKind.UNKNOWN, np.arange(3))])


def ethanol_mol2():
    """Independent explicit bond/H input, deliberately permuted atom records."""
    mol = Chem.MolFromSmiles("CCO")
    conf = Chem.Conformer(3)
    conf.Set3D(True)
    for i, xyz in enumerate(ethanol().structure.coordinates * 10):
        conf.SetAtomPosition(i, tuple(xyz))
    mol.AddConformer(conf)
    mol = Chem.AddHs(mol, addCoords=True)
    order = [2, 0, 1] + list(range(3, mol.GetNumAtoms()))
    names = ["C1", "C2", "O1"] + [f"H{i}" for i in range(3, mol.GetNumAtoms())]
    lines = [
        "@<TRIPOS>MOLECULE",
        "LIG",
        f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for i, idx in enumerate(order):
        atom = mol.GetAtomWithIdx(idx)
        xyz = mol.GetConformer().GetAtomPosition(idx)
        kind = "H" if atom.GetAtomicNum() == 1 else atom.GetSymbol() + ".3"
        # Deliberately arbitrary partial charges: identity must not adopt them.
        lines.append(f"{i + 1} {names[idx]} {xyz.x:.5f} {xyz.y:.5f} {xyz.z:.5f} {kind} 1 LIG 0.123")
    lines.append("@<TRIPOS>BOND")
    for i, bond in enumerate(mol.GetBonds()):
        lines.append(
            f"{i + 1} {order.index(bond.GetBeginAtomIdx()) + 1} "
            f"{order.index(bond.GetEndAtomIdx()) + 1} 1"
        )
    return ("\n".join(lines) + "\n").encode()


CCD = """data_LIG
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
LIG C1 C 0
LIG C2 C 0
LIG O1 O 0
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
LIG C1 C2 SING
LIG C2 O1 SING
#
"""


def test_mol2_preserves_original_order_coordinates_and_ignores_partial_charges(tmp_path):
    system = ethanol()
    before = system.structure.coordinates.copy()
    path = tmp_path / "ligand.mol2"
    path.write_bytes(ethanol_mol2())
    report = identity.resolve_identity("LIG", system.structure, [0, 1, 2], mol2_path=path, pH=13)
    assert report["smiles"] == "CCO"
    assert report["net_charge"] == 0
    assert report["source"] == "user_mol2"
    assert report["source_sha256"]
    mapped = Chem.MolFromSmiles(report["mapped_smiles"])
    assert [(a.GetSymbol(), a.GetAtomMapNum()) for a in mapped.GetAtoms()] == [
        ("C", 1),
        ("C", 2),
        ("O", 3),
    ]
    np.testing.assert_array_equal(before, system.structure.coordinates)


@pytest.mark.parametrize(
    "payload", [b"invalid", ethanol_mol2() * 2, ethanol_mol2().replace(b"C.3", b"N.3", 1)]
)
def test_invalid_or_wrong_mol2_cannot_supply_identity(tmp_path, payload):
    path = tmp_path / "input.mol2"
    path.write_bytes(payload)
    system = ethanol()
    with pytest.raises((identity.LigandIdentityError, ValueError)):
        identity.resolve_identity("LIG", system.structure, [0, 1, 2], mol2_path=path)


def test_explicit_state_bypasses_network_and_ph_model(monkeypatch):
    monkeypatch.setattr(identity, "_ccd", lambda *_: pytest.fail("unexpected network access"))
    monkeypatch.setattr(identity, "_at_ph", lambda *_: pytest.fail("explicit state was rewritten"))
    system = amp_system()
    result = identity.resolve_identity("AMP", system.structure, range(10), smiles=AMP_S, pH=13)
    assert result["smiles"] == AMP_S
    assert result["net_charge"] == 1
    with pytest.raises(identity.LigandIdentityError, match="stereochemistry"):
        identity.resolve_identity("AMP", system.structure, range(10), smiles=AMP_R)


@pytest.mark.parametrize("smiles", ["C[CH2:1]O", "[CH3:1][CH2:1][OH:3]", "CCN"])
def test_bad_map_and_wrong_chemistry_block(smiles):
    system = ethanol()
    with pytest.raises(identity.LigandIdentityError):
        identity.resolve_identity("LIG", system.structure, range(3), smiles=smiles)


def test_ambiguous_protonation_returns_clarification(monkeypatch):
    system = ethanol()
    monkeypatch.setattr(identity, "_ccd", lambda _: CCD)
    # A dictionary identity alone must not conceal an ambiguous pH model.
    monkeypatch.setattr(
        identity,
        "_convert",
        lambda payload, fmt, pH: Chem.MolFromSmiles("C[NH3+]" if pH <= 7 else "CN"),
    )
    with pytest.raises(identity.LigandIdentityError, match="protonation states"):
        identity.resolve_identity("LIG", system.structure, range(3))


def test_embedded_definition_precedes_ccd_and_survives_filtered_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "original.cif"
    path.write_text(CCD)
    monkeypatch.setattr(identity, "_ccd", lambda _: pytest.fail("embedded source was ignored"))
    monkeypatch.setattr(identity, "_at_ph", lambda mol, pH: mol)
    system = ethanol()
    result = identity.resolve_identity("LIG", system.structure, range(3), source_path=path)
    assert result["source"] == "input_mmcif"
    assert result["smiles"] == "CCO"


def test_ccd_retries_transient_failure_and_caches_success(tmp_path, monkeypatch):
    import io

    monkeypatch.setenv("GMXBUILDER_CCD_CACHE", str(tmp_path))
    urls = []

    def download(url, timeout):
        urls.append(url)
        if len(urls) == 1:
            raise OSError("temporarily offline")
        return io.BytesIO(CCD.replace("LIG", "EOH").encode())

    monkeypatch.setattr(identity.urllib.request, "urlopen", download)
    assert identity._ccd("EOH") is None
    assert identity._ccd("EOH") == CCD.replace("LIG", "EOH")
    assert identity._ccd("EOH") == CCD.replace("LIG", "EOH")
    assert urls == ["https://files.rcsb.org/ligands/download/EOH.cif"] * 2


def test_ccd_uses_managed_cache_home_and_keeps_verified_data_on_write_failure(
    tmp_path, monkeypatch
):
    import io

    raw = CCD.replace("LIG", "EOH")
    monkeypatch.delenv("GMXBUILDER_CCD_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        identity.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw.encode())
    )
    assert identity._ccd("EOH") == raw
    assert (tmp_path / "gmxbuilder/ccd/EOH.cif").read_text() == raw
    # A regular file cannot become a cache directory, independent of UID/mode.
    forbidden = tmp_path / "not-a-directory"
    forbidden.write_text("occupied")
    monkeypatch.setenv("GMXBUILDER_CCD_CACHE", str(forbidden / "ccd"))
    assert identity._ccd("EOH") == raw
    assert forbidden.read_text() == "occupied"


def test_name_is_a_hint_not_identity(monkeypatch):
    monkeypatch.setattr(identity, "_ccd", lambda _: CCD.replace("O1 O 0", "O1 N 0"))
    system = ethanol()
    monkeypatch.setattr(
        identity,
        "_perceive",
        lambda *_: (_ for _ in ()).throw(
            identity.LigandIdentityError("coordinate evidence remains unresolved")
        ),
    )
    with pytest.raises(identity.LigandIdentityError, match="coordinate evidence"):
        identity.resolve_identity("LIG", system.structure, range(3))


@pytest.mark.slow
@requires_gaff_runtime
def test_hydrogen_complete_pdb_and_covalent_link_detection(tmp_path, monkeypatch):
    from gmxbuilder.io.pdb import PDBWriter

    mol, _ = identity.read_mol2(ethanol_mol2())
    structure = Structure(
        coordinates=mol.GetConformer().GetPositions() / 10,
        box_vectors=np.eye(3) * 3,
        atom_names=[a.GetProp("_TriposAtomName") for a in mol.GetAtoms()],
        elements=[a.GetSymbol() for a in mol.GetAtoms()],
        resnames=["LIG"] * mol.GetNumAtoms(),
        resids=[1] * mol.GetNumAtoms(),
    )
    path = tmp_path / "original.pdb"
    PDBWriter.write(structure, path)
    monkeypatch.setattr(identity, "_ccd", lambda _: pytest.fail("explicit H were ignored"))
    report = identity.resolve_identity("LIG", ethanol().structure, range(3), source_path=path)
    assert report["source"] == "input_pdb"
    assert report["smiles"] == "CCO"
    with path.open("a") as handle:
        handle.write("CONECT    1  999\n")
    with pytest.raises(identity.LigandIdentityError, match="another residue or metal"):
        identity.resolve_identity("LIG", ethanol().structure, range(3), source_path=path)


@pytest.mark.slow
@requires_gaff_runtime
def test_heavy_only_mol2_uses_supplied_bonds_to_complete_hydrogens(tmp_path):
    payload = ethanol_mol2().decode()
    header, rest = payload.split("@<TRIPOS>ATOM")
    atoms = rest.split("@<TRIPOS>BOND")[0].splitlines()[1:4]
    payload = header.replace("9 8", "3 2") + "@<TRIPOS>ATOM\n" + "\n".join(atoms)
    payload += "\n@<TRIPOS>BOND\n1 2 3 1\n2 3 1 1\n"
    path = tmp_path / "heavy.mol2"
    path.write_text(payload)
    result = identity.resolve_identity("LIG", ethanol().structure, range(3), mol2_path=path)
    assert result["smiles"] == "CCO"
    assert result["net_charge"] == 0


def test_missing_perception_dependency_requests_input_but_smiles_still_works(monkeypatch):
    monkeypatch.setattr(
        identity,
        "_obabel",
        lambda: (_ for _ in ()).throw(identity.LigandIdentityError("Open Babel is unavailable")),
    )
    system = ethanol()
    with pytest.raises(identity.LigandIdentityError, match="Open Babel is unavailable"):
        identity.resolve_identity("LIG", system.structure, range(3), use_ccd=False)
    assert (
        identity.resolve_identity("LIG", system.structure, range(3), smiles="CCO")["smiles"]
        == "CCO"
    )


@pytest.mark.slow
@requires_gaff_runtime
@pytest.mark.parametrize("pH,charge", [(7.0, 1), (13.0, 0)])
def test_real_amp_auto_from_coordinates_and_ph(pH, charge, monkeypatch):
    monkeypatch.setattr(identity, "_ccd", lambda _: None)
    system = amp_system()
    report = identity.resolve_identity("AMP", system.structure, range(10), pH=pH)
    assert report["net_charge"] == charge
    assert "@" in report["smiles"]
    assert report["source"] == "coordinate_perception"


@pytest.mark.slow
@requires_gaff_runtime
def test_real_coordinate_tautomer_is_not_silently_selected(monkeypatch):
    from tests.test_charmm_compat import molecule_system

    system = molecule_system("CC(=O)O")
    with pytest.raises(identity.LigandIdentityError, match="tautomer"):
        identity.resolve_identity("LIG", system.structure, range(system.num_atoms), use_ccd=False)


def task(tmp_path, monkeypatch):
    monkeypatch.setattr(server.task_manager, "root", tmp_path)
    state = server.task_manager.create_task(filename="ligand.pdb")
    task_id = state["task_id"]
    ethanol().save_checkpoint(tmp_path / task_id / "steps/input")
    return task_id


def test_upload_preview_resume_and_trusted_inputs(tmp_path, monkeypatch):
    task_id = task(tmp_path, monkeypatch)
    with TestClient(server.app) as client:
        response = client.post(
            f"/api/ligand-chemistry-upload/{task_id}",
            data={"ligand_name": "LIG"},
            files={"mol2_file": ("ligand.mol2", ethanol_mol2())},
        )
        assert response.status_code == 200, response.text
        uploaded = response.json()
        config = {"charmm_compat_mol2": {"LIG": {"uploaded": True, "sha256": uploaded["sha256"]}}}
        response = client.post(f"/api/ligand-chemistry/{task_id}", json=config)
        assert response.status_code == 200, response.text
        assert response.json()["ligands"]["LIG"]["source"] == "user_mol2"
        invalid = client.post(
            f"/api/ligand-chemistry-upload/{task_id}",
            data={"ligand_name": "LIG"},
            files={"mol2_file": ("wrong.mol2", b"invalid")},
        )
        assert invalid.status_code == 400
        for bad in [{"uploaded": True, "sha256": "wrong"}, "/etc/passwd"]:
            response = client.post(
                f"/api/ligand-chemistry/{task_id}", json={"charmm_compat_mol2": {"LIG": bad}}
            )
            assert response.status_code == 400
    trusted = trusted_config(task_id, config, server.task_manager, server._validate_task_resource)
    assert trusted["charmm_compat_mol2"]["LIG"].endswith(".mol2")
    public = server._task_resource_helpers.public_task_state(server.task_manager.get_state(task_id))
    assert public["ligand_chemistry_uploads"]["LIG"]["sha256"] == uploaded["sha256"]
    assert "file" not in public["ligand_chemistry_uploads"]["LIG"]


def test_preview_reports_individual_ambiguity_and_never_uses_empty_override(tmp_path, monkeypatch):
    task_id = task(tmp_path, monkeypatch)
    with TestClient(server.app) as client:
        response = client.post(
            f"/api/ligand-chemistry/{task_id}", json={"charmm_compat_smiles": {"LIG": ""}}
        )
    assert response.status_code == 200
    assert response.json()["ligands"]["LIG"]["status"] == "needs_input"


@pytest.mark.slow
@requires_gaff_runtime
@requires_gromacs
@requires_forcefield("charmm36m")
@requires_forcefield("charmm36")
@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
def test_real_auto_selector_constructs_without_smiles(ff, tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "_ccd", lambda _: None)
    system = ethanol()
    before = dict(zip(system.structure.atom_names, system.structure.coordinates.copy()))
    result = ForceFieldSelector().run(system, {"name": ff, "_task_dir": str(tmp_path)})
    structure = result.system.structure
    for name, xyz in before.items():
        np.testing.assert_array_equal(structure.coordinates[structure.atom_names.index(name)], xyz)
    reports = list(tmp_path.rglob("report.json"))
    assert reports
    report = json.loads(reports[0].read_text())
    assert report["chemical_identification"]["source"] == "coordinate_perception"
    assert report["chemical_identification"]["net_charge"] == 0
    assert report["gromacs_preprocessing"] == "passed_maxwarn_0"


def test_ccd_survives_real_landlock_cache_denial(tmp_path):
    import subprocess
    import sys

    fixture = tmp_path / "ccd.cif"
    fixture.write_text(CCD.replace("LIG", "EOH"))
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    code = """
import io,os,sys
from pathlib import Path
from gmxbuilder.modules.forcefield import ligand_identity as identity
from gmxbuilder.web.write_sandbox import restrict_writes
raw = Path(sys.argv[1]).read_text()
os.environ['GMXBUILDER_CCD_CACHE'] = sys.argv[2]
identity.urllib.request.urlopen = lambda *a, **k: io.BytesIO(raw.encode())
restrict_writes(Path(sys.argv[3]))
assert identity._ccd('EOH') == raw
assert not Path(sys.argv[2]).exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture), str(tmp_path / "denied"), str(allowed)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

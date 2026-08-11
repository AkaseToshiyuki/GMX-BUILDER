from pathlib import Path
import subprocess

import numpy as np
import pytest
from rdkit import Chem

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.io.gro import GROWriter
from gmxbuilder.modules.forcefield.selector import ForceFieldSelector
from gmxbuilder.modules.forcefield.gaff_backend import (
    _itp_charges,
    estimate_gaff_net_charge,
    gaff_available,
    prepare_gaff_lipid,
    prepare_gaff_molecule,
)
from gmxbuilder.modules.forcefield.lipid_policy import (
    membrane_lipid_names,
    resolve_lipid_force_field,
)
from gmxbuilder.modules.membrane.lipids import CATEGORY_NAMES, LipidRegistry
from gmxbuilder.pipeline.config import PipelineConfig
from gmxbuilder.pipeline.pipeline import Pipeline
from tests.test_gromacs_smoke import _find_gmx
from tests.test_membrane_gromacs_smoke import _write_smoke_mdp


pytestmark = pytest.mark.skipif(
    not gaff_available(), reason="isolated AmberTools/ACPYPE environment unavailable"
)


def _small_molecule_system():
    structure = Structure(
        coordinates=np.array([[0.0, 0.0, 0.0], [0.145, 0.0, 0.0]]),
        box_vectors=np.eye(3) * 4.0,
        atom_names=["C01", "N02"], resnames=["LIG", "LIG"],
        resids=[1, 1], chain_ids=["L", "L"], elements=["C", "N"],
    )
    system = System(structure)
    system.add_component(Component(
        name="UNKNOWN", kind=ComponentKind.UNKNOWN,
        atom_indices=np.array([0, 1]),
    ))
    return system


def test_gaff2_coordinate_molecule_preserves_heavy_atoms_and_adds_hydrogens(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    system = _small_molecule_system()
    template = prepare_gaff_molecule("LIG", system.structure, [0, 1], 0)
    assert template.atom_names[:2] == ("C01", "N02")
    assert len(template.atom_names) > 2

    result = ForceFieldSelector().run(system, {
        "name": "amber99sb-ildn", "lipid_names": [], "lipid_ff": "none",
        "ligand_ff": "gaff2", "ligand_charges": {"LIG": 0},
    })
    assert result.system.num_atoms == len(template.atom_names)
    assert result.system.component_by_kind(ComponentKind.LIGAND)
    assert result.system.total_charge() == 0
    assert result.system.metadata["ligand_parameters"]["LIG"]["charge_method"] == "gas"

    output = tmp_path / "topology"
    output.mkdir()
    TopologyWriter("amber99sb-ildn", ff_config={
        "water_model": "tip3p",
        "ligand_parameters": result.system.metadata["ligand_parameters"],
    }).write_top(result.system.structure, output / "topol.top")
    assert '#include "LIG.itp"' in (output / "topol.top").read_text()
    GROWriter.write(result.system.structure, output / "input.gro")
    _write_smoke_mdp(output / "smoke.mdp")
    process = subprocess.run(
        [_find_gmx(), "grompp", "-f", "smoke.mdp", "-c", "input.gro",
         "-p", "topol.top", "-o", "smoke.tpr"],
        cwd=output, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr


def test_gaff_charge_suggestion_is_ph_dependent_for_primary_amine():
    system = _small_molecule_system()

    physiological = estimate_gaff_net_charge(
        "LIG", system.structure, [0, 1], pH=7.0,
    )
    basic = estimate_gaff_net_charge(
        "LIG", system.structure, [0, 1], pH=13.0,
    )

    assert physiological.net_charge == 1
    assert physiological.formula.endswith("+")
    assert basic.net_charge == 0


def test_gaff2_ph_protonation_preserves_uploaded_heavy_atom_names(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    system = _small_molecule_system()

    template = prepare_gaff_molecule(
        "LIG", system.structure, [0, 1], 1,
        charge_method="gas", target_pH=7.0,
    )

    assert template.atom_names[:2] == ("C01", "N02")
    assert sum(_itp_charges(template.itp_path)) == pytest.approx(1.0, abs=0.02)


def test_gaff_template_is_cached_and_namespaced(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path))
    lipid = LipidRegistry.get("CAMP")
    first = prepare_gaff_lipid("CAMP", lipid.smiles, lipid.charge, charge_method="gas")
    second = prepare_gaff_lipid("CAMP", lipid.smiles, lipid.charge, charge_method="gas")

    assert first.itp_path == second.itp_path
    assert first.coordinates.shape == (len(first.atom_names), 3)
    assert np.isfinite(first.coordinates).all()
    assert "[ atomtypes ]" in first.atomtypes_path.read_text()
    assert "g_camp_" in first.atomtypes_path.read_text()
    assert "[ atomtypes ]" not in first.itp_path.read_text()
    assert sum(_itp_charges(first.itp_path)) == pytest.approx(lipid.charge, abs=0.02)
    assert set(Path(tmp_path).glob("CAMP-*"))


def test_force_field_policy_preserves_rtp_and_uses_one_gaff_family():
    assert {
        LipidRegistry.get(name).category for name in LipidRegistry.list()
    } <= set(CATEGORY_NAMES)
    assert {
        name: (Chem.GetFormalCharge(Chem.MolFromSmiles(LipidRegistry.get(name).smiles)),
               LipidRegistry.get(name).charge)
        for name in LipidRegistry.list()
        if Chem.GetFormalCharge(Chem.MolFromSmiles(LipidRegistry.get(name).smiles))
        != LipidRegistry.get(name).charge
    } == {}
    rtp = resolve_lipid_force_field("charmm36m", ["POPC", "CHOL"])
    assert rtp.protein_force_field == "charmm36m"
    assert rtp.lipid_force_field == "charmm36m"
    assert not rtp.gaff_lipids

    camp = resolve_lipid_force_field("charmm36m", ["POPC", "CAMP"])
    assert camp.protein_force_field == "charmm36m"
    assert camp.lipid_force_field == "charmm36m"

    gaff = resolve_lipid_force_field("charmm36m", ["POPC", "20AHC"])
    assert gaff.protein_force_field == "amber14sb"
    assert gaff.lipid_force_field == "gaff2"
    assert gaff.gaff_lipids == ("20AHC", "POPC")

    assert membrane_lipid_names({"lipid_type": "popc"}) == ("POPC",)
    assert membrane_lipid_names({
        "lipid_composition": {
            "upper": [{"name": "POPC", "ratio": 50}, {"name": "camp", "ratio": 50}],
            "lower": [{"name": "POPC", "ratio": 100}],
        }
    }) == ("CAMP", "POPC")


def test_numeric_lipid_name_uses_gromacs_safe_molecule_type(monkeypatch, tmp_path):
    monkeypatch.setenv("GMXBUILDER_GAFF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GMXBUILDER_GAFF_CHARGE_METHOD", "gas")
    lipid = LipidRegistry.get("20AHC")
    template = prepare_gaff_lipid("20AHC", lipid.smiles, lipid.charge)
    structure = Structure(
        coordinates=template.coordinates,
        box_vectors=np.eye(3) * 12.0,
        atom_names=list(template.atom_names),
        resnames=["20AHC"] * len(template.atom_names),
        resids=[1] * len(template.atom_names),
    )
    top_path = tmp_path / "topol.top"
    TopologyWriter("amber99sb-ildn").write_top(structure, top_path)
    assert "L_20AHC" in top_path.read_text()


def test_pipeline_derives_policy_input_from_membrane_config():
    system = System(Structure(
        coordinates=np.empty((0, 3)), box_vectors=np.eye(3),
        atom_names=[], resnames=[], resids=[],
    ))
    pipeline = Pipeline().add_module(ForceFieldSelector())
    config = PipelineConfig(modules={
        "forcefield": {
            "name": "amber99sb-ildn", "lipid_ff": "lipid21",
            "ligand_ff": "none",
        },
        "membrane": {
            "lipid_composition": {
                "upper": [{"name": "POPC", "ratio": 50},
                          {"name": "DAPC", "ratio": 50}],
            }
        },
    })
    result = pipeline.run(system, config)
    assert result.system.metadata["force_field"] == "amber99sb-ildn"
    assert result.system.metadata["lipid_ff"] == "lipid21"
    assert result.system.metadata["lipid21_lipids"] == ["DAPC", "POPC"]
    assert result.system.metadata["ligand_ff"] == "none"

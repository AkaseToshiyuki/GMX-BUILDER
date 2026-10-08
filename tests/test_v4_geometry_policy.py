"""Physical population and budget boundaries, without starting dynamics."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.modules.membrane.lipid_equilibration import _orientation_gate
from gmxbuilder.modules.membrane.v4_trajectory import TrajectorySampler


def mixed_frame(target, horizontal_guests):
    sampler = TrajectorySampler.__new__(TrajectorySampler)
    sampler.protocol = {"lipid": target}
    sampler.n_leaflet = 200
    sampler.groups, blocks = [], []
    for name, count in (("POPC", 360), (target, 40)):
        upper = np.arange(count) < count // 2
        sign = np.where(upper, 1.0, -1.0)
        xyz = np.zeros((count, 2, 3))
        xyz[:, 0, 2] = sign * 2
        xyz[:, 1, 2] = sign * 0.05
        if name == target:
            xyz[:horizontal_guests, 1, 2] = xyz[:horizontal_guests, 0, 2]
            xyz[:horizontal_guests, 1, 0] = 1
        blocks.append(xyz)
        sampler.groups.append(
            {"name": name, "polar": [0], "tails": [1], "upper": upper, "anchor": 0}
        )
    return sampler, blocks


@pytest.mark.parametrize("name,passed", [("25OHC", True), ("CER18", False)])
def test_trajectory_and_final_frame_share_chemical_exception(name, passed):
    sampler, blocks = mixed_frame(name, 40)
    _, actual, evidence = sampler.metrics(np.eye(3) * 10, blocks)
    projections = np.r_[np.full(360, 1.95), np.zeros(40)]
    cosines = np.r_[np.ones(360), np.zeros(40)]
    final, profile, _, _ = _orientation_gate(
        name,
        projections,
        cosines,
        projections[:360],
        cosines[:360],
        upper_count=20,
        lower_count=20,
    )
    assert actual == final == passed
    assert profile == evidence["orientation"]["profile"]
    assert evidence["orientation"]["species_leaflet"][f"{name}_upper"]["correct_fraction"] == 0


def test_global_pass_does_not_hide_dilute_guest_outliers_in_evidence():
    sampler, blocks = mixed_frame("CER18", 4)
    _, passed, evidence = sampler.metrics(np.eye(3) * 10, blocks)
    assert passed  # Preserve the reviewed global threshold, rather than silently replace it.
    assert evidence["oriented_fraction"] == pytest.approx(0.99)
    guest = evidence["orientation"]["species_leaflet"]["CER18_upper"]
    assert guest["outlier_count"] == 4
    assert guest["correct_fraction"] == pytest.approx(0.8)


def test_oxysterol_without_host_cannot_receive_an_empty_population_exemption():
    passed, _, _, _ = _orientation_gate(
        "25OHC",
        np.ones(40),
        np.ones(40),
        np.array([]),
        np.array([]),
        upper_count=20,
        lower_count=20,
    )
    assert not passed


@pytest.mark.parametrize("maximum", ["200.01", "250", "inf", "nan"])
def test_queue_refuses_any_budget_above_200_before_starting_work(monkeypatch, maximum):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_v4_library as queue

    monkeypatch.setattr(sys, "argv", ["build_v4_library.py", "--maximum-ns", maximum])
    with pytest.raises(SystemExit) as error:
        queue.main()
    assert error.value.code == 2


def test_computed_uncertainty_does_not_claim_scientific_calibration():
    from gmxbuilder.modules.membrane.v4_sampling import assess_series, sampling_failures

    rng = np.random.default_rng(17)
    values = {
        key: mean + rng.normal(0, 0.001, 1000)
        for key, mean in (
            ("area_per_lipid_nm2", 0.6),
            ("head_to_head_nm", 4),
            ("box_z_nm", 8),
            ("volume_nm3", 400),
        )
    }
    result = assess_series(np.arange(1000) * 100, values)
    assert result["stationarity"]["uncertainty_estimated"]
    assert result["stationarity"]["calibration_status"] == "not-established"
    assert "calibrated" not in result["stationarity"]
    legacy = copy.deepcopy(result)
    legacy["stationarity"]["calibrated"] = legacy["stationarity"].pop("uncertainty_estimated")
    assert sampling_failures(legacy) == sampling_failures(result)


def test_collection_rejects_inverted_host_before_writing_target(tmp_path, monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from gmxbuilder.modules.membrane import v4_trajectory as module

    smiles = "C[C@H](O)C(=O)O"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=721) == 0
    xyz = molecule.GetConformer().GetPositions() / 10
    graph = {
        "smiles": smiles,
        "elements": tuple(a.GetSymbol() for a in molecule.GetAtoms()),
        "bonds": tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()),
    }
    sampler = TrajectorySampler.__new__(TrajectorySampler)
    sampler.protocol = {"lipid": "TARGET"}
    sampler.groups = [{**graph, "name": "TARGET"}, {**graph, "name": "HOST"}]
    monkeypatch.setattr(
        module,
        "assess_local",
        lambda _: {"analysis_window_ps": [0, 1000], "sample_spacing_ps": 100},
    )
    monkeypatch.setattr(
        sampler,
        "frames",
        lambda _: iter([(0, np.eye(3) * 10, [xyz[None], (xyz @ np.diag([-1, 1, 1]))[None]])]),
    )
    monkeypatch.setattr(sampler, "metrics", lambda *a, **kw: (None, True, None))
    with pytest.raises(ValueError, match="stereochemistry mismatch"):
        sampler.collect("unused", {"local_conformations": {}}, tmp_path)
    assert not list(tmp_path.glob("conf_*.npz"))

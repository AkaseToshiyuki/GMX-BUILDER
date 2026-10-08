"""Coordinate provenance is checked separately from statistical acceptance."""

import copy

import pytest

from gmxbuilder.modules.membrane.equilibrated_library import conformer_files
from gmxbuilder.modules.membrane.v4_evidence import (
    assembled_evidence_valid,
    replica_evidence_failures,
)
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from tests.local_relaxation_fixture import local_fixture
from tests.test_v4_atom_selection import synthetic_selection


@pytest.fixture
def evidence():
    protocol = resolve_protocol("POPC", "charmm36m-lipid")
    metadata = {
        "npt_ps": 200000,
        "n_conformations": 10,
        "conformer_validation": {"identity_passed": True, "intrinsic_geometry_passed": True},
        "conformer_provenance": [
            {"file": f"conf_{i:04d}.npz", "time_ps": 100000 + i * 1000} for i in range(10)
        ],
        "trajectory_analysis": {
            "atom_selections": [synthetic_selection()],
            "actual_leaflet_counts": {"upper": {"POPC": 64}, "lower": {"POPC": 64}},
            "observables": {"POPC_upper_axis_P2": {}, "POPC_lower_axis_P2": {}},
            "analysis_window_ps": [0, 1000],
            "trajectory_end_ps": 200000,
            "sample_spacing_ps": 100,
            "local_conformations": local_fixture(protocol["smiles"], "POPC"),
        },
    }
    return metadata, protocol


def test_ten_distinct_times_qualify_but_ten_molecules_at_one_time_do_not(evidence):
    metadata, protocol = evidence
    assert not replica_evidence_failures(metadata, protocol)
    for record in metadata["conformer_provenance"]:
        record["time_ps"] = 100000
    assert replica_evidence_failures(metadata, protocol) == [
        "fewer than ten retained sampling times"
    ]


@pytest.mark.parametrize("defect", ["count", "endpoint", "spacing", "duplicate", "identity"])
def test_inconsistent_provenance_is_not_repaired_by_running_longer(evidence, defect):
    metadata, protocol = copy.deepcopy(evidence)
    if defect == "count":
        metadata["trajectory_analysis"]["actual_leaflet_counts"]["upper"]["POPC"] = 63
    elif defect == "endpoint":
        metadata["trajectory_analysis"]["trajectory_end_ps"] = 900
    elif defect == "spacing":
        metadata["conformer_provenance"][1]["time_ps"] = 10
    elif defect == "duplicate":
        metadata["conformer_provenance"][1]["file"] = "conf_0000.npz"
    else:
        metadata["conformer_validation"]["identity_passed"] = False
    with pytest.raises(ValueError):
        replica_evidence_failures(metadata, protocol)


def test_replica_candidate_is_not_a_published_library(evidence):
    metadata, protocol = evidence
    metadata.update(lipid_name="POPC", parameter_family="charmm36m-lipid", v4_protocol=protocol)
    assert not assembled_evidence_valid(metadata)


def test_numeric_conformer_order_survives_five_digit_indices(tmp_path):
    for index in (9999, 10000, 1):
        (tmp_path / f"conf_{index:04d}.npz").touch()
    assert [p.stem for p in conformer_files(tmp_path)] == ["conf_0001", "conf_9999", "conf_10000"]

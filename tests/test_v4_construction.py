"""Construction admission contracts, independently of long-running MD.

Synthetic coordinates below exercise storage/reader contracts only. They do not
claim to validate molecular geometry; molecular-identity tests cover that layer.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.modules.membrane.equilibrated_library import (
    ACCEPTED_METHOD,
    SCHEMA_VERSION,
    EquilibratedLipidLibrary,
    topology_signature,
)
from gmxbuilder.modules.membrane.v4_construction import qualify_construction
from gmxbuilder.modules.membrane.v4_platform import construction_policy, platform_metrics
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from gmxbuilder.modules.membrane.v4_reanalysis import analysis_upgrade_allowed
from gmxbuilder.modules.membrane.v4_sampling import assess_series
from tests.dry_initial_fixture import dry_initial_fixture
from tests.local_relaxation_fixture import local_fixture, molecule_definition

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_v4_library as queue  # noqa: E402
from assemble_v4_library import assemble_entry  # noqa: E402


@pytest.fixture
def pair(parameter_sources):
    protocol = resolve_protocol("POPC", "charmm36m-lipid")
    result = []
    for seed in (6401, 6402):
        rng = np.random.default_rng(seed)
        series = {
            key: mean + rng.normal(0, mean * 0.001, 2001)
            for key, mean in {
                "area_per_lipid_nm2": 0.65,
                "head_to_head_nm": 3.8,
                "box_z_nm": 8.0,
                "volume_nm3": 400.0,
                "POPC_upper_axis_P2": 0.7,
                "POPC_lower_axis_P2": 0.7,
            }.items()
        }
        analysis = assess_series(np.arange(2001) * 100.0, series)
        from gmxbuilder.modules.membrane.local_relaxation import assess_local

        selection, _ = molecule_definition(protocol["smiles"], "POPC")
        analysis["local_conformations"] = local_fixture(protocol["smiles"], "POPC", seed=seed)
        local = assess_local(analysis["local_conformations"])
        analysis.update(
            actual_leaflet_counts={"upper": {"POPC": 64}, "lower": {"POPC": 64}},
            trajectory_end_ps=200000.0,
            atom_selections=[selection],
        )
        start = local["analysis_window_ps"][0]
        spacing = max(100.0, np.ceil(local["sample_spacing_ps"] / 100.0) * 100.0)
        names = selection["atom_names"]
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "method": ACCEPTED_METHOD,
                "coordinate_handedness": "preserved",
                "leaflet_transform": "proper_rotation",
                "lipid_name": "POPC",
                "parameter_family": "charmm36m-lipid",
                "force_field": "charmm36m",
                "lipid_ff": "charmm36m",
                "canonical_smiles": protocol["smiles"],
                "temperature_K": protocol["temperature_K"],
                "equilibration_host": None,
                "v4_protocol": protocol,
                "atom_names": names,
                "topology_sha256": topology_signature(names, "charmm36m", "charmm36m"),
                "status": "ready",
                "test_mode": False,
                "npt_ps": 200000.0,
                "n_conformations": 20,
                "quality": {
                    "initial_water_exclusion": dry_initial_fixture(),
                    "passed": True,
                    "apl_ratio": 1.0,
                    "dhh_ratio": 1.0,
                    "orientation": {
                        "passed": True,
                        "correct_fraction": 1.0,
                        "upper_lipids": 64,
                        "lower_lipids": 64,
                        "n_lipids_checked": 20,
                    },
                    "hydrophobic_core": {"passed": True, "tail_core_gap_nm": 0.1},
                },
                "conformer_validation": {
                    "identity_passed": True,
                    "intrinsic_geometry_passed": True,
                },
                "conformer_provenance": [
                    {
                        "file": f"conf_{i:04d}.npz",
                        "time_ps": start + i * spacing,
                        "leaflet": "upper",
                        "molecule_index": 0,
                    }
                    for i in range(20)
                ],
                "trajectory_analysis": analysis,
                "observables": {
                    "measures": "pure-bilayer-area-per-lipid",
                    "area_per_lipid_nm2": analysis["observables"]["area_per_lipid_nm2"],
                },
            }
        )
    from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

    for metadata in result:
        metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
    return result, protocol


def test_narrow_construction_scope_does_not_overwrite_uncertain_certification(pair):
    records, protocol = pair
    for metadata in records:
        metadata["trajectory_analysis"]["stationarity"]["uncertainty_estimated"] = False
    result = qualify_construction(records, protocol)
    assert result["status"] == "accepted", result
    assert result["warnings"]
    assert result["policy"]["equilibrium_certification"] is False
    assert records[0]["trajectory_analysis"]["stationarity"]["uncertainty_estimated"] is False


def test_old_bulk_statistics_do_not_discard_recomputed_local_evidence(pair):
    from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD

    records, protocol = pair
    for metadata in records:
        analysis = metadata["trajectory_analysis"]
        analysis.pop("autocorrelation_method")
        analysis["schema"] = "v4-common-window-3"
        analysis["stationarity"].pop("autocorrelation_method")
        for key, values in analysis["construction_platform"]["series"].items():
            centred = np.asarray(values) - np.mean(values)
            products = [
                sum(float(a * b) for a, b in zip(centred[: len(values) - lag], centred[lag:]))
                for lag in range(len(values))
            ]
            correlation = 1.0
            for lag in range(1, len(values)):
                if products[lag] <= 0:
                    break
                correlation += 2 * (1 - lag / len(values)) * products[lag] / products[0]
            measured = analysis["observables"][key]
            measured["autocorrelation_ps"] = max(1, correlation) * 100
            measured["effective_samples"] = len(values) / max(1, correlation)
    result = qualify_construction(records, protocol)
    assert result["accepted"], result["failures"]
    assert result["autocorrelation_method"] == AUTOCORRELATION_METHOD
    assert "reanalysis required" in "; ".join(result["warnings"])
    records[0]["trajectory_analysis"]["observables"]["area_per_lipid_nm2"]["effective_samples"] *= 2
    assert not qualify_construction(records, protocol)["accepted"]
    records[0]["trajectory_analysis"]["observables"]["area_per_lipid_nm2"]["effective_samples"] /= 2
    # Serialized acceptance cannot bypass the recomputed local spacing.
    records[0]["conformer_provenance"][1]["time_ps"] = (
        records[0]["conformer_provenance"][0]["time_ps"] + 100
    )
    assert not qualify_construction(records, protocol)["accepted"]


@pytest.mark.parametrize(
    "defect", ["identity", "geometry", "platform", "samples", "source", "budget"]
)
def test_false_ready_flags_do_not_override_invalid_evidence(pair, defect):
    records, protocol = pair
    target = records[0]
    if defect == "identity":
        target["canonical_smiles"] = "CCO"
    elif defect == "geometry":
        target["quality"]["hydrophobic_core"]["tail_core_gap_nm"] = 0.9
    elif defect == "platform":
        target["trajectory_analysis"]["construction_platform"]["series"]["area_per_lipid_nm2"][
            0
        ] = 20.0
    elif defect == "samples":
        target["trajectory_analysis"]["observables"]["area_per_lipid_nm2"]["effective_samples"] = (
            100000
        )
    elif defect == "source":
        target["conformer_provenance"][0]["time_ps"] = -10
    else:
        target["npt_ps"] = 210000
    assert qualify_construction(records, protocol)["status"] == "failed"


def test_valid_sample_deficit_is_capped_not_reseeded(pair):
    records, protocol = pair
    for metadata in records:
        metadata["n_conformations"] = 9
        metadata["conformer_provenance"] = metadata["conformer_provenance"][:9]
    result = qualify_construction(records, protocol)
    assert result["status"] == "capped_unaccepted"
    assert any("ten retained" in reason for reason in result["failures"])


def test_bulk_replica_difference_is_diagnostic_when_local_conformers_agree(pair):
    records, protocol = pair
    # Rebuild coherent primary statistics so this tests replica agreement,
    # rather than deliberately inconsistent raw/summary evidence.
    analysis = records[1]["trajectory_analysis"]
    values = analysis["construction_platform"]["series"]["head_to_head_nm"]
    analysis["construction_platform"]["series"]["head_to_head_nm"] = [v * 1.1 for v in values]
    analysis["observables"]["head_to_head_nm"]["mean"] *= 1.1
    analysis["observables"]["head_to_head_nm"]["standard_error"] *= 1.1
    result = qualify_construction(records, protocol)
    assert result["accepted"], result["failures"]
    assert result["replica_agreement"]["spread_sigma"]["head_to_head_nm"] > 4
    assert result["warnings"]


def test_local_replica_difference_is_a_hard_gate(pair):
    records, protocol = pair
    series = records[1]["trajectory_analysis"]["local_conformations"]["series"]
    for key, values in series.items():
        if key.startswith("shape:"):
            series[key] = [v * 1.15 for v in values]
    result = qualify_construction(records, protocol)
    assert not result["accepted"]
    assert any("replicas 1/2" in reason for reason in result["failures"])


def test_primary_shape_arithmetic_matches_independent_regression():
    x = np.linspace(0.0, 1.0, 2001)
    values = 3.8 + 0.025 * x + 0.001 * np.sin(53 * x)
    metrics = platform_metrics(values)
    assert metrics["linear_change"] == pytest.approx(
        np.polynomial.polynomial.polyfit(x, values, 1)[1] / values.mean()
    )
    assert metrics["quarter_mean_range"] == pytest.approx(
        np.ptp([v.mean() for v in np.array_split(values, 4)]) / values.mean()
    )


def test_terminal_step_cannot_hide_in_a_long_mean():
    rng = np.random.default_rng(32891)
    values = 1 + rng.normal(0, 0.001, 2000)
    values[-20:] -= 0.08
    metrics = platform_metrics(values)
    assert abs(metrics["terminal_half_shift"]) < 0.02
    assert abs(metrics["terminal_block_shift"]) > 0.02


def test_frozen_primary_is_invalid():
    with pytest.raises(ValueError, match="Frozen"):
        platform_metrics(np.ones(2000))


def test_queue_assembly_and_reader_use_the_same_admission(pair, tmp_path):
    records, protocol = pair
    entry = {"lipid": "POPC", "family": "charmm36m-lipid", "v4_protocol": protocol}
    roots = []
    for index, metadata in enumerate(records, 1):
        metadata["trajectory_analysis"]["stationarity"]["uncertainty_estimated"] = False
        root = tmp_path / f"replica-{index}"
        roots.append(root)
        target = root / entry["family"] / entry["lipid"]
        target.mkdir(parents=True)
        (target / "metadata.json").write_text(json.dumps(metadata))
        for source in metadata["conformer_provenance"]:
            np.savez(
                target / source["file"],
                coords=np.zeros((len(metadata["atom_names"]), 3)),
                atom_names=metadata["atom_names"],
            )
    result = queue.entry_assessment(tmp_path, entry, 2)
    assert result["queue_complete"], result
    assert not result["area_converged"]
    assembled = assemble_entry(
        tmp_path, roots, entry["family"], entry["lipid"], tmp_path / "library"
    )
    assert assembled["conformers"] == 40
    library = EquilibratedLipidLibrary(roots=[tmp_path / "library"])
    assert library.has("POPC", "charmm36m", "charmm36m")
    assert queue.assemble_completed(tmp_path, entry, 2)["status"] == "present"
    path = tmp_path / "library" / entry["family"] / entry["lipid"] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["replica_records"][1]["quality"]["passed"] = False
    path.write_text(json.dumps(metadata))
    assert not library.has("POPC", "charmm36m", "charmm36m")


def test_only_named_construction_analysis_upgrade_is_allowed(pair):
    import hashlib

    _, current = pair
    old = copy.deepcopy(current)
    old.pop("construction_policy")

    def sign(record):
        record.pop("sha256", None)
        record["sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return record

    assert analysis_upgrade_allowed(sign(old), current)
    old["construction_policy"] = {**construction_policy(), "relative_tolerance": 0.2}
    assert not analysis_upgrade_allowed(sign(old), current)


@pytest.mark.parametrize("defect", ["missing", "wet"])
def test_old_construction_cannot_be_readmitted_without_dry_initial_evidence(pair, defect):
    records, protocol = pair
    quality = records[0]["quality"]
    if defect == "missing":
        quality.pop("initial_water_exclusion")
    else:
        quality["initial_water_exclusion"]["files"]["ionized.gro"]["water_sites_in_membrane"] = 1
    result = qualify_construction(records, protocol)
    assert result["status"] != "accepted"
    assert "water exclusion" in str(result)


def test_early_missing_bulk_measurements_preserve_all_local_admission_checks(pair):
    from gmxbuilder.modules.membrane.v4_measurement import INCOMPLETE, assess_measured_series
    from tests.test_v4_early_measurements import missing_example

    records, protocol = pair
    for record in records:
        analysis = record["trajectory_analysis"]
        # Independent noisy bulk controls on the same complete 200 ns time axis.
        rng = np.random.default_rng(612)
        series = {
            key: value["mean"] + rng.normal(0, 0.001, 2001)
            for key, value in analysis["observables"].items()
        }
        _, _, errors = missing_example((0, 1))
        series["head_to_head_nm"][:2] = np.nan
        analysis.update(assess_measured_series(np.arange(2001) * 100.0, series, errors))
    result = qualify_construction(records, protocol)
    assert result["accepted"], result["failures"]
    assert INCOMPLETE in " ".join(result["warnings"])
    # Keeping a pass flag cannot authorize a trimmed local window.
    records[0]["trajectory_analysis"]["local_conformations"]["time_ps"][0] += 100
    result = qualify_construction(records, protocol)
    assert not result["accepted"]
    assert "local analysis window" in " ".join(result["failures"])

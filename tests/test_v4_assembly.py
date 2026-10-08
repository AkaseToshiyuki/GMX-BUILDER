"""Turning two replica roots into one library entry.

The queue writes each velocity seed separately so the two never contend for a
lock. Assembly has to put them back together without inventing anything: one
conformer ensemble from both trajectories, one pooled area, and a refusal when
the replicas are not actually of the same molecule.

The property that matters most is the last one. Pooling conformers of two
different molecules would produce an ensemble of neither, and nothing
downstream would notice -- the files would load and the counts would add up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.dry_initial_fixture import dry_initial_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from assemble_v4_library import assemble_entry, identity_mismatch, load_entry  # noqa: E402

ATOM_NAMES = ["N", "C12", "P", "O13", "C1"]


def _observables(mean: float, converged: bool = True) -> dict:
    from gmxbuilder.modules.membrane.area_observable import (
        AREA_OBSERVABLE_SCHEMA,
        AUTOCORRELATION_METHOD,
    )

    return {
        "schema_version": AREA_OBSERVABLE_SCHEMA,
        "autocorrelation_method": AUTOCORRELATION_METHOD,
        "measures": "pure-bilayer-area-per-lipid",
        "converged": converged,
        "rejection": None if converged else "synthetic",
        "area_per_lipid_nm2": {
            "mean": mean,
            "standard_error": 0.002,
            "n_frames": 40000,
            "n_blocks": 40,
            "effective_samples": 40.0,
            "autocorrelation_ps": 500.0,
            "relative_half_drift": 0.004,
            "drift_sigma": 0.8,
        },
        "analysis_window_ps": [10000.0, 50000.0],
        "equilibration_ps": 10000.0,
        "retained_fraction": 0.8,
    }


def _write_replica(
    root: Path,
    family: str,
    lipid: str,
    *,
    mean: float,
    conformers: int = 5,
    converged: bool = True,
    smiles: str = "CCO",
    topology: str = "abc123",
    checked: int = 50,
) -> Path:
    entry = root / family / lipid
    entry.mkdir(parents=True)
    for index in range(conformers):
        np.savez_compressed(
            entry / f"conf_{index:04d}.npz",
            coords=np.zeros((len(ATOM_NAMES), 3)),
            atom_names=np.asarray(ATOM_NAMES),
        )
    (entry / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "ready",
                "lipid_name": lipid,
                "canonical_smiles": smiles,
                "topology_sha256": topology,
                "parameter_family": family,
                "atom_names": ATOM_NAMES,
                "n_conformations": conformers,
                "npt_ps": 50000.0,
                "quality": {
                    "initial_water_exclusion": dry_initial_fixture(),
                    "passed": True,
                    "orientation": {"passed": True, "n_lipids_checked": checked},
                },
                "observables": _observables(mean, converged),
            }
        )
    )
    return entry


@pytest.fixture
def replicas(tmp_path):
    roots = [tmp_path / "replica-1", tmp_path / "replica-2"]
    return tmp_path, roots


# --------------------------------------------------------------------------
# The merge


def test_conformers_from_every_replica_land_in_one_ensemble(replicas):
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638, conformers=5)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.636, conformers=7)

    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")

    assert result["conformers"] == 12
    target = base / "assembled-candidates" / "charmm36m-lipid" / "POPC"
    files = sorted(target.glob("conf_*.npz"))
    assert len(files) == 12
    # Renumbered contiguously: each replica numbers its own from zero, so
    # copying by name would silently overwrite the first replica's ensemble.
    assert [f.name for f in files] == [f"conf_{i:04d}.npz" for i in range(12)]


def test_the_merged_entry_reports_the_count_it_actually_holds(replicas):
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638, conformers=5)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.636, conformers=7)
    assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")

    metadata = json.loads(
        (base / "assembled-candidates" / "charmm36m-lipid" / "POPC" / "metadata.json").read_text()
    )
    assert metadata["n_conformations"] == 12
    # The library validator compares this against the files on disk, and
    # against the orientation count, so both have to describe the merge.
    assert metadata["quality"]["orientation"]["n_lipids_checked"] == 100
    assert metadata["assembled_from"]["replicas"] == 2
    assert metadata["assembled_from"]["conformers_per_replica"] == [5, 7]


def test_the_pooled_area_replaces_the_per_replica_one(replicas):
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.640)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.636)
    assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")

    observables = json.loads(
        (base / "assembled-candidates" / "charmm36m-lipid" / "POPC" / "metadata.json").read_text()
    )["observables"]

    assert observables["pooled_replicas"] == 2
    assert observables["area_per_lipid_nm2"]["mean"] == pytest.approx(0.638, abs=0.001)
    assert len(observables["replicas"]) == 2


# --------------------------------------------------------------------------
# What it must refuse


@pytest.mark.parametrize(
    "difference",
    [
        {"smiles": "CCCO"},
        {"topology": "different"},
    ],
    ids=["different-molecule", "different-topology"],
)
def test_replicas_of_different_molecules_are_not_pooled(replicas, difference):
    """The files would load and the counts would add up. Nothing else notices."""
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.636, **difference)

    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")

    assert "skipped" in result
    assert "disagree" in result["skipped"]
    assert not (base / "assembled-candidates" / "charmm36m-lipid" / "POPC").exists()


def test_a_lone_replica_is_not_assembled(replicas):
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638)
    roots[1].mkdir()

    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")
    assert "only 1 replica" in result["skipped"]


def test_a_v3_entry_is_not_mistaken_for_a_replica(tmp_path):
    """V3 gets unpacked into these roots by any GAFF build; it is not input."""
    entry = tmp_path / "charmm36m-lipid" / "POPC"
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps({"status": "ready", "quality": {"area_per_lipid_nm2": 0.589}})
    )
    assert load_entry(tmp_path, "charmm36m-lipid", "POPC") is None


# --------------------------------------------------------------------------
# What it must not refuse


def test_an_unconverged_area_still_yields_its_conformers(replicas):
    """The ensemble and the area are two products of one run.

    A run whose area never settled still sampled real lipid conformations, and
    throwing those away would cost coverage for no gain. The area is marked
    unusable instead, which is what the area model already checks.
    """
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638, converged=False)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.601, converged=False)

    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "assembled-candidates")

    assert result["converged"] is False
    assert result["conformers"] == 10
    metadata = json.loads(
        (base / "assembled-candidates" / "charmm36m-lipid" / "POPC" / "metadata.json").read_text()
    )
    assert metadata["status"] == "ready"
    assert metadata["observables"]["converged"] is False


def test_report_only_writes_nothing(replicas):
    base, roots = replicas
    _write_replica(roots[0], "charmm36m-lipid", "POPC", mean=0.638)
    _write_replica(roots[1], "charmm36m-lipid", "POPC", mean=0.636)

    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", None)

    assert "area" in result and "conformers" not in result
    assert not (base / "assembled-candidates").exists()


def test_identity_mismatch_accepts_a_matched_pair():
    matched = [
        {
            "topology_sha256": "a",
            "canonical_smiles": "CCO",
            "parameter_family": "f",
            "lipid_name": "POPC",
            "status": "ready",
        }
    ] * 2
    assert identity_mismatch(matched) is None


def test_the_queue_does_not_skip_the_entries_it_exists_to_rebuild():
    """The build queue must not refuse a lipid for needing a rebuild.

    `policy_refusal` filters out pairings the project has quarantined. A lipid
    whose pre-equilibrated entry was built from a since-corrected structure
    reports as unavailable to ordinary callers, and this filter would then skip
    it -- reporting "current policy forbids the pairing" for the very entries
    the queue was started to produce. It skipped 41 of them before this was
    caught, so the question is asked inside the rebuild scope.
    """
    from build_v4_library import policy_refusal

    from gmxbuilder.modules.forcefield.lipid_policy import library_entry_superseded

    superseded = [
        name
        for name in ("DAPC", "DOPC", "POPC", "TOCL")
        if library_entry_superseded(name, "amber14sb", "gaff2")
    ]
    if not superseded:
        pytest.skip("no superseded entries installed; nothing to guard against here")
    for name in superseded:
        assert policy_refusal("amber-gaff2", name, "amber14sb") is None


def test_legacy_analysis_cannot_replace_runtime_library(replicas):
    base, roots = replicas
    for root in roots:
        _write_replica(root, "charmm36m-lipid", "POPC", mean=0.638)
    target = base / "library/charmm36m-lipid/POPC"
    target.mkdir(parents=True)
    old = target / "old-evidence.txt"
    old.write_text("preserve")
    result = assemble_entry(base, roots, "charmm36m-lipid", "POPC", base / "library")
    assert "runtime V4 reader gate" in result["skipped"]
    assert old.read_text() == "preserve"

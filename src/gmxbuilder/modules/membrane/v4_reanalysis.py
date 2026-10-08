"""Explicit analysis-only upgrades of retained, completed V4 trajectories."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from gmxbuilder.modules.membrane.v4_stationarity import METHOD, analysis_policy

LEGACY_METHOD = "joint-circular-block-bootstrap-1"


def _digest(protocol):
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def analysis_upgrade_allowed(previous, current):
    """Only the named analysis method may change; no chemistry/MD exemptions."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    if previous.get("sha256") != _digest(previous) or current.get("sha256") != _digest(current):
        return False
    if current.get("stationarity_method") != METHOD or previous.get("stationarity_method") not in {
        LEGACY_METHOD,
        METHOD,
    }:
        return False
    if current.get("stationarity_policy") != analysis_policy():
        return False
    if previous["stationarity_method"] == LEGACY_METHOD and "stationarity_policy" in previous:
        return False
    if (
        previous["stationarity_method"] == METHOD
        and previous.get("stationarity_policy") != analysis_policy()
    ):
        return False
    from gmxbuilder.modules.membrane.local_relaxation import construction_policy
    from gmxbuilder.modules.membrane.v4_platform import (
        construction_policy as legacy_construction_policy,
    )

    # A construction-use policy is an analysis-only addition. Never exempt an
    # arbitrary policy edit, changed geometry, temperature or molecular identity.
    if current.get("construction_policy") != construction_policy():
        return False
    if previous.get("construction_policy") not in (
        None,
        legacy_construction_policy(),
        construction_policy(),
    ):
        return False
    from gmxbuilder.modules.membrane.v4_atom_selection import METHOD as selection_method

    if current.get("observable_selection_method") != selection_method or previous.get(
        "observable_selection_method"
    ) not in (None, selection_method):
        return False
    ignored = {
        "stationarity_method",
        "stationarity_policy",
        "construction_policy",
        "sha256",
        "observable_selection_method",
    }
    return {k: v for k, v in previous.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored
    }


def archive_analysis(work: Path, output: Path):
    """Keep the previous assessment before its coordinates/metrics are replaced."""
    source = output / "metadata.json"
    if not source.is_file():
        return []
    payload = source.read_bytes()
    previous = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    archive = work / "analysis_history" / digest
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / "metadata.json"
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError("Existing analysis archive differs; refusing to overwrite evidence")
    if not target.exists():
        for name in (
            "whole.gro",
            "trajectory_observables.npz",
            "trajectory_geometry.npz",
            "trajectory_unmeasurable_frames.json",
            "trajectory_local_conformations.npz",
        ):
            if (work / name).is_file():
                shutil.copy2(work / name, archive / name)
        target.write_bytes(payload)
    history = list(previous.get("analysis_history", []))
    history.append(
        {
            "metadata_sha256": digest,
            "archive": str(archive.resolve()),
            "npt_ps": previous["npt_ps"],
            "analysis_method": previous.get("trajectory_analysis", {})
            .get("stationarity", {})
            .get("method"),
        }
    )
    return history


def reanalyse(builder, lipid_name, force_field, lipid_ff, *, work_dir, replica_seed):
    """Re-extract/revalidate coordinates without grompp, mdrun or new seeds.

    The caller holds the entry lock. The queue must be stopped at a completed
    extension boundary. An in-flight endpoint mismatch fails closed in the
    trajectory reader; it is never rounded down to the last published time.
    """
    context = builder._entry_context(lipid_name, force_field, lipid_ff, replica_seed=replica_seed)
    metadata = json.loads((context.output_dir / "metadata.json").read_text())
    from gmxbuilder.modules.membrane.parameter_provenance import validate_work_parameters

    validate_work_parameters(Path(work_dir), metadata)
    if not analysis_upgrade_allowed(metadata.get("v4_protocol"), builder.v4_protocol):
        raise RuntimeError("Reanalysis cannot change simulation conditions or molecular identity")
    if (
        metadata.get("test_mode")
        or metadata.get("status") != "ready"
        or metadata.get("quality", {}).get("passed") is not True
    ):
        raise RuntimeError(
            "Reanalysis requires a completed structurally valid production candidate"
        )
    if (
        metadata.get("canonical_smiles") != context.lipid.smiles
        or metadata.get("temperature_K") != builder.v4_protocol["temperature_K"]
    ):
        raise RuntimeError("Retained candidate identity/temperature mismatch")
    work = Path(work_dir)
    for name in (
        "npt.tpr",
        "npt.cpt",
        "npt.xtc",
        "npt.edr",
        "npt.gro",
        "topol.top",
        "v4-protocol.json",
    ):
        if not (work / name).is_file():
            raise RuntimeError(f"Reanalysis requires retained {name}")
    original = json.loads((work / "v4-protocol.json").read_text())
    if not analysis_upgrade_allowed(original, builder.v4_protocol):
        raise RuntimeError("Original simulation protocol differs from the reanalysis conditions")
    from gmxbuilder.modules.membrane.v4_tpr import verify_tpr

    verify_tpr(builder.gmx, work / "npt.tpr", original, force_field, "npt")
    return builder._measure_and_publish(
        work,
        "npt",
        name=context.name,
        lipid=context.lipid,
        force_field=force_field,
        lipid_ff=lipid_ff,
        temperature=builder.v4_protocol["temperature_K"],
        npt_steps=int(round(float(metadata["npt_ps"]) * 500)),
        host_lipid=context.host_lipid,
        membrane_lipid_names=context.membrane_lipid_names,
        simulation_resnames=context.simulation_resnames,
        target_apl=context.target_apl,
        target_dhh=context.target_dhh,
        genion_rmin=float(metadata.get("genion_rmin_nm") or 0),
        output_dir=context.output_dir,
        test_mode=False,
        build_started=time.time(),
        retain_work=work,
    )


def recover_replica_publication(journal: Path) -> None:
    """Roll back an interrupted multi-replica publication while holding the queue lock."""
    if not journal.is_file():
        return
    import os

    state = json.loads(journal.read_text())
    if state.get("status") == "committed":
        return
    for row in reversed(state["entries"]):
        target, backup = Path(row["target"]), Path(row["backup"])
        if backup.exists():
            if target.exists():
                recovered = backup.with_name(backup.name + ".uncommitted")
                if recovered.exists():
                    raise RuntimeError("Uncommitted analysis already exists; inspect transaction")
                os.replace(target, recovered)
            os.replace(backup, target)
    state["status"] = "rolled_back"
    journal.write_text(json.dumps(state, indent=2) + "\n")


def publish_replicas(candidates: list[tuple[Path, Path]], journal: Path) -> None:
    """Publish jointly accepted candidates, retaining originals and crash recovery evidence."""
    import os

    rows = [
        {
            "source": str(source),
            "target": str(target),
            "backup": str(journal.parent / f"previous-replica-{index}"),
        }
        for index, (source, target) in enumerate(candidates, 1)
    ]
    for row in rows:
        if not Path(row["source"]).is_dir() or not Path(row["target"]).is_dir():
            raise RuntimeError("Replica publication requires existing candidate and original")
        if Path(row["backup"]).exists():
            raise RuntimeError("Replica publication backup already exists")
    journal.write_text(json.dumps({"status": "pending", "entries": rows}, indent=2) + "\n")
    try:
        for row in rows:
            os.replace(row["target"], row["backup"])
            os.replace(row["source"], row["target"])
    except BaseException:
        recover_replica_publication(journal)
        raise
    temporary = journal.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": "committed", "entries": rows}, indent=2) + "\n")
    os.replace(temporary, journal)

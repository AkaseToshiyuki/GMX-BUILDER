#!/usr/bin/env python3
"""Assemble the replica roots the V4 queue produced into one library.

The queue writes each velocity seed to its own root so the two never contend
for a lock or a filename. This turns those into the single-directory layout
the runtime reads: conformers from every replica in one ensemble, and one
pooled area with the cross-seed agreement recorded.

Two things it will not do, both deliberate:

* It does not merge replicas that disagree about what molecule they built.
  Identical topology hash, canonical SMILES and parameter family are required,
  because pooling conformers of two different molecules would produce an
  ensemble of neither.
* It does not gate the entry on whether the *area* converged. The conformer
  ensemble and the area are two products of the same run and one failing does
  not invalidate the other; ``observables.converged`` says which areas may be
  used as construction targets, and the area model already refuses the rest.

    python scripts/assemble_v4_library.py                  # into <v4-root>/library
    python scripts/assemble_v4_library.py --report-only    # judge, write nothing

Nothing here touches V3, and nothing the runtime reads changes until the
assembled directory is deliberately installed.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_V4_ROOT = Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated_v4"

#: Fields that must agree across replicas before their conformers are pooled.
IDENTITY_FIELDS = (
    "topology_sha256",
    "canonical_smiles",
    "parameter_family",
    "lipid_name",
    "temperature_K",
    "equilibration_host",
    "v4_protocol",
)


def replica_roots(v4_root: Path) -> list[Path]:
    return sorted(p for p in v4_root.glob("replica-*") if p.is_dir())


def load_entry(root: Path, family: str, lipid: str) -> dict | None:
    path = root / family / lipid / "metadata.json"
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(metadata.get("observables"), dict):
        return None  # a V3 entry, not something this queue produced
    return metadata


def identity_mismatch(metadatas: list[dict]) -> str | None:
    for field in IDENTITY_FIELDS:
        values = {json.dumps(m.get(field), sort_keys=True) for m in metadatas}
        if len(values) > 1:
            return f"replicas disagree on {field}"
    if any(m.get("status") != "ready" for m in metadatas):
        return "one or more replicas are not ready"
    return None


def _assemble_entry(
    v4_root: Path, roots: list[Path], family: str, lipid: str, destination: Path
) -> dict:
    from gmxbuilder.modules.membrane.area_observable import (
        AreaMeasurement,
        pool_replicas,
        summarise_pooled,
    )

    metadatas, sources = [], []
    for root in roots:
        metadata = load_entry(root, family, lipid)
        if metadata is not None:
            metadatas.append(metadata)
            sources.append(root / family / lipid)

    result = {"family": family, "lipid": lipid, "replicas": len(metadatas)}
    if len(metadatas) < 2:
        result["skipped"] = f"only {len(metadatas)} replica(s) built"
        return result

    mismatch = identity_mismatch(metadatas)
    if mismatch:
        result["skipped"] = mismatch
        return result

    if any(m.get("v4_protocol") for m in metadatas):
        from build_v4_library import entry_assessment

        from gmxbuilder.modules.membrane.v4_sampling import quantitative_area

        assessment = entry_assessment(v4_root, {"family": family, "lipid": lipid}, len(roots))
        if not assessment["queue_complete"]:
            return {**result, "skipped": assessment["reason"]}
        area_metadata = quantitative_area([m["trajectory_analysis"] for m in metadatas])
        from types import SimpleNamespace

        pooled = SimpleNamespace(
            mean_nm2=area_metadata["area_per_lipid_nm2"]["mean"],
            converged=area_metadata["converged"],
            as_metadata=lambda: area_metadata,
        )
        area_summary = (
            area_metadata["rejection"] or "Quantitative APL passed 2% CI and replica gates"
        )
    else:
        pooled = pool_replicas([AreaMeasurement.from_metadata(m["observables"]) for m in metadatas])
        area_summary = summarise_pooled(pooled)
    result["area"] = pooled.mean_nm2
    result["converged"] = pooled.converged
    result["summary"] = area_summary

    if destination is None:
        return result

    final_target = destination / family / lipid
    final_target.parent.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=f".{lipid}.assembling-", dir=final_target.parent))

    # One ensemble from every replica. Renumbered rather than copied by name,
    # since each replica numbers its own conformers from zero.
    index = 0
    per_replica = []
    from gmxbuilder.modules.membrane.equilibrated_library import conformer_files

    for source in sources:
        count = 0
        for conformer in conformer_files(source):
            shutil.copy2(conformer, target / f"conf_{index:04d}.npz")
            index += 1
            count += 1
        per_replica.append(count)

    # Aggregating the top-level geometry count must not alter replica 1's
    # reviewed evidence through a shared nested quality dictionary.
    merged = copy.deepcopy(metadatas[0])
    merged["n_conformations"] = index
    merged["observables"] = {
        **metadatas[0]["observables"],
        **pooled.as_metadata(),
    }
    if merged.get("v4_protocol"):
        from gmxbuilder.modules.membrane.v4_construction import REPLICA_FIELDS

        merged["replica_analyses"] = [m["trajectory_analysis"] for m in metadatas]
        merged["replica_records"] = [
            {
                "replica": root.name,
                **{key: metadata[key] for key in REPLICA_FIELDS},
                **({"reused_from": metadata["reused_from"]} if "reused_from" in metadata else {}),
            }
            for root, metadata in zip(roots, metadatas, strict=True)
        ]
        merged["construction_acceptance"] = assessment["construction"]
        provenance = []
        for root, directory, metadata in zip(roots, sources, metadatas, strict=True):
            records = {item["file"]: item for item in metadata["conformer_provenance"]}
            for path in conformer_files(directory):
                source = records[path.name]
                provenance.append(
                    {**source, "replica": root.name, "file": f"conf_{len(provenance):04d}.npz"}
                )
        merged["conformer_provenance"] = provenance
        merged["conformer_sampling"] = "equal-replica-then-uniform-conformer"
    # The orientation gate counts molecules checked; the merged ensemble was
    # checked once per replica, so the totals add. Leaving replica 1's number
    # would make the entry fail its own library validity check.
    orientation = dict(merged.get("quality", {}).get("orientation", {}))
    if orientation:
        orientation["n_lipids_checked"] = sum(
            int(m.get("quality", {}).get("orientation", {}).get("n_lipids_checked", 0))
            for m in metadatas
        )
        merged.setdefault("quality", {})["orientation"] = orientation
    merged["assembled_from"] = {
        "replicas": len(metadatas),
        "conformers_per_replica": per_replica,
        "npt_ps_per_replica": [m.get("npt_ps") for m in metadatas],
        "seeds_agree": pooled.converged,
        "source_metadata_sha256": [
            hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest()
            for source in sources
        ],
    }
    (target / "metadata.json").write_text(json.dumps(merged, indent=1, allow_nan=False))
    # Only reader-valid V4 entries may replace the production library. Legacy
    # replicas remain assemblable as explicitly separate analysis artifacts.
    if merged.get("v4_protocol") or destination.resolve() == (v4_root / "library").resolve():
        from build_v4_library import _conformers_readable

        from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary

        library = EquilibratedLipidLibrary(roots=[destination])
        library.require_v4 = True
        try:
            accepted = library.inspect(
                lipid,
                merged.get("force_field", ""),
                merged.get("lipid_ff"),
                candidate_directory=target,
            )
            readable = accepted is not None and _conformers_readable(target, merged)
        except (OSError, ValueError, TypeError, KeyError):
            readable = False
        if not readable:
            shutil.rmtree(target)
            return {**result, "skipped": "Assembled candidate failed the runtime V4 reader gate"}
    # Preserve a previous valid entry until its replacement is complete. A
    # failed copy/serialization leaves only a hidden staging directory.
    backup = final_target.with_name(final_target.name + ".previous")
    if final_target.exists():
        final_target.rename(backup)
    try:
        target.rename(final_target)
    except OSError:
        if backup.exists():
            backup.rename(final_target)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    result["conformers"] = index
    return result


def assemble_entry(v4_root, roots, family, lipid, destination):
    """Serialize publishers and recover interrupted promotion before assembly."""
    if destination is None:
        return _assemble_entry(v4_root, roots, family, lipid, destination)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".publication.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = destination / family / lipid
        backup = final.with_name(final.name + ".previous")
        if backup.exists():
            if not final.exists():
                backup.rename(final)
            else:
                shutil.rmtree(backup)
        return _assemble_entry(v4_root, roots, family, lipid, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--install", action="store_true", help="publish validated entries to library"
    )
    arguments = parser.parse_args()

    roots = replica_roots(arguments.v4_root)
    if len(roots) < 2:
        raise SystemExit(f"expected at least two replica roots under {arguments.v4_root}")

    pairs = sorted(
        {
            (family.name, lipid.name)
            for root in roots
            for family in root.iterdir()
            if family.is_dir() and not family.name.startswith(".")
            for lipid in family.iterdir()
            if lipid.is_dir()
        }
    )
    destination = None
    if not arguments.report_only:
        destination = arguments.destination or (
            arguments.v4_root / ("library" if arguments.install else "assembled-candidates")
        )
        if (
            destination.resolve() == (arguments.v4_root / "library").resolve()
            and not arguments.install
        ):
            raise SystemExit("Publishing to library requires --install")
        destination.mkdir(parents=True, exist_ok=True)

    # Standalone publication must not race a queue that owns these replicas.
    queue_lock = (arguments.v4_root / "queue.lock").open("a+")
    try:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        queue_lock.close()
        raise SystemExit("A V4 queue owns these replicas; wait before assembling") from exc
    try:
        assembled, skipped, unconverged = 0, [], []
        for family, lipid in pairs:
            result = assemble_entry(arguments.v4_root, roots, family, lipid, destination)
            if "skipped" in result:
                skipped.append((family, lipid, result["skipped"]))
                continue
            assembled += 1
            if not result["converged"]:
                unconverged.append((family, lipid, result["summary"]))

        print(f"{assembled} entries assembled from {len(roots)} replicas")
        if destination:
            print(f"  written to {destination}")
        print(f"  {assembled - len(unconverged)} carry a converged area usable as a build target")
        print(f"  {len(unconverged)} built conformers but no converged area")
        if skipped:
            print(f"\n{len(skipped)} not assembled:")
            for family, lipid, why in skipped[:20]:
                print(f"    {family:18s} {lipid:6s} {why}")
        if unconverged:
            print("\nconformers kept, area not usable as a target:")
            for family, lipid, summary in unconverged[:20]:
                print(f"    {family:18s} {lipid:6s} {summary.split('--')[-1].strip()[:70]}")
        return 0
    finally:
        queue_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

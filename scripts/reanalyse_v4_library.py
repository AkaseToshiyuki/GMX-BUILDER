#!/usr/bin/env python3
"""Upgrade completed V4 analyses without running dynamics or changing MD inputs.

Run only after stopping the queue at a completed extension boundary. The queue
and entry locks prevent concurrent writers. Earlier analyses and the original
simulation protocol are retained, and the new result must pass current evidence
checks before it can be accepted by the queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_v4_library import (  # noqa: E402
    DEFAULT_V4_ROOT,
    capability_entries,
    entry_assessment,
    entry_seed,
    replica_root,
)
from v4_queue_state import queue_ownership, verify_source  # noqa: E402

from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary  # noqa: E402
from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder  # noqa: E402
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol  # noqa: E402


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lipid", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--gmx", required=True)
    args = parser.parse_args()
    args.v4_root = args.v4_root.resolve()
    if args.replicas < 2:
        parser.error("V4 requires at least two replicas")
    entries = [
        entry
        for entry in capability_entries()
        if entry["lipid"] == args.lipid.upper() and entry["family"] == args.family
    ]
    if len(entries) != 1:
        parser.error("No unique supported lipid/family pairing")
    entry = entries[0]
    configuration = {**vars(args), "v4_root": str(args.v4_root), "operation": "analysis_only"}
    with queue_ownership(args.v4_root, ROOT, configuration=configuration) as (
        source,
        manifest,
        _lock,
    ):
        from gmxbuilder.modules.membrane.v4_reanalysis import (
            publish_replicas,
            recover_replica_publication,
        )

        transactions = args.v4_root / "reanalysis-transactions"
        transactions.mkdir(exist_ok=True)
        for journal in transactions.glob("*/publication.json"):
            recover_replica_publication(journal)
        stage = transactions / uuid.uuid4().hex
        stage.mkdir()
        records = []
        candidates = []
        for replica in range(1, args.replicas + 1):
            original_root = replica_root(args.v4_root, replica)
            root = replica_root(stage, replica)
            original_entry = original_root / entry["family"] / entry["lipid"]
            staged_entry = root / entry["family"] / entry["lipid"]
            shutil.copytree(original_entry, staged_entry)
            candidates.append((staged_entry, original_entry))
            work = args.v4_root / "work" / entry["family"] / entry["lipid"] / f"replica-{replica}"
            # Work on independent copies: even analysis outputs cannot alter old evidence.
            original_work = work
            work = stage / "work" / entry["family"] / entry["lipid"] / f"replica-{replica}"
            shutil.copytree(original_work, work)
            # Hash the actual retained trajectory, checkpoint and compiled input.
            # An analysis upgrade must not overwrite or truncate any MD evidence.
            protected = [
                work / name
                for name in (
                    "npt.xtc",
                    "npt.cpt",
                    "npt.tpr",
                    "npt.edr",
                    "npt.gro",
                    "topol.top",
                    "v4-protocol.json",
                )
            ]
            before = {path.name: file_digest(path) for path in protected}
            builder = LipidEquilibrationBuilder(
                library=EquilibratedLipidLibrary(roots=[root, root]),
                gmx=args.gmx,
                v4_protocol=resolve_protocol(entry["lipid"], entry["family"]),
            )
            output = builder.reanalyse(
                entry["lipid"],
                entry["force_field"],
                entry["lipid_ff"],
                work_dir=work,
                replica_seed=entry_seed(entry["family"], entry["lipid"], replica),
            )
            after = {path.name: file_digest(path) for path in protected}
            if after != before:
                raise RuntimeError("MD evidence changed during reanalysis; inspection required")
            verify_source(ROOT, source)
            record = {"replica": replica, "output": str(output), "unchanged_md_sha256": after}
            records.append(record)
            manifest.with_suffix(".reanalysis.json").write_text(
                json.dumps({"records": records}, indent=2) + "\n"
            )
            print(f"replica {replica}: reanalysis complete; MD evidence unchanged", flush=True)
        result = entry_assessment(
            stage,
            {**entry, "v4_protocol": resolve_protocol(entry["lipid"], entry["family"])},
            args.replicas,
        )
        manifest.with_suffix(".reanalysis.json").write_text(
            json.dumps({"records": records, "assessment": result}, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)
        print(f"Reanalysis evidence: {manifest.with_suffix('.reanalysis.json')}", flush=True)
        if not result["queue_complete"]:
            raise RuntimeError(
                f"Reanalysis needs review; originals preserved, candidates at {stage}"
            )
        verify_source(ROOT, source)
        publish_replicas(candidates, stage / "publication.json")


if __name__ == "__main__":
    main()

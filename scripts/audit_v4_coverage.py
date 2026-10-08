#!/usr/bin/env python3
"""Report all registered lipid/family pairs without fitting or running dynamics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_v4_library import (  # noqa: E402
    capability_entries,
    policy_refusal,
    unresolved_stereochemistry,
)

from gmxbuilder.modules.forcefield.gaff_backend import (  # noqa: E402
    _cache_key,
    _cache_root,
    _load_cached,
    _safe_name,
    gaff_charge_method,
)
from gmxbuilder.modules.forcefield.gaff_lipid_identity import validate_template  # noqa: E402
from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability  # noqa: E402
from gmxbuilder.modules.forcefield.lipid_policy import (  # noqa: E402
    charmm_lipid_capability,
    gaff_lipid_capability,
    rebuilding_library_entry,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry  # noqa: E402
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol  # noqa: E402

FAMILIES = (
    ("charmm36m-lipid", "charmm36m", "charmm36m"),
    ("charmm36-lipid", "charmm36", "charmm36"),
    ("amber-lipid21", "amber14sb", "lipid21"),
    ("amber-gaff2", "amber14sb", "gaff2"),
)


def gaff_cache_evidence(name: str) -> dict:
    """Use the raw reader to distinguish missing files from a bad stereoisomer."""
    lipid = LipidRegistry.get(name)
    safe_name = _safe_name(name)
    key = _cache_key(safe_name, lipid.smiles, lipid.charge, gaff_charge_method())
    directory = _cache_root(safe_name, install=False) / f"{safe_name}-{key}"
    result = {"directory": str(directory), "status": "missing_or_invalid_files"}
    template = _load_cached(directory)
    if template is None:
        return result
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("smiles") != lipid.smiles:
            raise ValueError("Cached SMILES differs from registered identity")
        evidence = validate_template(lipid.smiles, template)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        return {**result, "status": "identity_failed", "reason": str(exc)}
    return {**result, "status": "identity_passed", "identity": evidence}


def audit(v4_root: Path) -> dict:
    selection_path = v4_root / "protocol-selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    selected = {(r["lipid"], r["family"]) for r in selection.get("candidates", [])}
    latest = {}
    progress = v4_root / "queue-progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            record = json.loads(line)
            latest[(record["lipid"], record["family"])] = record
    # Process evidence is deliberately separate from mere coordinate artifacts.
    active_directories = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (process / "cmdline").read_bytes().split(b"\0")
            if b"mdrun" in command:
                active_directories.append((process / "cwd").resolve())
        except (OSError, RuntimeError):
            continue
    queued_capabilities = {(row["family"], row["lipid"]): row for row in capability_entries()}
    rows = []
    with rebuilding_library_entry(retry_failed_validation=True):
        for name in sorted(LipidRegistry.list_builtin()):
            identity_reason = unresolved_stereochemistry(name)
            cache = gaff_cache_evidence(name)
            for family, force_field, lipid_ff in FAMILIES:
                queue_entry = queued_capabilities.get((family, name))
                selected_lipid_ff = queue_entry["lipid_ff"] if queue_entry else lipid_ff
                if lipid_ff == "gaff2":
                    capable, capability_reason = gaff_lipid_capability(name)
                elif lipid_ff == "lipid21":
                    capable, capability_reason = lipid21_capability(name)
                else:
                    capable, capability_reason = charmm_lipid_capability(name, force_field)
                try:
                    protocol = resolve_protocol(name, family)
                    protocol_reason = None
                except (ValueError, KeyError) as exc:
                    protocol, protocol_reason = None, str(exc)
                policy_reason = policy_refusal(family, name, force_field)
                reasons = list(
                    dict.fromkeys(
                        reason
                        for reason in (
                            None if capable else capability_reason,
                            identity_reason,
                            policy_reason,
                            protocol_reason,
                        )
                        if reason
                    )
                )
                work = v4_root / "work" / family / name
                running = any(work in directory.parents for directory in active_directories)
                rows.append(
                    {
                        "lipid": name,
                        "family": family,
                        "lipid_ff": selected_lipid_ff,
                        "queue_capable": queue_entry is not None,
                        "rebuild_capability": {"supported": capable, "reason": capability_reason},
                        "registered_identity": {
                            "passed": not identity_reason,
                            "reason": identity_reason,
                        },
                        "gaff_cache": cache if lipid_ff == "gaff2" else None,
                        "protocol": protocol,
                        "protocol_reason": protocol_reason,
                        "queue_eligible": not reasons and queue_entry is not None,
                        "hold_reasons": reasons,
                        "selected_by_current_run": (name, family) in selected,
                        "mdrun_observed": running,
                        "last_queue_record": latest.get((name, family)),
                    }
                )
    return {
        "schema": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Rebuild eligibility and cached identity are not physical validation.",
        "run_manifest": selection.get("run_manifest"),
        "counts": {
            "lipids": len(LipidRegistry.list_builtin()),
            "pairs": len(rows),
            "eligible_by_family": dict(Counter(r["family"] for r in rows if r["queue_eligible"])),
        },
        "records": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--v4-root", type=Path, default=Path.home() / ".cache/gmxbuilder/lipid_equilibrated_v4"
    )
    args = parser.parse_args()
    report = audit(args.v4_root)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["counts"]))

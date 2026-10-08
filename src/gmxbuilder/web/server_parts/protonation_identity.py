"""Reversible, task-local identity adapter for PROPKA's narrow PDB interface."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def prepare_analysis(structure, destination: Path) -> None:
    from gmxbuilder.io.pdb import PDBWriter

    if structure.num_atoms > 99999:
        raise ValueError("PROPKA analysis supports at most 99999 atoms")
    adapted = structure.copy()
    rows, lookup, identities = [], {}, {}
    sources = structure.source_info.get("atoms", {})
    for i in range(structure.num_atoms):
        chain = str(structure.chain_ids[i]).strip()
        resid = int(structure.resids[i])
        name = str(structure.resnames[i]).strip().upper()
        source = sources.get(str(structure.source_ids[i]), {})
        key = (chain, resid)
        identity = (source.get("segment", 0), source.get("icode", ""))
        if source:
            identities.setdefault(key, set()).add(identity)
            if len(identities[key]) > 1:
                raise ValueError(
                    "Ambiguous residue identity; repeat Input Check with distinct identifiers"
                )
        if key not in lookup:
            if len(rows) >= 9999:
                raise ValueError("PROPKA analysis supports at most 9999 distinct residues")
            lookup[key] = len(rows) + 1
            rows.append(
                {
                    "chain": chain,
                    "display_chain": chain or "A",
                    "resid": resid,
                    "resname": name,
                    "analysis_chain": "A",
                    "analysis_resid": len(rows) + 1,
                    "segment": identity[0],
                    "icode": identity[1],
                }
            )
        elif rows[lookup[key] - 1]["resname"] != name:
            raise ValueError("Conflicting names for the same residue identity")
        adapted.resids[i] = lookup[key]
    adapted.chain_ids = ["A"] * structure.num_atoms
    # Keep polymer boundaries distinct despite the globally unique residue numbering.
    chains = list(dict.fromkeys(row["chain"] for row in rows))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(chains) > len(alphabet):
        raise ValueError("PROPKA analysis supports at most 62 distinct chains")
    chain_codes = dict(zip(chains, alphabet))
    for row in rows:
        row["analysis_chain"] = chain_codes[row["chain"]]
    for i in range(structure.num_atoms):
        adapted.chain_ids[i] = chain_codes[str(structure.chain_ids[i]).strip()]
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".pdb")
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        PDBWriter.write(adapted, temporary, title="Mapped canonical structure for PROPKA")
        mapping = {
            "schema": 1,
            "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
            "residues": rows,
        }
        from .viewer_data import _atomic_write

        # A reader between replacements fails the digest check instead of mixing identities.
        _atomic_write(destination.with_suffix(".identities.json"), json.dumps(mapping).encode())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def map_predictions(predictions: list[dict], residues: list[dict], source: str) -> list[dict]:
    mapping_path = Path(source).with_suffix(".identities.json")
    if not mapping_path.exists():
        return predictions
    mapping = json.loads(mapping_path.read_text())
    if mapping.get("sha256") != hashlib.sha256(Path(source).read_bytes()).hexdigest():
        raise ValueError("PROPKA identity map is stale; recompute protonation")
    by_analysis = {(r["analysis_chain"], r["analysis_resid"]): r for r in mapping["residues"]}
    result = []
    for prediction in predictions:
        row = by_analysis.get((prediction.get("chain", ""), prediction.get("resid")))
        if row is None:
            continue
        candidates = [
            r
            for r in residues
            if r.get("chain", "").strip() in {row["chain"], row["display_chain"]}
            and int(r.get("resid", 0)) == row["resid"]
        ]
        if not candidates:
            continue
        # Display aliases cannot merge a real A chain with an unnamed chain.
        competing = [
            r
            for r in mapping["residues"]
            if r["resid"] == row["resid"] and r["display_chain"] == row["display_chain"]
        ]
        if len(candidates) != 1 or len(competing) != 1:
            raise ValueError(
                "Ambiguous PROPKA residue mapping; use distinct chain/residue identifiers"
            )
        target = candidates[0]
        aliases = {
            "HSD": "HIS",
            "HSE": "HIS",
            "HSP": "HIS",
            "HID": "HIS",
            "HIE": "HIS",
            "HIP": "HIS",
        }
        if aliases.get(target["resname"], target["resname"]) != prediction["residue_name"]:
            raise ValueError("PROPKA residue identity does not match the checked input")
        result.append(
            {
                **prediction,
                "chain": target.get("chain", ""),
                "resid": target["resid"],
                "residue_name": target["resname"],
            }
        )
    return result

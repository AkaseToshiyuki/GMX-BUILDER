"""Audited, condition-bound CHARMM36m to CHARMM36 V4 publication contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

SOURCE_FAMILY = "charmm36m-lipid"
TARGET_FAMILY = "charmm36-lipid"
PROVENANCE_SCHEMA = "v4-charmm-reuse-1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


@lru_cache(maxsize=16)
def _hashes_match(root, expected, stamp, compatibility=()):
    # The stat key invalidates cached hashes after any installed-file edit.
    from gmxbuilder.modules.membrane.parameter_provenance import _ADMISSION_ONLY_POLICY_HASHES

    def canonical(name, value):
        for compatible_name, current, prior in compatibility:
            if name == compatible_name and value == current:
                value = prior
        if name == "modules/forcefield/lipid_policy.py":
            return _ADMISSION_ONLY_POLICY_HASHES.get(value, value)
        return value

    return all(
        canonical(name, file_digest(Path(root) / name)) == canonical(name, checksum)
        for name, checksum in expected
    )


def installed_matches(root, files, *, compatibility=()):
    stamp = tuple(
        (name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        for name in sorted(files)
        for stat in [(root / name).stat()]
    )
    return _hashes_match(str(root), tuple(sorted(files.items())), stamp, compatibility)


def _exporter_compatibility(registry, name):
    """Bind unchanged lipid exports to one exact prior audit and source revision."""
    path = Path(__file__).parents[2] / "data" / "v4_charmm_reuse_compatibility.json"
    try:
        compatibility = json.loads(path.read_text())
        if (
            compatibility["schema"] != "v4-charmm-reuse-source-compatibility-1"
            or compatibility["audit_sha256"] != digest(registry)
            or name not in compatibility["entries"]
            or name in {"PPCPL", "PPEPL"}
        ):
            return ()
        # A compatible refactor may move source handling to a helper. Bind
        # those exact bytes too; a later helper edit must invalidate reuse.
        if not installed_matches(Path(__file__).parents[2], compatibility.get("support_files", {})):
            return ()
        return tuple(
            (relative, hashes["current"], hashes["prior"])
            for relative, hashes in sorted(compatibility["exporter_files"].items())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return ()


def audit_registry():
    path = Path(__file__).parents[2] / "data" / "v4_charmm_reuse.json"
    registry = json.loads(path.read_text())
    if registry["schema"] != "v4-charmm-reuse-audit-1":
        raise ValueError("Unknown CHARMM reuse audit")
    return registry


def matching_conditions(protocol, expected):
    from gmxbuilder.modules.membrane.v4_reanalysis import _digest

    if not isinstance(protocol, dict) or protocol.get("sha256") != _digest(protocol):
        return False
    conditions = {k: v for k, v in protocol.items() if k not in {"family", "sha256"}}
    if digest(conditions) == expected:
        return True
    from gmxbuilder.modules.membrane.local_relaxation import construction_policy
    from gmxbuilder.modules.membrane.v4_platform import construction_policy as legacy_policy

    # The audit binds physical conditions and exporter/parameter bytes. The one
    # named analysis-only transition cannot change those physical conditions.
    if conditions.get("construction_policy") != construction_policy():
        return False
    return digest({**conditions, "construction_policy": legacy_policy()}) == expected


def reuse_evidence(entry):
    """Fail closed on changed chemistry, protocols, installed files or MDP output."""
    from gmxbuilder.modules.forcefield.catalog import force_field_directory
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    registry = audit_registry()
    if entry.get("family") != TARGET_FAMILY or entry["lipid"] not in registry["entries"]:
        raise ValueError("Pair is outside the audited 58-entry CHARMM allowlist")
    if not installed_matches(
        Path(__file__).parents[2],
        registry["exporter_files"],
        compatibility=_exporter_compatibility(registry, entry["lipid"]),
    ):
        raise ValueError("Topology exporter differs from the reviewed CHARMM code")
    if not installed_matches(Path(__file__).parents[2], registry.get("local_parameter_files", {})):
        raise ValueError("Local CHARMM supplements differ from the reuse audit")
    name = entry["lipid"]
    audited = registry["entries"][name]
    source = resolve_protocol(name, SOURCE_FAMILY)
    target = resolve_protocol(name, TARGET_FAMILY)
    if entry.get("v4_protocol") != target or not all(
        matching_conditions(protocol, audited["conditions_sha256"]) for protocol in (source, target)
    ):
        raise ValueError("Source or target conditions differ from the CHARMM reuse audit")
    for ff, files in registry["parameter_files"].items():
        from gmxbuilder.modules.membrane.parameter_provenance import installed_forcefield_matches

        root = force_field_directory(ff)
        if root is None or not installed_forcefield_matches(ff, root, files):
            raise ValueError("Installed CHARMM parameters differ from the reuse audit")
        mdp = LipidEquilibrationBuilder._mdp("npt", 0, target["temperature_K"], force_field=ff)
        if hashlib.sha256(mdp.encode()).hexdigest() != audited["mdp_sha256"]:
            raise ValueError("Effective CHARMM simulation settings differ from the audit")
    return {
        "audit_sha256": digest(registry),
        "conditions_sha256": audited["conditions_sha256"],
        "source_protocol": source,
        "target_protocol": target,
        "itp_sha256": audited["itp_sha256"],
        "gmx_sha256": registry["gmx_sha256"],
    }


def source_entry(entry):
    try:
        evidence = reuse_evidence(entry)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {
        **entry,
        "family": SOURCE_FAMILY,
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
        "v4_protocol": evidence["source_protocol"],
    }


def target_metadata(source, evidence, files, simulation):
    """Change only the namespace; keep original MD and measurement provenance."""
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature

    target = copy.deepcopy(source)
    target.update(
        force_field="charmm36",
        lipid_ff="charmm36",
        parameter_family=TARGET_FAMILY,
        v4_protocol=evidence["target_protocol"],
        topology_sha256=topology_signature(source["atom_names"], "charmm36", "charmm36"),
    )
    from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

    target["parameter_fingerprint"] = fingerprint_from_metadata(target)
    target["reused_from"] = {
        "schema": PROVENANCE_SCHEMA,
        "audit_sha256": evidence["audit_sha256"],
        "source_metadata": copy.deepcopy(source),
        "source_metadata_sha256": digest(source),
        "source_conformer_sha256": files,
        "simulation": simulation,
        "independent_trajectory_added": False,
    }
    return target


def validate_reused_metadata(metadata):
    """A copied ready flag cannot replace accepted source evidence.

    Return the original candidate so the normal qualifier can evaluate it,
    including an exact maintainer approval, without granting a new waiver.
    """
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature
    from gmxbuilder.modules.membrane.v4_construction import IDENTITY_FIELDS, REPLICA_FIELDS

    proof = metadata["reused_from"]
    source = proof["source_metadata"]
    evidence = reuse_evidence(
        {
            "lipid": metadata["lipid_name"],
            "family": metadata["parameter_family"],
            "v4_protocol": metadata["v4_protocol"],
        }
    )
    if (
        proof["schema"] != PROVENANCE_SCHEMA
        or proof["audit_sha256"] != evidence["audit_sha256"]
        or proof["source_metadata_sha256"] != digest(source)
        or proof["independent_trajectory_added"] is not False
        or source.get("reused_from")
        or source["parameter_family"] != SOURCE_FAMILY
        or source["force_field"] != "charmm36m"
        or source["lipid_ff"] != "charmm36m"
        or source["v4_protocol"] != evidence["source_protocol"]
        or metadata["v4_protocol"] != evidence["target_protocol"]
        or proof["simulation"]["gmx_sha256"] != evidence["gmx_sha256"]
        or proof["simulation"]["itp_sha256"] != evidence["itp_sha256"]
    ):
        raise ValueError("Invalid CHARMM reuse provenance")
    for name in ("tpr_sha256", "original_protocol_sha256"):
        checksum = proof["simulation"][name]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ValueError("Missing original simulation provenance")
    original = source.get("simulation_protocol", source["v4_protocol"])
    if proof["simulation"]["original_protocol_sha256"] != original["sha256"]:
        raise ValueError("Reused entry changed its original simulation protocol")
    from gmxbuilder.modules.membrane.v4_tpr import validate_inputrec

    effective = proof["simulation"]["effective_inputrec"]
    text = "\n".join(
        f"{key} = " + (" ".join(map(str, value)) if isinstance(value, list) else str(value))
        for key, value in {**effective, "pcoupltype": "semiisotropic"}.items()
    )
    validate_inputrec(text, original["temperature_K"], "charmm36m", "npt")
    changes = {
        "force_field",
        "lipid_ff",
        "parameter_family",
        "v4_protocol",
        "topology_sha256",
        "parameter_fingerprint",
    }
    for field in (set(IDENTITY_FIELDS) | set(REPLICA_FIELDS) | {"trajectory_analysis"}) - changes:
        if metadata[field] != source[field]:
            raise ValueError(f"Reused candidate changed source evidence: {field}")
    if (
        metadata["force_field"] != "charmm36"
        or metadata["lipid_ff"] != "charmm36"
        or metadata["topology_sha256"]
        != topology_signature(metadata["atom_names"], "charmm36", "charmm36")
    ):
        raise ValueError("Invalid reused target namespace")

    def provenance(m):
        return [
            {k: v for k, v in record.items() if k not in {"file", "replica"}}
            for record in m["conformer_provenance"]
        ]

    if provenance(metadata) != provenance(source):
        raise ValueError("Reused conformers changed their physical source")
    if set(proof["source_conformer_sha256"]) != {p["file"] for p in source["conformer_provenance"]}:
        raise ValueError("Incomplete source conformer hashes")
    return source


def reused_coordinates_valid(directory, metadata):
    """Validate copies against original hashes, with stat-keyed cache invalidation."""
    from gmxbuilder.modules.membrane.v4_construction import assembled_replicas

    if not metadata.get("v4_protocol"):
        return True
    try:
        replicas = assembled_replicas(metadata) if "replica_records" in metadata else [metadata]
        if not any("reused_from" in replica for replica in replicas):
            return True
        files = {}
        for replica in replicas:
            proof = replica["reused_from"]
            source = proof["source_metadata"]
            for target, original in zip(
                replica["conformer_provenance"], source["conformer_provenance"], strict=True
            ):
                name = target["file"]
                if Path(name).name != name or not name.startswith("conf_"):
                    return False
                files[name] = proof["source_conformer_sha256"][original["file"]]
        return len(files) == metadata["n_conformations"] and installed_matches(directory, files)
    except (ValueError, OSError, KeyError, TypeError):
        return False

"""Require an explicit disposition for normalized uploaded protein chemistry."""

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.modifications.selection import validate_selection


def resolve_input_modifications(system, modifications, order, decisions):
    records = system.metadata.get("input_modifications", {}).get("records", [])
    normalized = {
        (record["chain"], int(record["resid"])): record
        for record in records
        if record.get("status") == "recognized"
    }
    removals = set()
    for entry in decisions:
        if not isinstance(entry, dict) or set(entry) != {"target", "action"}:
            raise ModuleConfigError("Input modification decisions require target and action")
        validate_selection(entry)
        target = entry["target"]
        key = (target["chain"], target["resid"])
        record = normalized.get(key)
        if (
            entry["action"] != "remove"
            or record is None
            or target["resname"] != record["standard_resname"]
            or key in removals
        ):
            raise ModuleConfigError(f"Invalid uploaded modification decision at {key}")
        removals.add(key)
    selected = {order[entry["index"]]: entry["patch_id"] for entry in modifications}
    resolved = []
    for key, record in normalized.items():
        if key not in order:
            raise ModuleConfigError(f"Uploaded modification at {key} lost its residue identity")
        preserved = selected.get(key) == record["patch_id"]
        if preserved and key in removals:
            raise ModuleConfigError(
                f"Conflicting keep/remove decisions for uploaded modification at {key}"
            )
        if not preserved and key not in removals:
            raise ModuleConfigError(
                f"Uploaded {record['original_resname']} at {key[0]}:{key[1]} requires "
                f"{record['patch_id']}. Restore it with a compatible force field, or "
                "explicitly accept removing this uploaded modification."
            )
        resolved.append(
            {
                "chain": key[0],
                "resid": key[1],
                "patch_id": record["patch_id"],
                "action": "preserve" if preserved else "remove",
            }
        )
    return resolved

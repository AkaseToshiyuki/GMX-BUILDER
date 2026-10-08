"""Stable protein residue selection across UI, API and checkpoint boundaries."""

from gmxbuilder.core.exceptions import ModuleConfigError


def validate_selection(entry, index_key="index", target_key="target"):
    target = entry.get(target_key)
    if target is not None:
        if (
            not isinstance(target, dict)
            or set(target) != {"chain", "resid", "resname"}
            or not isinstance(target["chain"], str)
            or isinstance(target["resid"], bool)
            or not isinstance(target["resid"], int)
            or not isinstance(target["resname"], str)
            or not target["resname"].strip()
        ):
            raise ModuleConfigError(f"Invalid residue identity in {target_key}")
    index = entry.get(index_key)
    if index is None and target is not None:
        return
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ModuleConfigError(f"Invalid residue index in {index_key}: {index!r}")


def resolve_selection(entry, order, names, *, index_key="index", target_key="target", mixed=False):
    """Identity is authoritative; legacy indices are safe only for pure proteins."""
    validate_selection(entry, index_key, target_key)
    result = dict(entry)
    target = entry.get(target_key)
    if target is not None:
        key = (target["chain"], target["resid"])
        if key not in order or names[key] != target["resname"].strip().upper():
            raise ModuleConfigError(
                f"Residue {key[0]}:{key[1]} no longer matches {target['resname']}; "
                "reload the checked input and select its current identity."
            )
        result[index_key] = order.index(key)
    elif mixed:
        raise ModuleConfigError(
            "Residue indices are ambiguous in a protein/non-protein mixture. "
            f"Supply {target_key} with chain, resid and resname."
        )
    elif result[index_key] >= len(order):
        raise ModuleConfigError(f"Residue index {result[index_key]} is outside the protein")
    return result


CHEMISTRY_POLICY_VERSION = 1


def chemistry_fingerprint(config):
    """Bind the checked chemistry to its replayable request, not mutable UI state."""
    import hashlib
    import json

    from gmxbuilder.pipeline.provenance import replayable_config

    request = replayable_config(config)
    # StepRunner injects a seed, but StructureProcessor has no stochastic seed input.
    request.pop("seed", None)
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()

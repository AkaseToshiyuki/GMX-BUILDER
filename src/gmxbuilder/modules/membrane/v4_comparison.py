"""Composition provenance for physically comparable replica observables."""

from __future__ import annotations

from collections import Counter

WATER_RESIDUES = frozenset({"SOL", "WAT", "TIP3", "HOH", "T3P"})
# The V4 builder currently adds sodium/chloride. Unrecognized solutes must not
# acquire an automatic exemption just because their residue name is unfamiliar.
SALT_RESIDUES = frozenset({"NA", "CL", "SOD", "CLA"})
EXTENSIVE_BOX_OBSERVABLES = frozenset({"volume_nm3", "box_z_nm"})


def system_composition(structure, resname_map):
    """Count contiguous residues, including GRO's five-digit residue wrap."""
    counts = Counter()
    previous = None
    for residue, name in zip(structure.resids, structure.resnames, strict=True):
        raw = str(name).strip().upper()
        key = (int(residue), raw)
        if key != previous:
            counts[resname_map.get(raw, raw)] += 1
        previous = key
    return {"source": "whole-coordinate-residues", "molecule_counts": dict(sorted(counts.items()))}


def comparison_policy(replicas):
    """Exclude raw box matching only for a documented hydration/salt difference.

    Per-run box sampling requirements remain active. Strict box drift checks
    depend on the caller's diagnostic policy. The construction qualifier keeps
    APL/thickness matching hard and order matching diagnostic. A lipid or
    unknown-solute mismatch requires review, not an MD extension.
    """
    contexts = [replica.get("system_composition") for replica in replicas]
    result = {"excluded": {}, "failures": [], "context": "identical_composition"}
    if not any(contexts):
        # Legacy records cannot justify a composition exemption. Preserve the
        # stricter comparison instead of assuming their different boxes are OK.
        return {**result, "context": "composition_unavailable; all observables compared"}
    try:
        counts = [context["molecule_counts"] for context in contexts]
        if any(
            not count or any(type(n) is not int or n <= 0 for n in count.values())
            for count in counts
        ):
            raise ValueError("Invalid molecule counts")
        solute = [
            {k: n for k, n in count.items() if k not in WATER_RESIDUES | SALT_RESIDUES}
            for count in counts
        ]
        if any(value != solute[0] for value in solute[1:]):
            return {
                **result,
                "failures": ["Replica lipid/solute compositions differ; review required"],
            }
        waters = [sum(n for k, n in count.items() if k in WATER_RESIDUES) for count in counts]
        if any(number != waters[0] for number in waters[1:]):
            if any(not any(k in WATER_RESIDUES for k in count) for count in counts):
                raise ValueError("Missing hydration evidence")
            reason = (
                "Different recorded water/ion counts; raw box size is not composition invariant"
            )
            result.update(
                context="different_hydration_or_salt",
                excluded={key: reason for key in sorted(EXTENSIVE_BOX_OBSERVABLES)},
            )
        return result
    except (KeyError, TypeError, ValueError, AttributeError):
        return {
            **result,
            "failures": ["Missing or invalid replica composition evidence; review required"],
        }

"""Select a coherent residue conformer from explicit alternate locations."""

from collections import defaultdict
from math import isfinite

from gmxbuilder.core.exceptions import ParseError


def select_residue_conformers(records):
    """Return row indices from (index, residue key, resname, atom, alt, occupancy).

    One label is selected per residue, by mean occupancy of its alternate
    atoms, with lexical label order breaking ties. Blank atoms are shared.
    Never fill holes using a different alternate conformation.
    """
    residues = defaultdict(list)
    issues = []
    for record in records:
        if not isfinite(record[5]) or record[5] < 0:
            issues.append(
                {
                    "code": "invalid_occupancy",
                    "record": record[0],
                    "residue": list(record[1]),
                    "resname": record[2],
                    "atom": record[3],
                    "altloc": record[4],
                    "value": str(record[5]),
                }
            )
        residues[record[1]].append(record)
    if issues:
        first = issues[0]
        raise ParseError(
            f"Invalid occupancy in {len(issues)} atom(s): {first['residue']} "
            f"{first['resname']} {first['atom']} has {first['value']}. "
            "Occupancy must be finite and non-negative. Correct the source model "
            "explicitly and recheck completeness; values are not changed automatically.",
            issues=issues,
        )
    selected = []
    for key, atoms in residues.items():
        if len({a[2] for a in atoms}) != 1:
            raise ParseError(
                f"Multiple residue identities at {key}; select one identity explicitly"
            )
        identities = [(a[3], a[4]) for a in atoms]
        if len(set(identities)) != len(identities):
            raise ParseError(
                f"Duplicate atom identity at {key}; distinct atoms require unique identities"
            )
        if any(not a[3].strip() for a in atoms):
            raise ParseError(f"Missing atom name at {key}")
        labels = sorted({a[4] for a in atoms if a[4]})
        chosen = (
            min(
                labels,
                key=lambda label: (
                    -sum(a[5] for a in atoms if a[4] == label) / sum(a[4] == label for a in atoms)
                ),
            )
            if labels
            else ""
        )
        candidates = {}
        for a in atoms:
            if a[4] not in {"", chosen}:
                continue
            previous = candidates.get(a[3])
            if previous is None or (bool(a[4]), a[5]) > (bool(previous[4]), previous[5]):
                candidates[a[3]] = a
        missing = {a[3] for a in atoms} - candidates.keys()
        if missing:
            raise ParseError(
                f"Alternate conformer {chosen!r} at {key} is incomplete "
                f"({', '.join(sorted(missing))}); "
                "provide one complete conformer instead of combining alternate atom positions"
            )
        selected.extend(a[0] for a in candidates.values())
    return sorted(selected)

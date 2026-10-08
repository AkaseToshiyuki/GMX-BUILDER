"""Bounded, explicit research analogies for missing alkyl/ammonium interactions.

CG314/CG324/CG334 -> CG311/CG321/CG331 keeps element, SP3 valence and hydrogen
count, but removes adjacent positive-nitrogen context. For proper torsion ends,
CG331 -> CG321 keeps the central bond but changes terminal carbon substitution.
These are NOT calibrated CGenFF equivalences.
It is used only for missing bonded terms in the opt-in aryl/ammonium model.
Atom types, LJ and charges remain unchanged. Every copied Fourier term is
retained and traced to its installed parameter-file line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gmxbuilder.modules.forcefield.charmm_compat import CharmmCompatError


@dataclass(frozen=True)
class Parameter:
    types: tuple[str, ...]
    function: str
    values: tuple[str, ...]
    line: int


def read_parameters(path: Path):
    counts = {"bondtypes": 2, "angletypes": 3, "dihedraltypes": 4}
    records = {name: [] for name in counts}
    section = ""
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
        elif code and not code.startswith("#") and section in counts:
            fields = code.split()
            n = counts[section]
            if len(fields) < n + 2:
                raise CharmmCompatError("FF_VERSION_MISMATCH", "malformed CHARMM bonded database")
            records[section].append(
                Parameter(tuple(fields[:n]), fields[n], tuple(fields[n + 1 :]), number)
            )
    return records


def matches(pattern, types):
    return any(
        all(a == "X" or a == b for a, b in zip(pattern, candidate, strict=True))
        for candidate in (types, types[::-1])
    )


def resolve_missing(topology: Path, parameter_file: Path):
    """Leave native exact/wildcard resolution intact; annotate only missing terms."""
    records = read_parameters(parameter_file)
    kinds = {
        "bonds": ("bondtypes", 2),
        "angles": ("angletypes", 3),
        "dihedrals": ("dihedraltypes", 4),
    }
    atom_types, output, audit = {}, [], []
    section = ""
    for raw in topology.read_text().splitlines():
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
        if section == "atoms" and code and not code.startswith(("[", "#")):
            fields = code.split()
            atom_types[fields[0]] = fields[1]
        if section not in kinds or not code or code.startswith(("[", "#")):
            output.append(raw)
            continue
        database, count = kinds[section]
        fields = code.split()
        if len(fields) > count + 1:  # Explicit native term already supplies its values.
            output.append(raw)
            continue
        indices, function = fields[:count], fields[count]
        types = tuple(atom_types[i] for i in indices)
        candidates = [p for p in records[database] if p.function == function]
        if any(matches(p.types, types) for p in candidates):
            output.append(raw)
            continue
        if section == "dihedrals" and function != "9":
            raise CharmmCompatError(
                "PARAMETER_MISSING",
                f"no bounded analogy for {section} {types}, function {function}",
            )
        substitutions = {"CG314": "CG311", "CG324": "CG321", "CG334": "CG331"}
        alternatives = []
        for index, atom_type in enumerate(types):
            replacement = substitutions.get(atom_type)
            if section == "dihedrals" and index in (0, 3) and atom_type == "CG331":
                replacement = "CG321"
            if replacement is None:
                continue
            candidate = tuple(replacement if i == index else t for i, t in enumerate(types))
            terms = [p for p in candidates if p.types in (candidate, candidate[::-1])]
            if terms:
                alternatives.append((int(index not in (0, count - 1)), candidate, terms))
        if not alternatives:
            raise CharmmCompatError(
                "PARAMETER_MISSING", f"no single-substitution source for {section} {types}"
            )
        alternatives.sort(key=lambda item: (item[0], item[1]))
        priority, analog, source = alternatives[0]
        if any(
            {p.values for p in other} != {p.values for p in source}
            for tier, _, other in alternatives
            if tier == priority
        ):
            raise CharmmCompatError(
                "PARAMETER_MISSING", "equally ranked research analogies disagree"
            )
        # Analogy must land on an exact source tuple, not stack another wildcard
        # or substitute additional types. No fitted force constants are invented.
        changes = ",".join(f"{a}->{b}" for a, b in zip(types, analog, strict=True) if a != b)
        if section != "dihedrals" and len({p.values for p in source}) != 1:
            raise CharmmCompatError("PARAMETER_MISSING", "ambiguous source parameters for analogy")
        terms = {}
        for parameter in source:
            key = parameter.values[-1] if section == "dihedrals" else "single"
            if key in terms and terms[key].values != parameter.values:
                raise CharmmCompatError(
                    "PARAMETER_MISSING", "conflicting source Fourier multiplicities"
                )
            terms[key] = parameter
        copied = sorted(terms.values(), key=lambda p: p.line)
        for parameter in copied:
            output.append(
                " ".join((*indices, function, *parameter.values))
                + f" ; RESEARCH {changes}, ffbonded.itp:{parameter.line}"
            )
        audit.append(
            {
                "section": section,
                "atom_indices": [int(i) for i in indices],
                "target_types": list(types),
                "source_types": list(analog),
                "function": function,
                "source_file": parameter_file.name,
                "source_lines": [p.line for p in copied],
                "parameters": [list(p.values) for p in copied],
                "fourier_terms_retained": len(copied),
                "transformation": changes + " for bonded lookup only",
                "uncertainty": "uncalibrated; charge context or terminal substitution differs",
                "physical_validation": "not_evaluated",
            }
        )
    topology.write_text("\n".join(output) + "\n")
    return audit

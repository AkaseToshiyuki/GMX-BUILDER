"""Apply supplemental CHARMM parameters only to their local chemical motif.

The West plasmalogen extension reuses ordinary CHARMM atom types. Loading its
dihedraltypes globally also changes an ester-linked oleoyl chain, including in
the same plasmalogen molecule. Explicit molecule terms avoid that collision.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=2)
def _parameters(force_field: str) -> tuple[tuple[str, ...], ...]:
    if force_field not in {"charmm36", "charmm36m"}:
        raise ValueError("Local CHARMM lipid terms require a CHARMM force field")
    root = Path(__file__).resolve().parents[2] / "data" / "forcefield_overlays"
    path = root / force_field / "gmxbuilder-plasmalogen-bonded.itp"
    section = ""
    terms = []
    sizes = {"bondtypes": 2, "angletypes": 3, "dihedraltypes": 4}
    for line in path.read_text().splitlines():
        code = line.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ")
        elif code and section in sizes:
            fields = tuple(code.split())
            terms.append((str(sizes[section]), *fields))
    return tuple(terms)


def local_bonded_parameters(
    record: dict, names: tuple[str, ...], function: int, force_field: str
) -> tuple[tuple[str, ...], ...]:
    """Return explicit function/parameters, or an empty tuple for native lookup.

    The template declares the actual vinyl-ether atoms. The name boundary keeps
    the unrelated 18:1 ester chain native even though its atom types coincide.
    No supplemental definitions are installed at global parameter scope.
    """
    scope = set(record.get("local_bonded_scope", ()))
    if not scope.intersection(names):
        return ()
    types = {atom[0]: atom[1] for atom in record["atoms"]}
    actual = tuple(types[name] for name in names)
    result = []
    for size, *fields in _parameters(force_field):
        count = int(size)
        if count != len(names) or int(fields[count]) != function:
            continue
        expected = tuple(fields[:count])
        if actual == expected or (function != 2 and actual[::-1] == expected):
            result.append(tuple(fields[count:]))
    return tuple(result)

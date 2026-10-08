"""Compare resolved CHARMM36 and CHARMM36m parameters before reusing a lipid run.

Build the same lipid/water/Na/Cl probe with each installed force field and
compare GROMACS TPR dumps. This includes bonded terms and multiplicities,
1-4 interactions, and the nonbonded matrix with ion-lipid NBFIX corrections.

Compare charges and types exactly, and numeric parameters with the declared
converter-rounding tolerance. Missing inputs or preprocessing failures prevent
reuse. Equivalence applies to this probe and these installations, not every
molecule supported by either force field."""

from __future__ import annotations

import re
import subprocess
import tempfile
from functools import cache, lru_cache
from pathlib import Path

#: Relative tolerance for resolved probe parameters, allowing converter rounding.
#: Passing this tolerance establishes numerical agreement only for the probe.
CONVERTER_ROUNDING = 1e-3

#: Include ions and water to expose cross interactions, including ion-lipid NBFIX.
PROBE_IONS = 2
PROBE_WATERS = 3

_GROMPP_MDP = """integrator = md
nsteps = 0
continuation = yes
cutoff-scheme = Verlet
rlist = 1.2
coulombtype = PME
rcoulomb = 1.2
vdwtype = cutoff
vdw-modifier = force-switch
rvdw-switch = 1.0
rvdw = 1.2
"""

_FUNCTYPE = re.compile(r"functype\[(\d+)\]=(\S+?),\s*(.*)$")
_VALUE = re.compile(r"([A-Za-z0-9_]+)=\s*(-?[\d.]+e?[-+]?\d*)")
_ATOM = re.compile(r"atom\[\s*(\d+)\]=\{type=\s*(\d+).*?m=\s*(\S+?),\s*q=\s*(\S+?),")
_TYPENAME = re.compile(r'type\[\s*(\d+)\]=\{name="([^"]*)"')
_MOLTYPE = re.compile(r"^\s{3}moltype \((\d+)\):")
_MOLNAME = re.compile(r'^\s+name="([^"]*)"')
_BLOCK = re.compile(r"^\s{6}([A-Za-z][A-Za-z0-9 .,()/'\-]*):$")
_ENTRY = re.compile(r"^\s+\d+ type=(\d+) \([^)]*\)((?:\s+\d+)+)\s*$")


@lru_cache(maxsize=4)
def _nonbonded(force_field: str) -> dict[str, tuple[float, float, float]]:
    from gmxbuilder.modules.forcefield.catalog import force_field_directory

    root = force_field_directory(force_field)
    if root is None:
        return {}
    table: dict[str, tuple[float, float, float]] = {}
    section = ""
    for line in (root / "ffnonbonded.itp").read_text(errors="replace").splitlines():
        code = line.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
            continue
        fields = code.split()
        if section == "atomtypes" and len(fields) >= 7:
            try:
                table[fields[0]] = (float(fields[-2]), float(fields[-1]), float(fields[2]))
            except ValueError:
                continue
    return table


@lru_cache(maxsize=4)
def _residues(force_field: str) -> dict[str, dict]:
    from gmxbuilder.modules.forcefield.catalog import force_field_directory
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser

    root = force_field_directory(force_field)
    if root is None:
        return {}
    found: dict[str, dict] = {}
    for filename in ("lipid.rtp", "merged.rtp"):
        path = root / filename
        if not path.is_file():
            continue
        parser = RTPParser(path)
        for name in parser.residue_names:
            record = parser.get_residue(name)
            if record:
                found.setdefault(name.upper(), record)
    return found


def _molecule_terms(record: dict) -> dict:
    """Every term the residue's own atoms and bonds imply.

    This is what pdb2gmx would write for the molecule: grompp generates no
    bonded terms of its own, so the topology has to name them and let each
    installation resolve their parameters.
    """
    order = [str(atom[0]) for atom in record.get("atoms") or []]
    types = {str(atom[0]): str(atom[1]) for atom in record.get("atoms") or []}
    charges = {str(atom[0]): float(atom[2]) for atom in record.get("atoms") or []}
    neighbours: dict[str, set[str]] = {name: set() for name in order}
    for first, second in record.get("bonds") or []:
        a, b = str(first), str(second)
        if a not in neighbours or b not in neighbours:
            continue  # crosses a residue boundary; grompp resolves it elsewhere
        neighbours[a].add(b)
        neighbours[b].add(a)

    bonds = sorted({tuple(sorted((a, b))) for a in neighbours for b in neighbours[a]})
    angles = []
    for b in order:
        around = sorted(neighbours[b])
        for index, a in enumerate(around):
            for c in around[index + 1 :]:
                angles.append((a, b, c))
    dihedrals: list[tuple[str, str, str, str]] = []
    pairs: set[tuple[str, str]] = set()
    for b in order:
        for c in sorted(neighbours[b]):
            if b >= c:
                continue
            for a in sorted(neighbours[b] - {c}):
                for d in sorted(neighbours[c] - {b}):
                    if a == d:
                        continue
                    dihedrals.append((a, b, c, d))
                    pairs.add(tuple(sorted((a, d))))
    impropers = [
        tuple(str(x) for x in term)
        for term in (record.get("impropers") or [])
        if len(term) == 4 and all(str(x) in types for x in term)
    ]
    return {
        "record": record,
        "order": order,
        "types": types,
        "charges": charges,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "pairs": sorted(pairs),
        "impropers": impropers,
    }


def _write_probe(terms: dict, root: Path, work: Path) -> None:
    """A one-molecule system with ions and water, ready for grompp."""
    order = terms["order"]
    index = {name: number for number, name in enumerate(order, start=1)}

    def bonded(names, function):
        from gmxbuilder.modules.forcefield.charmm_lipid_local import local_bonded_parameters

        rows = local_bonded_parameters(
            terms.get("record", {}), tuple(names), function, root.name.removesuffix(".ff")
        )
        return [
            " ".join(map(str, (*(index[name] for name in names), *row))) + "\n"
            for row in rows or ((str(function),),)
        ]

    lines = ["[ moleculetype ]\nLIP 3\n\n[ atoms ]\n"]
    for number, name in enumerate(order, start=1):
        lines.append(
            f"{number} {terms['types'][name]} 1 LIP {name} {number} {terms['charges'][name]:.6f}\n"
        )
    lines.append("\n[ bonds ]\n")
    lines += [line for names in terms["bonds"] for line in bonded(names, 1)]
    lines.append("\n[ pairs ]\n")
    lines += [f"{index[a]} {index[b]} 1\n" for a, b in terms["pairs"]]
    lines.append("\n[ angles ]\n")
    lines += [line for names in terms["angles"] for line in bonded(names, 5)]
    lines.append("\n[ dihedrals ]\n")
    lines += [line for names in terms["dihedrals"] for line in bonded(names, 9)]
    lines.append("\n[ dihedrals ]\n")
    lines += [line for names in terms["impropers"] for line in bonded(names, 2)]
    (work / "lipid.itp").write_text("".join(lines))

    # The ions are written here rather than included, because the two releases
    # package ions.itp differently and the probe must differ only in the
    # parameters, never in what molecules it contains.
    (work / "topol.top").write_text(
        f'#include "{root}/forcefield.itp"\n'
        '#include "lipid.itp"\n'
        "\n[ moleculetype ]\nIONP 1\n\n[ atoms ]\n1 SOD 1 SOD SOD 1 1.000000\n"
        "\n[ moleculetype ]\nIONM 1\n\n[ atoms ]\n1 CLA 1 CLA CLA 1 -1.000000\n"
        f'\n#include "{root}/tip3p.itp"\n'
        "\n[ system ]\nequivalence probe\n\n[ molecules ]\n"
        f"LIP 1\nIONP {PROBE_IONS}\nIONM {PROBE_IONS}\nSOL {PROBE_WATERS}\n"
    )

    # Coordinates only have to satisfy grompp: every excluded pair inside the
    # cut-off, which a tight lattice guarantees whatever the molecule is.
    total = len(order) + 2 * PROBE_IONS + 3 * PROBE_WATERS
    side = max(2, round(len(order) ** (1 / 3)) + 1)
    rows = [f"equivalence probe\n{total}\n"]

    def row(resid: int, residue: str, atom: str, x: float, y: float, z: float, serial: int):
        return f"{resid:5d}{residue:<5s}{atom[:5]:>5s}{serial:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"

    serial = 0
    for position, name in enumerate(order):
        rows.append(
            row(
                1,
                "LIP",
                name,
                0.08 * (position % side),
                0.08 * ((position // side) % side),
                0.08 * (position // (side * side)),
                serial + 1,
            )
        )
        serial += 1
    resid = 2
    for kind, residue in (("IONP", "SOD"), ("IONM", "CLA")):
        for _ in range(PROBE_IONS):
            rows.append(row(resid, residue, residue, 2.0 + 0.4 * resid, 2.0, 2.0, serial + 1))
            serial += 1
            resid += 1
    for water in range(PROBE_WATERS):
        for offset, atom in enumerate(("OW", "HW1", "HW2")):
            rows.append(
                row(resid, "SOL", atom, 6.0 + 0.4 * water + 0.1 * offset, 2.0, 2.0, serial + 1)
            )
            serial += 1
        resid += 1
    rows.append("  10.00000  10.00000  10.00000\n")
    (work / "conf.gro").write_text("".join(rows))
    (work / "grompp.mdp").write_text(_GROMPP_MDP)


def _parse_dump(text: str) -> dict:
    """The parameters a tpr holds, as grompp resolved them."""
    atnr = 0
    functypes: dict[int, tuple[str, tuple[float, ...]]] = {}
    molecules: dict[str, dict] = {}
    current: dict | None = None
    block = ""
    in_molecule = False
    for line in text.splitlines():
        if line.startswith("      atnr="):
            atnr = int(line.partition("=")[2])
            continue
        matched = _FUNCTYPE.search(line)
        if matched and not in_molecule:
            values = tuple(float(value) for _key, value in _VALUE.findall(matched.group(3)))
            functypes[int(matched.group(1))] = (matched.group(2), values)
            continue
        if _MOLTYPE.match(line):
            in_molecule = True
            current = {"atoms": {}, "typenames": {}, "interactions": {}}
            block = ""
            continue
        if current is None:
            continue
        name = _MOLNAME.match(line)
        if name and "name" not in current:
            current["name"] = name.group(1)
            molecules[name.group(1)] = current
            continue
        atom = _ATOM.search(line)
        if atom:
            current["atoms"][int(atom.group(1))] = (
                int(atom.group(2)),
                float(atom.group(3)),
                float(atom.group(4)),
            )
            continue
        typename = _TYPENAME.search(line)
        if typename:
            current["typenames"][int(typename.group(1))] = typename.group(2)
            continue
        heading = _BLOCK.match(line)
        if heading:
            block = heading.group(1)
            continue
        entry = _ENTRY.match(line)
        if entry and block:
            current["interactions"].setdefault(block, []).append(
                (tuple(int(x) for x in entry.group(2).split()), int(entry.group(1)))
            )
    return {"atnr": atnr, "functypes": functypes, "molecules": molecules}


def _canonical(parsed: dict) -> dict:
    """Rewrite a parsed tpr so the two installations can be compared directly.

    Force-field type numbering is internal to each tpr, and so is the set of
    types: grompp merges atom types whose non-bonded parameters agree, and the
    two converters print five and six decimals of the same numbers, so a type
    can merge under one release and stay separate under the other. Naming the
    types would therefore compare different things. The atoms, on the other
    hand, are the same atoms in the same order in both -- the topology here is
    generated from the residue -- so the Lennard-Jones parameters are recorded
    per pair of atoms rather than per pair of types, and every parameter by its
    values.
    """
    atnr = parsed["atnr"]
    functypes = parsed["functypes"]

    catalogue: list[tuple[tuple[str, int], int]] = []
    labels: dict[tuple[str, int], str] = {}
    for name in sorted(parsed["molecules"]):
        molecule = parsed["molecules"][name]
        for position in sorted(molecule["atoms"]):
            catalogue.append(((name, position), molecule["atoms"][position][0]))
            labels[(name, position)] = molecule["typenames"].get(position, "?")

    lennard_jones: dict[tuple[tuple[str, int], tuple[str, int]], tuple[float, ...]] = {}
    for index, (first, first_type) in enumerate(catalogue):
        for second, second_type in catalogue[index:]:
            entry = functypes.get(first_type * atnr + second_type)
            if entry is not None:
                lennard_jones[(first, second)] = entry[1]

    molecules: dict[str, dict] = {}
    for name, molecule in parsed["molecules"].items():
        atoms = [
            (
                position,
                molecule["typenames"].get(position, "?"),
                round(molecule["atoms"][position][1], 6),
                round(molecule["atoms"][position][2], 6),
            )
            for position in sorted(molecule["atoms"])
        ]
        interactions: dict[str, list] = {}
        for block, entries in molecule["interactions"].items():
            resolved = []
            for atoms_involved, functype in entries:
                kind, values = functypes.get(functype, ("?", ()))
                resolved.append((atoms_involved, kind, values))
            interactions[block] = sorted(resolved)
        molecules[name] = {"atoms": atoms, "interactions": interactions}
    return {"lj": lennard_jones, "molecules": molecules, "types": labels}


def _same_numbers(left, right) -> bool:
    """Whether two resolved parameters are the same number.

    The comparison is relative to the values themselves and not to unity: a
    dispersion coefficient is 1e-3 and a repulsion coefficient 1e-7, so a
    tolerance with a floor of 1 would call every Lennard-Jones pair in the force
    field identical -- including the ion-lipid NBFIX corrections that are 10%
    and 20% apart, which is exactly what this has to catch.
    """
    if len(left) != len(right):
        return False
    return all(
        abs(x - y) <= CONVERTER_ROUNDING * max(abs(x), abs(y), 1e-30) for x, y in zip(left, right)
    )


def _resolved(lipid_name: str, force_field: str) -> tuple[dict | None, str]:
    """What grompp makes of this lipid under one installation."""
    from gmxbuilder.modules.forcefield.catalog import force_field_directory
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    gmx = find_gromacs_executable()
    if gmx is None:
        return None, "GROMACS is unavailable, so the two cannot be compared"
    root = force_field_directory(force_field)
    if root is None:
        return None, f"{force_field} is not installed"
    _name, record = lipid_rtp_template(lipid_name, force_field)
    if record is None:
        return None, f"{lipid_name} has no resolvable template under {force_field}"

    terms = _molecule_terms(record)
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-equivalence-") as temporary:
        work = Path(temporary)
        _write_probe(terms, root, work)
        built = subprocess.run(
            [
                gmx,
                "grompp",
                "-f",
                "grompp.mdp",
                "-c",
                "conf.gro",
                "-p",
                "topol.top",
                "-o",
                "probe.tpr",
                "-maxwarn",
                "20",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if built.returncode != 0:
            detail = (built.stdout + built.stderr).strip().splitlines()
            complaint = next(
                (line.strip() for line in detail if "rror" in line and "file" in line),
                detail[-1].strip() if detail else "grompp failed",
            )
            return None, f"{force_field} cannot build {lipid_name}: {complaint}"
        dumped = subprocess.run(
            [gmx, "dump", "-s", "probe.tpr"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if dumped.returncode != 0:
            return None, f"{force_field} tpr for {lipid_name} could not be read back"
        return _canonical(_parse_dump(dumped.stdout)), ""


def _difference(left: dict, right: dict, labels: list[str]) -> str | None:
    """The first thing the two installations disagree about, in words."""

    def label(position: int) -> str:
        return labels[position] if 0 <= position < len(labels) else str(position)

    def pair_name(key) -> str:
        parts = []
        for molecule, position in key:
            atom = label(position) if molecule == "LIP" else molecule
            kind = left["types"].get((molecule, position)) or right["types"].get(
                (molecule, position)
            )
            parts.append(f"{atom} ({kind})" if kind else atom)
        return " and ".join(parts)

    for key in sorted(set(left["lj"]) | set(right["lj"])):
        first, second = left["lj"].get(key), right["lj"].get(key)
        if first is None or second is None:
            missing = "charmm36" if first is None else "charmm36m"
            return f"the pair {pair_name(key)} is absent from {missing}"
        if not _same_numbers(first, second):
            return f"the Lennard-Jones pair {pair_name(key)} differs: {first} against {second}"

    for name in sorted(set(left["molecules"]) | set(right["molecules"])):
        one, other = left["molecules"].get(name), right["molecules"].get(name)
        if one is None or other is None:
            return f"molecule {name} exists under only one release"
        if len(one["atoms"]) != len(other["atoms"]):
            return f"{name} has a different number of atoms"
        for a, b in zip(one["atoms"], other["atoms"]):
            # Charges are compared exactly -- they come from the residue
            # template and a difference in one is a different molecule. Masses
            # are compared to the converter tolerance: the two installations
            # write oxygen as 15.999 and 15.9994, which is the same element and
            # not a property the configurational ensemble depends on at all.
            if a[0] != b[0] or a[3] != b[3] or not _same_numbers((a[2],), (b[2],)):
                return f"{name} atom {label(a[0])} differs: {a[1:]} against {b[1:]}"
        blocks = set(one["interactions"]) | set(other["interactions"])
        for block in sorted(blocks):
            mine = one["interactions"].get(block, [])
            theirs = other["interactions"].get(block, [])
            if len(mine) != len(theirs):
                return (
                    f"{name} has {len(mine)} {block} terms under charmm36 and "
                    f"{len(theirs)} under charmm36m"
                )
            for (atoms, kind, values), (other_atoms, other_kind, other_values) in zip(mine, theirs):
                if atoms != other_atoms or kind != other_kind:
                    return f"{name} {block} terms do not correspond"
                if not _same_numbers(values, other_values):
                    involved = "-".join(label(position) for position in atoms)
                    return f"{name} {block} on {involved} differs: {values} against {other_values}"
    return None


def _counts(canonical: dict) -> tuple[int, int]:
    interactions = sum(
        len(entries)
        for molecule in canonical["molecules"].values()
        for entries in molecule["interactions"].values()
    )
    return interactions, len(canonical["lj"])


@cache
def equivalent_lipid_definition(lipid_name: str) -> tuple[bool, str]:
    """Whether CHARMM36 and CHARMM36m simulate this lipid identically.

    Returns the verdict and, when it is negative, what differs. A lipid that is
    absent from one of the two is not equivalent -- there is nothing to reuse.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    # The templates the builder actually uses, not the raw residue tables: a
    # CHARMM lipid may be composed from tail fragments, or patched, and for a
    # handful the classic release already defers to the modern one. Comparing
    # the resolved templates asks the question that matters -- would the two
    # force fields simulate the same molecule with the same parameters.
    residue, left = lipid_rtp_template(lipid_name, "charmm36")
    _modern_name, right = lipid_rtp_template(lipid_name, "charmm36m")
    residue = str(residue).upper()
    if left is None or right is None:
        missing = "charmm36" if left is None else "charmm36m"
        return False, f"{residue} has no resolvable template under {missing}"

    def described(record: dict) -> tuple:
        atoms = [(str(a[0]), str(a[1]), round(float(a[2]), 6)) for a in record.get("atoms") or []]
        bonds = sorted(tuple(sorted((str(u), str(v)))) for u, v in (record.get("bonds") or []))
        impropers = sorted(tuple(str(x) for x in t) for t in (record.get("impropers") or []))
        return atoms, bonds, impropers

    mine, theirs = described(left), described(right)
    for what, index in (("atoms, types or charges", 0), ("bond list", 1), ("improper list", 2)):
        if mine[index] != theirs[index]:
            return False, f"{residue} has a different {what}"

    old, reason = _resolved(lipid_name, "charmm36")
    if old is None:
        return False, reason
    new, reason = _resolved(lipid_name, "charmm36m")
    if new is None:
        return False, reason

    labels = [str(atom[0]) for atom in right.get("atoms") or []]
    difference = _difference(old, new, labels)
    if difference is not None:
        return False, f"{residue}: {difference}"
    interactions, pairs = _counts(new)
    return True, (
        "grompp resolves the same system under both releases: "
        f"{interactions} interactions and {pairs} Lennard-Jones pairs, "
        "ions and water included"
    )

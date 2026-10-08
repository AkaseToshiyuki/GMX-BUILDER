"""Merge nucleic-acid parameters without changing supported base residues.

Import OL24 DNA and OL3 RNA definitions into the installed protein force field.
A parameter is eligible only if nucleic residues use all its named atom types.
Wildcard keys must also contain a nucleic-only type. Do not add or replace a
parameter reachable by any base RTP residue, including modified residues.

These rules allow nucleic-only torsion refinements while protecting protein,
water and ion parameters. Standard library only: the installer imports this
module before project dependencies are installed."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: Files taken verbatim from the upstream port. These describe nucleic acids
#: and nothing else, so there is nothing in them for a protein to reach.
NUCLEIC_FILES = (
    "dna.rtp",
    "rna.rtp",
    "dna.hdb",
    "rna.hdb",
    "dna.r2b",
    "rna.r2b",
    "dna.arn",
    "rna.arn",
)

#: How many leading columns identify one entry, per parameter section.
_KEY_WIDTH = {
    "atomtypes": 1,
    "bondtypes": 2,
    "constrainttypes": 2,
    "angletypes": 3,
    "dihedraltypes": 5,
}

#: Sections merged out of ``ffbonded.itp``; ``atomtypes`` lives in
#: ``ffnonbonded.itp`` and is handled separately.
_BONDED_SECTIONS = ("bondtypes", "constrainttypes", "angletypes", "dihedraltypes")

# Relative tolerance for converter precision differences in numeric parameters.
# Preserve the reviewed value; a rounding tolerance is not permission to refit terms.
_RELATIVE_TOLERANCE = 1e-5


@dataclass
class MergeReport:
    """What the merge did, in enough detail to review it line by line."""

    added: dict[str, list[str]] = field(default_factory=dict)
    overridden: dict[str, list[str]] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    residues_dropped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    files_adopted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def summary(self) -> str:
        lines = [f"adopted {len(self.files_adopted)} nucleic-acid files"]
        if self.residues_dropped:
            lines.append(
                "  dropped colliding monatomic ion residues: " + ", ".join(self.residues_dropped)
            )
        for section in sorted(set(self.added) | set(self.overridden) | set(self.rejected)):
            lines.append(
                f"  {section:16s} +{len(self.added.get(section, ())):3d} added, "
                f"{len(self.overridden.get(section, ())):3d} overridden, "
                f"{self.kept.get(section, 0):4d} kept, "
                f"{len(self.rejected.get(section, ())):3d} not ours to take"
            )
        for conflict in self.conflicts:
            lines.append(f"  CONFLICT {conflict}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Reading GROMACS parameter files


def _strip(line: str) -> str:
    return line.split(";")[0].rstrip()


def parse_sections(text: str) -> dict[str, dict[tuple[str, ...], list[list[str]]]]:
    """Return ``{section: {key: [row, ...]}}`` for the sections we merge.

    Rows are kept as lists of columns and duplicates are preserved: a proper
    dihedral is several rows under one key, one per multiplicity, and dropping
    any of them would change the potential.
    """
    out: dict[str, dict[tuple[str, ...], list[list[str]]]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = _strip(raw)
        header = re.match(r"^\[\s*(\S+)\s*\]", line)
        if header:
            section = header.group(1)
            continue
        if section is None or not line.strip():
            continue
        width = _KEY_WIDTH.get(section)
        if width is None:
            continue
        parts = line.split()
        if len(parts) <= width:
            continue
        out.setdefault(section, {}).setdefault(tuple(parts[:width]), []).append(parts)
    return out


def _values(row: list[str], width: int) -> list[tuple[str, object]]:
    typed: list[tuple[str, object]] = []
    for token in row[width:]:
        try:
            typed.append(("n", float(token)))
        except ValueError:
            typed.append(("s", token))
    return typed


def _is_zero_row(row: list[str], width: int) -> bool:
    """A dihedral term with zero amplitude contributes nothing.

    The base carries a few of these where upstream simply omits them. They are
    not a disagreement about physics, so treating them as one would block the
    merge over nothing.
    """
    numbers = [value for kind, value in _values(row, width) if kind == "n"]
    # phase, force constant, multiplicity -- the force constant is the middle.
    return len(numbers) >= 2 and abs(numbers[1]) == 0.0


def _rows_equivalent(left: list[list[str]], right: list[list[str]], width: int) -> bool:
    """Do two sets of rows describe the same potential?

    Order-insensitive (upstream lists multiplicities in the opposite order),
    tolerant of truncated precision, and blind to zero-amplitude terms.
    """
    lhs = [row for row in left if not _is_zero_row(row, width)]
    rhs = [row for row in right if not _is_zero_row(row, width)]
    if len(lhs) != len(rhs):
        return False

    def matches(a: list[str], b: list[str]) -> bool:
        va, vb = _values(a, width), _values(b, width)
        if len(va) != len(vb):
            return False
        for (kind_a, value_a), (kind_b, value_b) in zip(va, vb, strict=True):
            if kind_a != kind_b:
                return False
            if kind_a == "s":
                if value_a != value_b:
                    return False
            else:
                assert isinstance(value_a, float) and isinstance(value_b, float)
                scale = max(1.0, abs(value_a), abs(value_b))
                if abs(value_a - value_b) > _RELATIVE_TOLERANCE * scale:
                    return False
        return True

    remaining = list(rhs)
    for row in lhs:
        for index, candidate in enumerate(remaining):
            if matches(row, candidate):
                del remaining[index]
                break
        else:
            return False
    return True


# --------------------------------------------------------------------------
# The reachability rule


def residue_atom_types(text: str) -> dict[str, set[str]]:
    """Return ``{residue: {atom type}}`` from an ``.rtp`` file."""
    out: dict[str, set[str]] = {}
    current: str | None = None
    section: str | None = None
    subsections = {"atoms", "bonds", "impropers", "dihedrals", "bondedtypes", "cmap", "exclusions"}
    for raw in text.splitlines():
        line = _strip(raw)
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
        if header:
            name = header.group(1)
            if name in subsections:
                section = name
            else:
                current, section = name, None
                out[current] = set()
            continue
        if section == "atoms" and current and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                out[current].add(parts[1])
    return out


#: The wildcard GROMACS matches against any atom type.
_WILDCARD = "X"


def _key_types(key: tuple[str, ...]) -> set[str]:
    """Atom types in *key*, dropping the trailing function-type column."""
    return {part for part in key if not part.isdigit() and part != _WILDCARD}


def base_reachable(key: tuple[str, ...], base_residues: dict[str, set[str]]) -> bool:
    """Can any single residue we already support supply every type in *key*?

    Deliberately per-residue rather than "is each type used somewhere". ``CT``
    appears in proteins and nucleic acids alike, so a union over all residues
    would call every nucleic dihedral containing ``CT`` reachable and forbid
    the whole merge. What matters is whether the bonded term can actually
    form, and it forms within one residue's reach: ``OS-CT-N*-CK`` cannot
    arise in any protein because no protein residue has ``N*`` or ``CK``.

    Wildcards are ignored rather than treated as matching everything. A
    wildcard position does match any atom, but the *named* positions still
    have to be present, so a key naming a nucleic-only type cannot arise in a
    protein however many wildcards surround it. Treating a wildcard as
    universally reachable rejected sixteen dihedrals the nucleic acids need --
    ``X-C1-CT-X`` and its siblings, whose ``C1``/``C2``/``C7``/``CJ`` no
    protein residue has. What keeps an all-shared wildcard such as
    ``X-CT-CT-X`` out is :func:`nucleic_eligible`, which requires a
    nucleic-only type before a wildcard key is considered at all.
    """
    return any(_key_types(key) <= types for types in base_residues.values())


def nucleic_eligible(key: tuple[str, ...], nucleic_types: set[str], nucleic_only: set[str]) -> bool:
    """Is this a parameter the nucleic acids actually need?

    Every named type must be one the nucleic residues use. A wildcard key must
    additionally contain a nucleic-only type, or it would match molecules this
    merge has no business changing.
    """
    named = _key_types(key)
    if not named or not named <= nucleic_types:
        return False
    return _WILDCARD not in key or bool(named & nucleic_only)


# --------------------------------------------------------------------------
# Rendering


def _render(section: str, rows: list[list[str]]) -> str:
    return "\n".join("  ".join(row) for row in rows)


def _append_section(text: str, section: str, rows: list[list[str]], note: str) -> str:
    """Append entries to *section*, creating it when the file has none."""
    if not rows:
        return text
    block = f"\n; {note}\n" + _render(section, rows) + "\n"
    # A repeated section header preserves the base file's original ordering.
    return text.rstrip("\n") + f"\n\n[ {section} ]{block}"


def _replace_rows(
    text: str, section: str, replacements: dict[tuple[str, ...], list[list[str]]]
) -> str:
    """Drop the base's rows for each replaced key; the new ones are appended."""
    if not replacements:
        return text
    width_by_key = {key: _KEY_WIDTH[section] for key in replacements}
    out: list[str] = []
    current: str | None = None
    for raw in text.splitlines():
        line = _strip(raw)
        header = re.match(r"^\[\s*(\S+)\s*\]", line)
        if header:
            current = header.group(1)
            out.append(raw)
            continue
        if current == section and line.strip():
            parts = line.split()
            for key, width in width_by_key.items():
                if len(parts) > width and tuple(parts[:width]) == key:
                    out.append(f"; superseded by the merged nucleic-acid parameters: {raw.strip()}")
                    break
            else:
                out.append(raw)
            continue
        out.append(raw)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# The merge


def merge_nucleic_parameters(base: Path, upstream: Path, target: Path) -> MergeReport:
    """Build *target* from the *base* force field plus *upstream*'s nucleic acids.

    *base* is copied first, so anything not named below is byte-identical to
    the force field already shipped.
    """
    base, upstream, target = Path(base), Path(upstream), Path(target)
    report = MergeReport()

    missing = [name for name in NUCLEIC_FILES if not (upstream / name).is_file()]
    if missing:
        report.conflicts.append(f"upstream is missing {', '.join(missing)}")
        return report

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base, target)

    # ---- 1. The residue databases, verbatim apart from one line -----------
    base_bondedtypes = _bondedtypes_block((base / "aminoacids.rtp").read_text(encoding="utf-8"))
    for name in NUCLEIC_FILES:
        text = (upstream / name).read_text(encoding="utf-8")
        if name.endswith(".rtp") and base_bondedtypes is not None:
            # GROMACS 2026 added a ninth bondedtypes column; upstream's files
            # predate it. Every force field must state one convention, and
            # this force field's convention is the base's.
            text = _replace_bondedtypes(text, base_bondedtypes)
        (target / name).write_text(text, encoding="utf-8")
        report.files_adopted.append(name)

    # ---- 1b. Residue names cannot mean two things ------------------------
    _resolve_residue_collisions(target, report)
    if not report.ok:
        return report

    # Every residue database the base ships, not only the amino acids. The
    # phosphorylated residues introduce `P`, which nucleic acids also use, and
    # reading one file would leave them unprotected.
    base_residues: dict[str, set[str]] = {}
    for database in sorted(base.glob("*.rtp")):
        base_residues.update(residue_atom_types(database.read_text(encoding="utf-8")))

    nucleic_residues: dict[str, set[str]] = {}
    for name in ("dna.rtp", "rna.rtp"):
        nucleic_residues.update(residue_atom_types((target / name).read_text(encoding="utf-8")))
    nucleic_types = set().union(*nucleic_residues.values()) if nucleic_residues else set()
    base_types = set().union(*base_residues.values()) if base_residues else set()
    nucleic_only = nucleic_types - base_types
    if not nucleic_types:
        report.conflicts.append("the adopted residue databases declare no atom types")
        return report

    # ---- 2. Atom types, then bonded parameters ---------------------------
    _merge_file(
        target / "ffnonbonded.itp",
        (base / "ffnonbonded.itp").read_text(encoding="utf-8"),
        (upstream / "ffnonbonded.itp").read_text(encoding="utf-8"),
        ("atomtypes",),
        base_residues,
        nucleic_types,
        nucleic_only,
        report,
    )
    _merge_atomtype_database(base, upstream, target, report)
    _merge_file(
        target / "ffbonded.itp",
        (base / "ffbonded.itp").read_text(encoding="utf-8"),
        (upstream / "ffbonded.itp").read_text(encoding="utf-8"),
        _BONDED_SECTIONS,
        base_residues,
        nucleic_types,
        nucleic_only,
        report,
    )
    return report


def _resolve_residue_collisions(target: Path, report: MergeReport) -> None:
    """A residue name must mean one thing, and here one of them means two.

    GROMACS's ff14SB bundles a monatomic residue for almost every ion in the
    periodic table, and one of them is ``RA`` -- radium. In ``rna.rtp`` ``RA``
    is adenosine. ``pdb2gmx`` refuses to load a force field that defines a
    residue twice, so one of them has to go.

    The nucleic residue wins, but only against a **monatomic ion**: this is a
    force field for nucleic-acid systems, ``RA`` in a nucleic-acid PDB means
    adenosine, and nobody is simulating radium. A collision with anything
    larger is not something to resolve by preference, so it is reported and
    the merge stops.
    """
    nucleic_names: set[str] = set()
    for name in ("dna.rtp", "rna.rtp"):
        nucleic_names.update(residue_atom_types((target / name).read_text(encoding="utf-8")))

    for database in sorted(target.glob("*.rtp")):
        if database.name in {"dna.rtp", "rna.rtp"}:
            continue
        residues = residue_atom_types(database.read_text(encoding="utf-8"))
        collisions = sorted(set(residues) & nucleic_names)
        if not collisions:
            continue
        polyatomic = [name for name in collisions if len(residues[name]) > 1]
        if polyatomic:
            report.conflicts.append(
                f"{database.name} defines {', '.join(polyatomic)}, which the nucleic-acid "
                "database also defines with more than one atom; a residue name that means "
                "two different molecules is not something this merge will pick between"
            )
            continue
        _drop_residues(database, collisions)
        report.residues_dropped.extend(f"{database.name}:{name}" for name in collisions)


def _drop_residues(database: Path, names: list[str]) -> None:
    """Comment out whole residue blocks, so the file still says what was there."""
    wanted = set(names)
    out: list[str] = []
    dropping = False
    for raw in database.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", _strip(raw))
        if header:
            name = header.group(1)
            subsection = name in {
                "atoms",
                "bonds",
                "impropers",
                "dihedrals",
                "bondedtypes",
                "cmap",
                "exclusions",
            }
            if not subsection:
                dropping = name in wanted
                if dropping:
                    out.append(
                        f"; {raw.strip()}  <- removed: the nucleic-acid database "
                        "defines this residue name"
                    )
                    continue
            elif dropping:
                out.append(f"; {raw.rstrip()}")
                continue
        elif dropping:
            out.append(f"; {raw.rstrip()}" if raw.strip() else raw)
            continue
        out.append(raw)
    database.write_text("\n".join(out) + "\n", encoding="utf-8")


def _merge_atomtype_database(base: Path, upstream: Path, target: Path, report: MergeReport) -> None:
    """Mirror the adopted atom types into ``atomtypes.atp``.

    ``pdb2gmx`` resolves an ``.rtp`` atom type against ``atomtypes.atp``, not
    against ``ffnonbonded.itp``. Adding a type to one and not the other builds
    a force field that ``grompp`` would accept and ``pdb2gmx`` refuses with
    "Atom type C2 (residue DA5) not found in atomtype database" -- which is
    exactly what the first version of this merge produced.
    """
    adopted = report.added.get("atomtypes", [])
    if not adopted:
        return

    def entries(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split(";")[0].strip()
            if line:
                out[line.split()[0]] = raw.rstrip()
        return out

    existing = entries(base / "atomtypes.atp")
    available = entries(upstream / "atomtypes.atp")
    missing = [name for name in adopted if name not in available]
    if missing:
        report.conflicts.append(
            "upstream ffnonbonded.itp declares "
            + ", ".join(missing)
            + " but its atomtypes.atp does not; pdb2gmx would reject the result"
        )
        return

    lines = [available[name] for name in adopted if name not in existing]
    if not lines:
        return
    destination = target / "atomtypes.atp"
    text = destination.read_text(encoding="utf-8").rstrip("\n")
    destination.write_text(
        text + "\n\n; merged nucleic-acid atom types\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _merge_file(
    destination: Path,
    base_text: str,
    upstream_text: str,
    sections: tuple[str, ...],
    base_residues: dict[str, set[str]],
    nucleic_types: set[str],
    nucleic_only: set[str],
    report: MergeReport,
) -> None:
    base_sections = parse_sections(base_text)
    upstream_sections = parse_sections(upstream_text)
    text = base_text

    for section in sections:
        width = _KEY_WIDTH[section]
        ours = base_sections.get(section, {})
        theirs = upstream_sections.get(section, {})
        additions: list[list[str]] = []
        overrides: dict[tuple[str, ...], list[list[str]]] = {}
        kept = 0

        for key in sorted(theirs):
            if not nucleic_eligible(key, nucleic_types, nucleic_only):
                # Upstream ships a whole force field; most of it is not ours
                # to take. Recorded rather than dropped silently, because what
                # lands here is the interesting part of the review.
                if key not in ours or not _rows_equivalent(ours[key], theirs[key], width):
                    report.rejected.setdefault(section, []).append("-".join(key))
                continue
            if key not in ours:
                # Adding is not automatically safe. GROMACS generates bonded
                # terms from connectivity, so a *new* type whose key a protein
                # residue can supply would start applying to proteins that
                # previously had no such term. Measured against the real
                # archive this refuses three constraint types (C-HO, CA-HO,
                # CT-HO) which are inert today only because no .rtp declares a
                # constraints section -- which is luck, not a guarantee.
                if base_reachable(key, base_residues):
                    report.rejected.setdefault(section, []).append("-".join(key))
                    continue
                additions.extend(theirs[key])
                report.added.setdefault(section, []).append("-".join(key))
                continue
            if _rows_equivalent(ours[key], theirs[key], width):
                kept += 1
                continue
            if base_reachable(key, base_residues):
                report.conflicts.append(
                    f"{section} {'-'.join(key)}: upstream disagrees about a parameter a "
                    "residue we already support can reach; this merge will not decide "
                    "that silently"
                )
                continue
            overrides[key] = theirs[key]
            report.overridden.setdefault(section, []).append("-".join(key))

        report.kept[section] = kept
        if overrides:
            text = _replace_rows(text, section, overrides)
        appended = additions + [row for rows in overrides.values() for row in rows]
        if appended:
            text = _append_section(
                text,
                section,
                appended,
                "merged nucleic-acid parameters (see forcefield.doc for provenance)",
            )

    if report.ok:
        destination.write_text(text, encoding="utf-8")


def _bondedtypes_block(text: str) -> list[str] | None:
    """Return the base's ``[ bondedtypes ]`` data rows."""
    rows: list[str] = []
    inside = False
    for raw in text.splitlines():
        line = _strip(raw)
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
        if header:
            if inside:
                break
            inside = header.group(1) == "bondedtypes"
            continue
        if inside and line.strip():
            rows.append(line.strip())
    return rows or None


def _replace_bondedtypes(text: str, rows: list[str]) -> str:
    out: list[str] = []
    inside = False
    written = False
    for raw in text.splitlines():
        line = _strip(raw)
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
        if header:
            if inside:
                inside = False
            elif header.group(1) == "bondedtypes":
                inside = True
                out.append(raw)
                continue
        if inside:
            if line.strip():
                if not written:
                    out.extend(rows)
                    written = True
                continue
            out.append(raw)
            continue
        out.append(raw)
    return "\n".join(out) + "\n"

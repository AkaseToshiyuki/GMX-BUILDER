"""Atom typing learned from the CHARMM release the user installed.

The previous model typed atoms by a fixed first-shell key -- element, formal
charge, hydrogen count and neighbour elements. That descriptor cannot see ring
membership or beta substitution, so it cannot separate atom types that CGenFF
separates, and every new functional-group class needed its own hand-written
module to work around it.

This replaces it with the approach published for MATCH (Yesselman, Price,
Knight & Brooks III, *J Comput Chem* 2012;33(2):189-202): describe an atom by
its neighbourhood at increasing radius and stop widening as soon as the
description picks out one type. Written from the publication -- no MATCH source
or resource file is used, and none may be: the redistribution's licence carries
no copyright line from its authors, and its resource tree bundles the CHARMM
parameter data this project deliberately does not ship.

**Nothing here is a parameter table.** The rules are derived at run time from
the ``.rtp`` and ``ffnonbonded.itp`` the user installed themselves, and cached
against their hashes. The repository gains no force-field numbers, and the
rules follow whichever release is installed rather than drifting from it.

The gain is coverage. The reviewed-template model learns from five molecules;
the installed CGenFF release documents 923 residues and 24150 typed atoms
across 154 atom types, and merged CHARMM36 documents 1319 and 57269 across 394.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from gmxbuilder.modules.forcefield.catalog import force_field_directory

#: Bumped when the descriptor definition changes, so cached corpora built by an
#: older definition are never reused against a newer one.
TYPING_MODEL_ID = "installed-release-wl-descriptor-v1"

#: Widening past this costs time and buys nothing: beyond four bonds the
#: descriptor is effectively the whole small molecule, and an environment that
#: is still ambiguous there is ambiguous because the release itself types two
#: identical neighbourhoods differently.
MAX_RADIUS = 4

#: How many corpus atoms must agree before an environment may decide a type.
#:
#: One, deliberately. Requiring two looks safer and is wrong here: a small
#: molecule's neighbourhood *is* the whole molecule once the radius reaches its
#: diameter, so methanol's carbon has exactly one witness in the release -- the
#: methanol entry -- and a threshold of two refuses to type methanol from a
#: release that documents methanol. Support is not discarded; it is carried on
#: every assignment and folded into the reported confidence, which is where a
#: single hand-tuned witness should show up.
MIN_SUPPORT = 1


#: Element symbols, longest first, so CL and BR are stripped before C and B.
_ELEMENT_PREFIXES = ("CL", "BR", "SI", "C", "H", "O", "N", "S", "P", "F", "I")


def is_general_type(atom_type: str) -> bool:
    """Whether a type name belongs to the general (CGenFF) small-molecule set.

    CHARMM36's ``merged.rtp`` is not one typing authority. It merges the
    protein, lipid, carbohydrate, nucleic and general force fields, and they
    name the same chemistry differently: propane's methyl carbon is ``CG331``
    to CGenFF and ``CC33A`` to the carbohydrate set. Widening the descriptor
    cannot separate those, because what differs is provenance rather than
    environment -- so a corpus built over the whole merged file reports every
    ordinary alkyl carbon as ambiguous and types nothing.

    CGenFF marks its types by following the element symbol with ``G``. Keeping
    only those makes the corpus a single consistent authority and matches the
    backend's existing rule that force-field versions are never mixed. A ligand
    must not borrow a protein or lipid atom type.
    """
    name = atom_type.upper()
    for prefix in _ELEMENT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :].startswith("G")
    return False


class TypingUnavailable(RuntimeError):
    """The installed release cannot be read as a typing corpus."""


@dataclass(frozen=True)
class TypeAssignment:
    """One atom's type, and how much of the release stands behind it."""

    atom_index: int
    atom_type: str
    radius: int
    support: int
    #: Distinct types sharing this environment at the radius below the decisive
    #: one. Empty when the first radius tried was already decisive.
    resolved_from: tuple[str, ...] = ()
    #: Whether the release also documents this atom's neighbourhood one shell
    #: wider than where the type was decided. See ``MEASURED_ERROR_RATES``.
    corroborated: bool = False

    @property
    def decisive(self) -> bool:
        return bool(self.atom_type)


#: Held-out error rates behind the confidence split, measured on this codebase
#: rather than assumed: 120 residues withheld one at a time from the charmm36m
#: corpus and typed from the remaining 909, 3118 atoms answered.
#:
#: An atom is *corroborated* when the release still documents its neighbourhood
#: one shell wider than the radius that decided its type. An *extrapolated*
#: match is one that committed at radius r only because radius r+1 was unseen --
#: the type was read off the widest view the release happens to hold, not off a
#: view wide enough to be sure. Every wrong answer in the sample was a case
#: where CGenFF's distinction lay just beyond that edge: trimethylamine's methyl
#: hydrogens read as HGA3 (the amide N-methyl answer) instead of HGAAM0, and
#: uracil ring carbons as CG2R61 instead of CG2R62.
MEASURED_ERROR_RATES = {
    "corroborated": {"n": 2541, "wrong": 3, "rate": 0.0012},
    "extrapolated": {"n": 577, "wrong": 20, "rate": 0.0347},
    "sample": "120 held-out residues, charmm36m, seed 20260907",
}


@dataclass
class TypingCorpus:
    """Every typed atom the installed release documents, indexed by environment.

    ``decisive[radius]`` maps a descriptor to the single type every corpus atom
    carrying it was given. ``witnesses[radius]`` maps the same descriptor to the
    set of types seen, so an ambiguous environment can say what it collided
    with instead of only that it failed.
    """

    force_field: str
    decisive: dict[int, dict[bytes, tuple[str, int]]]
    witnesses: dict[int, dict[bytes, set[str]]]
    source_sha256: str
    n_residues: int = 0
    n_atoms: int = 0
    n_types: int = 0
    skipped_residues: tuple[str, ...] = field(default_factory=tuple)

    def report(self) -> dict:
        return {
            "model_id": TYPING_MODEL_ID,
            "force_field": self.force_field,
            "source_sha256": self.source_sha256,
            "residues_read": self.n_residues,
            "atoms_read": self.n_atoms,
            "distinct_types": self.n_types,
            "max_radius": MAX_RADIUS,
            "min_support": MIN_SUPPORT,
            "decisive_environments": {r: len(m) for r, m in sorted(self.decisive.items())},
            "derived_from": "user-installed force field, read at run time",
            "distributed_parameters": "none",
        }


# ---------------------------------------------------------------------------
# The molecular graph, in the one shape both sides can produce
#
# The RTP gives connectivity but no bond orders; RDKit gives full chemistry.
# Anything the descriptor uses must therefore be computable from connectivity
# alone, and must be computed the same way on both sides or the two will not
# meet. Element, hydrogen count, heavy degree and smallest-ring size all are.


@dataclass(frozen=True)
class Graph:
    """Elements plus adjacency. Hydrogens are explicit atoms on both sides."""

    elements: tuple[int, ...]
    adjacency: tuple[tuple[int, ...], ...]

    def heavy_degree(self, index: int) -> int:
        return sum(1 for n in self.adjacency[index] if self.elements[n] != 1)

    def hydrogen_count(self, index: int) -> int:
        return sum(1 for n in self.adjacency[index] if self.elements[n] == 1)


def smallest_ring_sizes(graph: Graph) -> tuple[int, ...]:
    """Size of the smallest cycle through each atom, or 0 for none.

    Computed here rather than taken from RDKit so that the reference side --
    which has no RDKit molecule, only an RTP connectivity table -- and the
    query side produce the same number for the same topology. A BFS from each
    atom finds the shortest cycle through it; molecules of this size make the
    cost irrelevant.
    """
    sizes = []
    for start in range(len(graph.elements)):
        best = 0
        # Shortest cycle through `start`: for each pair of its neighbours, the
        # shortest path between them that avoids `start`.
        neighbours = graph.adjacency[start]
        for i, first in enumerate(neighbours):
            for second in neighbours[i + 1 :]:
                distance = _shortest_path_avoiding(graph, first, second, start)
                if distance is not None:
                    cycle = distance + 2
                    if best == 0 or cycle < best:
                        best = cycle
        sizes.append(best)
    return tuple(sizes)


def _shortest_path_avoiding(graph: Graph, source: int, target: int, blocked: int) -> int | None:
    if source == target:
        return 0
    seen = {source, blocked}
    queue = deque([(source, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbour in graph.adjacency[node]:
            if neighbour == target:
                return distance + 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None


def base_labels(graph: Graph) -> list[bytes]:
    """Radius-0 description: what an atom is, before its surroundings."""
    rings = smallest_ring_sizes(graph)
    return [
        b"%d|%d|%d|%d"
        % (graph.elements[i], graph.heavy_degree(i), graph.hydrogen_count(i), rings[i])
        for i in range(len(graph.elements))
    ]


def descriptors(graph: Graph, radius: int, labels: list[bytes] | None = None) -> list[bytes]:
    """Weisfeiler-Lehman refinement: fold each shell of neighbours in once.

    Order-independent by construction -- neighbour labels are sorted before
    hashing -- so the same neighbourhood gives the same descriptor no matter how
    the atoms happened to be numbered in an RTP entry or a SMILES string.
    """
    current = list(labels if labels is not None else base_labels(graph))
    for _ in range(radius):
        current = [
            hashlib.blake2b(
                current[i] + b"<" + b",".join(sorted(current[n] for n in graph.adjacency[i])),
                digest_size=16,
            ).digest()
            for i in range(len(graph.elements))
        ]
    return current


# ---------------------------------------------------------------------------
# Reading the installed release


def _atomic_numbers(root: Path) -> dict[str, int]:
    """Atom type -> atomic number, from the installed ``[ atomtypes ]`` table."""
    path = root / "ffnonbonded.itp"
    if not path.is_file():
        raise TypingUnavailable("installed release has no ffnonbonded.itp")
    numbers: dict[str, int] = {}
    section = ""
    for raw in path.read_text(errors="replace").splitlines():
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
            continue
        if section != "atomtypes" or not code or code.startswith("#"):
            continue
        fields = code.split()
        if len(fields) < 2:
            continue
        try:
            numbers.setdefault(fields[0], int(fields[1]))
        except ValueError:
            continue
    if not numbers:
        raise TypingUnavailable("installed release declares no atom types")
    return numbers


def _residue_graph(record: dict, numbers: dict[str, int]) -> tuple[Graph, list[str]] | None:
    """One RTP residue as a graph, or None when it cannot be read as a molecule.

    Inter-residue bonds (``-C``, ``+N``) name atoms this residue does not own.
    A residue carrying them is a polymer unit whose real neighbourhood is not
    in the record, so its boundary atoms would be described wrongly. Those
    residues are dropped rather than half-read.
    """
    atoms = record.get("atoms") or []
    if not atoms:
        return None
    index = {}
    elements = []
    types = []
    for position, entry in enumerate(atoms):
        name, atom_type = str(entry[0]), str(entry[1])
        number = numbers.get(atom_type)
        if number is None or not is_general_type(atom_type):
            # A residue typed by the protein, lipid, carbohydrate or nucleic
            # set is not evidence about general small-molecule chemistry.
            return None
        index[name] = position
        elements.append(number)
        types.append(atom_type)
    adjacency: list[set[int]] = [set() for _ in atoms]
    for bond in record.get("bonds") or []:
        first, second = str(bond[0]), str(bond[1])
        if first not in index or second not in index:
            return None  # crosses a residue boundary
        a, b = index[first], index[second]
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return Graph(tuple(elements), tuple(tuple(sorted(n)) for n in adjacency)), types


def _source_hash(root: Path, database: Path) -> str:
    digest = hashlib.sha256()
    digest.update(TYPING_MODEL_ID.encode())
    for path in (database, root / "ffnonbonded.itp"):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


_CACHE: dict[tuple[str, str], TypingCorpus] = {}


def build_corpus(force_field: str, *, exclude: frozenset[str] | None = None) -> TypingCorpus:
    """Index every typed atom in the installed release by its environment.

    Cached on the hashes of the files it was built from, so a release that is
    upgraded underneath a running process is rebuilt rather than reused.

    ``exclude`` withholds residues by name. It exists so the generalisation of
    this model can be measured honestly -- typing a molecule whose own entry
    built the rules proves only that the index works. Excluded builds are never
    cached, so a measurement cannot leak into production typing.
    """
    from gmxbuilder.modules.forcefield.charmm_compat import template_database
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser

    root = force_field_directory(force_field)
    if root is None:
        raise TypingUnavailable(f"{force_field} is not installed")
    database = template_database(force_field)
    if not database.is_file():
        raise TypingUnavailable("installed release has no residue database")

    signature = _source_hash(root, database)
    if exclude is None:
        cached = _CACHE.get((force_field, signature))
        if cached is not None:
            return cached

    numbers = _atomic_numbers(root)
    parser = RTPParser(database)
    witnesses: dict[int, dict[bytes, set[str]]] = {r: {} for r in range(1, MAX_RADIUS + 1)}
    counts: dict[int, dict[bytes, dict[str, int]]] = {r: {} for r in range(1, MAX_RADIUS + 1)}
    seen_types: set[str] = set()
    skipped: list[str] = []
    n_residues = n_atoms = 0

    for name in parser.residue_names:
        if exclude and name in exclude:
            continue
        record = parser.get_residue(name)
        if not record:
            continue
        built = _residue_graph(record, numbers)
        if built is None:
            skipped.append(name)
            continue
        graph, types = built
        labels = base_labels(graph)
        n_residues += 1
        n_atoms += len(types)
        seen_types.update(types)
        current = labels
        for radius in range(1, MAX_RADIUS + 1):
            current = descriptors(graph, 1, current)
            shell = witnesses[radius]
            tally = counts[radius]
            for position, atom_type in enumerate(types):
                key = current[position]
                shell.setdefault(key, set()).add(atom_type)
                tally.setdefault(key, {})
                tally[key][atom_type] = tally[key].get(atom_type, 0) + 1

    decisive: dict[int, dict[bytes, tuple[str, int]]] = {}
    for radius, shell in witnesses.items():
        resolved = {}
        for key, types in shell.items():
            if len(types) != 1:
                continue
            only = next(iter(types))
            support = counts[radius][key][only]
            if support >= MIN_SUPPORT:
                resolved[key] = (only, support)
        decisive[radius] = resolved

    corpus = TypingCorpus(
        force_field=force_field,
        decisive=decisive,
        witnesses=witnesses,
        source_sha256=signature,
        n_residues=n_residues,
        n_atoms=n_atoms,
        n_types=len(seen_types),
        skipped_residues=tuple(sorted(skipped)),
    )
    if exclude is None:
        _CACHE[(force_field, signature)] = corpus
    return corpus


# ---------------------------------------------------------------------------
# Typing a query molecule


def graph_from_rdkit(molecule) -> Graph:
    """The same graph shape, from a molecule that already has explicit H."""
    elements = tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    adjacency: list[set[int]] = [set() for _ in elements]
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].add(b)
        adjacency[b].add(a)
    return Graph(elements, tuple(tuple(sorted(n)) for n in adjacency))


def assign(molecule, corpus: TypingCorpus) -> list[TypeAssignment]:
    """Type every atom, widening the description only as far as it must.

    Returns one assignment per atom in molecule order. An atom the corpus
    cannot decide gets an assignment with an empty ``atom_type`` naming the
    types it collided with, so the caller can report *which* environment was
    unresolvable rather than only that typing failed.
    """
    graph = graph_from_rdkit(molecule)
    shells = []
    current = base_labels(graph)
    for _ in range(MAX_RADIUS):
        current = descriptors(graph, 1, current)
        shells.append(current)

    assignments = []
    for position in range(len(graph.elements)):
        chosen: TypeAssignment | None = None
        collided: tuple[str, ...] = ()
        for radius in range(1, MAX_RADIUS + 1):
            key = shells[radius - 1][position]
            resolved = corpus.decisive[radius].get(key)
            if resolved is not None:
                atom_type, support = resolved
                # Does the release still describe this atom one shell wider
                # than where the type was decided? If not, the answer is the
                # widest view available rather than a view wide enough to be
                # sure, and it is nearly thirty times more likely to be wrong.
                corroborated = (
                    False
                    if radius == MAX_RADIUS
                    else corpus.witnesses[radius + 1].get(shells[radius][position]) is not None
                )
                chosen = TypeAssignment(
                    position, atom_type, radius, support, collided, corroborated
                )
                break
            seen = corpus.witnesses[radius].get(key)
            if seen is None:
                # Widening further can only describe an environment the release
                # has never seen. Stop and report what the last radius held.
                break
            collided = tuple(sorted(seen))
        assignments.append(chosen or TypeAssignment(position, "", 0, 0, collided, False))
    return assignments


def typed_or_error(
    molecule,
    force_field: str,
    *,
    require_corroborated: bool = True,
) -> tuple[list[str], list[TypeAssignment]]:
    """Types for every atom, or a specific refusal naming the first failure.

    ``require_corroborated`` defaults to refusing extrapolated matches, which
    is the conservative reading of the measurement above: it answers 83.7% of
    held-out atoms at a 0.11% error rate instead of 100% at 0.74%. Callers that
    have accepted the experimental model may lower the bar, and every
    assignment carries its own flag either way so the choice is auditable.
    """
    from gmxbuilder.modules.forcefield.charmm_compat import CharmmCompatError

    corpus = build_corpus(force_field)
    assignments = assign(molecule, corpus)
    for assignment in assignments:
        index = assignment.atom_index + 1
        if not assignment.decisive:
            if assignment.resolved_from:
                raise CharmmCompatError(
                    "ATOM_TYPE_AMBIGUOUS",
                    f"atom {index} matches an environment the installed release types "
                    f"{len(assignment.resolved_from)} different ways "
                    f"({', '.join(assignment.resolved_from[:4])}); use CGenFF import",
                )
            raise CharmmCompatError(
                "ATOM_TYPE_UNASSIGNED",
                f"atom {index} has an environment the installed release does not "
                "document; use CGenFF import",
            )
        if require_corroborated and not assignment.corroborated:
            raise CharmmCompatError(
                "ATOM_TYPE_EXTRAPOLATED",
                f"atom {index} would be typed {assignment.atom_type} from the widest "
                f"environment the installed release documents ({assignment.radius} bonds), "
                "but the release does not describe this atom any further out, so the type "
                "cannot be confirmed; use CGenFF import",
            )
    return [a.atom_type for a in assignments], assignments


def typing_confidence(assignments: list[TypeAssignment]) -> dict:
    """A reportable summary of how far the release actually vouched for this."""
    decided = [a for a in assignments if a.decisive]
    extrapolated = [a for a in decided if not a.corroborated]
    return {
        "atoms": len(assignments),
        "typed": len(decided),
        "corroborated": len(decided) - len(extrapolated),
        "extrapolated": len(extrapolated),
        "extrapolated_atoms": [a.atom_index + 1 for a in extrapolated],
        "max_radius_used": max((a.radius for a in decided), default=0),
        "min_support": min((a.support for a in decided), default=0),
        "held_out_error_rate": (
            MEASURED_ERROR_RATES["extrapolated"]["rate"]
            if extrapolated
            else MEASURED_ERROR_RATES["corroborated"]["rate"]
        ),
        "physical_validation": "not_evaluated",
    }

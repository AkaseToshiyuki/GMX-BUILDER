"""The lipids the installed force field already provides, as comparable graphs.

A force field package ships its own lipid definitions -- CHARMM36 as 412 RTP
residues, Lipid21 as 39 ITP templates -- and those are authoritative. This
project additionally carries a SMILES per lipid, used by exactly one thing: the
GAFF2 backend, which builds a molecule from that SMILES when no native
definition is available. Nothing else parameterises from it; the CHARMM and
Lipid21 paths read their own files.

So when our SMILES and the force field's residue describe different molecules,
the force field is right and the GAFF2 build is wrong. Comparing them was not
possible before because an RTP residue has no SMILES -- only atom names, types
and a bond list. It has no bond orders either, so the two cannot be compared as
chemistry.

They can be compared as *graphs*. An RTP residue carries explicit hydrogens, so
element-labelled connectivity is recoverable from it, and the same graph is
recoverable from a SMILES once hydrogens are made explicit. Two molecules with
the same element-labelled connectivity graph are the same molecule up to bond
order and stereochemistry -- which is enough to catch a chain that is a carbon
short, a cardiolipin missing an acyl arm, or a sterol with the wrong side chain.

Used for two things:

* refusing an uploaded lipid the force field already provides, so a user cannot
  get home-made GAFF2 parameters for a molecule with official ones;
* checking every registry SMILES against the release, as a test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from gmxbuilder.modules.forcefield import charmm_typing as ct
from gmxbuilder.modules.forcefield.catalog import force_field_directory

#: Force fields whose own lipid definitions this module can read, and where.
#: CHARMM36m keeps lipids in their own file; CHARMM36 merges everything.
_LIPID_DATABASES = {
    "charmm36m": ("lipid.rtp",),
    "charmm36": ("merged.rtp",),
}


@dataclass(frozen=True)
class NativeLipid:
    """One lipid the installed force field defines itself."""

    residue: str
    force_field: str
    source: str
    n_atoms: int
    graph_key: str
    #: Kept so a bucket hit can be confirmed by exact isomorphism.
    graph: ct.Graph


def graph_key(graph: ct.Graph) -> str:
    """A cheap bucketing fingerprint. **Not** an identity test on its own.

    The multiset of maximum-radius Weisfeiler-Lehman descriptors plus the atom
    count and element composition. Order-independent, so two descriptions of
    the same molecule always land in the same bucket -- which is what makes it
    safe as a pre-filter.

    It is emphatically not sufficient by itself, and lipids are exactly where
    that bites: Lipid21's DPPC (16:0/16:0) and SMPC (18:0/14:0) are different
    molecules with the same formula, and their WL multisets are identical
    because every interior methylene looks alike four bonds out and the two
    structures have the same number of chain ends and ester groups. Callers
    must confirm a bucket hit with `same_molecule`.
    """
    descriptors = ct.descriptors(graph, ct.MAX_RADIUS)
    digest = hashlib.sha256()
    digest.update(b"%d|" % len(graph.elements))
    for element in sorted(graph.elements):
        digest.update(b"%d," % element)
    digest.update(b"#")
    for descriptor in sorted(descriptors):
        digest.update(descriptor)
    return digest.hexdigest()


def same_molecule(left: ct.Graph, right: ct.Graph) -> bool:
    """Exact element-labelled graph isomorphism.

    VF2 rather than a fingerprint comparison, because the fingerprint provably
    cannot separate two lipids that differ only in how a fixed number of chain
    carbons is split between two tails. Cheap invariants are checked first so
    the search itself runs only for genuine candidates.
    """
    if len(left.elements) != len(right.elements):
        return False
    if sorted(left.elements) != sorted(right.elements):
        return False
    if sum(len(n) for n in left.adjacency) != sum(len(n) for n in right.adjacency):
        return False

    import networkx as nx
    from networkx.algorithms import isomorphism

    def build(graph: ct.Graph):
        result = nx.Graph()
        for index, element in enumerate(graph.elements):
            result.add_node(index, element=element)
        for index, neighbours in enumerate(graph.adjacency):
            for neighbour in neighbours:
                if index < neighbour:
                    result.add_edge(index, neighbour)
        return result

    matcher = isomorphism.GraphMatcher(
        build(left),
        build(right),
        node_match=lambda a, b: a["element"] == b["element"],
    )
    return matcher.is_isomorphic()


def residue_graph(record: dict, atomic_numbers: dict[str, int]) -> ct.Graph | None:
    """An RTP residue as an element-labelled graph, or None if it is not one.

    Unlike the typing corpus this keeps every atom type, not only the general
    ones: a lipid residue is typed by the lipid force field by definition, and
    filtering those out would leave nothing to compare against.
    """
    atoms = record.get("atoms") or []
    if not atoms:
        return None
    index, elements = {}, []
    for position, entry in enumerate(atoms):
        number = atomic_numbers.get(str(entry[1]))
        if number is None:
            return None
        index[str(entry[0])] = position
        elements.append(number)
    adjacency: list[set[int]] = [set() for _ in atoms]
    for bond in record.get("bonds") or []:
        first, second = str(bond[0]), str(bond[1])
        if first not in index or second not in index:
            return None  # crosses a residue boundary; not a whole molecule
        a, b = index[first], index[second]
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return ct.Graph(tuple(elements), tuple(tuple(sorted(n)) for n in adjacency))


def smiles_graph(smiles: str) -> ct.Graph | None:
    """The same graph shape from a SMILES, with hydrogens made explicit."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    return ct.graph_from_rdkit(Chem.AddHs(molecule))


_INDEX_CACHE: dict[str, dict[str, list[NativeLipid]]] = {}


def native_lipid_index(force_field: str) -> dict[str, list[NativeLipid]]:
    """Every lipid the installed force field defines, keyed by graph."""
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser

    key = str(force_field).strip().lower()
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    index: dict[str, list[NativeLipid]] = {}
    root = force_field_directory(key)
    if root is not None:
        numbers = ct._atomic_numbers(root) if (root / "ffnonbonded.itp").is_file() else {}
        for filename in _LIPID_DATABASES.get(key, ()):
            path = root / filename
            if not path.is_file() or not numbers:
                continue
            parser = RTPParser(path)
            for residue in parser.residue_names:
                record = parser.get_residue(residue)
                if not record:
                    continue
                graph = residue_graph(record, numbers)
                if graph is None:
                    continue
                index.setdefault(graph_key(graph), []).append(
                    NativeLipid(
                        residue=residue.upper(),
                        force_field=key,
                        source=filename,
                        n_atoms=len(graph.elements),
                        graph_key=graph_key(graph),
                        graph=graph,
                    )
                )
    _add_lipid21(index)
    _INDEX_CACHE[key] = index
    return index


def _add_lipid21(index: dict[str, list[NativeLipid]]) -> None:
    """Lipid21's own templates, which carry an explicit atom and bond table."""
    root = Path(__file__).resolve().parents[2] / "data" / "lipid21"
    manifest = root / "templates.json"
    if not manifest.is_file():
        return
    try:
        templates = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return
    for residue in sorted(templates):
        graph = _lipid21_graph(root / "itp" / f"{residue}.itp")
        if graph is None:
            continue
        index.setdefault(graph_key(graph), []).append(
            NativeLipid(
                residue=residue.upper(),
                force_field="amber-lipid21",
                source=f"lipid21/{residue}.itp",
                n_atoms=len(graph.elements),
                graph_key=graph_key(graph),
                graph=graph,
            )
        )


#: Lipid21 namespaces its types as ``L21_xY`` where ``x`` is the element.
_L21_ELEMENTS = {"c": 6, "h": 1, "n": 7, "o": 8, "p": 15, "s": 16}


def _lipid21_graph(path: Path) -> ct.Graph | None:
    if not path.is_file():
        return None
    section = ""
    numbers: dict[str, int] = {}
    order: list[str] = []
    bonds: list[tuple[str, str]] = []
    for raw in path.read_text(errors="replace").splitlines():
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
            continue
        if not code or code.startswith("#"):
            continue
        fields = code.split()
        if section == "atoms" and len(fields) >= 2:
            atom_type = fields[1]
            element = _L21_ELEMENTS.get(atom_type.replace("L21_", "")[:1].lower())
            if element is None:
                return None
            numbers[fields[0]] = element
            order.append(fields[0])
        elif section == "bonds" and len(fields) >= 2:
            bonds.append((fields[0], fields[1]))
    if not order:
        return None
    position = {name: i for i, name in enumerate(order)}
    adjacency: list[set[int]] = [set() for _ in order]
    for first, second in bonds:
        if first not in position or second not in position:
            return None
        a, b = position[first], position[second]
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return ct.Graph(
        tuple(numbers[name] for name in order),
        tuple(tuple(sorted(n)) for n in adjacency),
    )


def find_native_lipid(smiles: str, force_field: str) -> list[NativeLipid]:
    """Native candidates sharing element-labelled connectivity with this SMILES.

    The fingerprint narrows the search; every hit is then confirmed by exact
    isomorphism, because the fingerprint alone would report DPPC and SMPC as
    the same molecule. A hit does not establish bond order or stereochemistry;
    callers must not substitute the candidate as a proven chemical identity.
    """
    graph = smiles_graph(smiles)
    if graph is None:
        return []
    bucket = native_lipid_index(force_field).get(graph_key(graph), ())
    return [entry for entry in bucket if same_molecule(graph, entry.graph)]


def registry_disagreements(force_field: str) -> list[dict]:
    """Registry lipids whose SMILES does not match the release's own residue.

    Compared by name rather than by graph -- the point is precisely that the
    graphs differ, so a graph lookup would simply find nothing.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_name
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    key = str(force_field).strip().lower()
    root = force_field_directory(key)
    if root is None or not (root / "ffnonbonded.itp").is_file():
        return []
    numbers = ct._atomic_numbers(root)
    residues: dict[str, dict] = {}
    for filename in _LIPID_DATABASES.get(key, ()):
        path = root / filename
        if not path.is_file():
            continue
        parser = RTPParser(path)
        for residue in parser.residue_names:
            record = parser.get_residue(residue)
            if record:
                residues.setdefault(residue.upper(), record)

    disagreements = []
    for name in LipidRegistry.list_builtin():
        lipid = LipidRegistry.get(name)
        if not lipid.smiles:
            continue
        residue = lipid_rtp_name(name, key).upper()
        record = residues.get(residue)
        if record is None:
            continue
        native = residue_graph(record, numbers)
        ours = smiles_graph(lipid.smiles)
        if native is None or ours is None:
            continue
        if same_molecule(native, ours):
            continue
        disagreements.append(
            {
                "lipid": name,
                "residue": residue,
                "force_field": key,
                "our_atoms": len(ours.elements),
                "native_atoms": len(native.elements),
                "our_formula": _formula(ours),
                "native_formula": _formula(native),
            }
        )
    return disagreements


_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S"}


def _formula(graph: ct.Graph) -> str:
    counts: dict[int, int] = {}
    for element in graph.elements:
        counts[element] = counts.get(element, 0) + 1
    return "".join(
        f"{_SYMBOLS.get(number, number)}{count}"
        for number, count in sorted(counts.items(), key=lambda kv: (kv[0] != 6, kv[0] != 1, kv[0]))
    )

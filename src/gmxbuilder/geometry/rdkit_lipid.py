"""RDKit-backed all-atom lipid geometry generation."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from gmxbuilder.modules.membrane.lipid_orientation import atom_element


def _element(atom_name: str) -> str:
    element = atom_element(atom_name)
    if not element:
        raise ValueError(f"Cannot infer an element from atom name {atom_name!r}")
    # RTP lipid names use organic elements; recognize the two-letter halogens
    # without misclassifying carbon labels such as CA/CB as calcium/boron.
    normalized = str(atom_name).strip().upper().lstrip("0123456789")
    return normalized[:2].title() if normalized.startswith(("CL", "BR")) else element


def _molecule_from_rtp(rtp: dict):
    """Create an explicit-atom RDKit molecule in exact RTP atom order."""
    from rdkit import Chem

    molecule = Chem.RWMol()
    atoms = rtp["atoms"]
    name_index = {}
    for name, _atom_type, _charge, _group in atoms:
        atom = Chem.Atom(_element(name))
        atom.SetNoImplicit(True)
        name_index[name] = molecule.AddAtom(atom)

    for left, right in rtp["bonds"]:
        molecule.AddBond(name_index[left], name_index[right], Chem.BondType.SINGLE)
    for name, atom_type, _charge, _group in atoms:
        atom = molecule.GetAtomWithIdx(name_index[name])
        if atom.GetSymbol() == "N" and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)
        elif atom.GetSymbol() == "P" and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)
        elif atom_type == "O2L" and atom.GetDegree() == 1:
            atom.SetFormalCharge(-1)
    result = molecule.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(result)
    return result, [atom[0] for atom in atoms]


def _seed_explicit_stereochemistry(molecule, smiles: str, seed: int) -> bool:
    """Seed an RTP-ordered conformer from an explicitly stereochemical SMILES.

    RTP files encode bonded parameters but not portable atom chirality tags.
    For registry entries with ``@`` stereochemistry, establish an exact
    element/connectivity graph mapping to an explicit-H SMILES conformer and
    reorder those coordinates into RTP atom order.  Returning random RTP
    stereochemistry would be scientifically unsafe, so an unmappable explicit
    stereoisomer is rejected by the caller.
    """
    if "@" not in smiles:
        return False
    from rdkit import Chem
    from rdkit.Chem import AllChem

    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if reference is None or reference.GetNumAtoms() != molecule.GetNumAtoms():
        return False

    def connectivity_graph(source):
        graph = Chem.RWMol(source)
        for atom in graph.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
        for bond in graph.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        result = graph.GetMol()
        result.UpdatePropertyCache(strict=False)
        return result

    match = connectivity_graph(molecule).GetSubstructMatch(
        connectivity_graph(reference),
        useChirality=False,
    )
    if len(match) != reference.GetNumAtoms() or len(set(match)) != len(match):
        return False

    embedded = False
    for attempt in range(4):
        status = AllChem.EmbedMolecule(
            reference,
            randomSeed=int(seed) + attempt * 104729,
            useRandomCoords=attempt > 0,
            maxAttempts=1000,
            clearConfs=True,
        )
        if status == 0:
            embedded = True
            break
    if not embedded:
        return False

    source = reference.GetConformer()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for reference_index, rtp_index in enumerate(match):
        conformer.SetAtomPosition(rtp_index, source.GetAtomPosition(reference_index))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return True


def _sample_lipid_tail_torsions(mol, conformer, names: list[str], rtp: dict, seed: int) -> None:
    """Generate reproducible membrane-like trans/gauche acyl-tail torsions.

    An all-trans hydrocarbon chain is a high-length crystal-like starting
    conformation, not a representative fluid bilayer conformation.  Preserve
    the exact RTP atom graph while sampling local gauche defects, and keep
    CHARMM ``CEL1-CEL1`` unsaturated bonds in their cis state.
    """
    from rdkit.Chem import rdMolTransforms

    index = {name: number for number, name in enumerate(names)}
    atom_types = {name: atom_type for name, atom_type, *_rest in rtp["atoms"]}
    rng = np.random.default_rng(int(seed) + 0x5EED)
    chains = []
    for prefix in ("C2", "C3"):
        chain = [
            (int(name[2:]), name)
            for name in names
            if name.startswith(prefix) and name[2:].isdigit()
        ]
        if chain:
            chains.append(sorted(chain))
    # CHARMM sphingolipids use C1F..CnF for the N-acyl chain and
    # C3S..CnS for the sphingosine hydrocarbon chain.  Leaving these out made
    # RDKit retain compact random coils and produced 1-2 nm bilayer core gaps.
    for suffix_letter, minimum_number in (("F", 1), ("S", 3)):
        chain = []
        for name in names:
            match = re.fullmatch(r"C(\d+)" + suffix_letter, name)
            if match and int(match.group(1)) >= minimum_number:
                chain.append((int(match.group(1)), name))
        if chain:
            chains.append(sorted(chain))

    for chain in chains:
        chain_names = [name for _suffix, name in sorted(chain)]
        rotatable_starts = []
        for start in range(len(chain_names) - 3):
            atoms = [index[name] for name in chain_names[start : start + 4]]
            if not all(
                mol.GetBondBetweenAtoms(left, right) is not None
                # strict=False on purpose: consecutive pairs: the tail is one shorter.
                for left, right in zip(atoms, atoms[1:], strict=False)
            ):
                continue
            central = chain_names[start + 1 : start + 3]
            central_types = [atom_types.get(name, "") for name in central]
            if central_types == ["CEL1", "CEL1"]:
                # Natural phospholipid double bonds are cis in the bundled
                # CHARMM templates.  The RTP bond list has no bond-order field,
                # so retain this explicitly in the coordinate generator.
                angle = 0.0
            elif start < 2:
                # Keep the ester-proximal chain directed into the membrane.
                angle = 180.0
            else:
                rotatable_starts.append(start)
                angle = float(rng.choice([180.0, 60.0, -60.0], p=[0.72, 0.14, 0.14]))
            rdMolTransforms.SetDihedralDeg(conformer, *atoms, angle)

        # Every long saturated segment needs at least one thermal gauche
        # defect; otherwise a random seed can still yield a fully extended
        # chain.  Use separated defects to avoid hairpin/self-clash geometry.
        if len(rotatable_starts) >= 4:
            forced = rotatable_starts[len(rotatable_starts) // 2]
            atoms = [index[name] for name in chain_names[forced : forced + 4]]
            current = rdMolTransforms.GetDihedralDeg(conformer, *atoms)
            if abs(abs(current) - 180.0) < 20.0:
                rdMolTransforms.SetDihedralDeg(
                    conformer, *atoms, 60.0 if rng.random() < 0.5 else -60.0
                )


def _orient_for_membrane(coords: np.ndarray, names: list[str]) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    name_index = {name: index for index, name in enumerate(names)}
    tail_names = []
    for prefix in ("C2", "C3"):
        candidates = [
            (int(name[2:]), name)
            for name in names
            if name.startswith(prefix) and name[2:].isdigit()
        ]
        if candidates:
            tail_names.append(max(candidates)[1])
    for suffix_letter in ("F", "S"):
        candidates = []
        for name in names:
            match = re.fullmatch(r"C(\d+)" + suffix_letter, name)
            if match:
                candidates.append((int(match.group(1)), name))
        if candidates:
            tail_names.append(max(candidates)[1])
    if "P" in name_index and tail_names:
        head = coords[name_index["P"]]
        tail = np.mean([coords[name_index[name]] for name in tail_names], axis=0)
        axis = head - tail
        axis /= np.linalg.norm(axis)
    else:
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, -1]
    polar = [i for i, name in enumerate(names) if _element(name) in {"N", "O", "P", "S"}]
    carbon = [i for i, name in enumerate(names) if _element(name) == "C"]
    if polar and carbon:
        direction = coords[polar].mean(axis=0) - coords[carbon].mean(axis=0)
        if np.dot(axis, direction) < 0:
            axis = -axis
    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [axis])
    oriented = rotation.apply(centered)
    return oriented - oriented.mean(axis=0)


def _align_tail_subtrees(coords, names, rtp, *, smiles):
    from gmxbuilder.geometry.tail_torsions import align_tail_torsions

    return align_tail_torsions(coords, names, rtp, smiles=smiles)


def _align_gaff_tail_subtrees(
    coords: np.ndarray,
    names: list[str],
    smiles: str,
) -> np.ndarray:
    """Point acyl/hydrocarbon branches inward using the SMILES bond graph.

    ACPYPE preserves the input SMILES heavy-atom order and appends hydrogens.
    Long carbon branches separated from a polar atom by a bridge bond are
    therefore identifiable without relying on GAFF serial names.  Ring systems
    such as sterols are deliberately skipped because cutting a ring edge does
    not isolate a tail subtree; their whole-molecule amphiphile axis is handled
    by :func:`_orient_for_membrane`.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return coords
    n_heavy = molecule.GetNumAtoms()
    if n_heavy > len(names):
        return coords
    expected_elements = [atom.GetSymbol().upper() for atom in molecule.GetAtoms()]
    actual_elements = [_element(name) for name in names[:n_heavy]]
    if expected_elements != actual_elements:
        return coords

    adjacency = {
        atom.GetIdx(): {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
        for atom in molecule.GetAtoms()
    }
    polar = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol().upper() in {"N", "O", "P", "S"}
    ]
    terminals = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetDegree() == 1
    ]
    candidates: dict[frozenset[int], tuple[tuple[int, ...], int, int]] = {}
    for terminal in terminals:
        paths = [Chem.GetShortestPath(molecule, start, terminal) for start in polar]
        path = min(paths, key=len, default=())
        if len(path) < 4:
            continue
        root, first = int(path[1]), int(path[2])

        # Find the component on the terminal side after removing root--first.
        component: set[int] = set()
        stack = [first]
        while stack:
            atom_index = stack.pop()
            if atom_index in component:
                continue
            component.add(atom_index)
            for neighbor in adjacency[atom_index]:
                if {atom_index, neighbor} == {root, first}:
                    continue
                stack.append(neighbor)
        if root in component or terminal not in component:
            continue
        carbon_count = sum(molecule.GetAtomWithIdx(index).GetSymbol() == "C" for index in component)
        if carbon_count < 6 or any(index in polar for index in component):
            continue
        key = frozenset(component)
        previous = candidates.get(key)
        if previous is None or len(path) > len(previous[0]):
            candidates[key] = (tuple(int(value) for value in path), root, terminal)

    # A bilayer amphiphile normally has one or two independent hydrocarbon
    # branches.  Keep the two largest to avoid moving short headgroup methyls.
    selected = sorted(
        candidates.items(), key=lambda item: (len(item[0]), len(item[1][0])), reverse=True
    )[:2]

    # Map appended GAFF hydrogens to their nearest heavy atom.  Every rigid
    # subtree/torsion operation must carry those hydrogens with the bonded
    # heavy atom or it would corrupt C-H bond lengths.
    attached_hydrogens: dict[int, list[int]] = {index: [] for index in range(n_heavy)}
    for index in range(n_heavy, len(names)):
        if _element(names[index]) != "H":
            continue
        nearest = int(np.linalg.norm(coords[:n_heavy] - coords[index], axis=1).argmin())
        attached_hydrogens[nearest].append(index)

    def with_hydrogens(heavy_indices: set[int] | frozenset[int]) -> list[int]:
        result = set(heavy_indices)
        for heavy_index in heavy_indices:
            result.update(attached_hydrogens.get(heavy_index, []))
        return sorted(result)

    def dihedral(atom_indices: tuple[int, int, int, int], values: np.ndarray) -> float:
        p0, p1, p2, p3 = values[list(atom_indices)]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        norm = np.linalg.norm(b1)
        if norm < 1e-10:
            return 0.0
        b1 /= norm
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

    for branch_number, (component, (path, root, terminal)) in enumerate(selected):
        # Remove folded cis-like single-bond torsions along the hydrocarbon
        # path.  Double bonds and rings keep their force-field geometry.
        for path_index in range(1, len(path) - 2):
            atoms = tuple(path[path_index - 1 : path_index + 3])
            left, right = atoms[1], atoms[2]
            bond = molecule.GetBondBetweenAtoms(left, right)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
                continue
            downstream: set[int] = set()
            stack = [right]
            while stack:
                atom_index = stack.pop()
                if atom_index in downstream:
                    continue
                downstream.add(atom_index)
                for neighbor in adjacency[atom_index]:
                    if {atom_index, neighbor} == {left, right}:
                        continue
                    stack.append(neighbor)
            if left in downstream:
                continue
            current = dihedral(atoms, coords)
            if abs(current) >= np.deg2rad(150.0):
                continue
            target = np.pi if current >= 0.0 else -np.pi
            delta = target - current
            axis = coords[right] - coords[left]
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10:
                continue
            axis /= axis_norm
            indices = with_hydrogens(downstream)
            origin = coords[left].copy()
            best_values = None
            best_error = float("inf")
            for signed_delta in (delta, -delta):
                trial = coords.copy()
                rotation = Rotation.from_rotvec(axis * signed_delta)
                trial[indices] = rotation.apply(trial[indices] - origin) + origin
                error = abs(abs(dihedral(atoms, trial)) - np.pi)
                if error < best_error:
                    best_error = error
                    best_values = trial[indices]
            if best_values is not None:
                coords[indices] = best_values

        origin = coords[root]
        # Turn about the attachment bond, not an arbitrary axis through the
        # root: the latter bends covalent angles and can invert ceramide C3.
        axis = coords[path[2]] - origin
        if np.linalg.norm(axis) < 1e-8:
            continue
        axis /= np.linalg.norm(axis)
        target = np.asarray([0.12 if branch_number == 0 else -0.12, 0.0, -1.0])
        target /= np.linalg.norm(target)
        indices = with_hydrogens(component)
        best, best_score = coords[indices].copy(), -np.inf
        for degrees in (0, 60, -60, 120, -120, 180):
            rotation = Rotation.from_rotvec(axis * np.deg2rad(degrees))
            direction = rotation.apply(coords[terminal] - origin)
            score = float(np.dot(direction, target))
            if score > best_score:
                best = rotation.apply(coords[indices] - origin) + origin
                best_score = score
        coords[indices] = best
    return coords


def _chirality_signature(coords: np.ndarray, adjacency: list[tuple[int, ...]]) -> tuple:
    """The handedness of every four-coordinate atom, as a tuple of signs.

    Rotating a rigid subtree about the bond that attaches it cannot change a
    stereocentre -- but only if the subtree really is everything on one side of
    that bond. Where the partition is wrong the rotation moves some of a
    centre's substituents and not others, and the centre can come out inverted.
    That is not a hypothetical: aligning the tails of C16 ceramide inverted its
    sphingoid C3 on one seed in four, turning D-erythro ceramide into its C3
    epimer in a conformer that was otherwise perfectly good.
    """
    signs = []
    for index, neighbours in enumerate(adjacency):
        if len(neighbours) != 4:
            continue
        a, b, c, d = (coords[n] for n in neighbours)
        # The volume spanned by all four substituents, not three of them and
        # the centre. Tail alignment re-points a whole branch, so the atom that
        # moves is often the fourth one; a triple that happens to leave it out
        # reports no change while the centre has inverted.
        volume = float(np.dot(np.cross(b - a, c - a), d - a))
        signs.append((index, volume))
    return tuple(signs)


#: How far a stereocentre may flatten before its rearrangement is refused.
#:
#: The sign of the substituent tetrahedron is not enough on its own. Aligning
#: C16 ceramide's tail left the sign of its sphingoid C3 alone but collapsed the
#: tetrahedron from 0.0077 to 0.0012 nm^3, and a centre that nearly planar has
#: no reliable handedness left -- RDKit read the flattened conformer as the C3
#: epimer. A rigid rotation of a branch about the bond that attaches it does not
#: touch the geometry at the centre at all, so any real collapse means the
#: rearrangement was not that.
MINIMUM_CHIRAL_VOLUME_RATIO = 0.5


def _double_bond_signature(coords: np.ndarray, adjacency: list[tuple[int, ...]]) -> tuple:
    """Which side of each sp2-sp2 bond its substituents sit on.

    Chirality alone is not enough: the same rearrangement that can invert a
    centre can swing a double bond from cis to trans, and a plasmalogen's vinyl
    ether flipped on two seeds in three. An RTP file carries no bond orders, so
    a bond between two three-coordinate atoms stands in for one -- it is the
    geometry that has to be preserved either way.
    """
    signs = []
    for first, neighbours in enumerate(adjacency):
        if len(neighbours) != 3:
            continue
        for second in neighbours:
            if second <= first or len(adjacency[second]) != 3:
                continue
            left = next(n for n in adjacency[first] if n != second)
            right = next(n for n in adjacency[second] if n != first)
            axis = coords[second] - coords[first]
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:
                continue
            axis = axis / norm
            a = coords[left] - coords[first]
            b = coords[right] - coords[second]
            a = a - axis * float(np.dot(a, axis))
            b = b - axis * float(np.dot(b, axis))
            if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
                continue
            signs.append(((first, second), float(np.sign(np.dot(a, b)))))
    return tuple(signs)


def _keeps_chirality(
    before: np.ndarray, after: np.ndarray, adjacency: list[tuple[int, ...]]
) -> bool:
    """Whether a rearrangement left the molecule's stereochemistry alone.

    Both kinds: every four-coordinate atom's handedness, and which side of each
    sp2-sp2 bond its substituents lie on.
    """
    first, second = (
        _chirality_signature(before, adjacency),
        _chirality_signature(after, adjacency),
    )
    if len(first) != len(second):
        return False
    for (index, was), (also, now) in zip(first, second, strict=True):
        if index != also:
            return False
        if np.sign(was) != np.sign(now):
            return False
        if abs(now) < MINIMUM_CHIRAL_VOLUME_RATIO * abs(was):
            return False
    return _double_bond_signature(before, adjacency) == _double_bond_signature(after, adjacency)


def _adjacency_from_bonds(names: list[str], bonds) -> list[tuple[int, ...]]:
    position = {str(name).strip(): index for index, name in enumerate(names)}
    neighbours: list[set[int]] = [set() for _ in names]
    for first, second in bonds:
        a, b = str(first), str(second)
        if a not in position or b not in position:
            continue
        neighbours[position[a]].add(position[b])
        neighbours[position[b]].add(position[a])
    return [tuple(sorted(entry)) for entry in neighbours]


def _adjacency_from_rdkit(smiles: str, names: list[str]) -> list[tuple[int, ...]] | None:
    """Bond graph for a GAFF conformer, in the atom order ACPYPE returned.

    Only where ACPYPE really did keep the SMILES heavy-atom order and append the
    hydrogens. It does not always: for digalactosyldiacylglycerol it interleaves
    them, and eight positions carry a different element from the SMILES. Equal
    atom counts do not establish the correspondence, so the elements are
    compared -- the same test `_align_gaff_tail_subtrees` applies before it will
    touch anything, so the two agree about when the assumption holds.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    heavy = molecule.GetNumAtoms()
    if heavy > len(names):
        return None
    expected = [atom.GetSymbol().upper() for atom in molecule.GetAtoms()]
    if expected != [_element(name) for name in names[:heavy]]:
        return None
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() != len(names):
        return None
    neighbours: list[set[int]] = [set() for _ in names]
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbours[a].add(b)
        neighbours[b].add(a)
    return [tuple(sorted(entry)) for entry in neighbours]


def _has_intramolecular_overlap(coords: np.ndarray, cutoff_nm: float = 0.05) -> bool:
    """Return whether a rigid-tail transform made two atoms interpenetrate."""
    if len(coords) < 2 or not np.isfinite(coords).all():
        return True
    differences = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return bool(float(distances.min()) < cutoff_nm)


def _validated_gaff_geometry(template, smiles):
    """Accept optional packing torsions only if the exact ITP geometry survives."""
    from itertools import combinations

    from gmxbuilder.geometry.molecular_identity import itp_graph, validate_stereochemistry

    names, elements, bonds = itp_graph(template.itp_path)
    if names != template.atom_names:
        raise ValueError("GAFF coordinate/topology atom order differs")
    coords = _orient_for_membrane(template.coordinates.copy(), list(names))
    validate_stereochemistry(smiles, elements, bonds, coords)
    aligned = _align_gaff_tail_subtrees(coords.copy(), list(names), smiles)
    adjacency = [set() for _ in names]
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    # Bond lengths plus the distances between each pair of bonded neighbours
    # establish all covalent bond angles, including hydrogens. This catches
    # a wrongly partitioned branch even when its CIP sign happens to survive.
    pairs = list(bonds) + [pair for neighbours in adjacency for pair in combinations(neighbours, 2)]
    indices = np.asarray(pairs, dtype=int)
    before = np.linalg.norm(coords[indices[:, 0]] - coords[indices[:, 1]], axis=1)
    after = np.linalg.norm(aligned[indices[:, 0]] - aligned[indices[:, 1]], axis=1)
    intact = np.allclose(before, after, atol=1e-7, rtol=0)
    try:
        validate_stereochemistry(smiles, elements, bonds, aligned)
    except ValueError:
        intact = False
    if intact and not _has_intramolecular_overlap(aligned):
        coords = aligned
    coords -= coords.mean(axis=0)
    return coords


@lru_cache(maxsize=256)
def _build_cached(
    lipid_name: str, smiles: str, force_field: str, seed: int, net_charge: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    _rtp_name, rtp = lipid_rtp_template(lipid_name, force_field)
    if rtp is None:
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid

        template = prepare_gaff_lipid(lipid_name, smiles, net_charge)
        coords = _validated_gaff_geometry(template, smiles)
        return coords, template.atom_names

    # Only generated RTP bootstrap coordinates may try another seed. Cached
    # GAFF2/Lipid21 coordinates above are authoritative and are never replaced.
    failures = []
    for attempt in dict.fromkeys((seed, *range(5))):
        try:
            result = _build_rtp_seed(smiles, rtp, attempt)
        except _BootstrapGeometryError as exc:
            failures.append(f"seed {attempt}: {exc}")
            continue
        if attempt != seed:
            import logging

            logging.getLogger(__name__).warning(
                "%s/%s bootstrap seed %s rejected; using validated seed %s (%s)",
                lipid_name,
                force_field,
                seed,
                attempt,
                "; ".join(failures),
            )
        return result
    raise _BootstrapGeometryError(
        f"No valid bootstrap geometry for {lipid_name}/{force_field}: " + "; ".join(failures)
    )


class _BootstrapGeometryError(ValueError):
    """A generated conformer failed; a different deterministic seed may work."""


def _validate_rtp_bootstrap(smiles, names, rtp, coords, *, check_overlap=True):
    from gmxbuilder.geometry.molecular_identity import validate_stereochemistry

    positions = {name: i for i, name in enumerate(names)}
    elements = tuple(_element(name) for name in names)
    bonds = tuple((positions[a], positions[b]) for a, b in rtp["bonds"])
    try:
        validate_stereochemistry(smiles, elements, bonds, coords)
    except ValueError as exc:
        raise _BootstrapGeometryError(str(exc)) from exc
    if check_overlap and _has_intramolecular_overlap(coords):
        raise _BootstrapGeometryError("Intramolecular overlap in generated coordinates")


def _build_rtp_seed(smiles, rtp, seed):
    """Generate and validate one seed; no retries or identity substitution here."""
    from rdkit.Chem import AllChem

    mol, names = _molecule_from_rtp(rtp)
    stereo_seeded = _seed_explicit_stereochemistry(mol, smiles, int(seed))
    if "@" in smiles and not stereo_seeded:
        raise _BootstrapGeometryError(
            "Explicit stereochemistry cannot be mapped exactly onto its force-field atom graph"
        )
    if not stereo_seeded:
        status = -1
        for attempt in range(4):
            status = AllChem.EmbedMolecule(
                mol,
                randomSeed=int(seed) + attempt * 104729,
                useRandomCoords=attempt > 0,
                maxAttempts=1000,
                clearConfs=True,
            )
            if status == 0:
                break
        if status != 0:
            raise _BootstrapGeometryError(
                "RDKit could not embed lipid after four deterministic attempts"
            )
    conformer = mol.GetConformer()
    _sample_lipid_tail_torsions(mol, conformer, names, rtp, seed)
    coords = (
        np.asarray([list(conformer.GetAtomPosition(index)) for index in range(mol.GetNumAtoms())])
        / 10.0
    )

    coords = _orient_for_membrane(np.asarray(coords), list(names))
    # A torsion may resolve a steric clash, but must start with intact identity.
    _validate_rtp_bootstrap(smiles, names, rtp, coords, check_overlap=False)
    prealigned = coords.copy()
    # Improve tail directions with torsions only. An arbitrary branch-root
    # rotation used to flatten PSM's C3S while still reporting the R isomer.
    aligned = _align_tail_subtrees(coords.copy(), list(names), rtp, smiles=smiles)
    # An optional packing operation cannot weaken the identity contract. Check
    # the proposed coordinates, not merely a relative tetrahedral-volume sign.
    adjacency = _adjacency_from_bonds(list(names), rtp.get("bonds") or [])
    try:
        _validate_rtp_bootstrap(smiles, names, rtp, aligned)
        intact = _keeps_chirality(prealigned, aligned, adjacency)
    except _BootstrapGeometryError:
        intact = False
    coords = aligned if intact else prealigned
    _validate_rtp_bootstrap(smiles, names, rtp, coords)
    coords -= coords.mean(axis=0)
    return coords, tuple(names)


def build_rdkit_lipid_geometry(
    lipid_name: str,
    smiles: str,
    force_field: str = "charmm36m",
    seed: int = 0,
    net_charge: int | None = None,
    lipid_ff: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build one optimized all-atom lipid conformation."""
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

    selected_lipid_ff = lipid_backend_for(lipid_name, lipid_ff)
    if selected_lipid_ff == "lipid21":
        from gmxbuilder.modules.forcefield.lipid21_backend import load_lipid21_geometry
        from gmxbuilder.modules.membrane.lipid_orientation import (
            orient_lipid_to_outward_normal,
        )

        coords, names = load_lipid21_geometry(lipid_name)
        coords = orient_lipid_to_outward_normal(coords, names, upper=True)
        coords -= coords.mean(axis=0)
        return coords, names
    if net_charge is None:
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        net_charge = LipidRegistry.get(lipid_name).charge
    if selected_lipid_ff == "gaff2":
        # An Amber force-field installation may also contain legacy lipid RTP
        # residues.  An explicit GAFF2 selection is authoritative: coordinates
        # and topology must originate from the same cached ACPYPE template,
        # even when a same-named RTP residue exists (POPC and CHOL are common
        # examples).  Falling through to _build_cached() previously produced
        # RTP atom order with a GAFF2 topology and was correctly rejected by
        # TopologyWriter.
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid

        template = prepare_gaff_lipid(lipid_name, smiles, int(net_charge))
        coords = _validated_gaff_geometry(template, smiles)
        return coords, list(template.atom_names)
    coords, names = _build_cached(
        lipid_name.upper(), smiles, force_field.lower(), int(seed), int(net_charge)
    )
    return coords.copy(), list(names)

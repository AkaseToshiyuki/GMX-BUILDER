"""RDKit-backed all-atom lipid geometry generation."""

from __future__ import annotations

from functools import lru_cache
import re
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
                for left, right in zip(atoms, atoms[1:])
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


def _align_tail_subtrees(coords: np.ndarray, names: list[str], rtp: dict) -> np.ndarray:
    """Rigidly point both complete acyl-tail subtrees down the membrane Z axis."""
    name_index = {name: index for index, name in enumerate(names)}
    adjacency = {name: set() for name in names}
    for left, right in rtp["bonds"]:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    for root, first, pattern, x_component, y_component in (
        ("C21", "C22", r"C2(\d+)", 0.12, 0.10),
        ("C31", "C32", r"C3(\d+)", -0.12, -0.10),
        ("C1F", "C2F", r"C(\d+)F", 0.12, 0.10),
        ("C3S", "C4S", r"C(\d+)S", -0.12, -0.10),
    ):
        if root not in name_index or first not in name_index or first not in adjacency[root]:
            continue
        terminal_candidates = []
        for name in names:
            match = re.fullmatch(pattern, name)
            if match:
                terminal_candidates.append((int(match.group(1)), name))
        if not terminal_candidates:
            continue
        terminal = max(terminal_candidates)[1]
        subtree = set()
        stack = [first]
        while stack:
            name = stack.pop()
            if name == root or name in subtree:
                continue
            subtree.add(name)
            stack.extend(adjacency[name] - subtree - {root})
        origin = coords[name_index[root]]
        axis = coords[name_index[terminal]] - origin
        if np.linalg.norm(axis) < 1e-8:
            continue
        axis /= np.linalg.norm(axis)
        # Give the two inward-facing tails a small opposing azimuthal spread.
        # This remains a rigid subtree rotation, preserves stereochemistry and
        # bond geometry, and avoids the artificial tail/head intersections
        # produced when both branches are forced into the same XZ plane.
        target = np.array([x_component, y_component, -1.0])
        target /= np.linalg.norm(target)
        rotation, _ = Rotation.align_vectors([target], [axis])
        indices = [name_index[name] for name in subtree]
        base_values = rotation.apply(coords[indices] - origin)
        fixed_indices = [index for index in range(len(coords)) if index not in indices]
        fixed_values = coords[fixed_indices]
        best_values = base_values
        best_clearance = -np.inf
        # Alignment fixes the root-to-terminal direction but leaves a free
        # roll around that axis.  Select the deterministic roll with greatest
        # clearance from the headgroup/other tail.  This prevents a valid
        # sphingolipid tail from being discarded merely because C2F happens to
        # land on the amide nitrogen in the zero-roll orientation.
        for degrees in (0, 60, -60, 120, -120, 180):
            roll = Rotation.from_rotvec(target * np.deg2rad(degrees))
            candidate = roll.apply(base_values) + origin
            if len(fixed_values):
                separations = candidate[:, None, :] - fixed_values[None, :, :]
                clearance = float(np.linalg.norm(separations, axis=2).min())
            else:
                clearance = np.inf
            if clearance > best_clearance:
                best_clearance = clearance
                best_values = candidate
        coords[indices] = best_values
    return coords


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
        axis = coords[terminal] - origin
        if np.linalg.norm(axis) < 1e-8:
            continue
        axis /= np.linalg.norm(axis)
        target = np.asarray([0.12 if branch_number == 0 else -0.12, 0.0, -1.0])
        target /= np.linalg.norm(target)
        rotation, _ = Rotation.align_vectors([target], [axis])
        indices = with_hydrogens(component)
        coords[indices] = rotation.apply(coords[indices] - origin) + origin
    return coords


def _has_intramolecular_overlap(coords: np.ndarray, cutoff_nm: float = 0.05) -> bool:
    """Return whether a rigid-tail transform made two atoms interpenetrate."""
    if len(coords) < 2 or not np.isfinite(coords).all():
        return True
    differences = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return bool(float(distances.min()) < cutoff_nm)


@lru_cache(maxsize=256)
def _build_cached(
    lipid_name: str, smiles: str, force_field: str, seed: int, net_charge: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    _rtp_name, rtp = lipid_rtp_template(lipid_name, force_field)
    if rtp is None:
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid

        template = prepare_gaff_lipid(lipid_name, smiles, net_charge)
        coords = _orient_for_membrane(template.coordinates.copy(), list(template.atom_names))
        aligned = _align_gaff_tail_subtrees(
            coords.copy(),
            list(template.atom_names),
            smiles,
        )
        # Tail rotations are an optional bootstrap improvement, not a reason
        # to corrupt a valid GAFF2 conformer.  Some branched headgroups (DPPS
        # in particular) can make independently rotated subtrees cross even
        # though every moved covalent bond remains intact.
        if not _has_intramolecular_overlap(aligned):
            coords = aligned
        coords -= coords.mean(axis=0)
        return coords, template.atom_names

    from rdkit.Chem import AllChem

    mol, names = _molecule_from_rtp(rtp)
    stereo_seeded = _seed_explicit_stereochemistry(mol, smiles, int(seed))
    if "@" in smiles and not stereo_seeded:
        raise ValueError(
            f"Explicit stereochemistry for lipid {lipid_name} cannot be mapped "
            "exactly onto its force-field atom graph"
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
            raise ValueError(
                f"RDKit could not embed lipid {lipid_name} after four deterministic attempts"
            )
    conformer = mol.GetConformer()
    _sample_lipid_tail_torsions(mol, conformer, names, rtp, seed)
    coords = (
        np.asarray([list(conformer.GetAtomPosition(index)) for index in range(mol.GetNumAtoms())])
        / 10.0
    )

    coords = _orient_for_membrane(np.asarray(coords), list(names))
    prealigned = coords.copy()
    # The RTP path has an exact bond graph, so orient each complete acyl-tail
    # subtree after the whole-molecule head-to-tail alignment.  This helper
    # existed previously but was never called, leaving many lipids lying in
    # the XY membrane plane and producing sub-angstrom intermolecular clashes.
    aligned = _align_tail_subtrees(coords.copy(), list(names), rtp)
    # Tail alignment is a packing aid, never authority to corrupt the exact
    # RTP conformer.  Fall back to the whole-molecule rigid orientation when
    # independently rotated subtrees interpenetrate.
    coords = prealigned if _has_intramolecular_overlap(aligned) else aligned
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
    selected_lipid_ff = str(lipid_ff or "").strip().lower()
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
        coords = _orient_for_membrane(template.coordinates.copy(), list(template.atom_names))
        aligned = _align_gaff_tail_subtrees(
            coords.copy(),
            list(template.atom_names),
            smiles,
        )
        if not _has_intramolecular_overlap(aligned):
            coords = aligned
        coords -= coords.mean(axis=0)
        return coords, list(template.atom_names)
    coords, names = _build_cached(
        lipid_name.upper(), smiles, force_field.lower(), int(seed), int(net_charge)
    )
    return coords.copy(), list(names)

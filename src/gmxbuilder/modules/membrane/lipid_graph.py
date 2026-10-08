"""Exact force-field atom graphs used for imaging and coordinate validation."""

from functools import lru_cache

import numpy as np

from gmxbuilder.geometry.molecular_identity import itp_graph, reference_mappings


@lru_cache(maxsize=512)
def molecular_graph(name: str, force_field: str, lipid_ff: str, smiles: str):
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

    lipid_ff = lipid_backend_for(name, lipid_ff)
    if lipid_ff == "lipid21":
        from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_itp_path

        graph = itp_graph(lipid21_itp_path(name))
    elif lipid_ff == "gaff2":
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        graph = itp_graph(prepare_gaff_lipid(name, smiles, LipidRegistry.get(name).charge).itp_path)
    else:
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template
        from gmxbuilder.modules.membrane.lipid_orientation import atom_element

        _, template = lipid_rtp_template(name, force_field)
        if template is None:
            raise ValueError(f"No molecular graph for {name}/{force_field}")
        names = tuple(a[0] for a in template["atoms"])
        indices = {n: i for i, n in enumerate(names)}
        graph = (
            names,
            tuple(atom_element(n) for n in names),
            tuple((indices[a], indices[b]) for a, b in template["bonds"]),
        )
    reference_mappings(smiles, graph[1], graph[2])
    return graph


def ordered_graph(name, force_field, lipid_ff, smiles, atom_names):
    names, elements, bonds = molecular_graph(name, force_field, lipid_ff, smiles)
    supplied = tuple(str(n).strip() for n in atom_names)
    if len(set(supplied)) != len(supplied) or set(supplied) != set(names):
        raise ValueError(f"Coordinate/topology atom names disagree for {name}")
    indices = {n: i for i, n in enumerate(supplied)}
    mapping = [indices[n] for n in names]
    return (
        tuple(elements[names.index(n)] for n in supplied),
        tuple((mapping[a], mapping[b]) for a, b in bonds),
    )


@lru_cache(maxsize=512)
def _tree_levels(n_atoms, bonds):
    adjacent = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        adjacent[a].append(b)
        adjacent[b].append(a)
    visited, frontier, levels = {0}, [0], []
    while frontier:
        children, parents = [], []
        for a in frontier:
            for b in adjacent[a]:
                if b not in visited:
                    visited.add(b)
                    parents.append(a)
                    children.append(b)
        if children:
            levels.append((np.array(parents), np.array(children)))
        frontier = children
    if len(visited) != n_atoms:
        raise ValueError("Disconnected lipid graph")
    return levels


def whole_batch(coordinates, bonds, box):
    """Image molecules together, preserving every bond including ring closures."""
    xyz = np.asarray(coordinates, dtype=float)
    lattice = np.asarray(box, dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite molecule/atom/XYZ coordinates")
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("Invalid periodic lattice")
    inverse = np.linalg.inv(lattice)
    result = xyz.copy()
    for parents, children in _tree_levels(xyz.shape[1], tuple(bonds)):
        delta = (xyz[:, children] - xyz[:, parents]) @ inverse
        result[:, children] = result[:, parents] + (delta - np.rint(delta)) @ lattice
    a, b = np.array(bonds).T
    original = (xyz[:, b] - xyz[:, a]) @ inverse
    expected = (original - np.rint(original)) @ lattice
    if not np.allclose(result[:, b] - result[:, a], expected, rtol=0, atol=1e-6):
        raise ValueError("Periodic molecular graph has inconsistent ring closure")
    return result

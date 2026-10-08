"""Bounded torsion proposals for newly reconstructed side-chain atoms only."""

import numpy as np


def _anchor_frame(origin, parent, neighbour):
    """Build a right-handed frame without fitting or moving observed atoms."""
    axis = parent - origin
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length < 1e-8:
        return None
    axis = axis / length
    transverse = neighbour - origin
    transverse = transverse - axis * np.dot(axis, transverse)
    length = np.linalg.norm(transverse)
    if not np.isfinite(length) or length < 1e-8:
        return None
    transverse = transverse / length
    return np.array([axis, transverse, np.cross(axis, transverse)])


def restore_added_sidechain_templates(original, repaired, candidates):
    """Propose rigid template subtrees when relaxed new atoms fail validation."""
    from gmxbuilder.modules.input.geometry_validation import _CLOSURES, reference
    from gmxbuilder.modules.input.protein_repair import _SIDECHAIN_PARENTS, _atom_key

    observed = {_atom_key(original, i) for i in range(original.num_atoms)}
    indices = {_atom_key(repaired, i): i for i in range(repaired.num_atoms)}
    changes = []
    for key, missing in sorted(candidates.items()):
        parents = {**_SIDECHAIN_PARENTS[key[2]], "CA": "N", "N": "C"}
        bonds = list(parents.items()) + _CLOSURES.get(key[2], [])
        anchors = {b if a in missing else a for a, b in bonds if (a in missing) != (b in missing)}
        # A rigid subtree needs one attachment. Partial rings and proline can
        # have multiple attachments; leave those to the backend and validator.
        if len(anchors) != 1:
            continue
        anchor = anchors.pop()
        parent = parents.get(anchor)
        siblings = sorted(
            atom
            for atom, upstream in parents.items()
            if upstream == anchor and (*key, atom) in observed
        )
        neighbour = siblings[0] if siblings else parents.get(parent)
        frame_atoms = (anchor, parent, neighbour)
        if not all((*key, atom) in observed for atom in frame_atoms):
            continue
        ref = reference(key[2])
        source_frame = _anchor_frame(*(ref[atom] for atom in frame_atoms))
        target_points = [repaired.coordinates[indices[(*key, atom)]] for atom in frame_atoms]
        target_frame = _anchor_frame(*target_points)
        if source_frame is None or target_frame is None:
            continue
        # Exact anchor/bond-axis alignment preserves template bond lengths,
        # angles and ring geometry; only missing atoms receive coordinates.
        vectors = np.array([ref[atom] for atom in missing]) - ref[anchor]
        added = [indices[(*key, atom)] for atom in missing]
        repaired.coordinates[added] = vectors @ source_frame.T @ target_frame + target_points[0]
        changes.append({"chain": key[0], "resid": key[1], "resname": key[2]})
    repaired.source_info["sidechain_template_replacements"] = changes


def relieve_added_sidechain_clashes(original, repaired, candidates):
    """Try rigid rotations about an observed anchor bond; final validation is separate."""
    from gmxbuilder.modules.input.geometry_validation import (
        added_atom_clashes,
        protein_geometry_issues,
    )
    from gmxbuilder.modules.input.protein_repair import _SIDECHAIN_PARENTS, _atom_key

    observed = {_atom_key(original, i) for i in range(original.num_atoms)}
    indices = {_atom_key(repaired, i): i for i in range(repaired.num_atoms)}
    changes = []
    # Ten-degree proposals sample a torsion, not a physical acceptance tolerance.
    # Two sweeps allow one repaired neighbour to move out of another's way.
    for _ in range(2):
        changed = False
        for key, missing in sorted(candidates.items()):
            added = [indices[(*key, atom)] for atom in missing]
            initial_score = len(added_atom_clashes(repaired, added))
            if initial_score == 0:
                continue
            parents = _SIDECHAIN_PARENTS[key[2]]
            roots = [atom for atom in missing if parents.get(atom) not in missing]
            anchors = {parents.get(atom) for atom in roots}
            if len(anchors) != 1:
                continue
            anchor = anchors.pop()
            parent = "CA" if anchor == "CB" else parents.get(anchor)
            if (*key, anchor) not in observed or (*key, parent) not in observed:
                continue
            origin = repaired.coordinates[indices[(*key, anchor)]].copy()
            axis = origin - repaired.coordinates[indices[(*key, parent)]]
            length = np.linalg.norm(axis)
            if not np.isfinite(length) or length < 1e-8:
                continue
            axis /= length
            start = repaired.coordinates[added].copy()
            vectors = start - origin
            best, score, best_angle = start, initial_score, 0
            residue_indices = [index for atom_key, index in indices.items() if atom_key[:3] == key]
            for degrees in range(10, 360, 10):
                angle = np.deg2rad(degrees)
                # Rodrigues rotation preserves bonds within the missing subtree.
                proposal = origin + (
                    vectors * np.cos(angle)
                    + np.cross(axis, vectors) * np.sin(angle)
                    + np.outer(vectors @ axis, axis) * (1 - np.cos(angle))
                )
                repaired.coordinates[added] = proposal
                proposal_score = len(added_atom_clashes(repaired, added))
                if proposal_score < score and not protein_geometry_issues(
                    repaired.take(residue_indices)
                ):
                    best, score, best_angle = proposal.copy(), proposal_score, degrees
                    if score == 0:
                        break
            repaired.coordinates[added] = best
            if best_angle:
                changes.append(
                    {
                        "chain": key[0],
                        "resid": key[1],
                        "resname": key[2],
                        "angle_degrees": best_angle,
                        "anchor_bond": [parent, anchor],
                    }
                )
                changed = True
        if not changed:
            break
    repaired.source_info["sidechain_rotamer_adjustments"] = changes

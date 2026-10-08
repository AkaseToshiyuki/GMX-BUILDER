"""Coordinate-independent chemical definitions for V4 trajectory observables."""

from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np

METHOD = "bond-graph-head-tail-1"
POLAR_ELEMENTS = frozenset({"N", "O", "P", "S"})
# Three covalent edges exclude the polar group and its first two carbon shells.
MIN_POLAR_BOND_DISTANCE = 3


def chemical_selection(name, names, elements, bonds):
    """Define one atom-name mapping for every frame, replica and continuation.

    Phosphorus anchors phospholipids. Other heads use the polar atom farthest
    from the hydrophobic region in graph distance; names break equivalent ties.
    Sterol heads use ring-bound oxygen, excluding side-chain oxysterol oxygen.
    This defines a molecular axis, not segmental S_CD or an experimental DHH.
    """
    names, elements = tuple(names), tuple(elements)
    if len(names) != len(elements) or len(set(names)) != len(names):
        raise ValueError("Observable selection requires unique chemical atom names")
    adjacency = [set() for _ in names]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    polar = [i for i, element in enumerate(elements) if element in POLAR_ELEMENTS]
    carbon = [i for i, element in enumerate(elements) if element == "C"]
    if not polar or len(carbon) < 3:
        raise ValueError(f"Cannot define a head and hydrophobic region for {name}")

    def distances(starts):
        result = np.full(len(names), len(names), dtype=int)
        queue = deque(starts)
        result[list(starts)] = 0
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if result[b] > result[a] + 1:
                    result[b] = result[a] + 1
                    queue.append(b)
        if np.any(result == len(names)):
            raise ValueError("Disconnected observable atom graph")
        return result

    distance_to_polar = distances(polar)
    tail = [i for i in carbon if distance_to_polar[i] >= MIN_POLAR_BOND_DISTANCE]
    if len(tail) < 3:
        raise ValueError(f"No resolved hydrophobic carbon region for {name}")
    head = polar
    if len(polar) <= 3 and len(carbon) >= 15 and all(elements[i] == "O" for i in polar):
        # Graph leaf pruning identifies cyclic carbon atoms without fitting bonds
        # to a fluctuating coordinate cutoff.
        cyclic = set(carbon)
        while True:
            leaves = {i for i in cyclic if len(adjacency[i] & cyclic) < 2}
            if not leaves:
                break
            cyclic -= leaves
        ring_oxygen = [i for i in polar if adjacency[i] & cyclic]
        if ring_oxygen:
            head = ring_oxygen
    phosphorus = [i for i in head if elements[i] == "P"]
    distance_to_tail = distances(tail)
    anchor = min(phosphorus or head, key=lambda i: (-distance_to_tail[i], names[i]))
    record = {
        "method": METHOD,
        "lipid": name,
        "atom_names": list(names),
        "elements": list(elements),
        "bonds": sorted([sorted([int(a), int(b)]) for a, b in bonds]),
        "polar_atom_names": [names[i] for i in head],
        "tail_atom_names": [names[i] for i in tail],
        "anchor_atom_name": names[anchor],
    }
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"polar": np.array(head), "tails": np.array(tail), "anchor": anchor, "record": record}


def validate_selections(analysis, protocol):
    """Reject missing, mutated or mixed definitions before comparing replicas."""
    if protocol.get("observable_selection_method") != METHOD:
        raise ValueError("Trajectory protocol lacks the current observable definition")
    records = analysis.get("atom_selections", [])
    if len(records) != len(protocol["composition"]) or {r["lipid"] for r in records} != set(
        protocol["composition"]
    ):
        raise ValueError("Missing chemical observable selections")
    for record in records:
        expected = chemical_selection(
            record["lipid"], record["atom_names"], record["elements"], record["bonds"]
        )["record"]
        if record != expected:
            raise ValueError("Changed chemical observable selection or version")


def selections_by_lipid(analysis, protocol):
    """Return validated definitions keyed by species, independent of record order."""
    validate_selections(analysis, protocol)
    return {record["lipid"]: record for record in analysis["atom_selections"]}

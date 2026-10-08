"""Exact append-only nearest neighbors with logarithmic tree rebuilds."""

import numpy as np
from scipy.spatial import cKDTree

from gmxbuilder.runtime.hardware import query_workers


class AppendOnlyNeighbors:
    """Merge equal-sized immutable blocks instead of rebuilding every prefix."""

    def __init__(self, **tree_options):
        self.blocks = []
        self.tree_options = tree_options

    def append(self, coordinates):
        points = np.array(coordinates, dtype=float, copy=True)
        weight = 1
        while self.blocks and self.blocks[-1][0] == weight:
            previous_weight, previous, _tree = self.blocks.pop()
            points = np.concatenate((previous, points))
            weight += previous_weight
        self.blocks.append((weight, points, cKDTree(points, **self.tree_options)))

    def distances(self, points):
        result = np.full(len(points), np.inf)
        workers = query_workers(len(points))
        for _weight, _coordinates, tree in self.blocks:
            np.minimum(result, tree.query(points, k=1, workers=workers)[0], out=result)
        return result

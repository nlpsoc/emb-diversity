from __future__ import annotations

from .._embed import accepts_text

### Graph-Based Diversity Measure

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform
import torch



@accepts_text
def mst_dispersion(data: TensorLike,
                   metric: DISTANCE_METRIC = "cosine"
                   ) -> float:
    # normalize input to numpy array
    if isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected shape (n, d), got {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute graph entropy for fewer than 2 datapoints")

    # now we create and adjacency matrix with a specified pairwise distance metric
    # by default its cosine distance
    dist_condensed = _compute_pairwise_distances(X, metric=metric)
    # need to convert it to square form
    # we use this as our adjacency matrix of a complete graph formed by all datapoints
    dist_square = squareform(dist_condensed)

    # we use the scipy minimum spanning tree implementation
    mst = minimum_spanning_tree(dist_square)

    # to obtain the mst dispersion
    # we sum the lengths of the edges required to connect all samples with the minimum total cost
    mst_dispersion = mst.sum()

    return float(mst_dispersion)

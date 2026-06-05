from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult, TensorLike
from .utils import _compute_pairwise_distances

### Graph-Based Diversity Measure


def mst_dispersion(data: TensorLike,
                   metric: DISTANCE_METRIC = "cosine",
                   *,
                   diversity_axis: str = "semantic",
                   embedding_model: str | None = None,
                   ) -> MeasureResult:
    """Compute the total edge weight of the minimum spanning tree.

    Args:
        data: Embedding vectors of shape (n, d), or raw text strings. Must
            contain at least 2 samples.
        metric: Distance metric for the complete-graph edge weights. Defaults to "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        total MST edge weight and ``parameters`` records the configuration used.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)

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

    return {
        "value": float(mst_dispersion),
        "parameters": {"metric": metric, "embedding_model": embedding_model},
    }

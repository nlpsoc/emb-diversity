from __future__ import annotations

import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult, TensorLike
from .utils import compute_pairwise_distances

### Graph-Based Diversity Measure


def mst_dispersion(data: TensorLike,
                   metric: DistanceMetric = "cosine",
                   *,
                   diversity_axis: str = "semantic",
                   embedding_model: str | None = None,
                   ) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0, grows with n; bounded by (n-1) times the largest edge under ``metric``.

    Compute the total edge weight of the minimum spanning tree (MST).

    1) Build a complete weighted graph: the weight of edge (i, j) is the
       pairwise distance d_ij under ``metric``.
    2) Compute the minimum spanning tree of that graph.
    3) Return the total weight of the MST edges.

    References:
        Cox, Samuel Rhys, et al. "Directed diversity: Leveraging language embedding distances for collective creativity in crowd ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021.
        Atwal, Tevin, Chan Nam Tieu, Yefeng Yuan, Zhan Shi, Yuhong Liu, and Liang Cheng. "Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs." arXiv preprint arXiv:2507.18055 (2025).

    Args:
        data:
            (Embedding) vectors of shape (n, d), or raw text strings. Must
            contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist, used as edge weights. Defaults to
            "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        total MST edge weight and ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is not 2D, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="mst_dispersion")

    # normalize input to numpy array; torch is checked via sys.modules so that
    # accepting tensor input does not force the (slow) torch import — if torch
    # was never imported, *data* cannot be a torch tensor.
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected shape (n, d), got {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute MST dispersion for fewer than 2 datapoints")

    # now we create and adjacency matrix with a specified pairwise distance metric
    # by default its cosine distance
    dist_condensed = compute_pairwise_distances(X, metric=metric)
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

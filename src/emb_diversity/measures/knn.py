from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult
from .utils import compute_pairwise_distances


def knn(
        data: Sequence[Sequence[float]],
        k: int = 2,
        metric: DistanceMetric = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        chunking_kwargs: dict | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; the upper bound depends on ``metric`` (e.g. [0, 2] for cosine distance).

    Compute the average k-th-nearest-neighbour distance across all datapoints.

    A generalisation of :func:`~emb_diversity.measures.chamfer_dist.chamfer_dist`
    (which is the special case ``k=1``): instead of each point's nearest
    neighbour, it looks at each point's *k*-th nearest neighbour.

    1) Compute all unique pairwise distances between datapoints.
    2) For each point, find the distance to its k-th nearest neighbour
       (excluding itself; ``k=1`` is the nearest neighbour, ``k=2`` the
       second-nearest, and so on).
    3) Return the mean of those k-th-nearest-neighbour distances.

    References:
        Yang, Yuming, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. "Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18530-18549. 2025.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least ``k + 1`` samples.
        k:
            Which nearest neighbour to use, counted from 1 (the closest
            other point). Defaults to 2 (the second-nearest neighbour).
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        mean k-th-nearest-neighbour distance (higher = more dispersed) and
        ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, ``k`` is not positive, or there
            are fewer than ``k + 1`` datapoints.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")

    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="knn", chunking_kwargs=chunking_kwargs)
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < k + 1:
        raise ValueError(f"KNN requires at least {k + 1} datapoints for k={k}, got {n}")

    # compute all pairwise distances
    dist_vec = compute_pairwise_distances(data, metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)

    # set the diagonal to inf, to force exclude j = i
    np.fill_diagonal(dist_mat, np.inf)

    # for each i, take the k-th smallest distance in the row (k=1 -> nearest)
    kth_dists = np.partition(dist_mat, k - 1, axis=1)[:, k - 1]

    # finally, take the average across all i
    return {
        "value": float(np.mean(kth_dists)),
        "parameters": {"k": k, "metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

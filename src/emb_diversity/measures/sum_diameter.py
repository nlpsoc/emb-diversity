from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def sum_diameter(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        normalize_by_n: bool = False,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    SumDiameter: for each sample x_i find its farthest other sample and sum these maxima:
        SumDiameter(X) = sum_i max_{j != i} d(x_i, x_j)

    Args:
        data: Iterable of vectors (n, d), or raw text strings.
        metric: Distance metric name or callable for scipy.spatial.distance.pdist.
        normalize_by_n: If True, return the average max distance (sum / n).
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs: Extra kwargs forwarded to pdist.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        sum (or average if normalized) of per-sample maximum distances and
        ``parameters`` records the configuration used.

    Raises:
        ValueError: If data is empty or contains fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 2:
        raise ValueError("SumDiameter requires at least 2 datapoints")

    # condensed pairwise distances -> square matrix
    condensed = _compute_pairwise_distances(X, metric=metric, **metric_kwargs)
    dist_mat = squareform(condensed)

    # exclude self-distance when taking per-row maxima
    np.fill_diagonal(dist_mat, -np.inf)
    max_per_row = np.max(dist_mat, axis=1)

    total = float(np.sum(max_per_row))
    if normalize_by_n:
        total = total / float(n)
    return {
        "value": float(total),
        "parameters": {
            "metric": metric,
            "normalize_by_n": normalize_by_n,
            "embedding_model": embedding_model,
            **metric_kwargs,
        },
    }

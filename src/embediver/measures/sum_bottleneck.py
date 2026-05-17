from __future__ import annotations

from typing import Any, Sequence

from .._accepts_text import accepts_text
from ._types import DISTANCE_METRIC

### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform



@accepts_text
def sum_bottleneck(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        normalize_by_n: bool = False,
        **metric_kwargs: Any
) -> float:
    """
    SumBottleneck: for each sample x_i find its nearest other sample and sum these minima:
        SumBottleneck(X) = sum_i min_{j != i} d(x_i, x_j)

    Args:
        data: Iterable of vectors (n, d).
        metric: Distance metric name or callable for scipy.spatial.distance.pdist.
        normalize_by_n: If True, return the average min distance (sum / n).
        **metric_kwargs: Extra kwargs forwarded to pdist.

    Returns:
        The sum (or average if normalized) of per-sample minimum distances to
        any other point (a.k.a. sum of nearest-neighbour distances).

    Raises:
        ValueError: If data is empty or contains fewer than 2 datapoints.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 2:
        raise ValueError("SumBottleneck requires at least 2 datapoints")

    # condensed pairwise distances -> square matrix
    condensed = _compute_pairwise_distances(X, metric=metric, **metric_kwargs)
    dist_mat = squareform(condensed)

    # exclude self-distance when taking per-row minima
    np.fill_diagonal(dist_mat, np.inf)
    min_per_row = np.min(dist_mat, axis=1)

    total = float(np.sum(min_per_row))
    if normalize_by_n:
        total = total / float(n)
    return float(total)

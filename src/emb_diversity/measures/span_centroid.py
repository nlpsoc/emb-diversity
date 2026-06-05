from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult

### Distance-Based Diversity Measure


def span_centroid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        percentile: float = 90.0,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    Span with Centroid diversity (Cox et al., 2021).
    Computes diversity as the specified percentile (90th by default) of distances from each
    datapoint to the dataset centroid.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d), or raw text strings.
        metric: Metric name or callable, as accepted by scipy.spatial.distance.cdist.
                Default is "cosine".
        percentile: Percentile value (0-100) to compute. Default is 90.0.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs: Extra keyword arguments passed to cdist.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        specified percentile of distances from datapoints to the centroid and
        ``parameters`` records the configuration used.

    Raises:
        ValueError: If data has wrong shape or fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute span_with_centroid for fewer than 2 datapoints")

    # Centroid μ = (1/n) * sum_i x_i, shape (1, d)
    centroid = X.mean(axis=0, keepdims=True)

    # Distances D_i = d(x_i, μ), shape (n, 1) → flatten to (n,)
    # cdist is called directly here (point→centroid), so it bypasses the NaN
    # guard in compute_pairwise_distances; replicate it. A NaN means a
    # degenerate input for this metric, e.g. a zero-norm (all-zero) vector
    # under cosine, where the distance divides by the vector norm.
    dists = cdist(X, centroid, metric=metric, **metric_kwargs).ravel()
    if np.isnan(dists).any():
        raise ValueError(
            f"distance computation produced NaN with metric={metric!r}; "
            "this usually means a degenerate input, e.g. a zero-norm "
            "(all-zero) vector under cosine distance"
        )

    # Span = Percentile_p(D)
    return {
        "value": float(np.percentile(dists, percentile)),
        "parameters": {
            "metric": metric,
            "percentile": percentile,
            "embedding_model": embedding_model,
            **metric_kwargs,
        },
    }

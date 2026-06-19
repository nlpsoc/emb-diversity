from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..embed import resolve_embeddings
from ..utility.validate import ensure_cosine_defined
from .types import DistanceMetric, MeasureResult

### Distance-Based Diversity Measure


def span_centroid(
        data: Sequence[Sequence[float]],
        metric: DistanceMetric = "cosine",
        percentile: float = 90.0,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        chunking_kwargs: dict | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; the upper bound depends on ``metric`` (e.g. [0, 2] for cosine distance).

    Compute Span with Centroid diversity: a percentile of the distances from
    each datapoint to the dataset centroid.

    1) Compute the centroid (mean) of all input vectors.
    2) Compute each point's distance to the centroid under ``metric``.
    3) Return the given ``percentile`` of those distances.

    References:
        Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. “Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation.” Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.cdist. Defaults to "cosine".
        percentile:
            Percentile (0–100) of the centroid distances to return. Defaults to 90.0.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to cdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        specified percentile of distances from datapoints to the centroid and
        ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is not 2D, empty, or has fewer than 2
            datapoints — or, under the cosine metric, if a datapoint or the
            centroid is the zero vector (cosine distance is undefined there).
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="span_centroid", chunking_kwargs=chunking_kwargs)
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute span_with_centroid for fewer than 2 datapoints")

    ensure_cosine_defined(X, metric)

    # Centroid μ = (1/n) * sum_i x_i, shape (1, d)
    centroid = X.mean(axis=0, keepdims=True)

    if metric == "cosine" and not np.linalg.norm(centroid):
        raise ValueError(
            "Cosine distance to the centroid is undefined: the centroid of "
            "this data is the zero vector (this happens for symmetric data, "
            "e.g. [[1, 0], [-1, 0]]). Use a different metric "
            "(e.g. metric='euclidean')."
        )

    # Distances D_i = d(x_i, μ), shape (n, 1) → flatten to (n,)
    dists = cdist(X, centroid, metric=metric, **metric_kwargs).ravel()

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

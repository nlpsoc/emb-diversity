from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult
from .utils import compute_pairwise_distances

### Geometry-Based Diversity Measure


def span_medoid(
        data: Sequence[Sequence[float]],
        metric: DistanceMetric = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        chunking_kwargs: dict | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; the upper bound depends on ``metric`` (e.g. [0, 2] for cosine distance).

    Compute Span with Medoid diversity: the mean distance from all datapoints to
    the medoid.

    1) Compute all pairwise distances between datapoints.
    2) Find the medoid: the point with the smallest sum of distances to all
       others.
    3) Return the mean distance from all points to the medoid.

    References:
        Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. “Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation.” Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        mean distance to the medoid and ``parameters`` records the configuration
        used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="span_medoid", chunking_kwargs=chunking_kwargs)
    # 1) pairwise distances (condensed) -> full matrix (n, n)
    dist_vec = compute_pairwise_distances(data, metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)  # symmetric, zeros on diagonal

    # sum of distances for each row
    row_sums = dist_mat.sum(axis=1)

    # 3) medoid = the row with the minimum sum of distances
    medoid_idx = int(np.argmin(row_sums))

    # 4) distances to the medoid, and take the average
    dists_to_medoid = dist_mat[:, medoid_idx]
    return {
        "value": float(np.mean(dists_to_medoid)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

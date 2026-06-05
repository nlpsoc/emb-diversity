from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Volume-Based Diversity Measure


def span_medoid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    Compute the Span with Medoid diversity measure (Cox et al., 2021).

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        mean distance to the medoid and ``parameters`` records the configuration
        used.

    Raises:
        ValueError:
            If data is empty or contains only one datapoint.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    # 1) pairwise distances (condensed) -> full matrix (n, n)
    dist_vec = _compute_pairwise_distances(data, metric, **metric_kwargs)
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

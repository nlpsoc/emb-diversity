from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def dist_dispersion(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    Compute the sum of all pairwise distances between datapoints using the helper function _compute_pairwise_distances.
    e.g., used in https://aclanthology.org/2022.coling-1.437.pdf

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d), or raw text strings.
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs: Extra keyword arguments passed to _compute_pairwise_distances.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        sum of all pairwise distances across all unique pairs and ``parameters``
        records the configuration used.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return {
        "value": float(np.sum(dists)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

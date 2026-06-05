from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def bottleneck(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """Compute the minimum pairwise distance in an embedding set.

    1) Compute all unique pairwise distances between datapoints.
    2) Return the smallest distance (the bottleneck distance).

    References:
        Mironov, Mikhail, and Liudmila Prokhorenkova. “Measuring Diversity: Axioms and Challenges.” arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
        Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. “How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules.” arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

    Args:
        data:
            Iterable/array-like of embedding vectors with shape (n, d), or raw
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
        minimum distance across all unique pairs and ``parameters`` records the
        configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return {
        "value": float(np.min(dists)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

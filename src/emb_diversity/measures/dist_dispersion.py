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
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0, grows with n; the upper bound depends on ``metric`` (e.g. <= n(n-1) for cosine distance).

    Compute the sum of all pairwise distances between datapoints.

    1) Compute all unique pairwise distances between datapoints.
    2) Return their sum.

    References:
        Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.
        Yang, Yuming, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. "Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric." In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 18530-18549. 2025.

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
        sum of all pairwise distances across all unique pairs and ``parameters``
        records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return {
        "value": float(np.sum(dists)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

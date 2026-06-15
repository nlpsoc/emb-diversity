from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def sum_diameter(
        data: Sequence[Sequence[float]],
        metric: DistanceMetric = "cosine",
        normalize_by_n: bool = False,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; grows with n unless ``normalize_by_n``; the upper bound depends on ``metric`` (e.g. <= 2n for cosine distance).

    Compute SumDiameter: the sum over samples of each sample's distance to its
    farthest other sample, SumDiameter(X) = sum_i max_{j != i} d(x_i, x_j).

    1) Compute all pairwise distances between datapoints.
    2) For each sample, take the distance to its farthest other sample.
    3) Return the sum of those per-sample maxima (or their average if
       ``normalize_by_n``).

    References:
        Mironov, Mikhail, and Liudmila Prokhorenkova. “Measuring Diversity: Axioms and Challenges.” arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
        Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. “How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules.” arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        normalize_by_n:
            If True, return the average per-sample maximum distance (sum / n)
            instead of the sum. Defaults to False.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        sum (or average if normalized) of per-sample maximum distances and
        ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
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

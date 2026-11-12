from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult
from .utils import compute_pairwise_distances


def mean_pw_dist(
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

    Compute the average of all pairwise distances between datapoints.

    1) Compute all unique pairwise distances between datapoints.
    2) Return their mean.

    References:
        Guy Tevet and Jonathan Berant. 2021. Evaluating the Evaluation of Diversity in Natural Language Generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 326–346, Online. Association for Computational Linguistics.
        Tianhui Zhang, Bei Peng, and Danushka Bollegala. 2024. Improving Diversity of Commonsense Generation by Large Language Models via In-Context Learning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9226–9242, Miami, Florida, USA. Association for Computational Linguistics.
        Miranda, Brando, Alycia Lee, Sudharsan Sundar, Allison Casasola, Rylan Schaeffer, Elyas Obbad, and Sanmi Koyejo. "Beyond scale: The diversity coefficient as a data quality metric for variability in natural language data." arXiv preprint arXiv:2306.13840 (2023).
        Cox, Samuel Rhys, et al. "Directed diversity: Leveraging language embedding distances for collective creativity in crowd ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021.
        
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
        average pairwise distance across all unique pairs and ``parameters``
        records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="mean_pw_dist", chunking_kwargs=chunking_kwargs)
    dists = compute_pairwise_distances(data, metric, **metric_kwargs)
    return {
        "value": float(np.mean(dists)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }

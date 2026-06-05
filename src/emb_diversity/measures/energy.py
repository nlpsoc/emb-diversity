from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def energy(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        gamma: float = 1.0,
        epsilon: float = 1e-12,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    Implements the energy-based diversity metric for a set of vector representations,
    as described in Velikonivtsev et al., NeurIPS 2024.
    When gamma is set to 1, this  can be interpreted as the average
    pairwise energy for a system of equally charged particles.
    Because of the multiplication by -1, the value is larger for
    more diverse datasets.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d), or raw text strings.
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        gamma: The exponent parameter in the energy calculation, default is 1.0 (as in the paper)
        epsilon: Ensures that each distance is at least epsilon for numerical stability
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        "energy" of a dataset and ``parameters`` records the configuration used.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    # The metric can blow up when the distance is 0 (e.g., duplicates, or vectors
    # pointing in the same direction). Add a small constant epsilon to
    # entries that are 0 or very small
    dists = np.maximum(dists, epsilon)
    return {
        "value": -float((1.0 / (dists ** gamma)).mean()),
        "parameters": {
            "metric": metric,
            "gamma": gamma,
            "epsilon": epsilon,
            "embedding_model": embedding_model,
            **metric_kwargs,
        },
    }

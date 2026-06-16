from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..embed import resolve_embeddings
from .types import DistanceMetric, MeasureResult
from .utils import compute_pairwise_distances

### Distance-Based Diversity Measure


def energy(
        data: Sequence[Sequence[float]],
        metric: DistanceMetric = "cosine",
        gamma: float = 1.0,
        epsilon: float = 1e-12,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value (closer to 0) = more diverse.
    **Range:** (-inf, 0]; always <= 0.

    Compute the energy-based diversity of a set of vector representations.

    1) Compute all unique pairwise distances between datapoints (floored at
       ``epsilon`` for numerical stability).
    2) Raise each distance to the power ``gamma`` and take its reciprocal (the
       pairwise energy).
    3) Return the negative mean of these pairwise energies.

    References:
        Velikonivtsev, Fedor, Mikhail Mironov, and Liudmila Prokhorenkova. "Challenges of generating structurally diverse graphs." Advances in Neural Information Processing Systems 37 (2024): 57993-58022.
        Mironov, Mikhail, and Liudmila Prokhorenkova. “Measuring Diversity: Axioms and Challenges.” arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        gamma:
            Exponent applied to each pairwise distance. Defaults to 1.0 (as in
            the paper).
        epsilon:
            Lower bound applied to each distance, so zero distances (e.g.
            duplicates) do not blow up the reciprocal. Defaults to 1e-12.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        energy of the dataset and ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.

    Note:
        When ``gamma`` is 1, the value can be interpreted as the average
        pairwise energy of a system of equally charged particles. The result is
        multiplied by -1 so that larger values correspond to more diverse
        datasets.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    dists = compute_pairwise_distances(data, metric, **metric_kwargs)
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

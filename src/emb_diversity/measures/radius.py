from __future__ import annotations

from typing import Sequence

import numpy as np

from ..embed import resolve_embeddings
from ._types import MeasureResult

### Volume-Based Diversity Measure


def radius(
        data: Sequence[Sequence[float]],
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
) -> MeasureResult:
    """
    Compute diversity as the geometric mean of per-dimension standard deviations,
    following Lai et al. (2020) "Diversity, Density, and Homogeneity:
    Quantitative Characteristic Metrics for Text Collections".

    Formula:
        M_diversity = (σ1 * σ2 * ... * σH)^(1/H)
    where σi is the standard deviation along dimension i.

    Args:
        data: Iterable of embedding vectors (lists/tuples/np.ndarrays), shape
            (n, d), or raw text strings.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        geometric mean of standard deviations across all embedding dimensions
        (higher = higher dispersion) and ``parameters`` records the
        configuration used.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    X = np.asarray(data, dtype=float)
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute radius diversity for fewer than 2 datapoints")

    # Standard deviation along each embedding dimension
    stds = np.std(X, axis=0, ddof=1)  # unbiased estimator

    # Avoid log(0) for degenerate dimensions (replace 0 with eps)
    stds = np.clip(stds, a_min=1e-12, a_max=None)

    # Geometric mean of stds across all dimensions
    geom_mean = float(np.exp(np.mean(np.log(stds))))

    return {
        "value": geom_mean,
        "parameters": {"embedding_model": embedding_model},
    }

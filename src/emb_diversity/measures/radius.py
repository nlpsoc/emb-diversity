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
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0, unbounded above.

    Compute diversity as the geometric mean of the per-dimension standard
    deviations along each (embedding) dimension.

    1) Compute the standard deviation σi along each (embedding) dimension i.
    2) Return their geometric mean: (σ1 * σ2 * ... * σH) ** (1/H).

    References:
        Lai, Yi-An, et al. "Diversity, density, and homogeneity: Quantitative characteristic metrics for text collections." Proceedings of the Twelfth Language Resources and Evaluation Conference. 2020.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        geometric mean of standard deviations across all embedding dimensions
        and ``parameters`` records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.

    Note:
        Because this is a geometric mean, it is very sensitive to low-variance
        dimensions: a single near-constant dimension (std clipped to 1e-12 to
        avoid log(0)) drags the whole value toward 0, even if every other
        dimension is well spread. The value also scales with the magnitude of
        the input vectors, so it is not comparable across differently scaled
        embeddings.
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

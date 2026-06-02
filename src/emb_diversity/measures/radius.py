from __future__ import annotations

from typing import Sequence

from .._accepts_text import accepts_text

### Volume-Based Diversity Measure

import numpy as np



@accepts_text
def radius(
        data: Sequence[Sequence[float]]
) -> float:
    """
    Compute diversity as the geometric mean of per-dimension standard deviations,
    following Lai et al. (2020) "Diversity, Density, and Homogeneity:
    Quantitative Characteristic Metrics for Text Collections".

    Formula:
        M_diversity = (σ1 * σ2 * ... * σH)^(1/H)
    where σi is the standard deviation along dimension i.

    Args:
        data: Iterable of embedding vectors (lists/tuples/np.ndarrays), shape (n, d).

    Returns:
        The geometric mean of standard deviations across all embedding dimensions.
        Higher values indicate higher dispersion (diversity).
        Returns 0.0 if fewer than 2 datapoints or zero variance in all dims.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
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

    return geom_mean

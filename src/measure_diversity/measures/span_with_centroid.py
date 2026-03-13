### Distance-Based Diversity Measure

import numpy as np
from scipy.spatial.distance import cdist


def span_with_centroid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        percentile: float = 90.0,
        **metric_kwargs: Any,
) -> float:
    """
    Span with Centroid diversity (Cox et al., 2021).
    Computes diversity as the specified percentile (90th by default) of distances from each
    datapoint to the dataset centroid.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.cdist.
                Default is "cosine".
        percentile: Percentile value (0-100) to compute. Default is 90.0.
        **metric_kwargs: Extra keyword arguments passed to cdist.

    Returns:
        The specified percentile of distances from datapoints to the centroid.
        Higher values indicate higher diversity/spread.

    Raises:
        ValueError: If data has wrong shape or fewer than 2 datapoints.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute span_with_centroid for fewer than 2 datapoints")

    # Centroid μ = (1/n) * sum_i x_i, shape (1, d)
    centroid = X.mean(axis=0, keepdims=True)

    # Distances D_i = d(x_i, μ), shape (n, 1) → flatten to (n,)
    dists = cdist(X, centroid, metric=metric, **metric_kwargs).ravel()

    # Span = Percentile_p(D)
    return float(np.percentile(dists, percentile))
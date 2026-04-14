from __future__ import annotations

from .._accepts_text import accepts_text

### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances



@accepts_text
def mean_pw_dist(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Compute the average pairwise distance between all datapoints using the helper function _compute_pairwise_distances.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The average pairwise distance across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.mean(dists))

from __future__ import annotations

from .._accepts_text import accepts_text
from .._types import DISTANCE_METRIC

### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances



@accepts_text
def dist_dispersion(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Compute the sum of all pairwise distances between datapoints using the helper function _compute_pairwise_distances.
    e.g., used in https://aclanthology.org/2022.coling-1.437.pdf

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to _compute_pairwise_distances.

    Returns:
        The sum of all pairwise distances across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.sum(dists))

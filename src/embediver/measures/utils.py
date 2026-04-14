from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from ._types import DISTANCE_METRIC


def _compute_pairwise_distances(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> np.ndarray:
    """
    Helper function to compute all pairwise distances, using scipy.spatial.distance.pdist
   (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)

    Returns:
        Array of pairwise distances.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot compute distances for empty data")
    if n == 1:
        raise ValueError("Cannot compute distances for single data point")

    return pdist(X, metric=metric, **metric_kwargs)

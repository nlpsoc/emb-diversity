from __future__ import annotations

from typing import Any, Sequence

from .._accepts_text import accepts_text
from ._types import DISTANCE_METRIC

### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances



@accepts_text
def energy(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        gamma: float = 1.0,
        epsilon: float = 1e-12,
        **metric_kwargs: Any
) -> float:
    """
    Implements the energy-based diversity metric for a set of vector representations,
    as described in Velikonivtsev et al., NeurIPS 2024.
    When gamma is set to 1, this  can be interpreted as the average
    pairwise energy for a system of equally charged particles.
    Because of the multiplication by -1, the value is larger for
    more diverse datasets.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        gamma: The exponent parameter in the energy calculation, default is 1.0 (as in the paper)
        epsilon: Ensures that each distance is at least epsilon for numerical stability
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The "energy" of a dataset.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    # The metric can blow up when the distance is 0 (e.g., duplicates, or vectors
    # pointing in the same direction). Add a small constant epsilon to
    # entries that are 0 or very small
    dists = np.maximum(dists, epsilon)
    return -float((1.0 / (dists ** gamma)).mean())

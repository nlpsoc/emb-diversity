from typing import Any, Callable, Sequence, Union

import numpy as np

from ..compute_pairwise import compute_pairwise_distances

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]


def _compute_pairwise_distances(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any,
) -> np.ndarray:
    """
    Helper function to compute all pairwise distances, delegating to the cached
    `compute_pairwise_distances` module so repeated calls on the same embedding
    matrix reuse the previous result instead of re-running scipy.pdist.

    Returns:
        Condensed distance array.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    return compute_pairwise_distances(data, metric=metric, **metric_kwargs)

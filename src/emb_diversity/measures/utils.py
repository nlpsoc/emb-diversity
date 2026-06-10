from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..compute_pairwise import compute_pairwise_distances
from ._types import DISTANCE_METRIC


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
        ValueError: If data is not numeric (strings are rejected), empty, a
            single datapoint, not 2-dimensional, or contains all-zero
            vectors while ``metric`` is ``"cosine"``.
    """
    return compute_pairwise_distances(data, metric=metric, **metric_kwargs)

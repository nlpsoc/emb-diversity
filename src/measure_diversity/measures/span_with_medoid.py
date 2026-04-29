### Volume-Based Diversity Measure

import numpy as np
from scipy.spatial.distance import squareform

from .utils import _compute_pairwise_distances


def span_with_medoid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any,
) -> float:
    """
    Compute the Span with Medoid diversity measure (Cox et al., 2021).

    Raises:
        ValueError:
            If data is empty or contains only one datapoint.
    """
    # 1) pairwise distances (condensed) -> full matrix (n, n)
    dist_vec = _compute_pairwise_distances(data, metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)  # symmetric, zeros on diagonal

    # sum of distances for each row
    row_sums = dist_mat.sum(axis=1)

    # 3) medoid = the row with the minimum sum of distances
    medoid_idx = int(np.argmin(row_sums))

    # 4) distances to the medoid, and take the average
    dists_to_medoid = dist_mat[:, medoid_idx]
    return float(np.mean(dists_to_medoid))

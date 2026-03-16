### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform


def chamfer_distance_diversity(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Chamfer-distance-based diversity: for each point, take the distance
    to its nearest neighbour (excluding itself), then average over points.

    This implements:
        Chamfer(X) = (1/n) * sum_i min_{j != i} d_ij

    Args:
        data: Iterable of vectors, shape (n, d).
        metric: Distance metric as in scipy.spatial.distance.pdist.
        **metric_kwargs: Extra keyword arguments passed to pdist.

    Returns:
        The average nearest-neighbour distance. Higher values indicate
        more dispersed datasets.

    Raises:
        ValueError: If there are fewer than 2 datapoints.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Cannot compute chamfer distance for fewer than 2 datapoints")

    # compute all pairwise distances
    dist_vec = _compute_pairwise_distances(data, metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)

    # set the diagonal to inf, to force exclude j = i
    np.fill_diagonal(dist_mat, np.inf)

    # for each i, take the minimum distance in the row min_{j≠i} d_ij
    min_dists = np.min(dist_mat, axis=1)

    # finally, take the average of all i (1/n) ∑_i min_{j≠i} d_ij
    return float(np.mean(min_dists))

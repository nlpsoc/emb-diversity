from __future__ import annotations

from .._accepts_text import accepts_text

### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform



@accepts_text
def chamfer_dist(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """Compute the average nearest-neighbour distance across all datapoints.

    1) Compute all unique pairwise distances between datapoints.
    2) For each point, find the minimum distance to any other point (excluding itself).
    3) Return the mean of those nearest-neighbour distances.

    References:
        Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. “Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation.” Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.
        Zhang, Tianhui, Bei Peng, and Danushka Bollegala. “Evaluating the Evaluation of Diversity in Commonsense Generation.” In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), edited by Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar. Association for Computational Linguistics, 2025. https://aclanthology.org/2025.acl-long.1181/.

    Args:
        data:
            Iterable/array-like of embedding vectors with shape (n, d).
            Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        float: Mean nearest-neighbour distance. Higher values indicate more dispersed datasets.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
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

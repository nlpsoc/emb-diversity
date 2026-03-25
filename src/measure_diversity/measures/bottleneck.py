### Distance-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances


def bottleneck(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """Compute the minimum pairwise distance in an embedding set.

    1) Computes all unique pairwise distances between datapoints 
    2) Returns the smallest distance (the bottleneck distance)

    References:
    Mironov, Mikhail, and Liudmila Prokhorenkova. “Measuring Diversity: Axioms and Challenges.” arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
    Xie, Yutong, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. “How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules.” arXiv:2112.12542. Preprint, arXiv, March 6, 2023. https://doi.org/10.48550/arXiv.2112.12542.

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
        float: Minimum distance across all unique pairs.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.min(dists))

from __future__ import annotations

from typing import Sequence

from .._accepts_text import accepts_text

### Distance-Based Diversity Measure

import numpy as np
from sklearn.cluster import KMeans



@accepts_text
def cluster_inertia(
        data: Sequence[Sequence[float]],
        n_clusters: int = 200 
) -> float:
    """Compute diversity as k-means inertia over embedding vectors.

    1) Fit k-means with n clusters on the data.
    2) Return the inertia (sum of squared Euclidean distances to cluster centres).

    References:
        Yang, Yuming, Yang Nan, Junjie Ye, et al. “Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric.” arXiv:2502.17184. Preprint, arXiv, February 28, 2025. https://doi.org/10.48550/arXiv.2502.17184.
        Du, Wenchao, and Alan W. Black. “Boosting Dialog Response Generation.” In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, edited by Anna Korhonen, David Traum, and Lluís Màrquez. Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/P19-1005.

    Args:
        data:
            Iterable/array-like of embedding vectors with shape (n, d).
            Must contain at least 2 samples.
        n_clusters:
            Number of clusters for k-means. 
            For 10k samples, 200 clusters is a common default in the literature (e.g. Yang et al 2025)
            Automatically reduced to n-1 if fewer datapoints than clusters are provided.

    Returns:
        float: Sum of squared Euclidean distances from each point to its assigned cluster centre. 
        Higher values indicate greater spread/diversity.

    Raises:
        ValueError: If data is empty or contains fewer than 2 datapoints.
    """

    X = np.asarray(data, dtype=float)
    if X.size == 0:
        raise ValueError("Cannot compute cluster inertia for empty data")

    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute cluster centers and thus inertia for fewer than 2 datapoints")

    # Adjust number of clusters only if fewer datapoints than requested clusters
    actual_clusters = n - 1 if n < n_clusters else n_clusters

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42) # , n_init=10 is a value to determine how many times the k-means algorithm will be run with different centroid seeds
    kmeans.fit(X)
    return float(kmeans.inertia_)

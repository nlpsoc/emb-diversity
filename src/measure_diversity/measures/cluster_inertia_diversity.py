### Distance-Based Diversity Measure

import numpy as np
from sklearn.cluster import KMeans


def cluster_inertia_diversity(
        data: Sequence[Sequence[float]],
        n_clusters: int = 10
) -> float:
    """
    Compute diversity as the inertia (sum of squared distances to cluster centers)
    from k-means clustering using Euclidean distance.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        n_clusters: Number of clusters for k-means. Automatically reduced if
                   fewer datapoints than clusters.

    Returns:
        The k-means inertia (sum of squared Euclidean distances to cluster centers).
        Higher values indicate higher diversity/spread.
        Returns 0.0 if there are fewer than 2 datapoints.

    Raises:
        ValueError: If data is empty.
    """

    if not data:
        raise ValueError("Cannot compute cluster inertia for empty data")

    X = np.asarray(data, dtype=float)
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute cluster centers and thus inertia for fewer than 2 datapoints")

    # Adjust number of clusters if we have fewer points
    actual_clusters = min(n_clusters, n - 1) # TODO @Tao can you check what ppl have done here?

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42) # , n_init=10 is a value to determine how many times the k-means algorithm will be run with different centroid seeds
    kmeans.fit(X)
    return float(kmeans.inertia_)

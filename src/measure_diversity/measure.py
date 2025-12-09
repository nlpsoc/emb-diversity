"""
    Diversity measures based on vector representations of data.
"""
from collections import Counter
from typing import List, Iterable, Any, Sequence, Union, Callable
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
### Distance-Based Diversity Measure

try:
    from vendi_score import vendi
    _HAS_VENDI = True
except ImportError:  
    _HAS_VENDI = False

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]


def _compute_pairwise_distances(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> np.ndarray:
    """
    Helper function to compute all pairwise distances.

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


def mean_pairwise_distance(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Compute the average pairwise distance between all datapoints using
    scipy.spatial.distance.pdist
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The average pairwise distance across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.mean(dists))


def distance_dispersion(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Compute the sum of all pairwise distances between datapoints using
    scipy.spatial.distance.pdist.
    e.g., used in https://aclanthology.org/2022.coling-1.437.pdf

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The sum of all pairwise distances across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.sum(dists))


def diameter(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Find the largest distance between any two instances in the embedding space
    of a dataset.
    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The largest distance among all pairwise distances across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.max(dists))


def bottleneck(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any
) -> float:
    """
    Find the smallest distance between any two instances in the embedding space
    of a dataset.
    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.pdist
                Default is "cosine".
        **metric_kwargs: Extra keyword arguments passed to pdist. (as in scipy docs)

    Returns:
        The minimum distance among all pairwise distances across all unique pairs.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return float(np.min(dists))


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
    from sklearn.cluster import KMeans

    if not data:
        raise ValueError("Cannot compute cluster inertia for empty data")

    X = np.asarray(data, dtype=float)
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute cluster centers and thus inertia for fewer than 2 datapoints")

    # Adjust number of clusters if we have fewer points
    actual_clusters = min(n_clusters, n - 1) # @Tao can you check what ppl have done here?

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42) # , n_init=10 is a value to determine how many times the k-means algorithm will be run with different centroid seeds
    kmeans.fit(X)
    return float(kmeans.inertia_)

def span_with_centroid(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        percentile: float = 90.0,
        **metric_kwargs: Any,
) -> float:
    """
    Span with Centroid diversity (Cox et al., 2021).
    Computes diversity as the specified percentile (90th by default) of distances from each
    datapoint to the dataset centroid.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric: Metric name or callable, as accepted by scipy.spatial.distance.cdist.
                Default is "cosine".
        percentile: Percentile value (0-100) to compute. Default is 90.0.
        **metric_kwargs: Extra keyword arguments passed to cdist.

    Returns:
        The specified percentile of distances from datapoints to the centroid.
        Higher values indicate higher diversity/spread.

    Raises:
        ValueError: If data has wrong shape or fewer than 2 datapoints.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute span_with_centroid for fewer than 2 datapoints")

    # Centroid μ = (1/n) * sum_i x_i, shape (1, d)
    centroid = X.mean(axis=0, keepdims=True)

    # Distances D_i = d(x_i, μ), shape (n, 1) → flatten to (n,)
    dists = cdist(X, centroid, metric=metric, **metric_kwargs).ravel()

    # Span = Percentile_p(D)
    return float(np.percentile(dists, percentile))
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

### Volume-Based Diversity Measure

def convex_hull_volume(
        data: Sequence[Sequence[float]]
) -> float:
    """
    Compute diversity as the volume (or area in 2D) of the convex hull (smallest convex shape that contains all points)
    formed by the datapoints.
    Used in: https://aclanthology.org/2022.coling-1.437/

    Attention: This function is not normed, i.e., it does not return a value in [0, 1].

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).

    Returns:
        The volume/area of the convex hull.
        Returns 0.0 if there are fewer than d+1 datapoints or if points are collinear.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Cannot compute convex hull for empty data")

    X = np.asarray(data, dtype=float)
    n, d = X.shape

    # Need at least d+1 points to form a non-degenerate convex hull
    if n < d + 1:
        raise ValueError(f"Cannot compute convex hull for fewer than dimension+1 {d + 1} points (got {n})")

    try:
        hull = ConvexHull(X)
        return float(hull.volume)
    except (ValueError, RuntimeError) as e:
        # Points are collinear/coplanar - hull has zero volume
        return 0.0

def radius_diversity(
        data: Sequence[Sequence[float]]
) -> float:
    """
    Compute diversity as the geometric mean of per-dimension standard deviations,
    following Lai et al. (2020) "Diversity, Density, and Homogeneity: 
    Quantitative Characteristic Metrics for Text Collections".

    Formula:
        M_diversity = (σ1 * σ2 * ... * σH)^(1/H)
    where σi is the standard deviation along dimension i.

    Args:
        data: Iterable of embedding vectors (lists/tuples/np.ndarrays), shape (n, d).

    Returns:
        The geometric mean of standard deviations across all embedding dimensions.
        Higher values indicate higher dispersion (diversity).
        Returns 0.0 if fewer than 2 datapoints or zero variance in all dims.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    X = np.asarray(data, dtype=float)
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute radius diversity for fewer than 2 datapoints")

    # Standard deviation along each embedding dimension
    stds = np.std(X, axis=0, ddof=1)  # unbiased estimator

    # Avoid log(0) for degenerate dimensions (replace 0 with eps)
    stds = np.clip(stds, a_min=1e-12, a_max=None)

    # Geometric mean of stds across all dimensions
    geom_mean = float(np.exp(np.mean(np.log(stds))))

    return geom_mean

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
    X = np.asarray(data, dtype=float)
    n = X.shape[0]

    if n == 0:
        raise ValueError("Cannot compute span_with_medoid for empty data")
    if n == 1:
        raise ValueError("Cannot compute span_with_medoid for a single datapoint")

    # 1) pairwise distances (condensed) -> full matrix (n, n)
    dist_vec = pdist(X, metric=metric, **metric_kwargs)
    dist_mat = squareform(dist_vec)  # symmetric, zeros on diagonal

    # sum of distances for each row
    row_sums = dist_mat.sum(axis=1)

    # 3) medoid = the row with the minimum sum of distances
    medoid_idx = int(np.argmin(row_sums))

    # 4) distances to the medoid, and take the average
    dists_to_medoid = dist_mat[:, medoid_idx]
    return float(np.mean(dists_to_medoid))

### Graph-Based Diversity Measure
def graph_entropy(data: Sequence[Sequence[float]]) -> float:
    """
    Compute Graph Entropy diversity measure as in Yu et al. (2022):
    - Treat each embedding as a node in a fully connected graph.
    - Edge weights are cosine distances d_ij = 1 - cos(x_i, x_j).
    - For each node i, define f_i(d_ij) = d_ij / sum_k d_ik.
    - Local entropy: I(x_i) = - sum_j f_i(d_ij) log f_i(d_ij).
    - GraphEntropy(X) = sum_i I(x_i).

    Args:
        data: Iterable of embedding vectors (lists/tuples/np.ndarrays), shape (n, d).

    Returns:
        The graph entropy of the dataset. Higher values indicate higher diversity.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    X = np.asarray(data, dtype=float)
    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute graph entropy for fewer than 2 datapoints")

    # normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    X_norm = X / norms

    sim = X_norm @ X_norm.T          # (n, n)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    row_sums = dist.sum(axis=1, keepdims=True)  # (n, 1)

    # 用 where 做安全除法，避免 bool index 形状不匹配
    p = np.divide(dist, row_sums, out=np.zeros_like(dist), where=row_sums > 0)

    p_safe = np.clip(p, 1e-12, 1.0)
    local_entropies = -np.sum(p * np.log(p_safe), axis=1)

    return float(np.sum(local_entropies))


### Distribution-Based Diversity Measure

def vendi_score_diversity(
        data: Sequence[Sequence[float]],
        q: float = 1.0,
        normalize: bool = True,
        use_dual: bool = True,
) -> float:
    """
    Compute diversity using the Vendi Score (Friedman & Dieng, 2023).

    The Vendi Score takes a set of samples and a similarity function/kernel,
    and returns an "effective number of unique elements" based on the
    von Neumann / Shannon entropy of the similarity matrix eigenvalues.

    Here we provide a wrapper around the official vendi_score implementation
    (https://github.com/vertaix/Vendi-Score) for embedding data.

    Args:
        data:
            Iterable of embedding vectors, shape (n, d).
        q:
            Order of the Vendi score (Renyi-style generalization).
            q = 1.0 corresponds to the original Vendi Score.
        normalize:
            Whether to L2-normalize rows of X when using dot-product similarity.
            For normalized embeddings, the dot product corresponds to cosine similarity.
        use_dual:
            If True, use vendi.score_dual(X, ...) which is efficient when d < n.
            If False, build a Gram matrix K and call vendi.score_K(K, ...).

    Returns:
        The Vendi Score as a float. Higher values indicate higher diversity.

    Raises:
        ImportError:
            If vendi_score is not installed.
        ValueError:
            If there are fewer than 2 datapoints.
    """
    if not _HAS_VENDI:
        raise ImportError(
            "vendi_score is not installed. Please `pip install vendi_score` "
            "to use vendi_score_diversity."
        )

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute Vendi Score for fewer than 2 datapoints")

    # Case 1: use dual formulation (recommended when d <= n, or in general for embeddings)
    if use_dual:
        # vendi.score_dual handles normalization internally
        return float(vendi.score_dual(X, q=q, normalize=normalize))

    # Case 2: explicitly build similarity matrix K and call score_K
    # Here we use (normalized) dot product as similarity.
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        X_norm = X / norms
    else:
        X_norm = X

    K = X_norm @ X_norm.T  # Gram matrix of similarities
    return float(vendi.score_K(K, q=q))


#### Count Based Diversity Measure
def dummy_diversity(data: List[List[Any]] | Iterable[Iterable[Any]]) -> float:
    """
    Diversity = 1 - (count of the most common row / total number of rows)

    Args:
        data: 2D data, e.g. [[1, 0], [1, 1], ...]

    Returns:
        A float in [0, 1]. Returns 0.0 for empty on one element inputs.
    """
    # materialize once and allow general iterables
    rows = [tuple(row) for row in data]
    if not rows:
        return 0.0
    counts = Counter(rows)
    most_common_count = counts.most_common(1)[0][1]
    return 1 - (most_common_count / len(rows))
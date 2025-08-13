from collections import Counter
from typing import List, Iterable, Any
from typing import Sequence, Union, Callable, Any
import numpy as np
from scipy.spatial.distance import pdist

### Distance Based Diversity Measure

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]

def pairwise_diversity(
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
        Returns 0.0 if there are fewer than 2 datapoints.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 2:
        return 0.0

    dists = pdist(X, metric=metric, **metric_kwargs)
    return float(np.mean(dists))


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
from __future__ import annotations

from .._embed import accepts_text

### Volume-Based Diversity Measure

import numpy as np
from scipy.spatial import ConvexHull



@accepts_text
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

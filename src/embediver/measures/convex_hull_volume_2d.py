from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .._accepts_text import accepts_text

### Volume-Based Diversity Measure (2D-projected)


@accepts_text
def convex_hull_volume_2d(
        data: Sequence[Sequence[float]],
        random_state: int = 42,
) -> float:
    """
    Compute diversity as the area of the convex hull of the data, after first
    projecting the input to 2 dimensions.

    Used in: https://aclanthology.org/2022.coling-1.437/

    Why project to 2D first
    -----------------------
    scipy's ``ConvexHull`` (Qhull) becomes intractable past ~10 dimensions:
    the number of facets grows as O(n^floor(d/2)). For typical embedding
    dimensions (e.g. 4096), the raw convex hull is both infeasible to compute
    and numerically meaningless (the "volume" is a product of many small
    lengths and underflows). Projecting to 2D yields a stable, comparable
    "area" measure.

    Dimension reduction
    -------------------
    * If the input already has 2 columns, it is used as-is (no projection).
    * Otherwise the input is projected with
      ``umap.UMAP(n_components=2, random_state=random_state)``.
    * If ``umap-learn`` is not installed, or if UMAP fails to fit (e.g. on
      very small inputs where ``n_neighbors`` exceeds ``n``), the function
      falls back to taking the first 2 columns of the input and emits a
      ``UserWarning``. This fallback is only sensible for toy data — for
      real embeddings, install ``umap-learn``.

    Comparability caveat
    --------------------
    The returned value is an area in the UMAP-projected space. UMAP is
    non-linear and its scale is data-dependent, so values are NOT comparable
    across separate UMAP fits (i.e. across different datasets, or across
    runs with different ``random_state``). Fix ``random_state`` and compare
    within a single experiment.

    Attention: This function is not normed, i.e., it does not return a value in [0, 1].

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        random_state: Seed passed to UMAP. Defaults to 42.

    Returns:
        The area of the 2D convex hull of the projected points.
        Returns 0.0 if the projected points are collinear.

    Raises:
        ValueError: If data is empty or has fewer than 3 points.
    """
    if not data:
        raise ValueError("Cannot compute convex hull for empty data")

    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n < 3:
        raise ValueError(
            f"Cannot compute 2D convex hull for fewer than 3 points (got {n})"
        )

    two_d = _reduce_to_2d(X, random_state=random_state)

    try:
        hull = ConvexHull(two_d)
        # In 2D, scipy's ConvexHull.volume returns the area.
        return float(hull.volume)
    except (ValueError, RuntimeError):
        # Projected points are collinear — hull has zero area.
        return 0.0


def _reduce_to_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project X to 2D via UMAP; fall back to the first 2 columns on failure."""
    if X.shape[1] == 2:
        return X

    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    except Exception as e:
        warnings.warn(
            f"UMAP reduction to 2D failed ({type(e).__name__}: {e}); "
            "falling back to the first 2 columns of the input. "
            "This fallback is only meaningful for toy data — "
            "for real embeddings, install umap-learn.",
            stacklevel=3,
        )
        if X.shape[1] >= 2:
            return X[:, :2]
        pad = np.zeros((X.shape[0], 2 - X.shape[1]))
        return np.concatenate([X, pad], axis=1)

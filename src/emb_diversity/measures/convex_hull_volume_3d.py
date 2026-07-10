from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from ..embed import resolve_embeddings
from .types import MeasureResult


def convex_hull_volume_3d(
        data: Sequence[Sequence[float]],
        random_state: int = 42,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        chunking_kwargs: dict | None = None,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0, unbounded above (0 if the projected points are coplanar).

    Compute diversity as the volume of the convex hull of the data, after first
    projecting the input to 3 dimensions.

    1) Project the input to 3D (UMAP, or use as-is if already 3D).
    2) Compute the convex hull of the projected points.
    3) Return its volume (0.0 if the points are coplanar).


    References:
        Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.

    Args:
        data: Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d), or raw text strings.
        random_state: Seed passed to UMAP. Defaults to 42.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        volume of the 3D convex hull of the projected points (0.0 if coplanar)
        and ``parameters`` records the configuration used. This value is not
        normalized (it is not in [0, 1]).

    Raises:
        ValueError: If data is empty or has fewer than 4 points.

    Note:
        **Why project to 3D first:** scipy's ``ConvexHull`` (Qhull) becomes
        intractable past ~10 dimensions: the number of facets grows as
        O(n^floor(d/2)). For typical embedding dimensions (e.g. 4096), the raw
        convex hull is both infeasible to compute and numerically meaningless
        (the "volume" is a product of many small lengths and underflows).
        Projecting to 3D yields a stable, comparable "volume" measure.

        **Dimension reduction:** if the input already has 3 columns, it is used
        as-is (no projection). Otherwise the input is projected with
        ``umap.UMAP(n_components=3, random_state=random_state)``. If
        ``umap-learn`` is not installed, or if UMAP fails to fit (e.g. on very
        small inputs where ``n_neighbors`` exceeds ``n``), the function falls
        back to taking the first 3 columns of the input and emits a
        ``UserWarning``. This fallback is only sensible for toy data — for real
        embeddings, install ``umap-learn``.

        **Comparability caveat:** the returned value is a volume in the
        UMAP-projected space. UMAP is non-linear and its scale is
        data-dependent, so values are NOT comparable across separate UMAP fits
        (i.e. across different datasets, or across runs with different
        ``random_state``). Fix ``random_state`` and compare within a single
        experiment.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="convex_hull_volume_3d", chunking_kwargs=chunking_kwargs)
    parameters = {"random_state": random_state, "embedding_model": embedding_model}

    # Convert first, so numpy-array inputs don't trigger the ambiguous-truth-value
    # error of `if not data:` when data is an ndarray with >1 element.
    X = np.asarray(data, dtype=float)
    if X.size == 0:
        raise ValueError("Cannot compute convex hull for empty data")

    n = X.shape[0]
    if n < 4:
        raise ValueError(
            f"Cannot compute 3D convex hull for fewer than 4 points (got {n})"
        )

    three_d = _reduce_to_3d(X, random_state=random_state)

    try:
        hull = ConvexHull(three_d)
        return {"value": float(hull.volume), "parameters": parameters}
    except (ValueError, RuntimeError):
        # Projected points are coplanar — hull has zero volume.
        return {"value": 0.0, "parameters": parameters}


def _reduce_to_3d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project X to 3D via UMAP; fall back to the first 3 columns on failure."""
    if X.shape[1] == 3:
        return X

    try:
        import umap as umap_lib
        from ._umap import fit_transform_umap
        reducer = umap_lib.UMAP(n_components=3, random_state=random_state)
        return fit_transform_umap(reducer, X)
    except Exception as e:
        warnings.warn(
            f"UMAP reduction to 3D failed ({type(e).__name__}: {e}); "
            "falling back to the first 3 columns of the input. "
            "This fallback is only meaningful for toy data — "
            "for real embeddings, install umap-learn.",
            stacklevel=3,
        )
        if X.shape[1] >= 3:
            return X[:, :3]
        pad = np.zeros((X.shape[0], 3 - X.shape[1]))
        return np.concatenate([X, pad], axis=1)

"""Shared UMAP projection helper for measures that reduce to 2D.

``bins_entropy`` and ``convex_hull_volume_2d`` can both project their input to two
dimensions with UMAP.
"""
from __future__ import annotations

import warnings

import numpy as np

# UMAP's wording of the small-dataset warning. Matched as a substring so a
# reworded message in a future umap-learn release falls through to the
# pass-through branch (re-emitted verbatim) rather than crashing.
_SMALL_DATASET_MARKER = "n_neighbors is larger than the dataset size"


def fit_transform_umap(reducer, X: np.ndarray) -> np.ndarray:
    """Run ``reducer.fit_transform(X)``, rewording UMAP's small-dataset warning.

    When the number of points is smaller than UMAP's neighborhood size, UMAP
    warns ``n_neighbors is larger than the dataset size; truncating to
    X.shape[0] - 1``. That message is replaced with one that names the fix (use
    more data). Every other warning UMAP raises is re-emitted unchanged, keeping
    its original origin (so an unfamiliar warning can still be traced back to
    UMAP).

    Args:
        reducer: A configured ``umap.UMAP`` instance (``n_components`` already
            set).
        X: Input array of shape ``(n, d)``.

    Returns:
        The 2D projection returned by ``reducer.fit_transform``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = reducer.fit_transform(X)

    for w in caught:
        if _SMALL_DATASET_MARKER in str(w.message):
            warnings.warn(
                f"Dataset is too small for UMAP ({X.shape[0]} points); "
                "using a bigger dataset will fix this.",
                UserWarning,
            )
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    return projected
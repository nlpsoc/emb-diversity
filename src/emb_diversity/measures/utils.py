from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..compute_pairwise import compute_pairwise_distances
from ._types import DISTANCE_METRIC


def _compute_pairwise_distances(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        **metric_kwargs: Any,
) -> np.ndarray:
    """
    Helper function to compute all pairwise distances, delegating to the cached
    `compute_pairwise_distances` module so repeated calls on the same embedding
    matrix reuse the previous result instead of re-running scipy.pdist.

    Returns:
        Condensed distance array.

    Raises:
        ValueError: If data is empty or contains only one datapoint.
    """
    return compute_pairwise_distances(data, metric=metric, **metric_kwargs)


def _require_nonzero_norms(
        X: np.ndarray,
        *,
        context: str = "cosine similarity",
) -> np.ndarray:
    """Return the row L2 norms of ``X``, raising if any row is all-zero.

    A zero-norm (all-zero) vector has no direction, so cosine similarity /
    normalized dot-product is undefined for it. Raising here keeps the
    normalize-based kernel measures (``vendi_score``, ``dcscore``,
    ``log_determinant``) consistent with the pdist-based measures, which raise
    on the NaN that a zero-norm vector produces under cosine distance (see
    ``compute_pairwise_distances``).

    Returns:
        1D array of per-row L2 norms, so callers can normalize without
        recomputing them: ``X / _require_nonzero_norms(X)[:, None]``.

    Raises:
        ValueError: If any row of ``X`` has zero L2 norm.
    """
    norms = np.linalg.norm(X, axis=1)
    if (norms == 0).any():
        bad = np.where(norms == 0)[0]
        raise ValueError(
            f"{context} is undefined for zero-norm (all-zero) vectors; "
            f"rows {bad.tolist()} have zero length"
        )
    return norms

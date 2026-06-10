"""Validation helpers for measure functions.

Checks that depend on measure-specific arguments (e.g. the distance
metric) live here; they cannot run in the shared input validation of
``to_numeric_array``, which never sees those arguments.
"""

from __future__ import annotations

import numpy as np


def ensure_cosine_defined(X: np.ndarray, metric) -> None:
    """Raise if *metric* is cosine and *X* contains all-zero rows.

    Cosine distance divides by the row norms, so an all-zero row would
    silently turn every distance involving it into nan.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        metric: The metric name or callable the caller will compute with;
            only the string ``"cosine"`` triggers the check.

    Raises:
        ValueError: If ``metric == "cosine"`` and X has at least one
            all-zero row.
    """
    if metric != "cosine":
        return
    zero_rows = np.flatnonzero(np.linalg.norm(X, axis=1) == 0)
    if zero_rows.size > 0:
        raise ValueError(
            "Cosine distance is undefined for all-zero vectors (their "
            "norm is 0, which would cause a division by zero); data "
            f"row(s) {zero_rows.tolist()} are all zeros. Remove these "
            "rows or use a different metric (e.g. metric='euclidean')."
        )

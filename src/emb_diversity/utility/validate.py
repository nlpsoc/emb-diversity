"""Validation helpers for measure functions.

Checks that depend on measure-specific arguments (e.g. the distance
metric) live here; they cannot run in the shared input validation of
``to_numeric_array``, which never sees those arguments.
"""

from __future__ import annotations

import warnings

import numpy as np


def warn_on_zero_norm_rows(X: np.ndarray, measure: str) -> None:
    """Warn if *X* contains all-zero rows before cosine-kernel normalization.

    The cosine-similarity kernels (``kernel_type="cs"`` with ``normalize=True``)
    divide each row by its L2 norm. A zero vector has no direction, so cosine
    similarity is undefined for it. The measure still proceeds — it clips the
    zero norm to a tiny positive value, so the zero row stays zero and
    contributes as if near-orthogonal to every other point — but its
    contribution to the score is not meaningful, so this warns.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        measure: Name of the calling measure, used in the warning message.
    """
    zero_rows = np.flatnonzero(np.linalg.norm(X, axis=1) == 0)
    if zero_rows.size > 0:
        warnings.warn(
            f"{measure}: data row(s) {zero_rows.tolist()} are all-zero vectors. "
            "Cosine similarity is undefined for them (their norm is 0), but "
            "instead of failing this measure clips the norm to a tiny value and "
            "still returns a score. That score is silently affected by these "
            "clipped zero rows, so it may not reflect your data and you get no "
            "error to flag it. Remove these rows, set normalize=False, or use a "
            "non-cosine kernel_type.",
            stacklevel=2,
        )

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

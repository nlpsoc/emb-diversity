"""Validation helpers for measure functions.

Checks that depend on measure-specific arguments (e.g. the distance
metric) live here; they cannot run in the shared input validation of
``to_numeric_array``, which never sees those arguments.
"""

from __future__ import annotations

import warnings

import numpy as np


def kernel_row_norms(X: np.ndarray, measure: str) -> np.ndarray:
    """Return L2 row norms for cosine-kernel normalization, warning on zeros.

    The cosine-similarity kernels (``kernel_type="cs"`` with ``normalize=True``)
    divide each row by its L2 norm. A zero vector has no direction, so cosine
    similarity is undefined for it. Rather than fail, the measure proceeds —
    the zero norm is clipped to a tiny positive value, so the zero row stays
    zero and contributes as if near-orthogonal to every other point — but its
    contribution to the score is not meaningful, so this warns.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        measure: Name of the calling measure, used in the warning message.

    Returns:
        Row norms with shape (n_samples, 1), clipped to a tiny positive
        minimum so the caller can divide ``X`` by them without producing nan.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms.ravel() == 0)
    if zero_rows.size > 0:
        warnings.warn(
            f"{measure}: cosine similarity is undefined for all-zero vectors "
            f"(their norm is 0); data row(s) {zero_rows.tolist()} are all "
            "zeros and are treated as near-orthogonal to every other point. "
            "Their contribution to the score is not meaningful — remove these "
            "rows, set normalize=False, or use a non-cosine kernel_type.",
            stacklevel=2,
        )
    return np.clip(norms, 1e-12, None)


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

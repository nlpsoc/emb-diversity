"""Shared validation for numeric measure input.

Every measure receives its data through ``resolve_embeddings``, which uses
``to_numeric_array`` so that invalid numeric input fails early with the same
message everywhere. ``compute_pairwise_distances`` applies the same checks for
callers that use it directly.
"""

from __future__ import annotations

import numpy as np


def to_numeric_array(data) -> np.ndarray:
    """Convert *data* to a float array, rejecting non-numeric content.

    String content is rejected rather than coerced, so that number-like
    strings (e.g. ``"1"``) do not silently pass as numbers.

    Raises:
        ValueError: If data contains strings or values that cannot be
            interpreted as floats.
    """
    X = np.asarray(data)
    contains_strings = X.dtype.kind in ("U", "S") or (
        X.dtype == object and any(isinstance(v, (str, bytes)) for v in X.flat)
    )
    if contains_strings:
        raise ValueError(
            "Data must be numeric, but it contains strings. Number-like "
            "strings (e.g. '1') are not converted automatically; convert "
            "them to numbers first. To measure diversity of texts, pass a "
            "flat list of strings instead."
        )
    try:
        return X.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Data must be numeric (conversion to float failed: "
            f"{exc}). Pass vectors of numbers with shape "
            "(n_samples, n_features)."
        ) from exc


def ensure_cosine_defined(X: np.ndarray, metric) -> None:
    """Raise if *metric* is cosine and *X* contains all-zero rows.

    Cosine distance divides by the row norms, so an all-zero row would
    silently turn every distance involving it into nan.

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

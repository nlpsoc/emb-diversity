"""Shared validation for numeric measure input.

Every measure receives its data through ``resolve_embeddings``, which uses
``to_numeric_array`` so that invalid numeric input fails early with the same
message everywhere. ``compute_pairwise_distances`` applies the same
conversion for callers that use it directly.
"""

from __future__ import annotations

import numbers

import numpy as np

# numpy dtype kinds that already hold numbers:
# b = bool, u = unsigned int, i = signed int, f = float
_NUMERIC_KINDS = "buif"


def to_numeric_array(data) -> np.ndarray:
    """Convert measure input to a validated float (n_samples, n_features) array.

    Anything this returns is safe to compute on: numeric content (strings
    are rejected, not coerced), at least 2 samples, and 2-dimensional.
    Measures with higher minimums (e.g. a convex hull needs 3 points)
    check those themselves afterwards.

    Raises:
        ValueError: If data is not numeric, has fewer than 2 samples, or is
            not 2-dimensional.
    """
    X = np.asarray(data)
    is_numeric = X.dtype.kind in _NUMERIC_KINDS or (
        X.dtype == object and all(isinstance(v, numbers.Real) for v in X.flat)
    )
    if not is_numeric:
        raise ValueError(
            f"Data must be numeric, got dtype '{X.dtype}'. Number-like "
            "strings (e.g. '1') are not converted automatically; convert "
            "them to numbers first. To measure diversity of texts, pass a "
            "flat list of strings instead."
        )
    try:
        X = X.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Data must be numeric (conversion to float failed: "
            f"{exc}). Pass vectors of numbers with shape "
            "(n_samples, n_features)."
        ) from exc
    n_samples = X.shape[0] if X.ndim > 0 else 1
    if n_samples < 2:
        raise ValueError(
            f"Measuring diversity needs at least 2 samples, got {n_samples}."
        )
    if X.ndim != 2:
        raise ValueError(
            "Data must be a 2-dimensional array of shape "
            f"(n_samples, n_features); got shape {X.shape}. For "
            "one-dimensional samples, pass one vector per row, e.g. "
            "[[0], [1]] instead of [0, 1]."
        )
    return X

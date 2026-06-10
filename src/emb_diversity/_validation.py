"""Shared validation for numeric measure input.

Every measure receives its data through ``resolve_embeddings``, which uses
``to_numeric_array`` so that invalid numeric input fails early with the same
message everywhere. ``compute_pairwise_distances`` applies the same
conversion for callers that use it directly.
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

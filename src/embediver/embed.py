"""Text embedding utilities."""

from __future__ import annotations

import numpy as np

from .axes_registry import get_axis
from .embeddings.SBERT import encode_sentences


# TODO: will be replaced by Menan's code
def embed_texts(
    texts: list[str],
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
) -> np.ndarray:
    """Embed a list of texts into vectors.

    Resolution order:
      1. If *embedding_model* is given, use it directly.
      2. Otherwise look up *diversity_axis* in the axis registry.

    Args:
        texts: Raw text strings.
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit HuggingFace model id; overrides *diversity_axis*.

    Returns:
        numpy array of shape ``(len(texts), embedding_dim)``.
    """
    if embedding_model is not None:
        model_name = embedding_model
    elif diversity_axis is not None:
        axis = get_axis(diversity_axis)
        model_name = axis.default_model
    else:
        raise ValueError(
            "Either diversity_axis or embedding_model must be provided"
        )

    vectors = encode_sentences(texts, model_name=model_name)
    return np.asarray(vectors, dtype=float)

"""Text embedding utilities."""

from __future__ import annotations

import numpy as np

from .axes_registry import axes
from .embeddings.embed import encode


def embed_texts(
    texts: list[str],
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
) -> np.ndarray:
    """Embed a list of texts into vectors, with disk caching.

    Resolution order:
      1. If *embedding_model* is given, use it directly.
      2. Otherwise look up *diversity_axis* in the axis registry.

    Embeddings are cached on disk under ``.cache/embeddings/<model>/`` and
    reused across calls — repeated runs over the same texts skip the model.

    Args:
        texts: Raw text strings.
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit HuggingFace / SentenceTransformer model id;
            overrides *diversity_axis*.

    Returns:
        numpy array of shape ``(len(texts), embedding_dim)``.
    """
    if embedding_model is not None:
        model_name = embedding_model
    elif diversity_axis is not None:
        axis = axes.get(diversity_axis)
        model_name = axis.default_model
    else:
        raise ValueError(
            "Either diversity_axis or embedding_model must be provided"
        )

    vectors = encode(texts, model_name=model_name)
    return np.asarray(vectors, dtype=float)

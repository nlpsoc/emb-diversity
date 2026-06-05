"""Text embedding utilities."""

from __future__ import annotations

import numpy as np

from .axes_registry import axes
from .embeddings.embed import encode


def resolve_model_name(
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
) -> str:
    """Resolve which embedding model id will actually be used.

    Resolution order:
      1. If *embedding_model* is given, use it directly.
      2. Otherwise look up *diversity_axis* in the axis registry.

    Args:
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit model id; overrides *diversity_axis*.

    Returns:
        The resolved model id string.
    """
    if embedding_model is not None:
        return embedding_model
    if diversity_axis is not None:
        return axes.get(diversity_axis).default_model
    raise ValueError(
        "Either diversity_axis or embedding_model must be provided"
    )


def _is_text_input(data) -> bool:
    """Return True if *data* looks like a list of strings."""
    return len(data) > 0 and isinstance(data[0], str)


def resolve_embeddings(
    data,
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
):
    """Turn raw text into vectors, reporting the model that was used.

    Text input is embedded and the resolved model id is returned alongside the
    vectors. Numeric input (already embeddings) is passed through unchanged with
    a ``None`` model id, since no embedding happened.

    Args:
        data: A list of text strings, or embedding vectors (n, d).
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit model id; overrides *diversity_axis*.

    Returns:
        Tuple ``(vectors, resolved_model_or_None)``.
    """
    if _is_text_input(data):
        model_name = resolve_model_name(diversity_axis, embedding_model)
        return embed_texts(data, diversity_axis, embedding_model), model_name
    return data, None


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
    model_name = resolve_model_name(diversity_axis, embedding_model)
    vectors = encode(texts, model_name=model_name)
    return np.asarray(vectors, dtype=float)

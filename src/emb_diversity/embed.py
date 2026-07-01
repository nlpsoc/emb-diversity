"""Text and audio embedding utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warnings

from .embeddings._embed_numpy import to_numeric_array
from .axes_registry import axes
from .embeddings.embed_text import encode
from .embeddings.embed_audio import encode as encode_audio
from .utility._progress import announce_calculation

LARGE_DATASET_THRESHOLD = 10_000
SLOW_DATASET_THRESHOLD = 1_000


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

    Raises:
        ValueError: If *diversity_axis* is not a registered axis, or if
            neither *diversity_axis* nor *embedding_model* is provided.
    """
    if embedding_model is not None:
        return embedding_model
    if diversity_axis is not None:
        if diversity_axis not in axes:
            available = ", ".join(sorted(axes.keys())) or "(none)"
            raise ValueError(
                f"Unknown diversity_axis {diversity_axis!r}. "
                f"Available axes: {available}."
            )
        return axes.get(diversity_axis).default_model
    raise ValueError(
        "Either diversity_axis or embedding_model must be provided"
    )


def _is_raw_input(data) -> bool:
    """Return True if *data* looks like a list of strings.

    Covers both raw text and audio file paths — both are plain strings that
    still need embedding, as opposed to numeric input that is already a
    vector. Which encoder a string list is routed to is decided separately,
    by the resolved axis's modality (see ``_modality_for_axis``).
    """
    return len(data) > 0 and isinstance(data[0], str)


def _modality_for_axis(diversity_axis: str | None) -> str:
    """Return the registered modality ("text" or "audio") for *diversity_axis*.

    Falls back to "text" when no axis is given (e.g. an explicit
    ``embedding_model`` was passed with ``diversity_axis=None``) or the axis
    is not registered — ``resolve_model_name`` raises on the latter case
    anyway, so this never has to.
    """
    if diversity_axis is not None and diversity_axis in axes:
        return axes.get(diversity_axis).modality
    return "text"


def resolve_embeddings(
    data,
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
    measure: str | None = None,
    chunking_kwargs: dict | None = None,
):
    """Turn raw text or audio into vectors, reporting the model that was used.

    Raw input (text strings, or audio file paths for an audio-modality axis
    like ``"speaker"``) is embedded and the resolved model id is returned
    alongside the vectors. Numeric input (already embeddings) is converted to
    a float numpy array and returned with a ``None`` model id, since no
    embedding happened. Every measure resolves its data through this
    function, so the numeric validation here covers the whole API.

    Args:
        data: A list of text strings, a list of audio file paths (for an
            audio-modality axis), or embedding vectors (n, d).
        diversity_axis: Registered axis name (default ``"semantic"``). Its
            registered ``modality`` decides whether raw input is embedded as
            text or as audio.
        embedding_model: Explicit model id; overrides *diversity_axis*.
        measure: Name of the calling measure. When given, an interactive notice
            ("Calculating measure '<measure>'…") is printed once embedding is
            done, just before the measure's calculation begins.
        chunking_kwargs: Long-text handling options forwarded to
            :func:`embed_texts` / :func:`encode` for text input — e.g.
            ``{"chunking": True, "chunks": 5, "pooling": "mean"}``. Ignored for
            audio and for numeric (already-embedded) input.

    Returns:
        Tuple ``(vectors, resolved_model_or_None)``.

    Raises:
        ValueError: If *data* is a single string instead of a list of
            texts/paths (a bare string is iterable, so it would otherwise be
            embedded character by character) — or if numeric input contains
            strings (number-like strings are rejected, not coerced), has
            fewer than 2 samples, or is not a 2-D (n_samples, n_features)
            matrix — or if the vectors (given or embedded) contain
            non-finite values (nan or inf).
    """
    if len(data) > LARGE_DATASET_THRESHOLD:
        warnings.warn(
            f"Dataset has {len(data)} samples, above {LARGE_DATASET_THRESHOLD}. "
            "Several measures build an O(n^2) matrix, so this may be slow or run "
            "out of memory. Support for large datasets is planned for a future "
            f"release. For now, we recommend repeated calculation on subsamples "
            f"of size up to {LARGE_DATASET_THRESHOLD}.",
            UserWarning,
            stacklevel=3,
        )
    elif len(data) > SLOW_DATASET_THRESHOLD:
        warnings.warn(
            f"You are calculating diversity for {len(data)} samples. "
            "It can take some time to embed text samples and and build similarity matrices.",
            UserWarning,
            stacklevel=3,
        )

    if _is_raw_input(data):
        # Resolve the model id once so it can be reported back, then pass it
        # down explicitly: embed_texts/embed_audio are the single
        # embedding code path per modality (and each runs its embedded
        # vectors through to_numeric_array itself).
        model_name = resolve_model_name(diversity_axis, embedding_model)
        if _modality_for_axis(diversity_axis) == "audio":
            vectors = embed_audio(data, embedding_model=model_name)
        else:
            vectors = embed_texts(data, embedding_model=model_name, **(chunking_kwargs or {}))
        resolved_model = model_name
    else:
        vectors, resolved_model = to_numeric_array(data), None

    # Embedding is done; the calculation is what follows in the caller. Announce
    # it now so a slow measure does not look like a hang.
    if measure is not None:
        announce_calculation(measure)
    return vectors, resolved_model


def embed_texts(
    texts: list[str],
    diversity_axis: str | None = "semantic",
    embedding_model: str | None = None,
    cache_dir: Path | None = None,
    **chunking_kwargs,
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
        cache_dir: Override the disk cache directory.
        **chunking_kwargs: Long-text handling options forwarded to
            :func:`encode` — ``chunking`` (bool), ``chunks`` (int), and
            ``pooling`` (str). By default texts are truncated to the model's
            max sequence length; pass ``chunking=True`` to chunk and pool
            instead.

    Returns:
        numpy array of shape ``(len(texts), embedding_dim)``.

    Raises:
        ValueError: If *texts* is a single string instead of a list of
            texts, contains fewer than 2 texts, or if the embedded vectors
            contain non-finite values (nan or inf — embedding models can
            emit those, e.g. on half-precision overflow).
    """
    # A bare string is iterable, so it would otherwise be embedded
    # character by character. This must fire before the model loads (and
    # before to_numeric_array, which would only see the embedded chars).
    if isinstance(texts, str):
        raise ValueError(
            "Expected a list of texts, not a single string. Wrap it in "
            "a list, e.g. [\"some text\", \"another text\"] — measuring "
            "diversity needs at least 2 texts."
        )
    model_name = resolve_model_name(diversity_axis, embedding_model)
    # to_numeric_array is the shared exit gate for measure data: embedded
    # vectors get the same validation (>= 2 samples, finite) as vector
    # input, with no extra guards here.
    kwargs = {"model_name": model_name, **chunking_kwargs}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return to_numeric_array(encode(texts, **kwargs))


def embed_audio(
    paths: list[str],
    diversity_axis: str | None = "speaker",
    embedding_model: str | None = None,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Embed a list of audio files into speaker embeddings, with disk caching.

    Resolution order:
      1. If *embedding_model* is given, use it directly.
      2. Otherwise look up *diversity_axis* in the axis registry.

    Embeddings are cached on disk under ``.cache/embeddings/<model>/`` and
    reused across calls — repeated runs over the same files skip the model.

    Args:
        paths: Paths to audio files (any format supported by ``soundfile``,
            e.g. .wav, .flac, .ogg — not .mp3).
        diversity_axis: Registered audio-modality axis name (default
            ``"speaker"``).
        embedding_model: Explicit SpeechBrain model id; overrides
            *diversity_axis*.
        cache_dir: Override the disk cache directory.

    Returns:
        numpy array of shape ``(len(paths), embedding_dim)``.

    Raises:
        ValueError: If *paths* is a single string instead of a list of
            paths, contains fewer than 2 paths, or if the embedded vectors
            contain non-finite values (nan or inf).
    """
    # A bare string is iterable, so it would otherwise be embedded
    # character by character. This must fire before the model loads (and
    # before to_numeric_array, which would only see the embedded chars).
    if isinstance(paths, str):
        raise ValueError(
            "Expected a list of audio file paths, not a single string. Wrap "
            "it in a list, e.g. [\"a.wav\", \"b.wav\"] — measuring diversity "
            "needs at least 2 files."
        )
    model_name = resolve_model_name(diversity_axis, embedding_model)
    # to_numeric_array is the shared exit gate for measure data: embedded
    # vectors get the same validation (>= 2 samples, finite) as vector
    # input, with no extra guards here.
    kwargs = {"model_name": model_name}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return to_numeric_array(encode_audio(paths, **kwargs))

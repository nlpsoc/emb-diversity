import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..utility._cache import cached_encode, DEFAULT_CACHE_DIR
from ..utility._progress import announce_embedding, load_with_spinner, progress_enabled


# Models that require the HuggingFace Transformers backend
_HF_MODELS = {
    "princeton-nlp/sup-simcse-roberta-large",
}

# Defaults for long-text handling. Changing them here changes the behaviour of
# encode()/embed_texts() everywhere; nothing downstream re-declares them.
DEFAULT_CHUNKING = False
DEFAULT_CHUNKS = 10
DEFAULT_POOLING = "mean"

# Pooling strategies that collapse a text's per-chunk embeddings (a 2-D array
# of shape (n_chunks, dim)) into a single vector of shape (dim,).
_POOLERS = {
    "mean": lambda M: M.mean(axis=0),
    "max": lambda M: M.max(axis=0),
    "first": lambda M: M[0],  # CLS-style: keep the first chunk
}


# Tokens reserved per window for the special tokens (e.g. CLS/SEP) that the
# backend re-adds when a decoded chunk is encoded, so a full window does not
# overflow the model's limit and get silently re-truncated.
_SPECIAL_TOKEN_MARGIN = 2


# private functions to load models. Memoized so a chunked encode that needs the
# model both to read its sequence limit and to embed misses loads it only once.
@lru_cache(maxsize=None)
def _load_st(model_name: str):
    def _load(stage):
        stage.loading_libraries()
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage.fetching_model(model_name)
        return SentenceTransformer(model_name, device=device)

    return load_with_spinner(model_name, _load)

@lru_cache(maxsize=None)
def _load_hf(model_name: str):
    def _load(stage):
        stage.loading_libraries()
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage.fetching_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        return tokenizer, model, device

    return load_with_spinner(model_name, _load)


@lru_cache(maxsize=None)
def _get_tokenizer(model_name: str, backend: str):
    """Tokenizer for *model_name*, memoized, resolved per backend.

    - **ST**: taken from the loaded SentenceTransformer (``model.tokenizer``).
      ST accepts shortcut names like ``all-MiniLM-L6-v2`` that are *not* valid
      HuggingFace repo ids, so ``AutoTokenizer`` cannot resolve them directly —
      the model does the resolution. The model is memoized and reused for
      encoding, so this adds no extra load.
    - **HF**: loaded standalone via ``AutoTokenizer`` (HF models are full repo
      ids), so token counting never loads the heavy model.
    """
    if backend == "st":
        return _load_st(model_name).tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=None)
def _window_size(model_name: str, backend: str) -> int:
    """Window size in content tokens for chunking, per backend.

    Must match the limit the backend's encode actually truncates to, or chunks
    would be silently re-truncated and content lost:

    - **ST**: ``model.max_seq_length`` (often smaller than the tokenizer's
      limit; reading it requires the model, which is memoized and reused for
      encoding).
    - **HF**: the tokenizer's ``model_max_length`` (no model load needed).

    A margin is reserved for the special tokens re-added on encode.
    """
    if backend == "st":
        n = int(_load_st(model_name).max_seq_length)
    else:
        n = int(_get_tokenizer(model_name, backend).model_max_length)
        # HF uses a huge sentinel (e.g. 1e30) when the limit is unset.
        n = n if 0 < n <= 100_000 else 512
    return max(1, n - _SPECIAL_TOKEN_MARGIN)


def _actual_chunks(text: str, model_name: str, backend: str, window: int, cap: int) -> int:
    """Number of windows *text* is split into, capped at *cap* (>= 1)."""
    n_tokens = len(
        _get_tokenizer(model_name, backend).encode(text, add_special_tokens=False)
    )
    return max(1, min(math.ceil(n_tokens / window), cap))


# private methods to calculate embeddings for the texts not existing in cache
def _raw_encode_st(texts: List[str], model_name: str) -> List[List[float]]:
    model = _load_st(model_name)
    announce_embedding(len(texts))
    # show_progress_bar surfaces SentenceTransformer's per-batch bar, so a long
    # encode does not look like a hang. Gated like the rest so pipes/CI stay quiet.
    return model.encode(
        texts, convert_to_numpy=True, show_progress_bar=progress_enabled()
    ).tolist()


def _raw_encode_hf(texts: List[str], model_name: str) -> List[List[float]]:
    import torch

    tokenizer, model, device = _load_hf(model_name)
    announce_embedding(len(texts))
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().tolist()


def _raw_encode(texts: List[str], model_name: str, backend: str) -> List[List[float]]:
    """Truncation-mode encode, dispatched by backend."""
    if backend == "hf":
        return _raw_encode_hf(texts, model_name)
    return _raw_encode_st(texts, model_name)


def _chunk_strings(text: str, model_name: str, backend: str, window: int, cap: int) -> List[str]:
    """Split *text* into up to *cap* windows of *window* tokens, as strings.

    Tokenizes once, slices the token-ids into windows, and decodes each window
    back to text so it can be re-embedded through either backend.
    """
    tokenizer = _get_tokenizer(model_name, backend)
    ids = tokenizer.encode(text, add_special_tokens=False)
    windows = [ids[i:i + window] for i in range(0, len(ids), window)][:cap]
    if not windows:  # empty text → keep a single (empty) chunk
        windows = [ids]
    return [tokenizer.decode(w) for w in windows]


def _encode_chunked(
    texts: List[str], model_name: str, backend: str, cap: int, pooling: str
) -> List[List[float]]:
    """Embed each text by windowing it, embedding all windows, then pooling.

    All windows across all texts are embedded in a single batch, then regrouped
    per text and pooled into one vector each. Output is aligned with *texts*.
    """
    window = _window_size(model_name, backend)
    pool = _POOLERS[pooling]

    # Build the flat batch of chunk strings, tracking each text's slice.
    flat_chunks: List[str] = []
    spans: List[tuple[int, int]] = []
    for text in texts:
        chunks = _chunk_strings(text, model_name, backend, window, cap)
        spans.append((len(flat_chunks), len(flat_chunks) + len(chunks)))
        flat_chunks.extend(chunks)

    chunk_vecs = np.array(_raw_encode(flat_chunks, model_name, backend), dtype=np.float32)

    return [pool(chunk_vecs[start:end]).tolist() for start, end in spans]


def encode(
    texts: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    chunking: bool = DEFAULT_CHUNKING,
    chunks: int = DEFAULT_CHUNKS,
    pooling: str = DEFAULT_POOLING,
) -> List[List[float]]:
    """
    Encode texts into embeddings using the appropriate backend, with disk caching.

    Uses HuggingFace Transformers for known HF-only models, SentenceTransformers otherwise.

    Long texts are handled in one of two ways:

    - **Truncation (default)**: each text is truncated to the model's maximum
      sequence length, discarding any tokens past the limit.
    - **Chunking** (``chunking=True``): each text is split into windows of the
      model's max sequence length (up to ``chunks`` windows), every window is
      embedded, and the per-window vectors are combined into one vector with
      the ``pooling`` strategy. The chunking mode and the *actual* number of
      windows used are folded into the cache key, so chunked and truncated
      embeddings of the same text never collide.

    Args:
        texts: Input texts to encode.
        model_name: Pretrained model name. Defaults to "all-MiniLM-L6-v2".
        cache_dir: Root directory for the disk cache.
        chunking: If True, chunk long texts instead of truncating them.
        chunks: Maximum number of windows per text when chunking. Texts shorter
            than this use fewer windows; longer texts are truncated at this cap.
        pooling: How to combine per-window vectors when chunking. One of
            "mean" (default), "max", or "first".

    Returns:
        List of embedding vectors as lists of floats.

    Raises:
        ValueError: If ``pooling`` is not a known strategy.
    """
    backend = "hf" if model_name in _HF_MODELS else "st"

    if chunking:
        if pooling not in _POOLERS:
            raise ValueError(
                f"Unknown pooling {pooling!r}. Options: {sorted(_POOLERS)}"
            )
        window = _window_size(model_name, backend)
        # Per-text suffix records the *actual* chunk count, not the cap, so two
        # calls with different caps that produce the same windows share a key.
        key_suffixes: Optional[List[str]] = [
            f"chunk={_actual_chunks(t, model_name, backend, window, chunks)}|pool={pooling}"
            for t in texts
        ]
        encode_fn = lambda t: _encode_chunked(t, model_name, backend, chunks, pooling)
    else:
        key_suffixes = None  # → "" per text (truncation)
        encode_fn = lambda t: _raw_encode(t, model_name, backend)

    return cached_encode(
        texts,
        encode_fn=encode_fn,
        model_name=model_name,
        cache_dir=cache_dir,
        key_suffixes=key_suffixes,
    )

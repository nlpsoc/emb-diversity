"""Internal caching layer. Not part of the public API."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import xxhash
from safetensors.numpy import save_file, load_file

DEFAULT_CACHE_DIR = Path(".cache/embeddings")


def _hash(text: str) -> str:
    return xxhash.xxh128(text.encode("utf-8")).hexdigest()


def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return load_file(path)["embedding"]


def _save(path: Path, embedding: np.ndarray) -> None:
    save_file({"embedding": embedding}, path)


def cached_encode(
    texts: Sequence[str],
    encode_fn: Callable[[List[str]], List[List[float]]],
    model_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_workers: int = 8,
) -> List[List[float]]:
    """
    Wrapper that adds disk caching to any encode function.

    Args:
        texts: Input texts to encode.
        encode_fn: Function that takes a list of strings and returns
                   List[List[float]].
        model_name: Used to namespace the cache (different models
                    get different cache folders).
        cache_dir: Root cache directory.
        max_workers: Threads for parallel I/O.

    Returns:
        List of embedding vectors as lists of floats.
    """
    if not texts:
        return []

    model_cache = cache_dir / model_name.replace("/", "_")
    model_cache.mkdir(parents=True, exist_ok=True)

    hashes = [_hash(t) for t in texts]
    paths = [model_cache / f"{h}.safetensors" for h in hashes]

    # Parallel cache lookup
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        cached = list(pool.map(_load, paths))

    uncached = [(i, texts[i]) for i, emb in enumerate(cached) if emb is None]

    if uncached:
        indices, missed_texts = zip(*uncached)
        new_results = encode_fn(list(missed_texts))
        new_arrays = [np.array(emb, dtype=np.float32) for emb in new_results]

        for idx, arr in zip(indices, new_arrays):
            cached[idx] = arr

        # Parallel save
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(_save, [paths[i] for i in indices], new_arrays)

    return [emb.tolist() for emb in cached]


def clear_cache(model_name: str = None, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Clear cached embeddings. If model_name given, only that model."""
    import shutil
    target = cache_dir / model_name.replace("/", "_") if model_name is not None else cache_dir
    if target.exists():
        shutil.rmtree(target)
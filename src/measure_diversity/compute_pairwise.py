"""
We designed this module to efficiently compute pairwise distances.
It uses a two-level caching strategy:
1. It checks whether the same embedding matrix is used again, then it will just use the previously calculated pairwise distance in Memory.
2. It also saves the most recenlty calculated pairwise distance to disk, so that if the same embedding matrix is used again in the future, it can load the pairwise distance from disk instead of recomputing it.

Yayy!
"""
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import numpy as np
import xxhash
from scipy.spatial.distance import pdist
from safetensors.numpy import save_file, load_file

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]
DEFAULT_CACHE_DIR = Path(".cache/pdist")
#how many chunks we feed into the hash function at a time, to keep memory usage constant regardless of input size
_HASH_CHUNK = 1_000_000
#how many distance matrices to keep in memory before evicting the oldest one (LRU)
_MEMORY_MAX = 0


def _fingerprint(X: np.ndarray) -> str:
    """Full-content xxhash of an array, chunked to keep memory constant."""
    h = xxhash.xxh64()
    h.update(str(X.shape).encode())
    h.update(str(X.dtype).encode())
    flat = X.ravel()
    for i in range(0, len(flat), _HASH_CHUNK):
        h.update(flat[i:i + _HASH_CHUNK].tobytes())
    return h.hexdigest()



# in memory cache

_memory: dict[str, np.ndarray] = {}
_memory_ids: dict[str, int] = {}


def _store_memory(key: str, obj_id: int, result: np.ndarray) -> None:
    if _MEMORY_MAX == 0:
        return
    if len(_memory) >= _MEMORY_MAX:
        oldest = next(iter(_memory))
        del _memory[oldest]
        _memory_ids.pop(oldest, None)
    _memory[key] = result
    _memory_ids[key] = obj_id


# user API

def compute_pairwise_distances(
    data: Sequence[Sequence[float]],
    metric: DISTANCE_METRIC = "cosine",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """
    Compute pairwise distances with two-level caching.

    Args:
        data: 2D array-like of shape (n_samples, n_features).
        metric: Distance metric name (e.g. "cosine", "euclidean").
        cache_dir: Root directory for disk cache.

    Returns:
        Condensed distance array (upper triangle from scipy.pdist).

    Raises:
        ValueError: If data is empty or single row.
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot compute distances for empty data")
    if n == 1:
        raise ValueError("Cannot compute distances for single data point")

    obj_id = id(X)

    # Level 1: id() fast path
    for k, cached_id in _memory_ids.items():
        if cached_id == obj_id and k.endswith(f"|{metric}"):
            return _memory[k]

    fp = _fingerprint(X)
    key = f"{fp}|{metric}"

    # Level 1b: fingerprint match in memory
    if key in _memory:
        _memory_ids[key] = obj_id
        return _memory[key]

    # Level 2: disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fp}_{metric}.safetensors"

    if path.exists():
        result = load_file(path)["distances"]
        _store_memory(key, obj_id, result)
        return result

    # Level 3: compute
    result = pdist(X, metric=metric)

    _store_memory(key, obj_id, result)
    save_file({"distances": result}, path)

    return result


def clear_distance_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Clear both memory and disk caches."""
    import shutil
    _memory.clear()
    _memory_ids.clear()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def distance_cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return cache statistics."""
    disk_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    return {
        "memory_entries": len(_memory),
        "memory_mb": round(sum(v.nbytes for v in _memory.values()) / 1024 / 1024, 2),
        "disk_files": len(disk_files),
        "disk_mb": round(sum(f.stat().st_size for f in disk_files) / 1024 / 1024, 2),
    }

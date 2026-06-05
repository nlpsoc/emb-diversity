"""
Two-level cached pairwise distance computation.

scipy.pdist is the bottleneck when several measures are run on the same
embedding matrix. This module wraps it with:

  Level 1 — in-process memory: a small bounded LRU dict keyed by full
    content fingerprint of the matrix plus the metric / kwargs. Up to
    _MEMORY_MAX entries are kept; oldest is evicted on overflow.

  Level 2 — disk: condensed distance array stored under .cache/pdist/
    as safetensors, keyed the same way. Survives across processes — a
    SLURM job that finished yesterday leaves a cache that today's job
    can pick up.

  Level 3 — compute: scipy.pdist + write through both layers.

The cache key folds in the metric and any metric_kwargs, so different
metrics on the same data do not collide.
"""
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import numpy as np
import xxhash
from scipy.spatial.distance import pdist
from safetensors.numpy import save_file, load_file

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]
DEFAULT_CACHE_DIR = Path(".cache/pdist")
# how many chunks we feed into the hash function at a time, to keep memory
# usage constant regardless of input size
_HASH_CHUNK = 1_000_000
# how many distance matrices to keep in memory before evicting the oldest one (LRU)
_MEMORY_MAX = 4

# in-memory cache (LRU)
_memory: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _fingerprint(X: np.ndarray) -> str:
    """Full-content xxhash of an array, chunked to keep memory constant."""
    h = xxhash.xxh64()
    h.update(str(X.shape).encode())
    h.update(str(X.dtype).encode())
    flat = X.ravel()
    for i in range(0, len(flat), _HASH_CHUNK):
        h.update(flat[i:i + _HASH_CHUNK].tobytes())
    return h.hexdigest()


def _metric_key(metric: DISTANCE_METRIC, metric_kwargs: dict) -> str:
    """Stable, filesystem-safe key for metric + kwargs."""
    if not metric_kwargs and isinstance(metric, str):
        return metric
    parts = [str(metric)]
    for k in sorted(metric_kwargs):
        parts.append(f"{k}={metric_kwargs[k]!r}")
    return xxhash.xxh64("|".join(parts).encode()).hexdigest()


def _store_memory(key: str, result: np.ndarray) -> None:
    if _MEMORY_MAX <= 0:
        return
    if key in _memory:
        _memory.move_to_end(key)
        return
    if len(_memory) >= _MEMORY_MAX:
        _memory.popitem(last=False)
    _memory[key] = result


def compute_pairwise_distances(
    data: Sequence[Sequence[float]],
    metric: DISTANCE_METRIC = "cosine",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    **metric_kwargs: Any,
) -> np.ndarray:
    """
    Compute pairwise distances with two-level (memory + disk) caching.

    Args:
        data: 2D array-like of shape (n_samples, n_features).
        metric: Distance metric name (e.g. "cosine", "euclidean") or callable.
        cache_dir: Root directory for the disk cache.
        **metric_kwargs: Extra keyword arguments forwarded to scipy.pdist.
            Included in the cache key so different kwargs do not collide.

    Returns:
        Condensed distance array (upper triangle from scipy.pdist).

    Raises:
        ValueError: If data is empty, a single row, or the metric produces NaN
            distances (e.g. a zero-norm vector under cosine).
    """
    X = np.asarray(data, dtype=float)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot compute distances for empty data")
    if n == 1:
        raise ValueError("Cannot compute distances for single data point")

    metric_id = _metric_key(metric, metric_kwargs)
    fp = _fingerprint(X)
    key = f"{fp}|{metric_id}"

    # Level 1: in-memory match by content fingerprint
    if key in _memory:
        _memory.move_to_end(key)
        return _memory[key]

    # Level 2: disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fp}_{metric_id}.safetensors"
    if path.exists():
        result = load_file(path)["distances"]
        _store_memory(key, result)
        return result

    # Level 3: compute, populate both layers
    result = pdist(X, metric=metric, **metric_kwargs)
    # A valid distance matrix never contains NaN. NaN means a degenerate input
    # for this metric (e.g. a zero-norm vector under cosine, where the distance
    # divides by the vector norm). Raise instead of silently caching/returning
    # NaN, which would otherwise poison every downstream measure.
    if np.isnan(result).any():
        raise ValueError(
            f"distance computation produced NaN with metric={metric!r}; "
            "this usually means a degenerate input, e.g. a zero-norm "
            "(all-zero) vector under cosine distance"
        )
    _store_memory(key, result)
    save_file({"distances": result}, path)
    return result


def clear_distance_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Clear both memory and disk caches."""
    import shutil
    _memory.clear()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def distance_cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return memory and disk cache statistics."""
    disk_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    return {
        "memory_entries": len(_memory),
        "memory_mb": round(sum(v.nbytes for v in _memory.values()) / 1024 / 1024, 2),
        "memory_max": _MEMORY_MAX,
        "disk_files": len(disk_files),
        "disk_mb": round(sum(f.stat().st_size for f in disk_files) / 1024 / 1024, 2),
    }

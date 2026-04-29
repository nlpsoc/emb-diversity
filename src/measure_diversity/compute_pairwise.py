"""
Disk-cached pairwise distance computation.

scipy.pdist is the bottleneck when several measures are run on the same
embedding matrix. This module wraps it with a content-addressed disk
cache: the matrix is fingerprinted with xxhash, and the resulting
condensed distance array is stored under .cache/pdist/. Subsequent calls
on the same data load the precomputed array from disk instead of
re-running pdist.

The cache is persistent across processes — a SLURM job that finished
yesterday leaves a cache that today's job can pick up. It is keyed by
both the matrix contents and the (metric, kwargs) pair, so different
metrics on the same data do not collide.

There is no in-memory layer: per-process speedup beyond the first call
comes from the OS file cache, which is fast enough for typical workloads.
"""
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


def compute_pairwise_distances(
    data: Sequence[Sequence[float]],
    metric: DISTANCE_METRIC = "cosine",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    **metric_kwargs: Any,
) -> np.ndarray:
    """
    Compute pairwise distances with disk caching.

    Args:
        data: 2D array-like of shape (n_samples, n_features).
        metric: Distance metric name (e.g. "cosine", "euclidean") or callable.
        cache_dir: Root directory for disk cache.
        **metric_kwargs: Extra keyword arguments forwarded to scipy.pdist.
            Included in the cache key so different kwargs do not collide.

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

    metric_id = _metric_key(metric, metric_kwargs)
    fp = _fingerprint(X)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fp}_{metric_id}.safetensors"

    if path.exists():
        return load_file(path)["distances"]

    result = pdist(X, metric=metric, **metric_kwargs)
    save_file({"distances": result}, path)
    return result


def clear_distance_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Remove the disk cache directory."""
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def distance_cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return disk cache statistics."""
    disk_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    return {
        "disk_files": len(disk_files),
        "disk_mb": round(sum(f.stat().st_size for f in disk_files) / 1024 / 1024, 2),
    }

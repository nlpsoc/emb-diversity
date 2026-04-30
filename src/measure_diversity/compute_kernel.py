"""
Two-level cached kernel / similarity matrix computation.

Several diversity measures (DCScore, log-determinant diversity,
Renyi kernel entropy, ...) build the same n x n kernel matrix from an
embedding matrix before applying their own post-processing. Computing
this kernel is O(n^2 d), which for 10k embeddings dominates the rest of
the measure cost. This module wraps the kernel construction step in
the same two-level cache used by compute_pairwise:

  Level 1 — in-process memory: a small bounded LRU dict keyed by full
    content fingerprint of the matrix plus the kernel parameters. Up to
    _MEMORY_MAX entries are kept; oldest is evicted on overflow.

  Level 2 — disk: kernel matrix stored under .cache/kernel/ as
    safetensors, keyed the same way. Survives across processes.

  Level 3 — compute the kernel via sklearn (or matmul for "cs") and
    populate both layers.

Supported kernel_type values: "cs" (linear kernel on optionally
L2-normalized rows, scaled by 1/tau), "rbf", "lap", "poly" (sklearn
gaussian / laplacian / polynomial kernels). The cached matrix is the
raw kernel; callers that need a symmetrized version do that themselves
(it is cheap relative to building the kernel).
"""
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import xxhash
from safetensors.numpy import save_file, load_file
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

DEFAULT_CACHE_DIR = Path(".cache/kernel")
_HASH_CHUNK = 1_000_000
# how many kernel matrices to keep in memory before evicting the oldest one (LRU)
_MEMORY_MAX = 4

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")

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


def _kernel_key(kernel_type: str, tau: float, normalize: bool) -> str:
    """Stable, filesystem-safe key for kernel parameters."""
    parts = [kernel_type, f"tau={tau!r}"]
    if kernel_type == "cs":
        parts.append(f"normalize={normalize!r}")
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


def compute_kernel_matrix(
    data: Sequence[Sequence[float]],
    kernel_type: str = "cs",
    tau: float = 1.0,
    normalize: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """
    Compute an n x n kernel / similarity matrix with two-level caching.

    Args:
        data: 2D array-like of shape (n, d), n >= 2.
        kernel_type: One of "cs", "rbf", "lap", "poly".
            - "cs"  : (X_norm @ X_norm.T) / tau, with optional L2-normalization
            - "rbf" : sklearn RBF kernel with gamma=tau
            - "lap" : sklearn Laplacian kernel with gamma=tau
            - "poly": sklearn polynomial kernel with degree=int(tau)
        tau: Kernel parameter (temperature for "cs", gamma for "rbf"/"lap",
            degree for "poly"). Must be positive.
        normalize: For "cs" only — L2-normalize rows so the dot product
            equals cosine similarity. Ignored for other kernel types.
        cache_dir: Root directory for the disk cache.

    Returns:
        Kernel matrix of shape (n, n).

    Raises:
        ValueError: If data has fewer than 2 rows, tau <= 0, or for
            "poly" if tau is not integer-valued.
        NotImplementedError: For unknown kernel_type.
    """
    if kernel_type not in _KERNEL_TYPES:
        raise NotImplementedError(
            f"Unknown kernel_type '{kernel_type}'. Use one of: {_KERNEL_TYPES}."
        )
    if tau <= 0:
        raise ValueError("tau must be positive")

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("kernel matrix requires at least 2 datapoints")

    if kernel_type == "poly" and not float(tau).is_integer():
        raise ValueError("For 'poly' kernel, tau must be an integer (degree).")

    kernel_id = _kernel_key(kernel_type, tau, normalize)
    fp = _fingerprint(X)
    key = f"{fp}|{kernel_id}"

    # Level 1: in-memory match by content fingerprint
    if key in _memory:
        _memory.move_to_end(key)
        return _memory[key]

    # Level 2: disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fp}_{kernel_id}.safetensors"
    if path.exists():
        result = load_file(path)["kernel"]
        _store_memory(key, result)
        return result

    # Level 3: compute, populate both layers
    K = _raw_kernel(X, kernel_type, tau, normalize)
    _store_memory(key, K)
    save_file({"kernel": K}, path)
    return K


def _raw_kernel(X: np.ndarray, kernel_type: str, tau: float, normalize: bool) -> np.ndarray:
    if kernel_type == "cs":
        if normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            X_use = X / norms
        else:
            X_use = X
        return (X_use @ X_use.T) / tau

    if kernel_type == "rbf":
        return rbf_kernel(X, X, gamma=tau)

    if kernel_type == "lap":
        return laplacian_kernel(X, X, gamma=tau)

    # "poly" is the only remaining type after the validation in compute_kernel_matrix
    return polynomial_kernel(X, X, degree=int(tau))


def clear_kernel_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Clear both memory and disk caches."""
    import shutil
    _memory.clear()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def kernel_cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return memory and disk cache statistics."""
    disk_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    return {
        "memory_entries": len(_memory),
        "memory_mb": round(sum(v.nbytes for v in _memory.values()) / 1024 / 1024, 2),
        "memory_max": _MEMORY_MAX,
        "disk_files": len(disk_files),
        "disk_mb": round(sum(f.stat().st_size for f in disk_files) / 1024 / 1024, 2),
    }

"""
Disk-cached kernel / similarity matrix computation.

Several diversity measures (DCScore, log-determinant diversity,
Renyi kernel entropy, ...) build the same n x n kernel matrix from an
embedding matrix before applying their own post-processing. Computing
this kernel is O(n^2 d), which for 10k embeddings dominates the rest of
the measure cost. This module wraps the kernel construction step in the
same content-addressed disk cache used by compute_pairwise: matrix
fingerprint plus kernel parameters become the cache key, and the n x n
result is stored as safetensors under .cache/kernel/.

Supported kernel_type values: "cs" (linear kernel on optionally
L2-normalized rows, scaled by 1/tau), "rbf", "lap", "poly" (sklearn
gaussian / laplacian / polynomial kernels). The cached matrix is the
raw kernel; callers that need a symmetrized version do that themselves
(it is cheap relative to building the kernel).
"""
from pathlib import Path
from typing import Sequence

import numpy as np
import xxhash
from safetensors.numpy import save_file, load_file
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

DEFAULT_CACHE_DIR = Path(".cache/kernel")
_HASH_CHUNK = 1_000_000

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")


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


def compute_kernel_matrix(
    data: Sequence[Sequence[float]],
    kernel_type: str = "cs",
    tau: float = 1.0,
    normalize: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """
    Compute an n x n kernel / similarity matrix with disk caching.

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

    fp = _fingerprint(X)
    key = _kernel_key(kernel_type, tau, normalize)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fp}_{key}.safetensors"

    if path.exists():
        return load_file(path)["kernel"]

    K = _raw_kernel(X, kernel_type, tau, normalize)
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
    """Remove the kernel disk cache directory."""
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def kernel_cache_info(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return disk cache statistics."""
    disk_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    return {
        "disk_files": len(disk_files),
        "disk_mb": round(sum(f.stat().st_size for f in disk_files) / 1024 / 1024, 2),
    }

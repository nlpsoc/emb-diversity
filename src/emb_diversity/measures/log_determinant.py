from __future__ import annotations

from typing import Sequence

from ..embed import resolve_embeddings
from ._types import MeasureResult
from .utils import _require_nonzero_norms

### Distribution-Based Diversity Measure

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")



def log_determinant(
        data: Sequence[Sequence[float]],
        kernel_type: str = "cs",
        tau: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-6,
        use_cholesky: bool = True,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
) -> MeasureResult:
    """
    Log-Determinant Diversity (LDD):
        LDD = log det(K + eps * I)

    K is a similarity / kernel matrix built from embeddings.
    The log-determinant of a positive definite kernel matrix measures
    the "volume" spanned by the data in the feature space, with higher
    values indicating greater diversity.

    Args:
        data:
            Iterable of embedding vectors, shape (n, d).
        kernel_type:
            Type of similarity/kernel:
                - "cs"  : cosine-similarity-like (X Xᵀ / tau, optionally normalized)
                - "rbf" : RBF kernel (uses sklearn.metrics.pairwise.rbf_kernel,
                          with gamma=tau)
                - "lap" : Laplacian kernel (laplacian_kernel, gamma=tau)
                - "poly": Polynomial kernel (polynomial_kernel, degree=tau)
        tau:
            Temperature / kernel parameter.
            For "cs", the similarity matrix is (X Xᵀ) / tau.
            For RBF / Laplacian it is passed as `gamma=tau`,
            for polynomial as `degree=tau`.
        normalize:
            If True and kernel_type=="cs", L2-normalize embeddings row-wise
            before computing X Xᵀ, so dot product equals cosine similarity.
        eps:
            Jitter term added to the diagonal (eps * I) for numerical stability.
            Makes the matrix more positive definite and prevents singular matrices.
        use_cholesky:
            If True, use Cholesky decomposition for efficient logdet computation
            when the matrix is positive definite. Falls back to slogdet if Cholesky fails.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        log-determinant of (K + eps * I) (higher = more diverse) and
        ``parameters`` records the configuration used.

    Raises:
        ValueError:
            If there are fewer than 2 datapoints, or if tau <= 0, or if eps <= 0.
        NotImplementedError:
            For unknown kernel_type.
        np.linalg.LinAlgError:
            If the matrix determinant is not positive (sign <= 0) after adding eps.
            Try increasing eps or re-checking kernel choice.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    parameters = {
        "kernel_type": kernel_type,
        "tau": tau,
        "normalize": normalize,
        "eps": eps,
        "use_cholesky": use_cholesky,
        "embedding_model": embedding_model,
    }

    # ---- Validate inputs ----
    if kernel_type not in _KERNEL_TYPES:
        raise NotImplementedError(
            f"Unknown kernel_type '{kernel_type}'. Use one of: {_KERNEL_TYPES}."
        )
    if tau <= 0:
        raise ValueError("tau must be positive")
    if len(data) < 2:
        raise ValueError("LDD requires at least 2 datapoints")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if kernel_type == "poly" and not float(tau).is_integer():
        raise ValueError("For 'poly' kernel, tau must be an integer (degree).")

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")

    # ---- 1) Build kernel matrix K ----
    if kernel_type == "cs":
        if normalize:
            X_use = X / _require_nonzero_norms(X)[:, None]
        else:
            X_use = X
        K = (X_use @ X_use.T) / tau
    elif kernel_type == "rbf":
        K = rbf_kernel(X, X, gamma=tau)
    elif kernel_type == "lap":
        K = laplacian_kernel(X, X, gamma=tau)
    else:  # poly
        K = polynomial_kernel(X, X, degree=int(tau))

    # Symmetrize for safety (numerical noise)
    K = 0.5 * (K + K.T)

    n = K.shape[0]
    A = K + eps * np.eye(n, dtype=K.dtype)

    # ---- Compute logdet ----
    if use_cholesky:
        try:
            L = np.linalg.cholesky(A)
            return {"value": float(2.0 * np.sum(np.log(np.diag(L)))), "parameters": parameters}
        except np.linalg.LinAlgError:
            # fallback below
            pass

    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        # If this happens often, increase eps or re-check kernel choice.
        raise np.linalg.LinAlgError(
            "logdet undefined: det(A) is not positive (sign <= 0). "
            "Try larger eps or re-check kernel choice."
        )
    return {"value": float(logdet), "parameters": parameters}

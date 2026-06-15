from __future__ import annotations

from typing import Sequence

from ..embed import resolve_embeddings
from ..utility.validate import kernel_row_norms
from ._types import MeasureResult

### Distribution-Based Diversity Measure

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")



def dcscore(
        data: Sequence[Sequence[float]],
        kernel_type: str = "cs",
        tau: float = 1.0,
        normalize: bool = True,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** [1, n] (1 when all points are identical; approaches n when each row's softmax is concentrated on the diagonal).
    The attainable maximum depends on ``kernel_type`` / ``normalize`` and ``tau``; for the default
    ``kernel_type="cs"`` with ``normalize=True`` and ``tau=1``, the maximum for well-separated points
    approaches e ≈ 2.718 as n grows.

    Compute DCScore, a diversity metric based on the self-similarity of the
    samples under a row-wise softmax (as in the original DCScore implementation).

    1) Build a similarity matrix K ∈ R^{n×n} from the input vectors.
    2) Apply a row-wise softmax to K to obtain a probability matrix P.
    3) Return the sum of the diagonal of P: DCScore = sum_i P_ii.

    References:
        Zhu, Yuchang, Huizhe Zhang, Bingzhe Wu, Jintang Li, Zibin Zheng, Peilin Zhao, Liang Chen, and Yatao Bian. "Measuring diversity in synthetic datasets." arXiv preprint arXiv:2502.08512 (2025).

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        kernel_type:
            Type of similarity/kernel:

            - ``"cs"``: cosine-similarity-like (X Xᵀ / tau, optionally normalized).
            - ``"rbf"``: RBF kernel (``rbf_kernel`` with ``gamma=tau``).
            - ``"lap"``: Laplacian kernel (``laplacian_kernel`` with ``gamma=tau``).
            - ``"poly"``: Polynomial kernel (``polynomial_kernel`` with ``degree=tau``).
        tau:
            Temperature / kernel parameter.
            For "cs", the similarity matrix is (X Xᵀ) / tau.
            For RBF / Laplacian it is passed as ``gamma=tau``,
            for polynomial as ``degree=tau``.
        normalize:
            If True and kernel_type=="cs", L2-normalize the input vectors row-wise
            before computing X Xᵀ, as in the original text-based DCScore.
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        scalar DCScore and ``parameters`` records the configuration used.

    Raises:
        ValueError: If there are fewer than 2 datapoints or tau <= 0.
        NotImplementedError: For unknown kernel_type.

    Warns:
        UserWarning: If ``kernel_type="cs"`` and ``normalize=True`` and the
            input contains an all-zero row (cosine similarity is undefined for
            it). The score is still returned, treating the zero row as
            near-orthogonal to every other point.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)

    # ---- Validate inputs ----
    if kernel_type not in _KERNEL_TYPES:
        raise NotImplementedError(
            f"Unknown kernel_type '{kernel_type}'. Use one of: {_KERNEL_TYPES}."
        )
    if tau <= 0:
        raise ValueError("tau must be positive")
    if len(data) < 2:
        raise ValueError("DCScore requires at least 2 datapoints")
    if kernel_type == "poly" and not float(tau).is_integer():
        raise ValueError("For 'poly' kernel, tau must be an integer (degree).")

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")

    # ---- 1) Build kernel matrix K ----
    if kernel_type == "cs":
        if normalize:
            norms = kernel_row_norms(X, "dcscore")
            X_use = X / norms
        else:
            X_use = X
        K = (X_use @ X_use.T) / tau
    elif kernel_type == "rbf":
        K = rbf_kernel(X, X, gamma=tau)
    elif kernel_type == "lap":
        K = laplacian_kernel(X, X, gamma=tau)
    else:  # poly
        K = polynomial_kernel(X, X, degree=int(tau))

    # ---- 2) Row-wise softmax over K ----
    # numerical stability: subtract row max
    K = K - np.max(K, axis=1, keepdims=True)
    exp_K = np.exp(K)
    row_sums = np.sum(exp_K, axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    P = exp_K / row_sums  # each row is a probability distribution

    # ---- 3) Sum of diagonal of P ----
    score = float(np.trace(P))
    return {
        "value": score,
        "parameters": {
            "kernel_type": kernel_type,
            "tau": tau,
            "normalize": normalize,
            "embedding_model": embedding_model,
        },
    }

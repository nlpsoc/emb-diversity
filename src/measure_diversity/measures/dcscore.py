### Distribution-Based Diversity Measure

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")


def dcscore(
        data: Sequence[Sequence[float]],
        kernel_type: str = "cs",
        tau: float = 1.0,
        normalize: bool = True,
) -> float:
    """
    Diversity metric based on DCScore (self-similarity with softmax over rows).

    It follows the logic of `calculate_dcscore_by_embedding` in the original
    DCScore implementation:
        1. Build a similarity matrix K ∈ R^{n×n}.
        2. Apply a row-wise softmax to K to obtain probabilities P.
        3. Return the sum of the diagonal of P: DCScore = ∑_i P_{ii}.

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
            before computing X Xᵀ, as in the original text-based DCScore.

    Returns:
        A scalar DCScore value (float).

    Raises:
        ValueError: If there are fewer than 2 datapoints or tau <= 0.
        NotImplementedError: For unknown kernel_type.
    """
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
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
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
    return score

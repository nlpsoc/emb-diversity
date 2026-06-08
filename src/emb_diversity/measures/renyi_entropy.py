from __future__ import annotations

from typing import Sequence

from ..embed import resolve_embeddings
from ._types import MeasureResult

### Distribution-Based Diversity Measure

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel

_KERNEL_TYPES = ("cs", "rbf", "lap", "poly")



def renyi_entropy(
        data: Sequence[Sequence[float]],
        alpha: float = 2.0,
        kernel_type: str = "cs",
        tau: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-12,
        use_eigendecomp: bool | None = None,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.

    Compute Rényi Kernel Entropy (RKE), a matrix-based Rényi entropy of the
    eigenvalue spectrum of a kernel matrix built from the input vectors. A more
    spread-out spectrum (more modes in the vector space) gives higher entropy.

    1) Build a PSD kernel/similarity matrix K (n x n) from the input vectors.
    2) Normalize to A = K / tr(K) so its eigenvalues lambda_i sum to 1.
    3) Compute the order-``alpha`` Rényi entropy of the eigenvalues of A:
       RKE = (1 / (1 - alpha)) * log(sum_i lambda_i ** alpha).

    Special cases (computed without a full eigendecomposition):

    - alpha = 2: RKE = -log(tr(A^2)) = -log(||A||_F^2).
    - alpha = 1: von Neumann entropy, -sum_i lambda_i * log(lambda_i).

    References:
        Mironov, Mikhail, and Liudmila Prokhorenkova. “Measuring Diversity: Axioms and Challenges.” arXiv:2410.14556. Preprint, arXiv, June 14, 2025. https://doi.org/10.48550/arXiv.2410.14556.
        Jalali, Mohammad, Cheuk Ting Li, and Farzan Farnia. "An information-theoretic evaluation of generative models in learning multi-modal distributions." Advances in Neural Information Processing Systems 36 (2023): 9931-9943.

    Args:
        data:
            (Embedding) vectors of shape (n, d), or raw text strings. Must
            contain at least 2 samples.
        alpha:
            Order of the Rényi entropy. Must be > 0. Defaults to 2.0.
        kernel_type:
            Type of similarity/kernel:

            - ``"cs"``: linear kernel on (optionally) L2-normalized vectors (PSD).
            - ``"rbf"``: RBF kernel with ``gamma=tau``.
            - ``"lap"``: Laplacian kernel with ``gamma=tau``.
            - ``"poly"``: Polynomial kernel with ``degree=int(tau)``.
        tau:
            Kernel parameter:

            - ``"cs"``: temperature scaling via division by tau (K = (X Xᵀ) / tau).
            - ``"rbf"`` / ``"lap"``: passed as ``gamma=tau``.
            - ``"poly"``: ``degree=tau`` (must be an integer).
        normalize:
            If True and kernel_type=="cs", L2-normalize rows so the dot product
            equals cosine similarity.
        eps:
            Jitter for numerical stability (clips eigenvalues, avoids division
            by zero).
        use_eigendecomp:
            If None, choose automatically (False for alpha==2, True otherwise).
            If True, force the eigendecomposition even for alpha==2. If False,
            forbid it (errors if alpha is not in {1, 2}).
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        RKE score and ``parameters`` records the configuration used.

    Raises:
        ValueError:
            If there are fewer than 2 datapoints, if tau <= 0, if alpha <= 0, or
            if use_eigendecomp=False with alpha not in {1, 2}.
        NotImplementedError:
            For unknown kernel_type.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    parameters = {
        "alpha": alpha,
        "kernel_type": kernel_type,
        "tau": tau,
        "normalize": normalize,
        "eps": eps,
        "use_eigendecomp": use_eigendecomp,
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
        raise ValueError("RKE requires at least 2 datapoints")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
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

    # Symmetrize (numerical safety)
    K = 0.5 * (K + K.T)

    # ---- 2) Normalize to trace-1 matrix A ----
    tr = float(np.trace(K))
    if not np.isfinite(tr) or tr <= eps:
        # Degenerate: everything zero-ish or numerical blow-up
        return {"value": 0.0, "parameters": parameters}

    A = K / tr

    # Decide whether to eigendecompose
    if use_eigendecomp is None:
        use_eigendecomp = (abs(alpha - 2.0) > 1e-12) and (abs(alpha - 1.0) > 1e-12)
    parameters["use_eigendecomp"] = use_eigendecomp

    # ---- 3) Compute entropy ----
    # Fast path: alpha = 2
    if abs(alpha - 2.0) <= 1e-12 and not use_eigendecomp:
        frob_sq = float(np.sum(A * A))  # ||A||_F^2 = tr(A^2)
        frob_sq = max(frob_sq, eps)
        return {"value": float(-np.log(frob_sq)), "parameters": parameters}

    # von Neumann entropy: alpha = 1
    if abs(alpha - 1.0) <= 1e-12:
        # Need eigenvalues for log
        evals = np.linalg.eigvalsh(A)
        evals = np.clip(evals, eps, 1.0)
        evals = evals / float(np.sum(evals))  # keep sum=1 (numerical)
        return {"value": float(-np.sum(evals * np.log(evals))), "parameters": parameters}

    # General Rényi: need eigenvalues
    if not use_eigendecomp:
        raise ValueError("use_eigendecomp=False but alpha is not in {1, 2}; cannot compute RKE.")

    evals = np.linalg.eigvalsh(A)  # symmetric PSD-ish
    evals = np.clip(evals, eps, 1.0)
    evals = evals / float(np.sum(evals))  # enforce sum=1

    s = float(np.sum(evals ** alpha))
    s = max(s, eps)
    return {"value": float((1.0 / (1.0 - alpha)) * np.log(s)), "parameters": parameters}

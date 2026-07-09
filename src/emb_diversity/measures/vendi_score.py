from __future__ import annotations

from typing import Sequence

import numpy as np
from vendi_score import vendi

from ..embed import resolve_embeddings
from .types import MeasureResult
from ..utility.validate import warn_on_zero_norm_rows

### Distribution-Based Diversity Measure


def vendi_score(
        data: Sequence[Sequence[float]],
        q: float = 1.0,
        normalize: bool = True,
        use_dual: bool = True,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        chunking_kwargs: dict | None = None,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** [1, n] (the effective number of unique elements).

    Compute diversity using the Vendi Score: the "effective number of unique
    elements" in a set, derived from the entropy of the eigenvalues of a
    similarity matrix over the samples.

    1) Build a similarity matrix over the samples (a dot-product / cosine kernel
       on the input vectors; the dual formulation avoids forming it explicitly).
    2) Take the eigenvalues of the similarity matrix, normalized to sum to 1.
    3) Return the exponential of their order-``q`` (Rényi) entropy — the
       effective number of unique elements.

    References:
        Friedman, Dan, and Adji Bousso Dieng. "The Vendi Score: A Diversity Evaluation Metric for Machine Learning." Transactions on Machine Learning Research (2023).
        Pasarkar, A. P., & Dieng, A. B. (2023). Cousins of the vendi score: A family of similarity-based diversity metrics for science and machine learning. arXiv preprint arXiv:2310.12952.
        
    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        q:
            Order of the Vendi score (Renyi-style generalization).
            q = 1.0 corresponds to the original Vendi Score.
        normalize:
            Whether to L2-normalize rows of X when using dot-product similarity.
            For normalized vectors, the dot product corresponds to cosine similarity.
        use_dual:
            If True, use vendi.score_dual(X, ...) which is efficient when d < n.
            If False, build a Gram matrix K and call vendi.score_K(K, ...).

        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        Vendi Score and ``parameters`` records the configuration used.

    Raises:
        ImportError:
            If vendi_score is not installed.
        ValueError:
            If input is not 2D, or has fewer than 2 datapoints.

    Warns:
        UserWarning: If ``normalize=True`` and the input contains an all-zero
            row (cosine similarity is undefined for it). The score is still
            returned, treating the zero row as near-orthogonal to every other
            point. Applies to both the dual and explicit paths.

    Note:
        Wraps the official ``vendi_score`` implementation
        (https://github.com/vertaix/Vendi-Score).
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="vendi_score", chunking_kwargs=chunking_kwargs)
    parameters = {
        "q": q,
        "normalize": normalize,
        "use_dual": use_dual,
        "embedding_model": embedding_model,
    }

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")
    n, d = X.shape

    if n < 2:
        raise ValueError("Cannot compute Vendi Score for fewer than 2 datapoints")

    # When normalizing, warn (once) on any all-zero row before either path so
    # the dual (library-normalized) and explicit branches behave the same.
    if normalize:
        warn_on_zero_norm_rows(X, "vendi_score")

    # Case 1: use dual formulation (recommended when d <= n, or in general for embeddings)
    if use_dual:
        # vendi.score_dual handles normalization internally
        return {"value": float(vendi.score_dual(X, q=q, normalize=normalize)), "parameters": parameters}

    # Case 2: explicitly build similarity matrix K and call score_K
    # Here we use (normalized) dot product as similarity.
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        X_norm = X / norms
    else:
        X_norm = X

    K = X_norm @ X_norm.T  # Gram matrix of similarities
    return {"value": float(vendi.score_K(K, q=q)), "parameters": parameters}

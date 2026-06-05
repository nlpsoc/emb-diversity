from __future__ import annotations

from typing import Sequence

from ..embed import resolve_embeddings
from ._types import MeasureResult
from .utils import _require_nonzero_norms

### Distribution-Based Diversity Measure

import numpy as np
from vendi_score import vendi



def vendi_score(
        data: Sequence[Sequence[float]],
        q: float = 1.0,
        normalize: bool = True,
        use_dual: bool = True,
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
) -> MeasureResult:
    """
    Compute diversity using the Vendi Score (Friedman & Dieng, 2023).

    The Vendi Score takes a set of samples and a similarity function/kernel,
    and returns an "effective number of unique elements" based on the
    von Neumann / Shannon entropy of the similarity matrix eigenvalues.

    Here we provide a wrapper around the official vendi_score implementation
    (https://github.com/vertaix/Vendi-Score) for embedding data.

    Args:
        data:
            Iterable of embedding vectors, shape (n, d).
        q:
            Order of the Vendi score (Renyi-style generalization).
            q = 1.0 corresponds to the original Vendi Score.
        normalize:
            Whether to L2-normalize rows of X when using dot-product similarity.
            For normalized embeddings, the dot product corresponds to cosine similarity.
        use_dual:
            If True, use vendi.score_dual(X, ...) which is efficient when d < n.
            If False, build a Gram matrix K and call vendi.score_K(K, ...).

        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        Vendi Score (higher = more diverse) and ``parameters`` records the
        configuration used.

    Raises:
        ImportError:
            If vendi_score is not installed.
        ValueError:
            If there are fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
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

    # Validate up front (when normalizing) so the dual and explicit paths behave
    # the same: a zero-norm vector makes the normalized dot-product undefined.
    norms = _require_nonzero_norms(X) if normalize else None

    # Case 1: use dual formulation (recommended when d <= n, or in general for embeddings)
    if use_dual:
        # vendi.score_dual handles normalization internally
        return {"value": float(vendi.score_dual(X, q=q, normalize=normalize)), "parameters": parameters}

    # Case 2: explicitly build similarity matrix K and call score_K
    # Here we use (normalized) dot product as similarity.
    if normalize:
        X_norm = X / norms[:, None]
    else:
        X_norm = X

    K = X_norm @ X_norm.T  # Gram matrix of similarities
    return {"value": float(vendi.score_K(K, q=q)), "parameters": parameters}

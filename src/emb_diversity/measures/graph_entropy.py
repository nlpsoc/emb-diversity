from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from ._types import DistanceMetric, MeasureResult, TensorLike
from .utils import _compute_pairwise_distances

### Graph-Based Diversity Measure


def graph_entropy(data: TensorLike,
                  metric: DistanceMetric = "cosine",
                  *,
                  diversity_axis: str = "semantic",
                  embedding_model: str | None = None,
                  ) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0, grows with n (a sum of per-node entropies, each <= log(n-1)).

    Compute graph entropy over a complete weighted graph whose vertices are the
    data samples and whose edge weights are pairwise distances.

    1) Build a complete weighted graph: the weight of edge (i, j) is the
       pairwise distance d_ij under ``metric``.
    2) Turn each node's edge weights into a local probability distribution: f_ij = d_ij / sum_k d_ik, i.e. normalize node i's distances to
       all other nodes so they sum to 1.
    3) Compute each node's local Shannon entropy:
       H_i = -sum_j f_ij * log(f_ij).
    4) Return the total graph entropy: the sum of all local node entropies.

    References:
        Yu, Yu, Shahram Khadivi, and Jia Xu. "Can data diversity enhance learning generalization?." Proceedings of the 29th international conference on computational linguistics. 2022.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist, used as edge weights. Defaults to
            "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        total graph entropy (sum of all local node entropies) and ``parameters``
        records the configuration used.

    Raises:
        ValueError: If input is not 2D, empty, or has fewer than 2 datapoints.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)

    # normalize input to numpy array; torch is checked via sys.modules so that
    # accepting tensor input does not force the (slow) torch import — if torch
    # was never imported, *data* cannot be a torch tensor.
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected shape (n, d), got {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute graph entropy for fewer than 2 datapoints")

    # calulate essentials

    # 1. pairwise distances
    # only issue with pairwise distances is that it returns a condensed matrix
    # (basically a flattened upper triangular matrix)
    # need to write logic to get a particular distance from the condensed matrix
    dist_condensed = _compute_pairwise_distances(X, metric=metric)

    # 2.lets get the square matrix from the condensed matrix
    dist_square = squareform(dist_condensed)

    # 3. calulate the sum of all distances for each node
    # denomianator of eqaution 30 in Tao's notes
    node_distance_sums = dist_square.sum(axis=1, keepdims=True)  # (n, 1)

    # since all the essentials are calculated, we can now do f_i(d_ij) = d_ij / sum_k d_ik.
    # essentially the local probabilities from a node to all other nodes
    F = np.divide(dist_square, node_distance_sums, out=np.zeros_like(dist_square), where=node_distance_sums != 0)
    F_safe = np.clip(F, 1e-12, 1.0)  # avoid log(0)

    # local entropies
    local_entropies = -np.sum(F * np.log(F_safe), axis=1)

    # finally the graph entropy
    # we can choose mean as well, but strictly follwoing Tao's notes in page 11 and 12
    return {
        "value": float(np.sum(local_entropies)),
        "parameters": {"metric": metric, "embedding_model": embedding_model},
    }

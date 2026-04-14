from __future__ import annotations

from .._accepts_text import accepts_text
from .._types import DISTANCE_METRIC

### Graph-Based Diversity Measure

import numpy as np
from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform
import torch



@accepts_text
def graph_entropy(data: TensorLike,
                  metric: DISTANCE_METRIC = "cosine"
                  ) -> float:
    """
    Computes the graph entropy of a dataset, a metric for structural diversity.

    This implementation follows the methodology described in Tao's notes (Pages 11-12).
    It constructs a complete weighted graph where vertices correspond to data samples
    and edge weights correspond to pairwise distances.

    The calculation proceeds in two main steps:
    1. Local Probabilities (Eq. 30): Determines the relative contribution of each
       edge to a node's total connectivity.
    2. Local Entropy (Eq. 31): Calculates the Shannon entropy for each node based
       on its distance distribution.

    Args:
        data (TensorLike): The input dataset of shape (N, D), where N is the number
            of samples and D is the dimensionality. Must contain at least 2 samples.
        metric (DISTANCE_METRIC, optional): The distance metric to use for edge weights.
            Defaults to "cosine".

    Returns:
        float: The total graph entropy, calculated as the sum of all local node
        entropies.
    """

    # normalize input to numpy array
    if isinstance(data, torch.Tensor):
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
    dist_sqaure = squareform(dist_condensed)

    # 3. calulate the sum of all distances for each node
    # denomianator of eqaution 30 in Tao's notes
    node_distance_sums = dist_sqaure.sum(axis=1, keepdims=True)  # (n, 1)

    # since all the essentials are calculated, we can now do f_i(d_ij) = d_ij / sum_k d_ik.
    # essentially the local probabilities from a node to all other nodes
    F = np.divide(dist_sqaure, node_distance_sums, out=np.zeros_like(dist_sqaure), where=node_distance_sums != 0)
    F_safe = np.clip(F, 1e-12, 1.0)  # avoid log(0)

    # local entropies
    local_entropies = -np.sum(F * np.log(F_safe), axis=1)

    # finally the graph entropy
    # we can choose mean as well, but strictly follwoing Tao's notes in page 11 and 12
    return float(np.sum(local_entropies))

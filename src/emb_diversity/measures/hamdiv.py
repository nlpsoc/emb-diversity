from __future__ import annotations

from typing import Any, Literal, Sequence

import networkx as nx
from networkx.algorithms.approximation import greedy_tsp, christofides
from scipy.spatial.distance import squareform

from ..embed import resolve_embeddings
from ._types import DISTANCE_METRIC, MeasureResult
from .utils import _compute_pairwise_distances

### Distance-Based Diversity Measure


def hamdiv(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        solver: Literal["greedy", "christofides"] = "christofides",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """
    Compute geometric diversity as the length of the shortest Hamiltonian circuit
    (Traveling Salesman Problem tour) through all points.

    Uses NetworkX TSP solvers to find an (approximately) shortest tour.
    NetworkX provides a much simpler API than Google's Or-Tools.
    TODO: In the future & for larger datasets,
        it might be interesting to test Google's or-tools implementation as it uses Cython.

    Args:
        data:
            Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d), or raw text strings.
        metric:
            Distance metric name or callable, as accepted by ``scipy.spatial.distance.pdist``.
            Default is ``"cosine"``.
        solver:
            NetworkX solver strategy. Options:

            - ``"greedy"``: Greedy nearest neighbor heuristic
            - ``"christofides"``: Christofides algorithm (default)

        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to ``pdist``.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        length of the Hamiltonian circuit and ``parameters`` records the
        configuration used.

    Raises:
        ValueError:
            If data is empty or contains fewer than 2 datapoints, or if solver is invalid.
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
    if len(data) < 2:
        raise ValueError("hamdiv requires at least 2 datapoints to compute a Hamiltonian circuit")

    # Validate solver option
    valid_solvers = {"greedy", "christofides"}
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {valid_solvers}, got '{solver}'")

    # Compute pairwise distances
    condensed = _compute_pairwise_distances(data, metric, **metric_kwargs)

    # Convert to full distance matrix
    dist_matrix = squareform(condensed)
    n = len(dist_matrix)

    # Create a complete weighted graph from the distance matrix
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])

    # Solve TSP based on chosen solver
    if solver == "greedy":
        # Greedy/nearest neighbor approach
        tour = greedy_tsp(G, weight='weight', source=0)
    elif solver == "christofides":
        # Christofides algorithm
        tour = christofides(G, weight='weight')

    # Calculate total tour length
    # NetworkX returns a cycle that includes returning to start, so we iterate through pairs
    total_length = 0.0
    for i in range(len(tour) - 1):
        total_length += dist_matrix[tour[i], tour[i + 1]]

    return {
        "value": float(total_length),
        "parameters": {
            "metric": metric,
            "solver": solver,
            "embedding_model": embedding_model,
            **metric_kwargs,
        },
    }

from __future__ import annotations

from .._embed import accepts_text

### Distance-Based Diversity Measure

from .utils import _compute_pairwise_distances
from scipy.spatial.distance import squareform
import networkx as nx
from networkx.algorithms.approximation import greedy_tsp, christofides



@accepts_text
def hamdiv(
        data: Sequence[Sequence[float]],
        metric: DISTANCE_METRIC = "cosine",
        solver: Literal["greedy", "christofides"] = "christofides",
        **metric_kwargs: Any
) -> float:
    """
    Compute geometric diversity as the length of the shortest Hamiltonian circuit
    (Traveling Salesman Problem tour) through all points.

    Uses NetworkX TSP solvers to find an (approximately) shortest tour.
    NetworkX provides a much simpler API than Google's Or-Tools.
    TODO: In the future & for larger datasets,
        it might be interesting to test Google's or-tools implementation as it uses Cython.

    Args:
        data:
            Iterable of vectors (lists/tuples/np.ndarrays), shape (n, d).
        metric:
            Distance metric name or callable, as accepted by ``scipy.spatial.distance.pdist``.
            Default is ``"cosine"``.
        solver:
            NetworkX solver strategy. Options:

            - ``"greedy"``: Greedy nearest neighbor heuristic
            - ``"christofides"``: Christofides algorithm (default)

        **metric_kwargs:
            Extra keyword arguments forwarded to ``pdist``.

    Returns:
        The length of the Hamiltonian circuit as a Python float.

    Raises:
        ValueError:
            If data is empty or contains fewer than 2 datapoints, or if solver is invalid.

    Examples:
        Basic usage with default settings (christofides, cosine distance):

        >>> from embediver import hamdiv
        >>> embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        >>> diversity = hamdiv(embeddings)
        >>> print(f"Diversity: {diversity:.4f}")
        Diversity: 0.0678

        Using different solvers:

        >>> # Fast nearest neighbor
        >>> diversity_nn = hamdiv(embeddings, solver="greedy")
        >>>
        >>> # Best quality (Christofides)
        >>> diversity_chris = hamdiv(embeddings, solver="christofides")
    """
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

    return float(total_length)

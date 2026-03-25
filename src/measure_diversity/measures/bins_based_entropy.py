### Distribution-Based Diversity Measure

import numpy as np
from sklearn.decomposition import PCA

try:
    from umap import UMAP
except Exception:
    UMAP = None


def bins_based_entropy(
    data,
    projection: str = "umap",
    pca_kwargs=None,
    umap_kwargs=None,
    n_bins_x: int = 5,
    n_bins_y: int = 5,
    normalize: bool = True,
    normalization: str = "effective",  # "effective" -> log(min(n,B)), "bins" -> log(B)
) -> float:
    """Compute bins-based entropy diversity from a 2D projection.

    1) Project embeddings to 2D with UMAP or PCA.
    2) Bin points into a n_bins_x × n_bins_y grid.
    3) Compute Shannon entropy over bin occupancies.
    4) Optionally normalize.

    References:
        Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. “Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation.” Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

    Args:
        data:
            Iterable/array-like of embedding vectors with shape (n, d).
            Must contain at least 2 samples.
            Accepts numpy arrays and (optionally) torch tensors.
        projection:
            "umap" or "pca". Defaults to "umap".
        umap_kwargs:
            Extra kwargs passed to UMAP(...).
            Defaults to None (treated as {}).
            If random_state is not provided, random_state=42 is used.
        pca_kwargs:
            Extra kwargs passed to PCA(...).
            Defaults to None (treated as {}).
            PCA is deterministic for full SVD solver.
        n_bins_x:
            Number of bins along x-axis. Must be > 0.
        n_bins_y:
            Number of bins along y-axis. Must be > 0.
        normalize:
            If True, normalize entropy by a log factor.
        normalization:
            - "effective": (default) divide by log(min(n, B)) ensures result in [0,1]
            - "bins": divide by log(B) (paper-style; when n<B max < 1)

    Returns:
        float: bins-based-entropy (normalized if normalize=True) as a float between 0 and 1 (if effective normalization, which is default), where larger values indicate greater diversity.

    Raises:
        ImportError: If projection is "umap" but UMAP is not installed.
        ValueError: If input shape, bins, projection, or normalization is invalid.
    """


    # Normalize input to numpy array
    X = np.asarray(data, dtype=float)
    if X.size == 0:
        raise ValueError("Cannot compute bins_based_entropy for fewer than 2 datapoints")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute bins_based_entropy for fewer than 2 datapoints")

    if n_bins_x <= 0 or n_bins_y <= 0:
        raise ValueError("n_bins_x and n_bins_y must be positive integers")

    total_bins = int(n_bins_x) * int(n_bins_y)

    if projection not in {"umap", "pca"}:
        raise ValueError('projection must be either "umap" or "pca"')

    # Projection kwargs 
    if pca_kwargs is None:
        pca_kwargs = {}
    else:
        pca_kwargs = dict(pca_kwargs)  # copy

    if umap_kwargs is None:
        umap_kwargs = {}
    else:
        umap_kwargs = dict(umap_kwargs)  # copy

    # 2D projection
    # (User can still override solver/whiten/etc via kwargs)
    if projection == "umap":
        if UMAP is None:
            raise ImportError("UMAP is not installed.")
        umap_kwargs.setdefault("random_state", 42)
        reducer = UMAP(n_components=2, **umap_kwargs)
    else:
        reducer = PCA(n_components=2, **pca_kwargs)

    Y = reducer.fit_transform(X)  # shape (n, 2)

    # Compute bounds and ranges for binning
    min_x, min_y = Y.min(axis=0)
    max_x, max_y = Y.max(axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Assign each point to a bin
    eps = 1e-10

    if range_x <= 0:
        bin_x = np.zeros(n, dtype=int)
    else:
        bin_x = np.floor((Y[:, 0] - min_x) / (range_x + eps) * n_bins_x).astype(int)

    if range_y <= 0:
        bin_y = np.zeros(n, dtype=int)
    else:
        bin_y = np.floor((Y[:, 1] - min_y) / (range_y + eps) * n_bins_y).astype(int)

    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_y - 1)

    # Map 2D bins to 1D labels
    bin_labels = bin_x * n_bins_y + bin_y

    # Count occurrences in each occupied bin
    _, counts = np.unique(bin_labels, return_counts=True)

    # Empirical distribution over occupied bins
    f_b = counts / n

    # Shannon entropy over occupied bins (empty bins contribute 0)
    entropy = -np.sum(f_b * np.log(f_b))

    # Optional normalization 
    if normalize:
        if total_bins <= 1:
            # With a 1x1 grid, entropy is always 0; avoid division by zero
            return 0.0

        if normalization not in {"effective", "bins"}:
            raise ValueError('normalization must be either "effective" or "bins"')

        denom_bins = min(n, total_bins) if normalization == "effective" else total_bins
        denom = np.log(denom_bins)
        if denom <= 0:
            return 0.0

        entropy = entropy / denom

        # Numerical safety: keep in [0,1] for effective normalization
        if normalization == "effective":
            entropy = float(np.clip(entropy, 0.0, 1.0))

    return float(entropy)

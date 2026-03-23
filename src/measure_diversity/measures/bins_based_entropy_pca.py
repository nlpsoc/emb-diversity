### Distribution-Based Diversity Measure

import numpy as np
from sklearn.decomposition import PCA


def bins_based_entropy_pca(
    data,
    n_bins_x: int = 5,
    n_bins_y: int = 5,
    normalize: bool = True,
    normalization: str = "effective",  # "effective" -> log(min(n,B)), "bins" -> log(B)
    pca_kwargs=None,
) -> float:
    """
    Compute bins-based entropy diversity using 2D UMAP projection.

    Reference: 
    Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. “Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation.” Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

    Steps:
      1) Project embeddings to 2D with PCA.
      2) Bin points into a n_bins_x × n_bins_y grid.
      3) Compute Shannon entropy over bin occupancies.
      4) Optionally normalize.

    Args:
        data:
            Iterable/array-like of embedding vectors, shape (n, d).
            Must contain at least 2 samples.
            Accepts numpy arrays and (optionally) torch tensors.
        n_bins_x:
            Number of bins along x-axis. Must be > 0.
        n_bins_y:
            Number of bins along y-axis. Must be > 0.
        normalize:
            If True, normalize entropy by a log factor.
        normalization:
            - "effective": divide by log(min(n, B)) ensures result in [0,1]
            - "bins": divide by log(B) (paper-style; when n<B max < 1)
        pca_kwargs:
            Extra kwargs passed to PCA(...).
            Defaults to {}. (PCA is deterministic for full SVD solver.)

    Returns:
        float: entropy (normalized if normalize=True).

    Raises:
        ImportError:
            If scikit-learn is not installed.
        ValueError:
            If input shapes/bins are invalid, or if normalization is invalid.
    """


    # --- Normalize input to numpy array ---
    X = np.asarray(data, dtype=float)
    if X.size == 0:
        raise ValueError("Cannot compute bins_based_entropy for fewer than 2 datapoints")

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, d), got shape {X.shape}")

    n, d = X.shape
    if n < 2:
        raise ValueError("Cannot compute bins_based_entropy_pca for fewer than 2 datapoints")

    if n_bins_x <= 0 or n_bins_y <= 0:
        raise ValueError("n_bins_x and n_bins_y must be positive integers")

    total_bins = int(n_bins_x) * int(n_bins_y)

    # --- PCA kwargs (deterministic by default) ---
    if pca_kwargs is None:
        pca_kwargs = {}
    else:
        pca_kwargs = dict(pca_kwargs)  # copy

    # Ensure we always do 2D projection
    # (User can still override solver/whiten/etc via pca_kwargs)
    pca = PCA(n_components=2, **pca_kwargs)
    Y = pca.fit_transform(X)  # shape (n, 2)

    # --- Compute bounds and ranges ---
    min_x, min_y = Y.min(axis=0)
    max_x, max_y = Y.max(axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y

    # --- Assign each point to a bin (robust to degenerate projections) ---
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

    # Map 2D bins -> 1D labels
    bin_labels = bin_x * n_bins_y + bin_y

    # --- Count occurrences in each occupied bin ---
    _, counts = np.unique(bin_labels, return_counts=True)

    # Empirical distribution over occupied bins
    f_b = counts / n

    # Shannon entropy over occupied bins (empty bins contribute 0)
    entropy = -np.sum(f_b * np.log(f_b))

    # --- Optional normalization ---
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

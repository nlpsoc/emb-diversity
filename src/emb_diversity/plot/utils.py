"""Utilities for visualization functions."""

import numpy as np
import colorsys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from typing import Iterable, List



def _reduce_dim(
    embeddings: np.ndarray,
    method: str,
    n_components: int,
    *,
    tsne_perplexity: float = 30.0,
    tsne_lr: float = 200.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce dimensionality of an embedding matrix.

    Apply PCA, t‑SNE, or UMAP to project embeddings into a lower‑dimensional
    space.

    Args:
        embeddings:
            A numpy array of shape (n, d) containing embeddings.
        method:
            Dimensionality reduction method. One of {"pca", "tsne", "umap"}.
        n_components:
            Number of output dimensions.
        tsne_perplexity:
            Perplexity parameter for t‑SNE.
        tsne_lr:
            Learning rate for t‑SNE.
        umap_n_neighbors:
            Number of neighbors for UMAP.
        umap_min_dist:
            Minimum distance parameter for UMAP.
        random_state:
            Random seed for reproducibility.

    Returns:
        A numpy array of shape (n, n_components) containing the projected
        coordinates.

    Raises:
        ValueError: If an unknown method is provided.

    Example:
        coords = _reduce_dim(embeddings, method="pca", n_components=2)
    """

    method = method.lower()

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

        if tsne_perplexity >= embeddings.shape[0]:
            raise ValueError(
                f"tsne_perplexity must be < n_samples ({embeddings.shape[0]}), got {tsne_perplexity}"
            )
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_lr,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    if method == "umap":
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    raise ValueError("method must be one of: 'pca', 'tsne', 'umap'")


def _validate_embeddings(embeddings: Iterable[np.ndarray]) -> List[np.ndarray]:
    """Validate and normalize embedding input.

    Convert a numpy array or iterable of numpy arrays into a validated list of
    2‑dimensional embedding matrices. Ensures each array is of shape (n, d).

    Args:
        embeddings:
            A numpy array of shape (n, d) or an iterable of such arrays. Each
            array must be 2‑dimensional.

    Returns:
        A list of numpy arrays, each of shape (n, d).

    Raises:
        TypeError: If the input is not a numpy array or iterable of arrays.
        TypeError: If any element is not a numpy array.
        ValueError: If any array is not 2‑dimensional.

    Example:
        validated = _validate_embeddings([np.random.randn(10, 768)])
        len(validated)
    """

    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]

    if not isinstance(embeddings, Iterable):
        raise TypeError("embeddings must be a numpy array or iterable of arrays")

    validated = []
    for e in embeddings:
        if not isinstance(e, np.ndarray):
            raise TypeError("each embedding must be a numpy array")
        if e.ndim != 2:
            raise ValueError("each embedding must be 2D (N, D)")
        validated.append(e)

    return validated


def _assign_color(rgb, factor=0.6):
    """Return a lighter version of an RGB color.

    Args:
        rgb (tuple[float, float, float]):
            The base color expressed as an RGB tuple, where each component
            is in the range [0.0, 1.0].
        factor (float, optional):
            A value in the range [0.0, 1.0] that determines how much lighter
            the output color becomes. Higher values produce lighter colors.
            Defaults to 0.6.

    Returns:
        tuple[float, float, float]:
            A new RGB tuple representing the lighter color.

    Raises:
        ValueError:
            If `rgb` does not contain exactly three numeric components.

    """
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = min(1.0, l + (1 - l) * factor)
    return colorsys.hls_to_rgb(h, l, s)

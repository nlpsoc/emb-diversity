"""Visualize embedding spaces. Assumes that embeddings are numeric."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import Iterable, List, Tuple
from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap
from emb_diversity.plot.utils import _validate_embeddings, _reduce_dim, _assign_color


def plot_2d(
    embeddings: Iterable[np.ndarray] | np.ndarray,
    method: str = "pca",
    *,
    tsne_perplexity: float = 30.0,
    tsne_lr: float = 200.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    labels: list[str] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:

    embeddings = _validate_embeddings(embeddings)
    all_embs = np.vstack(embeddings)

    coords = _reduce_dim(
        all_embs,
        method,
        n_components=2,
        tsne_perplexity=tsne_perplexity,
        tsne_lr=tsne_lr,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )

    sizes = [e.shape[0] for e in embeddings]
    coords_split = np.split(coords, np.cumsum(sizes)[:-1])
    colors = sns.color_palette("Set1", len(embeddings))

    fig, ax = plt.subplots(figsize=(8, 6))

    # KDE background per group
    for group_coords, color in zip(coords_split, colors):
        # KDE needs enough samples to estimate a density

        if group_coords.shape[0] < 2:

            continue



        light = _assign_color(color, factor=0.55)
        cmap = LinearSegmentedColormap.from_list(
            "lighter_kde", [(1, 1, 1), light], N=256
        )

        sns.kdeplot(
            x=group_coords[:, 0],
            y=group_coords[:, 1],
            fill=True,
            cmap=cmap,
            alpha=0.35,
            bw_adjust=0.7,
            thresh=0.05,
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(embeddings))]
    elif len(labels) != len(embeddings):
        raise ValueError(f"labels must have length {len(embeddings)}, got {len(labels)}")

        )

    # Scatter groups
    labels = labels or [f"Dataset {i + 1}" for i in range(len(embeddings))]
    for i, (group_coords, color) in enumerate(zip(coords_split, colors)):
        ax.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            s=40,
            alpha=0.8,
            color=color,
            label=labels[i],
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0)

    ax.set_title(f"2D Projection ({method.upper()})")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_3d(
    embeddings: Iterable[np.ndarray] | np.ndarray,
    method: str = "pca",
    *,
    point_size: int = 4,
    tsne_perplexity: float = 30.0,
    tsne_lr: float = 200.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    labels: list[str] | None = None,
) -> go.Figure:
    """Create an interactive 3D projection of embedding matrices.

    Generate a 3D visualization using PCA, t‑SNE, or UMAP. All embedding
    matrices are projected jointly and displayed using Plotly.

    Args:
        embeddings:
            A numpy array of shape (n, d) or a list of such arrays. Each array
            must be 2‑dimensional.
        method:
            Dimensionality reduction method. One of {"pca", "tsne", "umap"}.
        point_size:
            Marker size for the 3D scatter plot.
        tsne_perplexity:
            Perplexity parameter for t‑SNE.
        tsne_lr:
            Learning rate for t‑SNE.
        umap_n_neighbors:
            Number of neighbors for UMAP.
        umap_min_dist:
            Minimum distance parameter for UMAP.
        labels:
            (Optional) names for each dataset as shown in the legend.

    Returns:
        A Plotly ``Figure`` containing the interactive 3D scatter plot.

    Raises:
        TypeError: If embeddings is not an array or iterable of arrays.
        ValueError: If any embedding array is not 2‑dimensional.

    Example:
        fig = plot_3d(embeddings, method="pca")
        fig.write_html("projection.html")
    """

    embeddings = _validate_embeddings(embeddings)
    all_embs = np.vstack(embeddings)

    coords = _reduce_dim(
        all_embs,
        method,
        n_components=3,
        tsne_perplexity=tsne_perplexity,
        tsne_lr=tsne_lr,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )

    sizes = [e.shape[0] for e in embeddings]
    coords_split = np.split(coords, np.cumsum(sizes)[:-1])
    colors = [to_hex(c) for c in sns.color_palette("Set1", len(embeddings))]

    fig = go.Figure()

    for i, (group_coords, color) in enumerate(zip(coords_split, colors)):
        label = labels[i] if labels else f"Dataset {i + 1}"
        fig.add_trace(
            go.Scatter3d(
                x=group_coords[:, 0],
                y=group_coords[:, 1],
                z=group_coords[:, 2],
                mode="markers",
                marker=dict(size=point_size, opacity=0.85, color=color),
                name=label,
            )
        )

    fig.update_layout(
        title=f"3D Projection ({method.upper()})",
        scene=dict(
            xaxis_title=f"{method.upper()}‑1",
            yaxis_title=f"{method.upper()}‑2",
            zaxis_title=f"{method.upper()}‑3",
        ),
        width=900,
        height=700,
    )

    return fig

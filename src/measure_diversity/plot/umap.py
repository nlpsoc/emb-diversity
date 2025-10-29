import numpy as np
import pandas as pd
from typing import List, Optional, Dict


def plot_umap_plotnine(
    vectors: List[List[float]] | np.ndarray,
    title: str = "UMAP Projection",
    labels: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
):
    """
    Create a UMAP plot from a list of vectors using plotnine.

    Args:
        vectors: List of vectors or numpy array with shape (n_samples, n_features)
        title: Plot title
        labels: Optional labels for each point (for coloring)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random state for reproducibility

    Returns:
        plotnine ggplot object
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP plotting. "
            "Install it with: pip install umap-learn"
        ) from e

    try:
        from plotnine import ggplot, aes, geom_point, labs, theme_minimal, theme, element_text
    except ImportError as e:
        raise ImportError(
            "plotnine is required for plotting. "
            "Install it with: pip install plotnine"
        ) from e

    # Convert to numpy array if needed
    if isinstance(vectors, list):
        vectors = np.array(vectors)

    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state
    )
    embedding = reducer.fit_transform(vectors)

    # Create dataframe for plotting
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1]
    })

    if labels is not None:
        df['label'] = labels
        plot = (
            ggplot(df, aes(x='UMAP1', y='UMAP2', color='label'))
            + geom_point(size=3, alpha=0.7)
            + labs(title=title, x='UMAP 1', y='UMAP 2')
            + theme_minimal()
            + theme(
                text=element_text(size=18),
                axis_title=element_text(size=22, weight='bold'),
                axis_text=element_text(size=18),
                plot_title=element_text(size=24, weight='bold'),
                legend_title=element_text(size=20, weight='bold'),
                legend_text=element_text(size=18)
            )
        )
    else:
        plot = (
            ggplot(df, aes(x='UMAP1', y='UMAP2'))
            + geom_point(size=3, alpha=0.7, color='steelblue')
            + labs(title=title, x='UMAP 1', y='UMAP 2')
            + theme_minimal()
            + theme(
                text=element_text(size=18),
                axis_title=element_text(size=22, weight='bold'),
                axis_text=element_text(size=18),
                plot_title=element_text(size=24, weight='bold')
            )
        )

    return plot


def plot_umap_comparable(
    datasets: Dict[str, List[List[float]] | np.ndarray],
    title: str = "UMAP Projection",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
):
    """
    Create comparable UMAP plots from multiple datasets using the same UMAP reduction.

    This function concatenates all datasets, fits UMAP once, then creates separate plots
    for each dataset that are directly comparable because they share the same embedding space.

    Args:
        datasets: Dictionary mapping dataset names to their vectors
        title: Base title for plots (dataset name will be appended)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random state for reproducibility

    Returns:
        Dictionary mapping dataset names to plotnine ggplot objects
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP plotting. "
            "Install it with: pip install umap-learn"
        ) from e

    try:
        from plotnine import ggplot, aes, geom_point, labs, theme_minimal, theme, element_text
    except ImportError as e:
        raise ImportError(
            "plotnine is required for plotting. "
            "Install it with: pip install plotnine"
        ) from e

    # Convert all datasets to numpy arrays
    converted_datasets = {}
    for name, vectors in datasets.items():
        if isinstance(vectors, list):
            converted_datasets[name] = np.array(vectors)
        else:
            converted_datasets[name] = vectors

    # Concatenate all datasets
    all_vectors = np.vstack(list(converted_datasets.values()))

    # Create labels to track which dataset each point belongs to
    dataset_labels = []
    for name, vectors in converted_datasets.items():
        dataset_labels.extend([name] * len(vectors))

    # Fit UMAP on all data
    reducer = umap.UMAP(
        # n_neighbors=n_neighbors,
        # min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric="cosine"
    )
    embedding = reducer.fit_transform(all_vectors)

    # Create combined plot with all datasets
    df_combined = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Dataset': dataset_labels
    })

    combined_plot = (
        ggplot(df_combined, aes(x='UMAP1', y='UMAP2', color='Dataset'))
        + geom_point(size=3, alpha=0.7)
        + labs(title=f"{title} - All Datasets", x='UMAP 1', y='UMAP 2')
        + theme_minimal()
        + theme(
            text=element_text(size=18),
            axis_title=element_text(size=22, weight='bold'),
            axis_text=element_text(size=18),
            plot_title=element_text(size=24, weight='bold'),
            legend_title=element_text(size=20, weight='bold'),
            legend_text=element_text(size=18)
        )
    )

    # Split embeddings back into separate datasets
    plots = {'combined': combined_plot}
    start_idx = 0
    for name, vectors in converted_datasets.items():
        end_idx = start_idx + len(vectors)
        dataset_embedding = embedding[start_idx:end_idx]

        # Create dataframe for this dataset
        df = pd.DataFrame({
            'UMAP1': dataset_embedding[:, 0],
            'UMAP2': dataset_embedding[:, 1]
        })

        # Create plot
        plot = (
            ggplot(df, aes(x='UMAP1', y='UMAP2'))
            + geom_point(size=3, alpha=0.7, color='steelblue')
            + labs(title=f"{title} - {name}", x='UMAP 1', y='UMAP 2')
            + theme_minimal()
            + theme(
                text=element_text(size=18),
                axis_title=element_text(size=22, weight='bold'),
                axis_text=element_text(size=18),
                plot_title=element_text(size=24, weight='bold')
            )
        )

        plots[name] = plot
        start_idx = end_idx

    return plots
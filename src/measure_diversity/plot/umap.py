import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple


def compute_umap(
    vectors: List[List[float]] | np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP embedding for vectors.

    Args:
        vectors: List of vectors or numpy array with shape (n_samples, n_features)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP distance metric (default: "cosine", use "euclidean" for non-text embeddings)
        random_state: Random state for reproducibility

    Returns:
        UMAP embedding as numpy array with shape (n_samples, 2)
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for UMAP. "
            "Install it with: pip install umap-learn"
        ) from e

    # Convert to numpy array if needed
    if isinstance(vectors, list):
        vectors = np.array(vectors)

    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(vectors)

    return embedding


def create_umap_plot(
    embedding: np.ndarray,
    title: str = "UMAP Projection",
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_axis_labels: bool = False,
    show_axis_ticks: bool = False,
    show_legend: bool = True,
    color: str = 'steelblue',
    point_size: float = 7,
    point_alpha: float = 0.45,
    font_size: int = 39,
    title_size: int = 42,
    axis_title_size: int = 39,
    legend_title_size: int = 39,
    legend_text_size: int = 39,
    width: float = 10,
    height: float = 8,
    dpi: int = 300,
    label_order: Optional[List[str]] = None
):
    """
    Create a plot from UMAP embeddings.

    Args:
        embedding: UMAP embedding array with shape (n_samples, 2)
        title: Plot title
        labels: Optional labels for each point (for coloring by category)
        save_path: Optional path to save the plot as PNG. If None, plot is not saved.
        show_axis_labels: Whether to show x and y axis labels
        show_axis_ticks: Whether to show x and y axis ticks
        show_legend: Whether to show the legend (only applies when labels provided)
        color: Color for points when labels is None
        point_size: Size of points
        point_alpha: Transparency of points (0-1)
        font_size: Base font size for axis text
        title_size: Font size for plot title
        axis_title_size: Font size for axis titles
        legend_title_size: Font size for legend title (only used when labels provided)
        legend_text_size: Font size for legend text (only used when labels provided)
        width: Width of saved plot in inches
        height: Height of saved plot in inches
        dpi: DPI (resolution) for saved plot
        label_order: Optional list specifying the order of labels in the legend

    Returns:
        plotnine ggplot object
    """
    try:
        from plotnine import ggplot, aes, geom_point, labs, theme_minimal, theme, element_text, element_blank, scale_color_discrete
    except ImportError as e:
        raise ImportError(
            "plotnine is required for plotting. "
            "Install it with: pip install plotnine"
        ) from e

    # Create dataframe for plotting
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1]
    })

    # Add labels if provided
    if labels is not None:
        df['Dataset'] = labels

    # Determine axis labels
    x_label = 'UMAP 1' if show_axis_labels else ''
    y_label = 'UMAP 2' if show_axis_labels else ''

    # Create plot
    if labels is not None:
        plot = (
            ggplot(df, aes(x='UMAP1', y='UMAP2', color='Dataset'))
            + geom_point(size=point_size, alpha=point_alpha)
            + labs(title=title, x=x_label, y=y_label)
            + theme_minimal()
        )
        # Add scale_color_discrete to control both color assignment and legend order
        # - limits: controls the mapping of categories to colors (assigns colors in this order)
        # - breaks: controls the display order in the legend
        if label_order is not None:
            plot = plot + scale_color_discrete(limits=label_order, breaks=label_order)
    else:
        plot = (
            ggplot(df, aes(x='UMAP1', y='UMAP2'))
            + geom_point(size=point_size, alpha=point_alpha, color=color)
            + labs(title=title, x=x_label, y=y_label)
            + theme_minimal()
        )

    # Build theme
    theme_dict = {
        'text': element_text(size=font_size),
        'plot_title': element_text(size=title_size, weight='bold', ha='center'),
        'figure_size': (width, height),
        'aspect_ratio': 1.0,  # Keep plot area square
        'plot_title_position': 'plot'  # Position title above entire figure (plot + legend)
    }

    # Add legend styling if labels provided
    if labels is not None and show_legend:
        theme_dict['legend_title'] = element_text(size=legend_title_size, weight='bold')
        theme_dict['legend_text'] = element_text(size=legend_text_size)
        theme_dict['legend_box_margin'] = 0
        theme_dict['legend_position'] = 'right'
    elif labels is not None and not show_legend:
        theme_dict['legend_position'] = 'none'

    # Add axis styling
    if show_axis_labels:
        theme_dict['axis_title'] = element_text(size=axis_title_size, weight='bold')
    else:
        theme_dict['axis_title'] = element_blank()

    if show_axis_ticks:
        theme_dict['axis_text'] = element_text(size=font_size)
    else:
        theme_dict['axis_text'] = element_blank()
        theme_dict['axis_ticks'] = element_blank()

    plot = plot + theme(**theme_dict)

    # Save plot if path provided
    if save_path is not None:
        plot.save(save_path, width=width, height=height, dpi=dpi, verbose=False)

    return plot


def plot_umap(
    vectors: List[List[float]] | np.ndarray,
    title: str = "UMAP Projection",
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    show_axis_labels: bool = False,
    show_axis_ticks: bool = False,
    show_legend: bool = True,
    color: str = 'steelblue',
    point_size: float = 4,
    point_alpha: float = 0.7,
    font_size: int = 22,
    title_size: int = 24,
    axis_title_size: int = 22,
    legend_title_size: int = 20,
    legend_text_size: int = 22,
    width: float = 10,
    height: float = 8,
    dpi: int = 300,
    return_embedding: bool = False
):
    """
    Compute UMAP and create a customizable plot (convenience function).

    This function combines compute_umap() and create_umap_plot() for convenience.

    Args:
        vectors: List of vectors or numpy array with shape (n_samples, n_features)
        title: Plot title
        labels: Optional labels for each point (for coloring by category)
        save_path: Optional path to save the plot as PNG. If None, plot is not saved.
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP distance metric (default: "cosine", use "euclidean" for non-text embeddings)
        random_state: Random state for reproducibility
        show_axis_labels: Whether to show x and y axis labels
        show_axis_ticks: Whether to show x and y axis ticks
        color: Color for points when labels is None
        point_size: Size of points
        point_alpha: Transparency of points (0-1)
        font_size: Base font size for axis text
        title_size: Font size for plot title
        axis_title_size: Font size for axis titles
        legend_title_size: Font size for legend title (only used when labels provided)
        legend_text_size: Font size for legend text (only used when labels provided)
        width: Width of saved plot in inches
        height: Height of saved plot in inches
        dpi: DPI (resolution) for saved plot
        return_embedding: If True, returns (plot, embedding), else just plot

    Returns:
        plotnine ggplot object, or tuple of (plot, embedding) if return_embedding=True
    """
    # Compute UMAP embedding
    embedding = compute_umap(
        vectors=vectors,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    # Create plot
    plot = create_umap_plot(
        embedding=embedding,
        title=title,
        labels=labels,
        save_path=save_path,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_legend=show_legend,
        color=color,
        point_size=point_size,
        point_alpha=point_alpha,
        font_size=font_size,
        title_size=title_size,
        axis_title_size=axis_title_size,
        legend_title_size=legend_title_size,
        legend_text_size=legend_text_size,
        width=width,
        height=height,
        dpi=dpi
    )

    if return_embedding:
        return plot, embedding
    return plot


def plot_umap_comparable(
    datasets: Dict[str, List[List[float]] | np.ndarray],
    title: str = "UMAP Projection",
    save_dir: Optional[str] = None,
    create_individual_plots: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    show_axis_labels: bool = False,
    show_axis_ticks: bool = False,
    show_legend: bool = True,
    width: float = 10,
    height: float = 8,
    dpi: int = 300
):
    """
    Create comparable UMAP plots from multiple datasets using the same UMAP reduction.

    This function concatenates all datasets, computes UMAP once, then optionally creates
    separate plots for each dataset that are directly comparable because they share the
    same embedding space.

    Args:
        datasets: Dictionary mapping dataset names to their vectors.
                  The order of keys in the dictionary determines:
                  1. Legend order (first key appears first in legend)
                  2. Color assignment (first key gets first color)
                  3. Plotting order (first key plotted LAST, appearing on top)
        title: Base title for plots (dataset name will be appended)
        save_dir: Optional directory to save plots. If provided, saves:
                  - {save_dir}/{title}_combined.png (always)
                  - {save_dir}/{title}_{dataset_name}.png (only if create_individual_plots=True)
        create_individual_plots: Whether to create individual plots for each dataset (default: False)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP distance metric (default: "cosine")
        random_state: Random state for reproducibility
        show_axis_labels: Whether to show x and y axis labels (default: False)
        show_axis_ticks: Whether to show x and y axis ticks (default: False)
        show_legend: Whether to show the legend in combined plot (default: True)
        width: Width of saved plots in inches
        height: Height of saved plots in inches
        dpi: DPI (resolution) for saved plots

    Returns:
        plotnine ggplot object (combined plot), or dict with 'combined' and individual plots
        if create_individual_plots=True
    """
    # Convert all datasets to numpy arrays
    converted_datasets = {}
    for name, vectors in datasets.items():
        if isinstance(vectors, list):
            converted_datasets[name] = np.array(vectors)
        else:
            converted_datasets[name] = vectors

    # Preserve the order for legend (as provided in input)
    dataset_order = list(converted_datasets.keys())

    # Concatenate datasets in REVERSE order for plotting
    # This way, the first dataset in the legend will be plotted last (on top)
    reversed_datasets = list(reversed(list(converted_datasets.items())))
    all_vectors = np.vstack([vectors for name, vectors in reversed_datasets])

    # Create labels in reversed order to match the concatenation
    dataset_labels = []
    for name, vectors in reversed_datasets:
        dataset_labels.extend([name] * len(vectors))

    # Compute UMAP embedding once for all data
    embedding = compute_umap(
        vectors=all_vectors,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    # Prepare save path for combined plot if needed
    combined_save_path = None
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        safe_title = title.replace(' ', '_').replace('/', '_')
        combined_save_path = os.path.join(save_dir, f"{safe_title}_combined.png")

    # Create combined plot with all datasets
    # Pass label_order to preserve the order of datasets in the legend
    combined_plot = create_umap_plot(
        embedding=embedding,
        title=title,
        labels=dataset_labels,
        save_path=combined_save_path,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_legend=show_legend,
        width=width,
        height=height,
        dpi=dpi,
        label_order=dataset_order
    )

    # Only create individual plots if requested
    if not create_individual_plots:
        return combined_plot

    # Split embeddings back into separate datasets and create individual plots
    # Note: embeddings are in reversed order, so iterate through reversed_datasets
    plots = {'combined': combined_plot}
    start_idx = 0
    for name, vectors in reversed_datasets:
        end_idx = start_idx + len(vectors)
        dataset_embedding = embedding[start_idx:end_idx]

        # Prepare save path for individual plot if needed
        individual_save_path = None
        if save_dir is not None:
            safe_name = name.replace(' ', '_').replace('/', '_')
            individual_save_path = os.path.join(save_dir, f"{safe_title}_{safe_name}.png")

        # Create individual plot using pre-computed embeddings
        plot = create_umap_plot(
            embedding=dataset_embedding,
            title=f"{title} - {name}",
            labels=None,
            save_path=individual_save_path,
            show_axis_labels=show_axis_labels,
            show_axis_ticks=show_axis_ticks,
            width=width,
            height=height,
            dpi=dpi
        )

        plots[name] = plot
        start_idx = end_idx

    return plots
"""embediver – embedding-based diversity measures for text and vector data."""

from __future__ import annotations

### Distance-Based Diversity Measures
from .measures.mean_pw_dist import mean_pw_dist
from .measures.dist_dispersion import dist_dispersion
from .measures.hamdiv import hamdiv
from .measures.diameter import diameter
from .measures.bottleneck import bottleneck
from .measures.sum_diameter import sum_diameter
from .measures.energy import energy
from .measures.cluster_inertia import cluster_inertia
from .measures.span_centroid import span_centroid
from .measures.chamfer_dist import chamfer_dist

### Volume-Based Diversity Measures
from .measures.convex_hull_volume import convex_hull_volume
from .measures.radius import radius
from .measures.span_medoid import span_medoid

### Distribution-Based Diversity Measures
from .measures.vendi_score import vendi_score
from .measures.renyi_entropy import renyi_entropy
from .measures.dcscore import dcscore
from .measures.log_determinant import log_determinant
from .measures.bins_entropy import bins_entropy

### Graph-Based Diversity Measures
from .measures.graph_entropy import graph_entropy
from .measures.mst_dispersion import mst_dispersion

### Axes API
from ._axes import register_axis, get_axis, list_axes

### Embedding helper
from ._embed import embed_texts

### Convenience functions
from ._registry import MEASURES, DEFAULT_MEASURE, CORE_MEASURES


def measure_all(
    data,
    measures: list[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Run multiple diversity measures and return results as a dict.

    Args:
        data: Embedding vectors (n, d) or a list of text strings.
        measures: Measure names to run.  If ``None``, runs all 20.
        **kwargs: Extra arguments forwarded to each measure
            (e.g. ``metric``, ``diversity_axis``, ``embedding_model``).

    Returns:
        Dict mapping measure name to score.
    """
    registry = MEASURES
    names = measures if measures is not None else list(registry)
    results: dict[str, float] = {}
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown measure {name!r}. Available: {sorted(registry)}")
        try:
            results[name] = registry[name](data, **kwargs)
        except Exception as exc:
            results[name] = float("nan")
    return results


def measure_text(
    texts: list[str],
    measures: list[str] | None = None,
    diversity_axis: str = "semantic",
    embedding_model: str | None = None,
    **kwargs,
) -> dict[str, float]:
    """Embed texts and measure diversity in one call.

    Args:
        texts: List of text strings.
        measures: Measure names to run (default: all).
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit model id; overrides *diversity_axis*.
        **kwargs: Extra arguments forwarded to each measure.

    Returns:
        Dict mapping measure name to score.
    """
    vectors = embed_texts(texts, diversity_axis=diversity_axis, embedding_model=embedding_model)
    return measure_all(vectors, measures=measures, **kwargs)


__all__ = [
    # Measures
    "mean_pw_dist", "dist_dispersion", "hamdiv", "diameter", "bottleneck",
    "sum_diameter", "energy", "cluster_inertia", "span_centroid", "chamfer_dist",
    "convex_hull_volume", "radius", "span_medoid", "vendi_score",
    "renyi_entropy", "dcscore", "log_determinant", "bins_entropy",
    "graph_entropy", "mst_dispersion",
    # Convenience
    "measure_all", "measure_text", "embed_texts",
    # Axes
    "register_axis", "get_axis", "list_axes",
]

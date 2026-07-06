"""emb-diversity – embedding-based diversity measures for text and vector data."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .measures_registry import MEASURE_NAMES

# Public names resolve lazily on first attribute access (PEP 562), keeping
# `import emb_diversity` fast: the heavy dependencies (torch, scikit-learn,
# umap, …) load only when something that needs them is actually called.
# Maps each public name to the submodule that defines it.
_ATTR_TO_SUBMODULE = {
    # Main entry point
    "measure_diversity": "convenience",
    # Individual measures
    **{name: f"measures.{name}" for name in MEASURE_NAMES},
    # Embedding helpers
    "embed_texts": "embed",
    "embed_audio": "embed",
    # Axes registry
    "axes": "axes_registry",
    # Pairwise distance caching
    "compute_pairwise_distances": "measures.utils",
    "clear_distance_cache": "measures.utils",
    "distance_cache_info": "measures.utils",
}

__all__ = list(_ATTR_TO_SUBMODULE)


def __getattr__(name: str):
    try:
        submodule = _ATTR_TO_SUBMODULE[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    attr = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = attr  # cache so later accesses skip __getattr__
    return attr


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ATTR_TO_SUBMODULE))


if TYPE_CHECKING:
    # Static-only mirror of the lazy attributes above, so IDEs and type
    # checkers can resolve `from emb_diversity import <name>`.
    from .axes_registry import axes
    from .measures.utils import (
        clear_distance_cache,
        compute_pairwise_distances,
        distance_cache_info,
    )
    from .convenience import measure_diversity
    from .embed import embed_texts, embed_audio
    from .measures.bins_entropy import bins_entropy
    from .measures.bottleneck import bottleneck
    from .measures.chamfer_dist import chamfer_dist
    from .measures.cluster_inertia import cluster_inertia
    from .measures.convex_hull_volume_2d import convex_hull_volume_2d
    from .measures.dcscore import dcscore
    from .measures.diameter import diameter
    from .measures.energy import energy
    from .measures.geo_mean_std import geo_mean_std
    from .measures.graph_entropy import graph_entropy
    from .measures.hamdiv import hamdiv
    from .measures.knn import knn
    from .measures.log_determinant import log_determinant
    from .measures.mean_pw_dist import mean_pw_dist
    from .measures.mst_dispersion import mst_dispersion
    from .measures.renyi_entropy import renyi_entropy
    from .measures.span_centroid import span_centroid
    from .measures.span_medoid import span_medoid
    from .measures.sum_bottleneck import sum_bottleneck
    from .measures.sum_diameter import sum_diameter
    from .measures.sum_pairwise_dist import sum_pairwise_dist
    from .measures.vendi_score import vendi_score

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
from .embed import embed_texts

### Main entry point
from ._convenience import measure_diversity


__all__ = [
    # Main entry point
    "measure_diversity",
    # Individual measures
    "mean_pw_dist", "dist_dispersion", "hamdiv", "diameter", "bottleneck",
    "sum_diameter", "energy", "cluster_inertia", "span_centroid", "chamfer_dist",
    "convex_hull_volume", "radius", "span_medoid", "vendi_score",
    "renyi_entropy", "dcscore", "log_determinant", "bins_entropy",
    "graph_entropy", "mst_dispersion",
    # Helpers
    "embed_texts",
    # Axes
    "register_axis", "get_axis", "list_axes",
]

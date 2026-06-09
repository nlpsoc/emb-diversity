"""emb-diversity – embedding-based diversity measures for text and vector data."""

from __future__ import annotations

### Registries
# Import the (empty) measures subpackage first so the import system binds
# ``emb_diversity.measures`` to it, then immediately rebind that name to the
# registry below. Without this, the first *lazy* import of a measure module
# would set ``emb_diversity.measures`` to the subpackage and clobber the
# registry; importing the subpackage up front means later measure imports find
# it already loaded and leave the registry binding intact.
from . import measures as _measures_subpackage  # noqa: F401
from .axes_registry import axes
from .measures_registry import measures

### Embedding helper
from .embed import embed_texts

### Main entry point
from .convenience import measure_diversity

### Caching utilities
from .compute_pairwise import compute_pairwise_distances, clear_distance_cache, distance_cache_info


# Individual measure functions are imported lazily on first attribute access
# (e.g. ``emb_diversity.vendi_score``), so ``import emb_diversity`` does not pull
# in every measure's dependencies. Resolution is delegated to the measures
# registry, which imports each measure's module on demand and caches the result.
def __getattr__(name: str):
    if name in measures:
        value = measures.get(name)
        globals()[name] = value  # cache so __getattr__ is not called again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(measures.keys()))


__all__ = [
    # Main entry point
    "measure_diversity",
    # Individual measures
    "mean_pw_dist", "dist_dispersion", "hamdiv", "diameter", "bottleneck",
    "sum_bottleneck", "sum_diameter", "energy", "cluster_inertia", "span_centroid", "chamfer_dist",
    "convex_hull_volume_2d", "radius", "span_medoid", "vendi_score",
    "renyi_entropy", "dcscore", "log_determinant", "bins_entropy",
    "graph_entropy", "mst_dispersion",
    # Helpers
    "embed_texts",
    # Registries
    "axes", "measures",
    # Pairwise distance caching
    "compute_pairwise_distances", "clear_distance_cache", "distance_cache_info",
]

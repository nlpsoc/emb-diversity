"""emb-diversity – embedding-based diversity measures for text and vector data."""

from __future__ import annotations

### Registries
from .axes_registry import axes
from .measures_registry import measure_registry

### Embedding helper
from .embed import embed_texts

### Main entry point
from .convenience import measure_diversity

### Caching utilities
from .compute_pairwise import compute_pairwise_distances, clear_distance_cache, distance_cache_info


# Individual measure functions are imported lazily on first attribute access
# (e.g. ``emb_diversity.vendi_score``), so ``import emb_diversity`` does not pull
# in every measure's dependencies. Resolution is delegated to the measure
# registry, which imports each measure's module on demand and caches the result.
# (The registry is named ``measure_registry`` rather than ``measures`` so it does
# not collide with the ``emb_diversity.measures`` code subpackage.)
def __getattr__(name: str):
    if name in measure_registry:
        value = measure_registry.get(name)
        globals()[name] = value  # cache so __getattr__ is not called again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(measure_registry.keys()))


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
    "axes", "measure_registry",
    # Pairwise distance caching
    "compute_pairwise_distances", "clear_distance_cache", "distance_cache_info",
]

"""Central measure registry used by the CLI and convenience functions."""

from __future__ import annotations

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
from .measures.convex_hull_volume import convex_hull_volume
from .measures.radius import radius
from .measures.span_medoid import span_medoid
from .measures.vendi_score import vendi_score
from .measures.renyi_entropy import renyi_entropy
from .measures.dcscore import dcscore
from .measures.log_determinant import log_determinant
from .measures.bins_entropy import bins_entropy
from .measures.graph_entropy import graph_entropy
from .measures.mst_dispersion import mst_dispersion

# All 20 measures keyed by their public name
MEASURES: dict[str, callable] = {
    "mean_pw_dist": mean_pw_dist,
    "dist_dispersion": dist_dispersion,
    "hamdiv": hamdiv,
    "diameter": diameter,
    "bottleneck": bottleneck,
    "sum_diameter": sum_diameter,
    "energy": energy,
    "cluster_inertia": cluster_inertia,
    "span_centroid": span_centroid,
    "chamfer_dist": chamfer_dist,
    "convex_hull_volume": convex_hull_volume,
    "radius": radius,
    "span_medoid": span_medoid,
    "vendi_score": vendi_score,
    "renyi_entropy": renyi_entropy,
    "dcscore": dcscore,
    "log_determinant": log_determinant,
    "bins_entropy": bins_entropy,
    "graph_entropy": graph_entropy,
    "mst_dispersion": mst_dispersion,
}

# The single default measure
DEFAULT_MEASURE = "log_determinant"

# Curated representative set across categories
CORE_MEASURES: list[str] = [
    "log_determinant",
    "mean_pw_dist",
    "vendi_score",
    "convex_hull_volume",
    "graph_entropy",
    "energy",
]

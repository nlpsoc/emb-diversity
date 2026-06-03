"""Central measure registry used by the CLI and convenience functions."""

from __future__ import annotations

from ._registry import Registry

from .measures.mean_pw_dist import mean_pw_dist
from .measures.dist_dispersion import dist_dispersion
from .measures.hamdiv import hamdiv
from .measures.diameter import diameter
from .measures.bottleneck import bottleneck
from .measures.sum_bottleneck import sum_bottleneck
from .measures.sum_diameter import sum_diameter
from .measures.energy import energy
from .measures.cluster_inertia import cluster_inertia
from .measures.span_centroid import span_centroid
from .measures.chamfer_dist import chamfer_dist
from .measures.convex_hull_volume_2d import convex_hull_volume_2d
from .measures.radius import radius
from .measures.span_medoid import span_medoid
from .measures.vendi_score import vendi_score
from .measures.renyi_entropy import renyi_entropy
from .measures.dcscore import dcscore
from .measures.log_determinant import log_determinant
from .measures.bins_entropy import bins_entropy
from .measures.graph_entropy import graph_entropy
from .measures.mst_dispersion import mst_dispersion

# All 21 measures
measures = Registry()

measures.register("mean_pw_dist", mean_pw_dist)
measures.register("dist_dispersion", dist_dispersion)
measures.register("hamdiv", hamdiv)
measures.register("diameter", diameter)
measures.register("bottleneck", bottleneck)
measures.register("sum_bottleneck", sum_bottleneck)
measures.register("sum_diameter", sum_diameter)
measures.register("energy", energy)
measures.register("cluster_inertia", cluster_inertia)
measures.register("span_centroid", span_centroid)
measures.register("chamfer_dist", chamfer_dist)
measures.register("convex_hull_volume_2d", convex_hull_volume_2d)
measures.register("radius", radius)
measures.register("span_medoid", span_medoid)
measures.register("vendi_score", vendi_score)
measures.register("renyi_entropy", renyi_entropy)
measures.register("dcscore", dcscore)
measures.register("log_determinant", log_determinant)
measures.register("bins_entropy", bins_entropy)
measures.register("graph_entropy", graph_entropy)
measures.register("mst_dispersion", mst_dispersion)

# The default measure(s) run when no measure is specified.
# A list so more can be added later.
DEFAULT_MEASURE: list[str] = ["graph_entropy"]

# Named measure sets, selectable as measure="<name>" or CLI -m <name>.
# Add a set here and it becomes usable in both places automatically.
MEASURE_SETS: dict[str, list[str]] = {
    "variety": ["chamfer_dist", "sum_bottleneck", "mst_dispersion"],
    "balance": ["graph_entropy"],
    "disparity": ["graph_entropy", "mst_dispersion"],
}

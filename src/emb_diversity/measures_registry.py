"""Central measure registry used by the CLI and convenience functions."""

from __future__ import annotations

import importlib

from ._registry import Registry

# All 21 measures. Each lives in ``.measures.<name>`` and exposes a function of
# the same name. They are registered lazily so that importing this module (and
# therefore the package) does not pull in every measure's dependencies — each
# measure's module is imported only the first time that measure is looked up.
_MEASURE_NAMES = [
    "mean_pw_dist",
    "dist_dispersion",
    "hamdiv",
    "diameter",
    "bottleneck",
    "sum_bottleneck",
    "sum_diameter",
    "energy",
    "cluster_inertia",
    "span_centroid",
    "chamfer_dist",
    "convex_hull_volume_2d",
    "radius",
    "span_medoid",
    "vendi_score",
    "renyi_entropy",
    "dcscore",
    "log_determinant",
    "bins_entropy",
    "graph_entropy",
    "mst_dispersion",
]


def _measure_loader(name: str):
    """Build a loader that imports ``.measures.<name>`` and returns its function."""

    def load():
        module = importlib.import_module(f".measures.{name}", __package__)
        return getattr(module, name)

    return load


measures = Registry()
for _name in _MEASURE_NAMES:
    measures.register_lazy(_name, _measure_loader(_name))

# NOTE: If you change DEFAULT_MEASURE or MEASURE_SETS below, you must also update
# the hardcoded CLI help, docstrings, docs, and tests. See CLAUDE.md
# ("Keep the measure sets and the default measure in sync") for the full checklist.

# The default measure(s) run when no measure is specified.
# A list so more can be added later.
DEFAULT_MEASURE: list[str] = ["graph_entropy", "vendi_score", "mean_pw_dist"]

# Named measure sets, selectable as measure="<name>" or CLI -m <name>.
# Add a set here and it becomes usable in both places automatically.
MEASURE_SETS: dict[str, list[str]] = {
    "variety": ["chamfer_dist", "sum_bottleneck", "mst_dispersion"],
    "balance": ["graph_entropy"],
    "disparity": ["graph_entropy", "mst_dispersion"],
}

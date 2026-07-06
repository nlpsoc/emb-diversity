"""Central measure registry used by the CLI and convenience functions.

Each measure lives in ``.measures.<name>`` and exposes a function of the same
name. Measures are looked up lazily: :func:`get_measure` imports a measure's
module on first use, so importing this module (and therefore the package) does
not pull in the measures' heavy dependencies.
"""

from __future__ import annotations

import importlib
from typing import Callable

# All 22 measures.
MEASURE_NAMES: tuple[str, ...] = (
    # Distance-based
    "mean_pw_dist",
    "sum_pairwise_dist",
    "hamdiv",
    "diameter",
    "bottleneck",
    "sum_bottleneck",
    "sum_diameter",
    "energy",
    "cluster_inertia",
    "span_centroid",
    "chamfer_dist",
    "knn",
    # Volume-based
    "convex_hull_volume_3d",
    "geo_mean_std",
    "span_medoid",
    # Distribution-based
    "vendi_score",
    "renyi_entropy",
    "dcscore",
    "log_determinant",
    "bins_entropy",
    # Graph-based
    "graph_entropy",
    "mst_dispersion",
)


def get_measure(name: str) -> Callable:
    """Return the measure function registered under *name*.

    The measure's module is imported on first use and cached by Python's
    import system, so repeated lookups are cheap.

    Args:
        name: A measure name from :data:`MEASURE_NAMES`.

    Raises:
        KeyError: If *name* is not a registered measure.
    """
    if name not in MEASURE_NAMES:
        registered = ", ".join(sorted(MEASURE_NAMES))
        raise KeyError(f"Unknown measure {name!r}. Registered: {registered}")
    module = importlib.import_module(f".measures.{name}", __package__)
    return getattr(module, name)


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

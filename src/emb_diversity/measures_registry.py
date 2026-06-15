"""Central measure registry used by the CLI and convenience functions.

Each measure lives in ``.measures.<name>`` and exposes a function of the same
name. Measures are looked up lazily: :func:`get_measure` imports a measure's
module on first use, so importing this module (and therefore the package) does
not pull in the measures' heavy dependencies.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Callable

# All 21 measures.
MEASURE_NAMES: tuple[str, ...] = (
    # Distance-based
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
    # Volume-based
    "convex_hull_volume_2d",
    "radius",
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

# Measures added at runtime (via :func:`register_measure`), keyed by name. The
# built-in catalog above stays the static source of truth; this dict only holds
# user-supplied measures, so it is empty until something is registered.
_REGISTRY: dict[str, Callable] = {}


def get_measure(name: str) -> Callable:
    """Return the measure callable registered under *name*.

    Built-in measures live in ``.measures.<name>`` and are imported on first use
    (and cached by Python's import system). Runtime measures registered with
    :func:`register_measure` are looked up in :data:`_REGISTRY`.

    Args:
        name: A built-in name from :data:`MEASURE_NAMES`, or a runtime-registered
            measure name.

    Raises:
        KeyError: If *name* is not a known measure.
    """
    if name in MEASURE_NAMES:
        module = importlib.import_module(f".measures.{name}", __package__)
        return getattr(module, name)
    if name in _REGISTRY:
        return _REGISTRY[name]
    known = ", ".join(sorted(set(MEASURE_NAMES) | _REGISTRY.keys()))
    raise KeyError(f"Unknown measure {name!r}. Registered: {known}")


def measure(fn: Callable) -> Callable:
    """Turn a core diversity function into a full measure callable.

    The decorated ``fn`` receives an already-validated ``(n, d)`` array as its
    first argument and returns a ``{"value": float, "parameters": {...}}`` dict
    (without ``embedding_model`` — the wrapper adds it). The wrapper returned
    here owns the rest: it accepts text or vectors, runs them through
    :func:`resolve_embeddings` (the single shared input-validation and embedding
    step), calls ``fn``, checks the result shape, coerces ``value`` to ``float``,
    and records the resolved ``embedding_model`` under ``parameters``. Because
    the wrapper is the only code that validates the input and the result, a
    measure cannot skip validation or return the wrong shape.

    This does not register the measure. Built-in measures are found by name via
    their module; use :func:`register_measure` to add one at runtime.
    """
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    if first is None or first.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD,
    ):
        raise TypeError(
            f"measure {getattr(fn, '__name__', fn)!r} must take the data array "
            "as its first positional parameter"
        )

    @functools.wraps(fn)
    def wrapper(data, *args, diversity_axis="semantic", embedding_model=None, **kwargs):
        # Imported lazily so importing this module (and the package) stays light.
        from .embed import resolve_embeddings

        vectors, model = resolve_embeddings(data, diversity_axis, embedding_model)
        result = fn(vectors, *args, **kwargs)
        if not (
            isinstance(result, dict)
            and "value" in result
            and isinstance(result.get("parameters"), dict)
        ):
            raise TypeError(
                f"measure {fn.__name__!r} must return a dict of the form "
                "{'value': float, 'parameters': dict}, got "
                f"{type(result).__name__}"
            )
        result["value"] = float(result["value"])
        result["parameters"]["embedding_model"] = model
        return result

    wrapper._is_measure = True
    return wrapper


def register_measure(fn: Callable, *, name: str | None = None) -> Callable:
    """Register *fn* as a runtime measure, selectable by name.

    *fn* may be an ``@measure``-decorated callable or a plain core function (it
    is wrapped with :func:`measure` if it is not already a measure).

    Args:
        fn: The measure (or core function) to register.
        name: Name to register it under; defaults to ``fn.__name__``.

    Returns:
        The registered measure callable.

    Raises:
        ValueError: If the name is already taken by a built-in or a previously
            registered measure. There is no override path — pick a free name.
    """
    wrapped = fn if getattr(fn, "_is_measure", False) else measure(fn)
    key = name or fn.__name__
    if key in MEASURE_NAMES or key in _REGISTRY:
        raise ValueError(
            f"A measure named {key!r} already exists; choose a different name."
        )
    _REGISTRY[key] = wrapped
    return wrapped


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

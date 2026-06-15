"""Shared logic for measuring diversity, used by both Python API and CLI."""

from __future__ import annotations

import warnings

from .measures_registry import (
    DEFAULT_MEASURE,
    MEASURE_NAMES,
    MEASURE_SETS,
    get_measure,
)


def measure_diversity(
    data,
    measure: str | list[str] | None = None,
    diversity_axis: str = "semantic",
    embedding_model: str | None = None,
) -> dict[str, dict]:
    """Measure diversity of texts or embeddings.

    This is the main entry point for measuring diversity. It resolves measure
    names (including named set and "all" shortcuts) and runs the measures. Each
    measure embeds text itself; the embedding disk cache means text is encoded
    only once and reused across measures.

    Args:
        data: A list of text strings, or embedding vectors (n, d).
        measure: Which measure(s) to run. Options:
            - ``None``: run the default measure(s) (``graph_entropy``)
            - a named set: ``"variety"``, ``"balance"`` or ``"disparity"``
            - ``"all"``: run every built-in measure
            - a measure name like ``"mean_pw_dist"``
            - a measure callable defined with ``@measure`` (or registered with
              ``register_measure``)
            - a list mixing names and callables, like
              ``["mean_pw_dist", my_measure]``
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit model id; overrides *diversity_axis*.

    Returns:
        Dict mapping each measure name to its result, a dict of the form
        ``{"value": float, "parameters": {...}}`` where ``parameters`` records
        the configuration used (including the resolved ``embedding_model``).
        If a measure fails, its ``value`` is ``nan``, the entry gains an
        ``"error"`` key with the failure message, and a ``UserWarning`` is
        emitted — so other measures still run, but failures stay visible.

    Example:
        >>> from emb_diversity import measure_diversity
        >>> measure_diversity(["The cat sat.", "Dogs play fetch."])
        {'graph_entropy': {'value': ..., 'parameters': {'metric': 'cosine', 'embedding_model': ...}}}
        >>> measure_diversity(texts, measure="variety")
        {'chamfer_dist': {'value': ..., 'parameters': {...}}, 'sum_bottleneck': {...}, 'mst_dispersion': {...}}
    """
    # ── Resolve measures ─────────────────────────────────────────
    items = _resolve_measure_names(measure)

    # Resolve every item to a (name, callable) pair up front, so an unknown name
    # or a non-measure callable fails fast here instead of being recorded as a
    # per-measure NaN in the compute loop below.
    resolved = [_resolve_measure(item) for item in items]

    # ── Compute ──────────────────────────────────────────────────
    # Each measure embeds its input itself. When *data* is text, the first
    # measure populates the embedding disk cache and the rest hit it, so the
    # model runs only once.
    results: dict[str, dict] = {}
    for name, measure_fn in resolved:
        try:
            results[name] = measure_fn(
                data, diversity_axis=diversity_axis, embedding_model=embedding_model
            )
        except Exception as exc:
            # A failing measure must not abort the others (e.g. measure="all"),
            # but the failure has to stay visible: warn and record the message.
            error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Measure {name!r} failed — {error}", stacklevel=2)
            results[name] = {
                "value": float("nan"),
                "parameters": {},
                "error": error,
            }
    return results


def _resolve_measure_names(measure):
    """Convert the measure argument into a list of names and/or callables.

    Args:
        measure: None for default; a named set, "all", a single name, or a
            measure callable; or a list mixing names and callables.

    Returns:
        List whose items are measure name strings or measure callables.
    """
    if measure is None:
        return list(DEFAULT_MEASURE)
    if callable(measure):
        return [measure]
    if isinstance(measure, str):
        if measure == "all":
            return list(MEASURE_NAMES)
        if measure in MEASURE_SETS:
            return list(MEASURE_SETS[measure])
        return [measure]
    # It's a list (of names and/or callables).
    return list(measure)


def _resolve_measure(item):
    """Resolve a name or measure callable to a ``(name, callable)`` pair.

    Raises:
        KeyError: If *item* is a name that is not a known measure.
        TypeError: If *item* is a callable that was not built with ``@measure``
            (and so would bypass the shared validation and result envelope).
    """
    if callable(item):
        if not getattr(item, "_is_measure", False):
            raise TypeError(
                f"{getattr(item, '__name__', item)!r} is not a measure. Define it "
                "with the @measure decorator, or register it with register_measure()."
            )
        return item.__name__, item
    return item, get_measure(item)

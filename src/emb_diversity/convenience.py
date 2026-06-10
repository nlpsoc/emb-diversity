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
            - ``"all"``: run every registered measure
            - a measure name like ``"mean_pw_dist"``
            - a list of measure names like ``["mean_pw_dist", "diameter"]``
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
    # ── Resolve measure names ────────────────────────────────────
    measure_names = _resolve_measure_names(measure)

    # ── Validate ─────────────────────────────────────────────────
    for name in measure_names:
        if name not in MEASURE_NAMES:
            raise KeyError(
                f"Unknown measure {name!r}. Available: {sorted(MEASURE_NAMES)}"
            )

    # ── Compute ──────────────────────────────────────────────────
    # Each measure resolves + embeds its input itself. When *data* is text, the
    # first measure populates the embedding disk cache and the rest hit it, so
    # the model runs only once.
    results: dict[str, dict] = {}
    for name in measure_names:
        # Resolve outside the try block so an import failure surfaces as an
        # error instead of being silently recorded as NaN.
        measure_fn = get_measure(name)
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


def _resolve_measure_names(measure: str | list[str] | None) -> list[str]:
    """Convert the measure argument into a list of measure names.

    Args:
        measure: None for default, a named set, "all", a single name,
            or a list of names.

    Returns:
        List of measure name strings.
    """
    if measure is None:
        return list(DEFAULT_MEASURE)
    if isinstance(measure, str):
        if measure == "all":
            return list(MEASURE_NAMES)
        if measure in MEASURE_SETS:
            return list(MEASURE_SETS[measure])
        return [measure]
    # It's a list
    return list(measure)

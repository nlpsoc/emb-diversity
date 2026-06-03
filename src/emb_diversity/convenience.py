"""Shared logic for measuring diversity, used by both Python API and CLI."""

from __future__ import annotations

from .measures_registry import DEFAULT_MEASURE, MEASURE_SETS, measures
from .embed import embed_texts


def measure_diversity(
    data,
    measure: str | list[str] | None = None,
    diversity_axis: str = "semantic",
    embedding_model: str | None = None,
) -> dict[str, float]:
    """Measure diversity of texts or embeddings.

    This is the main entry point for measuring diversity. It handles
    embedding (if given text), resolving measure names (including named
    set and "all" shortcuts), and running the measures.

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
        Dict mapping measure name to score.

    Example:
        >>> from emb_diversity import measure_diversity
        >>> measure_diversity(["The cat sat.", "Dogs play fetch."])
        {'graph_entropy': ...}
        >>> measure_diversity(texts, measure="variety")
        {'chamfer_dist': ..., 'sum_bottleneck': ..., 'mst_dispersion': ...}
    """
    # ── Resolve measure names ────────────────────────────────────
    measure_names = _resolve_measure_names(measure)

    # ── Validate ─────────────────────────────────────────────────
    for name in measure_names:
        if name not in measures:
            raise KeyError(
                f"Unknown measure {name!r}. Available: {sorted(measures)}"
            )

    # ── Embed once if text, then pass vectors to all measures ────
    # We embed here rather than letting each measure's @accepts_text
    # decorator do it, to avoid re-embedding for every measure.
    if len(data) > 0 and isinstance(data[0], str):
        data = embed_texts(data, diversity_axis=diversity_axis, embedding_model=embedding_model)

    # ── Compute ──────────────────────────────────────────────────
    results: dict[str, float] = {}
    for name in measure_names:
        try:
            results[name] = measures[name](data)
        except Exception:
            results[name] = float("nan")
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
            return list(measures)
        if measure in MEASURE_SETS:
            return list(MEASURE_SETS[measure])
        return [measure]
    # It's a list
    return list(measure)

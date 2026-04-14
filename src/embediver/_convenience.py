"""Convenience functions for running multiple measures at once."""

from __future__ import annotations

from ._registry import MEASURES


def measure_all(
    data,
    measures: list[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Run multiple diversity measures and return results as a dict.

    Args:
        data: Embedding vectors (n, d) or a list of text strings.
        measures: Measure names to run.  If ``None``, runs all 20.
        **kwargs: Extra arguments forwarded to each measure
            (e.g. ``metric``, ``diversity_axis``, ``embedding_model``).

    Returns:
        Dict mapping measure name to score.
    """
    registry = MEASURES
    names = measures if measures is not None else list(registry)
    results: dict[str, float] = {}
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown measure {name!r}. Available: {sorted(registry)}")
        try:
            results[name] = registry[name](data, **kwargs)
        except Exception as exc:
            results[name] = float("nan")
    return results

"""Convenience functions for running multiple measures at once."""

from __future__ import annotations

from ._registry import MEASURES
from .embed import embed_texts


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


def measure_text(
    texts: list[str],
    measures: list[str] | None = None,
    diversity_axis: str = "semantic",
    embedding_model: str | None = None,
    **kwargs,
) -> dict[str, float]:
    """Embed texts and measure diversity in one call.

    Args:
        texts: List of text strings.
        measures: Measure names to run (default: all).
        diversity_axis: Registered axis name (default ``"semantic"``).
        embedding_model: Explicit model id; overrides *diversity_axis*.
        **kwargs: Extra arguments forwarded to each measure.

    Returns:
        Dict mapping measure name to score.
    """
    vectors = embed_texts(texts, diversity_axis=diversity_axis, embedding_model=embedding_model)
    return measure_all(vectors, measures=measures, **kwargs)

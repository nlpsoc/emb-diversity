"""Reusable engine for content-matched diversity comparisons.

A *class* is a labelled bag of texts: ``(label, list[Item])``. Each :class:`Item`
carries one text plus two keys:

- ``match``: the unit the classes are balanced within. It is held fixed across
  classes so the comparison is fair — e.g. a genre, a prompt id, or the id of a
  source text. Classes are downsampled so that, for every ``match`` value, each
  class contributes the same number of (distinct) texts.
- ``group``: the unit the results are broken down by in the printed tables —
  e.g. a genre or a corpus name.

For many datasets ``match`` and ``group`` are the same; they differ when you
want to match on something finer (say, a prompt) than you display (say, a
corpus).

The typical flow is::

    classes = [("Human", human_items), ("AI", ai_items)]
    run_comparison(classes)

which balances the classes, prints the per-group counts, and prints one table
per measure and axis with the classes side by side.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import NamedTuple

from emb_diversity import measure_diversity

# Seed for the resampling in ``balance_classes`` so runs are reproducible.
SEED = 0

# Diversity axes to measure, as (axis, embedding_model) pairs. The model
# overrides the axis's default (see ``emb_diversity.axes_registry``); ``None``
# keeps the default. "semantic" uses the axis default (all-mpnet-base-v2);
# "style" uses the StyleDistance model rather than the axis default.
AXES = (
    ("semantic", None),
    ("style", "StyleDistance/styledistance"),
)


class Item(NamedTuple):
    """One text with its balancing key (``match``) and display group (``group``)."""

    text: str
    match: object
    group: str


# A class is a (label, items) pair; a comparison is a list of classes.
Class = tuple[str, list[Item]]


def _subsample(items: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Return ``n`` items: all of them if already ``n`` or fewer, otherwise a
    random subset drawn **without replacement** (so no text is repeated)."""
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def balance_classes(classes: list[Class], seed: int = SEED) -> list[Class]:
    """Downsample every class to a common per-``match`` size, so they're comparable.

    For each ``match`` value present in *all* classes, the target size is the
    smallest class's count there; every class is downsampled to it **without
    replacement** (a random subset — no text is duplicated or invented). ``match``
    values missing from any class are dropped, since they cannot be compared
    across all classes. The result: for every kept ``match`` value each class has
    the same, unique-text count — so each class also has the same total size.

    Args:
        classes: List of ``(label, items)`` pairs to equalise.
        seed: Seed for the resampling, for reproducibility.

    Returns:
        The classes in the same order, each downsampled to the shared sizes.
    """
    rng = random.Random(seed)

    by_match: list[dict[object, list[Item]]] = []
    for _, items in classes:
        grouped: dict[object, list[Item]] = defaultdict(list)
        for item in items:
            grouped[item.match].append(item)
        by_match.append(grouped)

    shared = set.intersection(*[set(g) for g in by_match]) if by_match else set()

    balanced: list[list[Item]] = [[] for _ in classes]
    for match in sorted(shared, key=str):
        n = min(len(g[match]) for g in by_match)
        if n == 0:
            continue
        for i, grouped in enumerate(by_match):
            balanced[i].extend(_subsample(grouped[match], n, rng))

    return [(label, balanced[i]) for i, (label, _) in enumerate(classes)]


def groups(classes: list[Class]) -> list[str]:
    """Sorted union of the display groups present across the given classes."""
    seen: set[str] = set()
    for _, items in classes:
        for item in items:
            seen.add(item.group)
    return sorted(seen)


def _texts(items: list[Item], group: str | None = None) -> list[str]:
    """The texts in ``items``, optionally restricted to one display group."""
    return [item.text for item in items if group is None or item.group == group]


def print_stats(classes: list[Class], title: str) -> None:
    """Print each class's total count and the per-group breakdown."""
    labels = [label for label, _ in classes]
    totals = [len(items) for _, items in classes]
    print(f"{title} ({sum(totals)} texts)")
    for label, total in zip(labels, totals):
        print(f"  {label:<8}: {total}")

    print(f"  per group ({' / '.join(labels)}):")
    header = f"    {'group':<28}" + "".join(f"{label:>9}" for label in labels) + f"{'total':>9}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for group in groups(classes):
        counts = [sum(1 for item in items if item.group == group) for _, items in classes]
        cells = "".join(f"{c:>9}" for c in counts)
        print(f"    {group:<28}{cells}{sum(counts):>9}")


def measure_scopes(
    classes: list[Class], axis: str, embedding_model: str | None = None
) -> list[tuple[str, list[dict]]]:
    """Measure each class's diversity along ``axis`` for every scope.

    The scopes are the pooled "all" set followed by one scope per display group.
    ``embedding_model`` overrides the axis's default model when set.

    Returns:
        A list of ``(scope, results_per_class)`` tuples, where ``results_per_class``
        holds one ``measure_diversity`` result dict per class (empty if that class
        has no texts in the scope). Repeated embedding of the same text across
        scopes hits the on-disk cache, so each text is encoded once.
    """
    scopes: list[tuple[str, str | None]] = [("all", None)]
    scopes += [(group, group) for group in groups(classes)]

    rows: list[tuple[str, list[dict]]] = []
    for scope, group in scopes:
        results = []
        for _, items in classes:
            texts = _texts(items, group)
            results.append(
                measure_diversity(texts, diversity_axis=axis, embedding_model=embedding_model)
                if texts
                else {}
            )
        rows.append((scope, results))
    return rows


def _fmt(value: float | None) -> str:
    """Format a score for a table cell, or "-" when missing/undefined."""
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.4f}"


def _fmt_delta(a: float | None, b: float | None) -> str:
    """Format the b-minus-a gap, or "-" if either side is unavailable."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return "-"
    return f"{b - a:+.4f}"


def print_diversity_table(
    axis: str,
    labels: list[str],
    rows: list[tuple[str, list[dict]]],
    embedding_model: str | None = None,
) -> None:
    """Print one table per measure, with one column per class side by side.

    Each table compares one measure across scopes (all, then each display group).
    For a two-class comparison a final delta column (second minus first) is added.
    """
    # Collect measure names in the order the measures returned them.
    measures: list[str] = []
    for _, results in rows:
        for result in results:
            for measure in result:
                if measure not in measures:
                    measures.append(measure)

    two_class = len(labels) == 2
    delta_label = f"{labels[1]} - {labels[0]}" if two_class else ""

    suffix = f" ({embedding_model})" if embedding_model else ""
    print(f"\n======== {axis} diversity{suffix} ========")
    for measure in measures:
        print(f"\n  {measure}")
        head = f"    {'scope':<28}" + "".join(f"{label:>12}" for label in labels)
        if two_class:
            head += f"{delta_label:>14}"
        print(head)
        print("    " + "-" * (len(head) - 4))
        for scope, results in rows:
            values = [result.get(measure, {}).get("value") for result in results]
            cells = "".join(f"{_fmt(v):>12}" for v in values)
            if two_class:
                cells += f"{_fmt_delta(values[0], values[1]):>14}"
            print(f"    {scope:<28}{cells}")


def run_comparison(classes: list[Class], title: str | None = None) -> None:
    """Balance the given classes and print their diversity comparison.

    Args:
        classes: The labelled classes to compare.
        title: Optional banner; defaults to the class labels joined with "vs".
    """
    labels = [label for label, _ in classes]
    banner = title or "  vs  ".join(labels)
    print(f"\n################  {banner}  ################")

    balanced = balance_classes(classes)
    print_stats(balanced, "Balanced to a common per-match size")

    print("\nMeasuring diversity (embeds every text once; cached across scopes)...")
    for axis, embedding_model in AXES:
        rows = measure_scopes(balanced, axis, embedding_model)
        print_diversity_table(axis, labels, rows, embedding_model)

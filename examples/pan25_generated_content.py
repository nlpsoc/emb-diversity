"""Measure the diversity of human- vs AI-written texts in the PAN 2025 data.

This example works on the validation set of PAN 2025 subtask 1, "Voight-Kampff
Generated Content Analysis" (https://pan.webis.de/clef25/pan25-web/
generated-content-analysis.html). Each line of ``val.jsonl`` is a JSON object:

    {"id": str, "text": str, "model": str, "label": int, "genre": str}

where ``label`` is ``0`` for human-written and ``1`` for AI-written text, and
``genre`` is one of "essays", "news" or "fiction".

The script:

1. loads ``val.jsonl`` and splits it into human, all-AI, and GPT-only texts
   (GPT = any model whose id contains "gpt"), grouped by genre,
2. prints the class counts, the per-genre breakdown, and which models produced
   the AI texts,
3. runs several comparisons; in each, the classes are downsampled within every
   genre to a common size (``balance_classes``) so the diversity scores are
   comparable — only equal-sized sets can be compared,
4. measures each balanced class with ``measure_diversity`` along both registered
   axes — ``semantic`` (meaning) and ``style`` (writing style) — pooled over all
   genres and separately per genre, printing one table per measure with the
   classes side by side.

The comparisons run are:

- human vs all AI,
- human vs GPT-only,
- human vs GPT-only vs a mix of all AI models.

"Mix" is a sample of the full AI pool (all models, GPT included); change the
``("Mix", ai_by_genre)`` class in ``main`` to a non-GPT pool if you would rather
contrast GPT against the other models only.

Run it with::

    uv run python examples/pan25_generated_content.py path/to/val.jsonl

If no path is given it looks for ``val.jsonl`` in the current directory.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from emb_diversity import measure_diversity

# Seed for the resampling in ``balance_classes`` so runs are reproducible.
SEED = 0

# Diversity axes to measure, each with its own embedding model (see
# ``emb_diversity.axes_registry``): "semantic" captures meaning, "style"
# captures writing style.
AXES = ("semantic", "style")

# A class is a (label, genre -> texts) pair; a comparison is a list of classes.
Class = tuple[str, dict[str, list[str]]]


def _is_gpt(model: str) -> bool:
    """True if a model id names a GPT-family model (any version)."""
    return "gpt" in model.lower()


def load_split(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], Counter]:
    """Load val.jsonl and split it into human and AI texts, grouped by genre.

    Args:
        path: Path to the PAN 2025 subtask-1 ``val.jsonl`` file.

    Returns:
        ``(human_by_genre, ai_by_genre, gpt_by_genre, ai_models)`` — three dicts
        mapping each genre to the list of its document texts (human, all AI, and
        the GPT-family subset of the AI texts respectively), and a ``Counter`` of
        how many AI texts each model (the ``"model"`` field) produced.
    """
    human_by_genre: dict[str, list[str]] = {}
    ai_by_genre: dict[str, list[str]] = {}
    gpt_by_genre: dict[str, list[str]] = {}
    ai_models: Counter = Counter()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            genre = record.get("genre", "unknown")
            if record["label"] == 0:  # 0 = human-written
                human_by_genre.setdefault(genre, []).append(text)
            else:  # 1 = AI-written
                model = record.get("model", "unknown")
                ai_by_genre.setdefault(genre, []).append(text)
                ai_models[model] += 1
                if _is_gpt(model):
                    gpt_by_genre.setdefault(genre, []).append(text)

    return human_by_genre, ai_by_genre, gpt_by_genre, ai_models


def balance_classes(classes: list[Class], seed: int = SEED) -> list[Class]:
    """Downsample every class to a common per-genre size, so they're comparable.

    For each genre that *all* classes contain, the target size is the smallest
    class's count there; every class is downsampled to it **without replacement**
    (a random subset — no text is duplicated or invented). Genres missing from
    any class are dropped, since they cannot be compared across all classes. The
    result: within each kept genre every class has the same, unique-text count,
    so each class also has the same total size.

    Args:
        classes: List of ``(label, genre -> texts)`` pairs to equalise.
        seed: Seed for the resampling, for reproducibility.

    Returns:
        The classes in the same order, each downsampled to the shared sizes.
    """
    rng = random.Random(seed)
    maps = [by_genre for _, by_genre in classes]
    shared_genres = sorted(set.intersection(*[set(m) for m in maps])) if maps else []

    balanced: list[dict[str, list[str]]] = [{} for _ in classes]
    for genre in shared_genres:
        n = min(len(m[genre]) for m in maps)
        if n == 0:
            continue
        for i, m in enumerate(maps):
            balanced[i][genre] = _subsample(m[genre], n, rng)

    return [(label, balanced[i]) for i, (label, _) in enumerate(classes)]


def _subsample(items: list[str], n: int, rng: random.Random) -> list[str]:
    """Return ``n`` items: all of them if already ``n`` or fewer, otherwise a
    random subset drawn **without replacement** (so no text is repeated)."""
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def flatten(by_genre: dict[str, list[str]]) -> list[str]:
    """Collapse a genre -> texts dict into one flat list of texts."""
    return [text for texts in by_genre.values() for text in texts]


def all_genres(classes: list[Class]) -> list[str]:
    """Sorted union of the genres present across the given classes."""
    genres: set[str] = set()
    for _, by_genre in classes:
        genres |= set(by_genre)
    return sorted(genres)


def print_stats(classes: list[Class], title: str) -> None:
    """Print each class's total count and the per-genre breakdown."""
    labels = [label for label, _ in classes]
    totals = [sum(len(v) for v in by_genre.values()) for _, by_genre in classes]
    print(f"{title} ({sum(totals)} texts)")
    for label, total in zip(labels, totals):
        print(f"  {label:<8}: {total}")

    print(f"  per genre ({' / '.join(labels)}):")
    header = f"    {'genre':<12}" + "".join(f"{label:>9}" for label in labels) + f"{'total':>9}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for genre in all_genres(classes):
        counts = [len(by_genre.get(genre, [])) for _, by_genre in classes]
        cells = "".join(f"{c:>9}" for c in counts)
        print(f"    {genre:<12}{cells}{sum(counts):>9}")


def print_models(ai_models: Counter) -> None:
    """Print which models produced the AI texts and how many each produced."""
    total = sum(ai_models.values())
    print(f"\nAI models ({total} texts, {len(ai_models)} distinct):")
    header = f"  {'model':<28}{'count':>8}{'share':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model, count in ai_models.most_common():
        share = count / total if total else 0.0
        print(f"  {model:<28}{count:>8}{share:>9.1%}")


def measure_scopes(classes: list[Class], axis: str) -> list[tuple[str, list[dict]]]:
    """Measure each class's diversity along ``axis`` for every scope.

    The scopes are the pooled "all genres" set followed by one scope per genre.

    Returns:
        A list of ``(scope, results_per_class)`` tuples, where ``results_per_class``
        holds one ``measure_diversity`` result dict per class (empty if that class
        has no texts in the scope). Repeated embedding of the same text across
        scopes hits the on-disk cache, so each text is encoded once.
    """
    scopes: list[tuple[str, list[list[str]]]] = [
        ("all genres", [flatten(by_genre) for _, by_genre in classes])
    ]
    for genre in all_genres(classes):
        scopes.append((genre, [by_genre.get(genre, []) for _, by_genre in classes]))

    rows: list[tuple[str, list[dict]]] = []
    for scope, texts_per_class in scopes:
        results = [
            measure_diversity(texts, diversity_axis=axis) if texts else {}
            for texts in texts_per_class
        ]
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
    axis: str, labels: list[str], rows: list[tuple[str, list[dict]]]
) -> None:
    """Print one table per measure, with one column per class side by side.

    Each table compares one measure across scopes (all genres, then each genre).
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

    print(f"\n======== {axis} diversity ========")
    for measure in measures:
        print(f"\n  {measure}")
        head = f"    {'scope':<12}" + "".join(f"{label:>12}" for label in labels)
        if two_class:
            head += f"{delta_label:>14}"
        print(head)
        print("    " + "-" * (len(head) - 4))
        for scope, results in rows:
            values = [result.get(measure, {}).get("value") for result in results]
            cells = "".join(f"{_fmt(v):>12}" for v in values)
            if two_class:
                cells += f"{_fmt_delta(values[0], values[1]):>14}"
            print(f"    {scope:<12}{cells}")


def run_comparison(classes: list[Class]) -> None:
    """Balance the given classes and print their diversity comparison."""
    labels = [label for label, _ in classes]
    print(f"\n################  {'  vs  '.join(labels)}  ################")

    balanced = balance_classes(classes)
    print_stats(balanced, "Balanced to a common per-genre size")

    print("\nMeasuring diversity (embeds every text once; cached across scopes)...")
    for axis in AXES:
        rows = measure_scopes(balanced, axis)
        print_diversity_table(axis, labels, rows)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("val.jsonl")
    if not path.exists():
        sys.exit(
            f"Could not find {path}. Download the PAN 2025 subtask-1 validation "
            "set and pass its path, e.g.:\n"
            "  uv run python examples/pan25_generated_content.py path/to/val.jsonl"
        )

    human_by_genre, ai_by_genre, gpt_by_genre, ai_models = load_split(path)
    print_stats([("human", human_by_genre), ("AI", ai_by_genre)], "Loaded")
    print_models(ai_models)

    # 1) Human vs all AI.
    run_comparison([("Human", human_by_genre), ("AI", ai_by_genre)])

    if any(gpt_by_genre.values()):
        # 2) Human vs the GPT-only subset.
        run_comparison([("Human", human_by_genre), ("GPT", gpt_by_genre)])
        # 3) Human vs GPT vs a mix of all AI models (all sampled to one size).
        run_comparison(
            [("Human", human_by_genre), ("GPT", gpt_by_genre), ("Mix", ai_by_genre)]
        )
    else:
        print("\nNo GPT-family models found in the data; skipping the GPT comparisons.")


if __name__ == "__main__":
    main()

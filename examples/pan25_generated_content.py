"""Measure the diversity of human- vs AI-written texts in the PAN 2025 data.

This example works on the validation set of PAN 2025 subtask 1, "Voight-Kampff
Generated Content Analysis" (https://pan.webis.de/clef25/pan25-web/
generated-content-analysis.html). Each line of ``val.jsonl`` is a JSON object:

    {"id": str, "text": str, "model": str, "label": int, "genre": str}

where ``label`` is ``0`` for human-written and ``1`` for AI-written text, and
``genre`` is one of "essays", "news" or "fiction".

The script:

1. loads ``val.jsonl`` and splits it into human-written and AI-generated texts,
2. prints how many texts fall in each class, how they break down by genre, and
   which models produced the AI texts,
3. balances the two classes within each genre by downsampling the larger to
   the smaller (equal, unique-text counts per genre — see ``balance_per_genre``),
4. measures the diversity of each balanced class with ``measure_diversity``
   along both registered axes — ``semantic`` (meaning) and ``style`` (writing
   style) — and prints the scores side by side.

Run it with::

    uv run python examples/pan25_generated_content.py path/to/val.jsonl

If no path is given it looks for ``val.jsonl`` in the current directory.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from emb_diversity import measure_diversity

# Seed for the (re)sampling in ``balance_per_genre`` so runs are reproducible.
SEED = 0

# Diversity axes to measure, each with its own embedding model (see
# ``emb_diversity.axes_registry``): "semantic" captures meaning, "style"
# captures writing style.
AXES = ("semantic", "style")


def load_split(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], Counter]:
    """Load val.jsonl and split it into human and AI texts, grouped by genre.

    Args:
        path: Path to the PAN 2025 subtask-1 ``val.jsonl`` file.

    Returns:
        ``(human_by_genre, ai_by_genre, ai_models)`` — two dicts mapping each
        genre to the list of its document texts (one dict per class), and a
        ``Counter`` of how many AI texts each model (the ``"model"`` field)
        produced.
    """
    human_by_genre: dict[str, list[str]] = {}
    ai_by_genre: dict[str, list[str]] = {}
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
                ai_by_genre.setdefault(genre, []).append(text)
                ai_models[record.get("model", "unknown")] += 1

    return human_by_genre, ai_by_genre, ai_models


def balance_per_genre(
    human_by_genre: dict[str, list[str]],
    ai_by_genre: dict[str, list[str]],
    seed: int = SEED,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Equalise the human and AI counts within each genre.

    For every genre the target size is the *minimum* of the human and AI counts.
    The smaller class is kept as-is; the larger class is downsampled **without
    replacement** (a random subset) to that target. Every text in the result is
    a distinct original text — nothing is duplicated or invented — so the two
    classes have equal, unique-text counts per genre.

    (Downsampling discards data. If keeping every text matters more than avoiding
    duplicates, upsample the smaller class to ``n = max(len(h), len(a))`` with
    ``rng.choices`` instead — but note exact duplicates deflate diversity scores.)

    Args:
        human_by_genre: Genre -> human texts, as returned by ``load_split``.
        ai_by_genre: Genre -> AI texts, as returned by ``load_split``.
        seed: Seed for the resampling, for reproducibility.

    Returns:
        ``(human_by_genre, ai_by_genre)`` — the same shape as the input, but with
        the two classes equalised within each genre. A genre present in only one
        class is left untouched (it cannot be balanced).
    """
    rng = random.Random(seed)
    balanced_human: dict[str, list[str]] = {}
    balanced_ai: dict[str, list[str]] = {}

    for genre in sorted(set(human_by_genre) | set(ai_by_genre)):
        h = human_by_genre.get(genre, [])
        a = ai_by_genre.get(genre, [])
        n = min(len(h), len(a)) if h and a else max(len(h), len(a))
        balanced_human[genre] = _subsample(h, n, rng)
        balanced_ai[genre] = _subsample(a, n, rng)

    return balanced_human, balanced_ai


def _subsample(items: list[str], n: int, rng: random.Random) -> list[str]:
    """Return ``n`` items: all of them if already ``n`` or fewer, otherwise a
    random subset drawn **without replacement** (so no text is repeated)."""
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def flatten(by_genre: dict[str, list[str]]) -> list[str]:
    """Collapse a genre -> texts dict into one flat list of texts."""
    return [text for texts in by_genre.values() for text in texts]


def print_stats(
    human_by_genre: dict[str, list[str]],
    ai_by_genre: dict[str, list[str]],
    title: str,
) -> None:
    """Print the human/AI counts and the per-genre breakdown."""
    n_human = sum(len(v) for v in human_by_genre.values())
    n_ai = sum(len(v) for v in ai_by_genre.values())
    print(f"{title} ({n_human + n_ai} texts)")
    print(f"  human-written : {n_human}")
    print(f"  AI-generated  : {n_ai}")

    all_genres = sorted(set(human_by_genre) | set(ai_by_genre))
    print("  per genre (human / AI):")
    header = f"    {'genre':<12}{'human':>8}{'AI':>8}{'total':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for genre in all_genres:
        h = len(human_by_genre.get(genre, []))
        a = len(ai_by_genre.get(genre, []))
        print(f"    {genre:<12}{h:>8}{a:>8}{h + a:>8}")


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


def print_diversity(label: str, results: dict) -> None:
    """Print one measure_diversity result block."""
    print(f"\n{label} diversity:")
    for measure, result in results.items():
        value = result["value"]
        print(f"  {measure:<16}{value:.4f}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("val.jsonl")
    if not path.exists():
        sys.exit(
            f"Could not find {path}. Download the PAN 2025 subtask-1 validation "
            "set and pass its path, e.g.:\n"
            "  uv run python examples/pan25_generated_content.py path/to/val.jsonl"
        )

    human_by_genre, ai_by_genre, ai_models = load_split(path)
    print_stats(human_by_genre, ai_by_genre, "Loaded")
    print_models(ai_models)

    bal_human, bal_ai = balance_per_genre(human_by_genre, ai_by_genre)
    print()
    print_stats(bal_human, bal_ai, "Balanced per genre (min of the two)")

    human_texts = flatten(bal_human)
    ai_texts = flatten(bal_ai)

    for axis in AXES:
        print(f"\nMeasuring {axis} diversity (this embeds every text once)...")
        human_results = measure_diversity(human_texts, diversity_axis=axis)
        ai_results = measure_diversity(ai_texts, diversity_axis=axis)

        print_diversity(f"Human-written [{axis}]", human_results)
        print_diversity(f"AI-generated [{axis}]", ai_results)


if __name__ == "__main__":
    main()

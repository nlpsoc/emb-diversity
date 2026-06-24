"""Measure the diversity of human- vs AI-written texts in the PAN 2025 data.

This example works on the validation set of PAN 2025 subtask 1, "Voight-Kampff
Generated Content Analysis" (https://pan.webis.de/clef25/pan25-web/
generated-content-analysis.html). Each line of ``val.jsonl`` is a JSON object:

    {"id": str, "text": str, "model": str, "label": int, "genre": str}

where ``label`` is ``0`` for human-written and ``1`` for AI-written text, and
``genre`` is one of "essays", "news" or "fiction".

The script:

1. loads ``val.jsonl`` and splits it into human-written and AI-generated texts,
2. prints how many texts fall in each class and how they break down by genre,
3. measures the semantic diversity of each class with ``measure_diversity`` and
   prints the scores side by side.

Run it with::

    uv run python examples/pan25_generated_content.py path/to/val.jsonl

If no path is given it looks for ``val.jsonl`` in the current directory.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from emb_diversity import measure_diversity


def load_split(path: Path) -> tuple[list[str], list[str], Counter, Counter]:
    """Load val.jsonl and split it into human and AI texts.

    Args:
        path: Path to the PAN 2025 subtask-1 ``val.jsonl`` file.

    Returns:
        ``(human_texts, ai_texts, human_genres, ai_genres)`` — the two lists of
        document texts and a genre histogram (``Counter``) for each class.
    """
    human_texts: list[str] = []
    ai_texts: list[str] = []
    human_genres: Counter = Counter()
    ai_genres: Counter = Counter()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            genre = record.get("genre", "unknown")
            if record["label"] == 0:  # 0 = human-written
                human_texts.append(text)
                human_genres[genre] += 1
            else:  # 1 = AI-written
                ai_texts.append(text)
                ai_genres[genre] += 1

    return human_texts, ai_texts, human_genres, ai_genres


def print_stats(
    human_texts: list[str],
    ai_texts: list[str],
    human_genres: Counter,
    ai_genres: Counter,
) -> None:
    """Print the human/AI counts and the per-genre breakdown."""
    total = len(human_texts) + len(ai_texts)
    print(f"Loaded {total} texts")
    print(f"  human-written : {len(human_texts)}")
    print(f"  AI-generated  : {len(ai_texts)}")

    all_genres = sorted(set(human_genres) | set(ai_genres))
    print("\nGenre breakdown (human / AI):")
    header = f"  {'genre':<12}{'human':>8}{'AI':>8}{'total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for genre in all_genres:
        h = human_genres.get(genre, 0)
        a = ai_genres.get(genre, 0)
        print(f"  {genre:<12}{h:>8}{a:>8}{h + a:>8}")


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

    human_texts, ai_texts, human_genres, ai_genres = load_split(path)
    print_stats(human_texts, ai_texts, human_genres, ai_genres)

    print("\nMeasuring semantic diversity (this embeds every text once)...")
    human_results = measure_diversity(human_texts)
    ai_results = measure_diversity(ai_texts)

    print_diversity("Human-written", human_results)
    print_diversity("AI-generated", ai_results)


if __name__ == "__main__":
    main()

"""Measure the diversity of human- vs AI-written texts in the PAN 2025 data.

This example works on the validation set of PAN 2025 subtask 1, "Voight-Kampff
Generated Content Analysis" (https://pan.webis.de/clef25/pan25-web/
generated-content-analysis.html). Each line of ``val.jsonl`` is a JSON object:

    {"id": str, "text": str, "model": str, "label": int, "genre": str}

where ``label`` is ``0`` for human-written and ``1`` for AI-written text, and
``genre`` is one of "essays", "news" or "fiction".

The script:

1. loads ``val.jsonl`` and splits it into human, all-AI, and GPT-only texts
   (GPT = any model whose id contains "gpt"); each text is tagged with its genre,
2. prints the class counts, the per-genre breakdown, and which models produced
   the AI texts,
3. runs several comparisons; in each, the classes are downsampled within every
   genre to a common size (see ``diversity_compare.balance_classes``) so the
   diversity scores are comparable — only equal-sized sets can be compared,
4. measures each balanced class along the semantic and style axes, pooled and
   per genre, printing one table per measure with the classes side by side.

The comparisons run are human vs all AI, human vs GPT-only, and human vs GPT-only
vs a mix of all AI models. "Mix" is a sample of the full AI pool (all models, GPT
included); change the ``("Mix", ai)`` class in ``main`` to a non-GPT pool to
contrast GPT against the other models only.

The generic comparison engine lives in ``diversity_compare`` (shared with the
other examples); this file only handles the PAN-specific loading.

Run it with::

    uv run python examples/pan25_generated_content.py path/to/val.jsonl

If no path is given it looks for ``val.jsonl`` in the current directory.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from diversity_compare import Item, print_stats, run_comparison


def _is_gpt(model: str) -> bool:
    """True if a model id names a GPT-family model (any version)."""
    return "gpt" in model.lower()


def load_split(
    path: Path,
) -> tuple[list[Item], list[Item], list[Item], Counter]:
    """Load val.jsonl into human, all-AI, and GPT-only items (matched by genre).

    Args:
        path: Path to the PAN 2025 subtask-1 ``val.jsonl`` file.

    Returns:
        ``(human, ai, gpt, ai_models)`` — three lists of :class:`Item` (human, all
        AI, and the GPT-family subset of the AI texts; ``match`` and ``group`` are
        both the genre), and a ``Counter`` of how many AI texts each model
        produced.
    """
    human: list[Item] = []
    ai: list[Item] = []
    gpt: list[Item] = []
    ai_models: Counter = Counter()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            genre = record.get("genre", "unknown")
            item = Item(record["text"], genre, genre)
            if record["label"] == 0:  # 0 = human-written
                human.append(item)
            else:  # 1 = AI-written
                model = record.get("model", "unknown")
                ai.append(item)
                ai_models[model] += 1
                if _is_gpt(model):
                    gpt.append(item)

    return human, ai, gpt, ai_models


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


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("val.jsonl")
    if not path.exists():
        sys.exit(
            f"Could not find {path}. Download the PAN 2025 subtask-1 validation "
            "set and pass its path, e.g.:\n"
            "  uv run python examples/pan25_generated_content.py path/to/val.jsonl"
        )

    human, ai, gpt, ai_models = load_split(path)
    print_stats([("human", human), ("AI", ai)], "Loaded")
    print_models(ai_models)

    # 1) Human vs all AI.
    run_comparison([("Human", human), ("AI", ai)])

    if gpt:
        # 2) Human vs the GPT-only subset.
        run_comparison([("Human", human), ("GPT", gpt)])
        # 3) Human vs GPT vs a mix of all AI models (all sampled to one size).
        run_comparison([("Human", human), ("GPT", gpt), ("Mix", ai)])
    else:
        print("\nNo GPT-family models found in the data; skipping the GPT comparisons.")


if __name__ == "__main__":
    main()

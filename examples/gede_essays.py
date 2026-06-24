"""Content-matched human-vs-LLM diversity on the GEDE essays dataset.

GEDE (from "Assessing LLM Text Detection in Educational Contexts",
https://arxiv.org/abs/2508.08096, https://github.com/lukasgehring/
Assessing-LLM-Text-Detection-in-Educational-Contexts) collects student essays
and many LLM variants of them. Each record in ``gede_essays.json`` has:

    id, rewrite_of, dataset, contribution_level, text_author,
    question_id, question, answer, temperature, max_tokens

``answer`` is the text, ``question_id`` is the prompt it answers, ``dataset`` is
the source corpus (BAWE / argument-annotated-essays / persuade), ``text_author``
is ``human`` or an LLM id, and ``contribution_level`` says how the text was made.

Unlike PAN, this dataset lets us hold *content* fixed, so we can separate "what
the text is about" from "how it is written". Two comparisons are run, each with
four classes — Human / gpt-4o-mini / Llama-3.3 / Mix (Mix = all of that AI type
pooled) — downsampled to a common size:

1. **Human vs Task** — ``Task`` essays are written by the LLM from the prompt
   alone (``rewrite_of == "human"``), the same way a human answers it. The
   classes are matched on ``question_id`` (same prompt), so this compares
   *independent* human and AI writing on the same topics.
2. **Human vs Rewrite-Human** — ``Rewrite-Human`` essays are LLM rewordings of a
   specific human essay (``rewrite_of`` points to that essay's ``id``). The
   classes are matched on the source essay id, so the content is identical and
   the comparison isolates rewording/style.

In both, results are reported pooled and broken down by ``dataset``. The generic
comparison engine lives in ``diversity_compare`` (shared with the PAN example).

Run it with::

    uv run python examples/gede_essays.py path/to/gede_essays.json

If no path is given it looks for ``gede_essays.json`` in the current directory.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from diversity_compare import Item, run_comparison

# The two LLMs in the dataset, mapped to short column labels.
MODELS = {
    "gpt-4o-mini-2024-07-18": "gpt",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama",
}


def model_label(text_author: str) -> str:
    """Short label for an LLM author id (e.g. the gpt/Llama column names)."""
    return MODELS.get(text_author, text_author)


def load_records(path: Path) -> list[dict]:
    """Load the GEDE essays JSON (a list of record dicts)."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def task_classes(records: list[dict]) -> list[tuple[str, list[Item]]]:
    """Build the Human-vs-Task classes, matched on ``question_id``.

    Human essays and the from-the-prompt ``Task`` essays (split per model, plus a
    pooled mix) are tagged with ``match = question_id`` and ``group = dataset``.
    """
    human: list[Item] = []
    per_model: dict[str, list[Item]] = {label: [] for label in MODELS.values()}
    mix: list[Item] = []

    for r in records:
        level = r["contribution_level"]
        if level == "Human":
            human.append(Item(r["answer"], r["question_id"], r["dataset"]))
        elif level == "Task":
            item = Item(r["answer"], r["question_id"], r["dataset"])
            per_model[model_label(r["text_author"])].append(item)
            mix.append(item)

    return [
        ("Human", human),
        ("gpt", per_model["gpt"]),
        ("Llama", per_model["Llama"]),
        ("Mix", mix),
    ]


def rewrite_human_classes(records: list[dict]) -> list[tuple[str, list[Item]]]:
    """Build the Human-vs-Rewrite-Human classes, matched on the source essay id.

    Each human essay is tagged with ``match = id``; each ``Rewrite-Human`` essay
    (split per model, plus a pooled mix) is tagged with ``match = rewrite_of`` —
    the id of the human essay it rewords — so a human and its rewrites align.
    ``group`` is the ``dataset`` in both cases.
    """
    human: list[Item] = []
    per_model: dict[str, list[Item]] = {label: [] for label in MODELS.values()}
    mix: list[Item] = []

    for r in records:
        level = r["contribution_level"]
        if level == "Human":
            human.append(Item(r["answer"], r["id"], r["dataset"]))
        elif level == "Rewrite-Human" and str(r["rewrite_of"]).isdigit():
            source_id = int(r["rewrite_of"])
            item = Item(r["answer"], source_id, r["dataset"])
            per_model[model_label(r["text_author"])].append(item)
            mix.append(item)

    return [
        ("Human", human),
        ("gpt", per_model["gpt"]),
        ("Llama", per_model["Llama"]),
        ("Mix", mix),
    ]


def print_overview(records: list[dict]) -> None:
    """Print how the records break down by contribution level and author."""
    levels = Counter(r["contribution_level"] for r in records)
    authors = Counter(r["text_author"] for r in records)
    datasets = Counter(r["dataset"] for r in records)

    print(f"Loaded {len(records)} records")
    print(f"  datasets: {dict(datasets)}")
    print("  contribution levels:")
    for level, count in levels.most_common():
        print(f"    {level:<16}{count:>7}")
    print("  text authors:")
    for author, count in authors.most_common():
        print(f"    {author:<40}{count:>7}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gede_essays.json")
    if not path.exists():
        sys.exit(
            f"Could not find {path}. Pass the path to the GEDE essays JSON, e.g.:\n"
            "  uv run python examples/gede_essays.py path/to/gede_essays.json"
        )

    records = load_records(path)
    print_overview(records)

    # 1) Independent generation on the same prompts.
    run_comparison(
        task_classes(records),
        title="Human vs Task  (independent writing, matched by prompt)",
    )
    # 2) LLM rewordings of the same human essays.
    run_comparison(
        rewrite_human_classes(records),
        title="Human vs Rewrite-Human  (same content, matched by source essay)",
    )


if __name__ == "__main__":
    main()

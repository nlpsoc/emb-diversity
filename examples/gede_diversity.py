"""Human vs LLM essay diversity on GEDE, content-matched by prompt.

A self-contained experiment on the GEDE essays dataset (from "Assessing LLM Text
Detection in Educational Contexts", https://arxiv.org/abs/2508.08096). It
compares the diversity of human-written essays against LLM-written ones, holding
*topic* fixed: it uses the ``Task`` essays — written by an LLM from the prompt
alone, the way a human answers it — and matches them to the human essays on
``question_id`` (the prompt). Within each prompt the two classes are downsampled
to the same size, so the comparison is fair.

For each class it reports diversity along two axes — ``semantic`` (meaning) and
``style`` (writing style) — with three measures (``graph_entropy``,
``vendi_score``, ``mean_pw_dist``), as a single overall table, and saves a PCA
scatter of the embeddings on both axes.

Run it with::

    uv run --with gdown python examples/gede_diversity.py

With no argument it downloads the GEDE dataset (via ``gdown``, into
``gede_essay_detection/`` in the current directory) if it is not already
present. Pass a path to an existing ``gede_essays.json`` to skip the download::

    uv run python examples/gede_diversity.py path/to/gede_essays.json
"""

from __future__ import annotations

import json
import math
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to a file without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from emb_diversity import embed_texts, measure_diversity

SEED = 0  # for the downsampling, so runs are reproducible

# GEDE dataset download (Google Drive). Used when no data path is given.
GEDE_URL = "https://drive.google.com/file/d/1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD/view?usp=drive_link"
DATA_DIR = Path("gede_essay_detection")
TARBALL = Path("gede_essay_detection.tar.gz")
MEASURES = ["graph_entropy", "vendi_score", "mean_pw_dist"]
# Axis -> embed/measure kwargs. Both use the registered axis defaults
# (semantic: all-mpnet-base-v2; style: the Style-Embedding model).
AXES = {
    "semantic": {"diversity_axis": "semantic"},
    "style": {"diversity_axis": "style"},
}
COLORS = {"Human": "#3B6FB0", "AI": "#E0723C"}


def ensure_dataset() -> Path:
    """Download and extract the GEDE dataset if missing; return the JSON path.

    Mirrors the shell recipe: if ``gede_essay_detection/`` is absent, fetch the
    tarball from Google Drive with ``gdown``, extract it, and remove the
    tarball. Returns the path to ``gede_essays.json`` inside the dataset.
    """
    if DATA_DIR.exists():
        print(f"Skipping download ({DATA_DIR}/ already exists)")
    else:
        try:
            import gdown
        except ImportError:
            sys.exit(
                "Downloading the dataset needs gdown. Run with "
                "`uv run --with gdown python examples/gede_diversity.py`, or pass "
                "the path to an existing gede_essays.json as the first argument."
            )
        print("Downloading gede_essay_detection...")
        gdown.download(GEDE_URL, str(TARBALL), fuzzy=True)
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(filter="data")
        TARBALL.unlink()

    matches = sorted(DATA_DIR.rglob("gede_essays.json"))
    if not matches:
        sys.exit(f"Could not find gede_essays.json under {DATA_DIR}/ after download.")
    return matches[0]


def load_task_split(path: Path) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Group human and LLM ``Task`` essays by their prompt (``question_id``).

    Returns ``(human, ai)``, each a dict ``question_id -> [essay texts]``. ``ai``
    pools the LLM ``Task`` essays of all models (gpt-4o-mini and Llama-3.3).
    """
    human: dict[int, list[str]] = defaultdict(list)
    ai: dict[int, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for record in json.load(fh):
            level = record["contribution_level"]
            if level == "Human":
                human[record["question_id"]].append(record["answer"])
            elif level == "Task":
                ai[record["question_id"]].append(record["answer"])
    return human, ai


def balance_by_prompt(
    human: dict[int, list[str]], ai: dict[int, list[str]], seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Downsample to equal counts per prompt, returning flat human/AI text lists.

    For each prompt present in both classes, both are cut to the smaller count
    (a random subset, without replacement — no text is repeated or invented).
    """
    rng = random.Random(seed)
    human_texts: list[str] = []
    ai_texts: list[str] = []
    for q in sorted(set(human) & set(ai)):
        n = min(len(human[q]), len(ai[q]))
        human_texts += rng.sample(human[q], n) if len(human[q]) > n else list(human[q])
        ai_texts += rng.sample(ai[q], n) if len(ai[q]) > n else list(ai[q])
    return human_texts, ai_texts


def _fmt(value: float | None, sign: bool = False) -> str:
    """Format a score for a table cell, or "-" when missing/undefined."""
    if value is None or math.isnan(value):
        return "-"
    return f"{value:+.4f}" if sign else f"{value:.4f}"


def print_table(human_texts: list[str], ai_texts: list[str]) -> None:
    """Print one table with every measure x axis, Human vs AI side by side."""
    print(
        f"\nDiversity — Human vs AI (Task, matched by prompt, "
        f"{len(human_texts)} texts/class)\n"
    )
    header = f"{'axis':<10}{'measure':<16}{'Human':>12}{'AI':>12}{'AI - Human':>14}"
    print(header)
    print("-" * len(header))
    for axis, kwargs in AXES.items():
        human = measure_diversity(human_texts, **kwargs)
        ai = measure_diversity(ai_texts, **kwargs)
        for measure in MEASURES:
            h = human.get(measure, {}).get("value")
            a = ai.get(measure, {}).get("value")
            delta = a - h if (h is not None and a is not None) else None
            print(f"{axis:<10}{measure:<16}{_fmt(h):>12}{_fmt(a):>12}{_fmt(delta, sign=True):>14}")


def plot_pca(human_texts: list[str], ai_texts: list[str], out: Path) -> None:
    """Save a PCA scatter of the embeddings (one panel per axis)."""
    fig, axes = plt.subplots(1, len(AXES), figsize=(13, 6))
    for ax, (axis, kwargs) in zip(axes, AXES.items()):
        embeddings = {
            "Human": embed_texts(human_texts, **kwargs),
            "AI": embed_texts(ai_texts, **kwargs),
        }
        pca = PCA(n_components=2).fit(np.vstack([embeddings["Human"], embeddings["AI"]]))
        for label, emb in embeddings.items():
            xy = pca.transform(emb)
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.45,
                       c=COLORS[label], edgecolors="none", label=label)
        ax.set_title(f"{axis} — PCA of embeddings")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, framealpha=0.9)

    fig.suptitle(
        f"GEDE — Human vs AI (Task), content-matched ({len(human_texts)} texts/class)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            sys.exit(f"Could not find {path}.")
    else:
        path = ensure_dataset()

    human, ai = load_task_split(path)
    human_texts, ai_texts = balance_by_prompt(human, ai)
    print(
        f"GEDE Human vs AI (Task), matched by prompt: {len(human_texts)} texts/class "
        f"({len(human_texts) + len(ai_texts)} total)"
    )

    print_table(human_texts, ai_texts)
    plot_pca(human_texts, ai_texts, Path("gede_diversity_pca.png"))


if __name__ == "__main__":
    main()

"""Measure human-vs-LLM essay diversity on the GEDE dataset.

Final step of a small example comparing the diversity of human- vs LLM-written
essays on the GEDE dataset (from "Assessing LLM Text Detection in Educational
Contexts", https://arxiv.org/abs/2508.08096). It downloads the dataset (step 1),
loads the human essays and the LLM ``Task`` essays — written from the prompt
alone, the way a human answers it — matches them on ``question_id`` and
downsamples to equal size, prints the dataset stats, reports the semantic and
style diversity of each class as a table, and saves a PCA scatter of the
embeddings on both axes.

The download needs the ``examples`` extra (``gdown``); install it with
``pip install emb-diversity[examples]`` or, from a checkout,
``uv sync --extra examples``. Run it with::

    uv run --extra examples python examples/gede_diversity.py
"""

from __future__ import annotations

import json
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

# GEDE dataset on Google Drive — the file id from the share URL
# https://drive.google.com/file/d/1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD/view
GEDE_FILE_ID = "1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD"
DATA_DIR = Path("gede_essay_detection")
TARBALL = Path("gede_essay_detection.tar.gz")

# Seed for the prompt-matched downsampling, so runs are reproducible.
SEED = 0


# ── Dataset download ─────────────────────────────────────────────────────────
def ensure_dataset() -> Path:
    """Download and extract the GEDE dataset if missing; return the JSON path.

    If ``gede_essay_detection/`` is absent, fetch the tarball from Google Drive
    with ``gdown``, extract it, and remove the tarball. Returns the path to
    ``gede_essays.json`` inside the dataset.
    """
    if DATA_DIR.exists():
        print(f"Skipping download ({DATA_DIR}/ already exists)")
    else:
        try:
            import gdown
        except ImportError:
            sys.exit(
                "Downloading the dataset needs gdown (the 'examples' extra). "
                "Install it with `pip install emb-diversity[examples]`, or run "
                "`uv run --extra examples python examples/gede_diversity.py`."
            )
        print("Downloading gede_essay_detection...")
        gdown.download(id=GEDE_FILE_ID, output=str(TARBALL))
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(filter="data")
        TARBALL.unlink()

    matches = sorted(DATA_DIR.rglob("gede_essays.json"))
    if not matches:
        sys.exit(f"Could not find gede_essays.json under {DATA_DIR}/ after download.")
    return matches[0]


# ── Loading & prompt-matching ────────────────────────────────────────────────
def load_human_ai_split(path: Path) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Group human and LLM ``Task`` essays by their prompt (``question_id``).

    Returns ``(human, ai)``, each a dict ``question_id -> [essay texts]``; ``ai``
    pools the ``Task`` essays of all models (gpt-4o-mini and Llama-3.3).
    """
    human: dict[int, list[str]] = defaultdict(list)
    ai: dict[int, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for record in json.load(fh):
            level = record["contribution_level"]  # "Human" for human-written, anything else is LLM-written
            if level == "Human":
                human[record["question_id"]].append(record["answer"])
            elif level == "Task":  # independent generation from the prompt, not rewriting a human answer
                ai[record["question_id"]].append(record["answer"])
    return human, ai  # human/AI have different counts because persuade consists of 15 human vs 75 AI essays (see org. paper)


def balance_by_prompt(
    human: dict[int, list[str]], ai: dict[int, list[str]], seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Downsample to equal counts per prompt; return flat human/AI text lists.

    For each prompt present in both classes, both are cut to the smaller count
    (a random subset, without replacement), so the two classes match in size.
    """
    rng = random.Random(seed)
    human_texts: list[str] = []
    ai_texts: list[str] = []
    for q in sorted(set(human) & set(ai)):
        human_texts += rng.sample(human[q], 1)
        ai_texts += rng.sample(ai[q], 1)
    return human_texts, ai_texts


# ── Stats ────────────────────────────────────────────────────────────────────
def print_stats(
    human: dict[int, list[str]],
    ai: dict[int, list[str]],
    human_texts: list[str],
    ai_texts: list[str],
) -> None:
    """Print the dataset composition and the balanced per-class size."""
    n_human = sum(len(v) for v in human.values())
    n_ai = sum(len(v) for v in ai.values())
    print(f"Human essays:       {n_human}  (across {len(human)} prompts)")
    print(f"LLM 'Task' essays:  {n_ai}  (across {len(ai)} prompts)")
    print(f"Prompts in both:    {len(set(human) & set(ai))}")
    print(f"Balanced per class: {len(human_texts)}  ({len(human_texts) + len(ai_texts)} total)")


# ── Diversity measures ───────────────────────────────────────────────────────
# Embedding axes to measure; each uses the registered axis's default model
# (semantic: all-mpnet-base-v2, style: the Style-Embedding model).
AXES = ("semantic", "style")
MEASURES = ("graph_entropy", "vendi_score", "mean_pw_dist")


def measure_human_ai(
    human_texts: list[str], ai_texts: list[str]
) -> dict[str, dict[str, dict]]:
    """Measure the diversity of the human and AI sets along each axis.

    Returns ``{axis: {"Human": results, "AI": results}}``, where each ``results``
    is the ``measure_diversity`` output (measure name -> ``{"value": ...}``).
    """
    return {
        axis: {
            "Human": measure_diversity(human_texts, measure=list(MEASURES), diversity_axis=axis),
            "LLM": measure_diversity(ai_texts, measure=list(MEASURES), diversity_axis=axis),
        }
        for axis in AXES
    }


# ── Results table ────────────────────────────────────────────────────────────
def print_results_table(results: dict[str, dict[str, dict]], n_per_class: int) -> None:
    """Print one table: every measure and axis, Human vs LLM with the gap."""
    print(f"\nDiversity (Human vs LLM, {n_per_class} texts/class)\n")
    header = f"{'axis':<10}{'measure':<15}{'Human':>11}{'LLM':>11}{'LLM - Human':>14}"
    print(header)
    print("-" * len(header))
    for axis in AXES:
        for measure in MEASURES:
            human = results[axis]["Human"][measure]["value"]
            llm = results[axis]["LLM"][measure]["value"]
            print(f"{axis:<10}{measure:<15}{human:>11.4f}{llm:>11.4f}{llm - human:>+14.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────
# Marker colours for the two classes in the scatter.
COLORS = {"Human": "#3B6FB0", "LLM": "#E0723C"}


def plot_pca(human_texts: list[str], ai_texts: list[str], out: Path) -> None:
    """Save a PCA scatter of the embeddings, one panel per axis.

    PCA preserves scale, so the relative spread of the two classes is faithful:
    on the style axis the LLM essays form a tight cluster while the human essays
    stay spread out.
    """
    fig, axes = plt.subplots(1, len(AXES), figsize=(13, 6))
    for ax, axis in zip(axes, AXES):
        embeddings = {
            "Human": embed_texts(human_texts, diversity_axis=axis),
            "LLM": embed_texts(ai_texts, diversity_axis=axis),
        }
        pca = PCA(n_components=2).fit(np.vstack(list(embeddings.values())))
        for label, emb in embeddings.items():
            xy = pca.transform(emb)
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.45,
                       c=COLORS[label], edgecolors="none", label=label)
        ax.set_title(f"{axis} — PCA of embeddings")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, framealpha=0.9)

    fig.suptitle(
        f"GEDE — Human vs LLM, content-matched ({len(human_texts)} texts/class)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    path = ensure_dataset()
    human, ai = load_human_ai_split(path)
    human_texts, ai_texts = balance_by_prompt(human, ai)
    print_stats(human, ai, human_texts, ai_texts)

    results = measure_human_ai(human_texts, ai_texts)
    print_results_table(results, len(human_texts))

    plot_pca(human_texts, ai_texts, Path("gede_diversity_pca.png"))


if __name__ == "__main__":
    main()

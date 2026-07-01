"""Measure speaker diversity as more distinct voices read the same sentences.

Demonstrates the "speaker" diversity axis: ``measure_diversity()`` accepts a
list of audio file paths just like a list of texts — pass
``diversity_axis="speaker"`` and it embeds each file with a
speaker-discriminative model, then runs the same generic measures on the
resulting vectors.

Content is held fixed across all four conditions below — the same set of
sentences from the CMU ARCTIC corpus — so content is not a confound; only the
number of distinct speakers changes from 2 up to 5. The sentences are split
into N roughly-equal contiguous chunks for the N-speaker condition, one chunk
per speaker (e.g. with 4 speakers each reads about a quarter of the
sentences). Sample size (total recordings) is held fixed across conditions,
since the measures are sensitive to it — otherwise that would confound the
"number of speakers" comparison.

The dataset is located via the ``CMU_ARCTIC_DIR`` environment variable if set
(handy when the data is already present, e.g. on an HPC cluster); otherwise
it is downloaded automatically from Kaggle with ``kagglehub`` (cached after
the first download, requires ``~/.kaggle/kaggle.json``). Needs the ``audio``
extra (``speechbrain`` / ``torchaudio`` / ``soundfile``) and the ``examples``
extra (``kagglehub``). Install them with::

    pip install emb-diversity[audio,examples]

or from a checkout::

    uv sync --extra audio --extra examples

Run it with::

    uv run --extra audio --extra examples python examples/speaker_diversity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed on HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emb_diversity import measure_diversity

# ── Experiment configuration ──────────────────────────────────────────────────

SPEAKERS = ["bdl", "clb", "awb", "slt", "rms"]
N_SPEAKER_CONDITIONS = range(2, len(SPEAKERS) + 1)  # 2, 3, 4, 5
MEASURES = ("graph_entropy", "vendi_score", "mean_pw_dist")
KAGGLE_DATASET = "mrgabrielblins/speaker-recognition-cmu-arctic"


# ── Dataset ───────────────────────────────────────────────────────────────────

def ensure_dataset() -> Path:
    """Return the CMU ARCTIC dataset root, downloading it if needed.

    Uses ``kagglehub``, which caches the dataset locally and skips the
    download on subsequent runs. Requires a Kaggle account and API token
    (``~/.kaggle/kaggle.json``).
    """
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "Downloading the dataset needs kagglehub (the 'examples' extra).\n"
            "Install it with:  pip install emb-diversity[examples]"
        )
    print(f"Fetching {KAGGLE_DATASET} via kagglehub (cached after first download) …")
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


# ── Sentence / speaker table ──────────────────────────────────────────────────

def _load_common_sentence_table(dataset_dir: Path) -> dict[str, dict[str, str]]:
    """Map each sentence that all ``SPEAKERS`` share to ``{speaker: file_path}``.

    Reads ``train.csv``, restricts to ``SPEAKERS``, groups by exact transcript
    text, and keeps only sentences with a recording from every speaker.
    Matching is by text, not by ``arctic_aXXXX`` id — at least one speaker's
    file numbering is offset relative to the others, so the same id can hold
    different text per speaker.
    """
    df = pd.read_csv(dataset_dir / "train.csv")
    df["speech_norm"] = df["speech"].str.strip()
    sub = df[df["speaker"].isin(SPEAKERS)]

    table: dict[str, dict[str, str]] = {}
    for sentence, group in sub.groupby("speech_norm"):
        per_speaker = dict(zip(group["speaker"], group["file_path"]))
        if set(per_speaker) == set(SPEAKERS):
            table[sentence] = per_speaker
    return table


def _paths_for_condition(
    sentences: list[str],
    table: dict[str, dict[str, str]],
    n_speakers: int,
    dataset_dir: Path,
) -> list[str]:
    """Return audio paths for the ``n_speakers`` condition.

    Splits *sentences* into ``n_speakers`` contiguous chunks of equal size
    and assigns one chunk to each speaker, so the total recording count stays
    fixed regardless of how many speakers participate.
    """
    paths = []
    for speaker, chunk in zip(SPEAKERS[:n_speakers], np.array_split(sentences, n_speakers)):
        for sentence in chunk:
            paths.append(str(dataset_dir / table[sentence][speaker]))
    return paths


# ── Diversity measures ────────────────────────────────────────────────────────

def measure_all_conditions(
    sentences: list[str],
    table: dict[str, dict[str, str]],
    dataset_dir: Path,
) -> dict[int, dict]:
    """Run ``measure_diversity`` for each speaker-count condition.

    Returns ``{n_speakers: measure_diversity_output}`` for each n in
    ``N_SPEAKER_CONDITIONS``.
    """
    results: dict[int, dict] = {}
    for n in N_SPEAKER_CONDITIONS:
        paths = _paths_for_condition(sentences, table, n, dataset_dir)
        print(f"  {n} speaker(s): {len(paths)} recordings …")
        results[n] = measure_diversity(
            paths, measure=list(MEASURES), diversity_axis="speaker"
        )
    return results


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(results: dict[int, dict], n_sentences: int) -> None:
    """Print diversity scores for each condition as a formatted table."""
    ns = sorted(results)
    header = f"{'measure':<15}" + "".join(f"{n} speaker(s)".rjust(15) for n in ns)
    print(f"\nSpeaker diversity  ({n_sentences} sentences per condition)\n")
    print(header)
    print("-" * len(header))
    for name in MEASURES:
        row = "".join(f"{results[n][name]['value']:.4f}".rjust(15) for n in ns)
        print(f"{name:<15}{row}")
    print()
    print(
        f"Diversity should climb from left (2 speakers) to right ({max(ns)} speakers): "
        "same sentences throughout, only the number of distinct voices changes."
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_diversity(results: dict[int, dict], out: Path) -> None:
    """Save a line chart of diversity vs number of speakers, one panel per measure.

    Each panel shows how one measure changes as more distinct speakers
    contribute to the same fixed set of sentences. The figure is sized for a
    single-column layout and saved as a vector-friendly PNG.
    """
    x = sorted(results)
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8,
        "legend.fontsize": 6, "font.family": "serif",
    })
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(6.5, 2.2))
    for ax, measure in zip(axes, MEASURES):
        y = [results[n][measure]["value"] for n in x]
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=4)
        ax.set_title(measure)
        ax.set_xlabel("speakers")
        ax.set_xticks(x)
        ax.set_ylabel("diversity")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=1.0)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    dataset_dir = ensure_dataset()

    print("Building sentence/speaker table …")
    table = _load_common_sentence_table(dataset_dir)
    sentences = sorted(table)
    print(f"Found {len(sentences)} sentences common to all {len(SPEAKERS)} speakers.")

    print(f"\nMeasuring diversity for {len(list(N_SPEAKER_CONDITIONS))} conditions …")
    results = measure_all_conditions(sentences, table, dataset_dir)

    print_results_table(results, len(sentences))
    plot_diversity(results, Path("speaker_diversity.png"))


if __name__ == "__main__":
    main()

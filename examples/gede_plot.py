"""Visualize why human and LLM essays differ in style but not in meaning.

Companion to ``gede_essays.py``. For one content-matched comparison it builds a
2x2 figure:

- top row: a 2-D PCA projection of the embeddings (semantic, then style),
  coloured by class — you can see the LLM essays collapse into a tight blob on
  the style axis while the humans stay spread out;
- bottom row: the distribution of within-class pairwise cosine distances
  (semantic, then style), with each class's mean — the numeric "spread" behind
  ``mean_pw_dist``.

Embeddings come from the same disk cache ``gede_essays.py`` populates, so this is
fast to re-run once that has been run.

Run it with::

    uv run python examples/gede_plot.py path/to/gede_essays.json [rewrite|task] [pca|umap]

``rewrite`` (default) plots the Human vs Rewrite-Human comparison (identical
content); ``task`` plots Human vs Task (independent writing, same prompts).

The projection defaults to ``pca``. PCA preserves scale, so the tight LLM
style-blob faithfully reflects its small spread (the thing ``mean_pw_dist`` and
``vendi`` measure). ``umap`` reveals local cluster structure / separability
better, but it normalizes density and distorts cluster sizes and gaps — so don't
read absolute spread off a UMAP plot; the distance histograms below stay the
quantitative ground truth in both cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from diversity_compare import balance_classes
from emb_diversity import embed_texts
from gede_essays import load_records, rewrite_human_classes, task_classes

# Axis name -> embed_texts kwargs (semantic uses the axis default; style overrides
# the model with StyleDistance, matching gede_essays.py).
AXES = [
    ("semantic", {"diversity_axis": "semantic"}),
    ("style", {"embedding_model": "StyleDistance/styledistance"}),
]
PLOT_CLASSES = ["Human", "gpt", "Llama"]  # Mix omitted (it is gpt+Llama pooled)
COLORS = {"Human": "#3B6FB0", "gpt": "#E08A3C", "Llama": "#4C9A56"}


def class_texts(comparison: list) -> dict[str, list[str]]:
    """Balance a comparison and return {label: texts} for the plotted classes."""
    balanced = balance_classes(comparison)
    return {
        label: [item.text for item in items]
        for label, items in balanced
        if label in PLOT_CLASSES
    }


def project(emb: dict[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    """Project each class's embeddings to 2-D with a shared PCA or UMAP fit.

    PCA preserves scale (faithful relative spreads); UMAP reveals local cluster
    structure but distorts cluster sizes and inter-cluster distances.
    """
    stacked = np.vstack([emb[c] for c in PLOT_CLASSES])
    if method == "umap":
        import umap  # imported lazily; only needed for this branch

        coords = umap.UMAP(n_components=2, metric="cosine", random_state=0).fit_transform(stacked)
    else:
        coords = PCA(n_components=2).fit_transform(stacked)

    out, start = {}, 0
    for label in PLOT_CLASSES:
        out[label] = coords[start : start + len(emb[label])]
        start += len(emb[label])
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gede_essays.json")
    which = sys.argv[2] if len(sys.argv) > 2 else "rewrite"
    if not path.exists():
        sys.exit(f"Could not find {path}. Pass the path to the GEDE essays JSON.")

    method = sys.argv[3] if len(sys.argv) > 3 else "pca"
    records = load_records(path)
    builder = task_classes if which == "task" else rewrite_human_classes
    texts = class_texts(builder(records))
    n_per_class = len(next(iter(texts.values())))

    fig, axes = plt.subplots(2, len(AXES), figsize=(14, 11))
    for col, (axis_name, embed_kwargs) in enumerate(AXES):
        emb = {label: embed_texts(t, **embed_kwargs) for label, t in texts.items()}

        # ── top: 2-D projection (one shared fit across all classes) ──
        coords = project(emb, method)
        ax = axes[0][col]
        for label in PLOT_CLASSES:
            xy = coords[label]
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.45,
                       c=COLORS[label], edgecolors="none", label=label)
        ax.set_title(f"{axis_name} — {method.upper()} of embeddings")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, framealpha=0.9)

        # ── bottom: within-class pairwise cosine-distance distribution ──
        ax2 = axes[1][col]
        for label in PLOT_CLASSES:
            dists = pdist(emb[label], metric="cosine")
            ax2.hist(dists, bins=60, density=True, histtype="step", linewidth=1.8,
                     color=COLORS[label], label=f"{label}  (mean {dists.mean():.3f})")
        ax2.set_title(f"{axis_name} — within-class pairwise cosine distance")
        ax2.set_xlabel("cosine distance")
        ax2.set_ylabel("density")
        ax2.legend()

    fig.suptitle(
        f"GEDE '{which}' comparison — Human vs LLM, content-matched "
        f"({n_per_class} texts/class, {method.upper()})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"gede_{which}_semantic_vs_style_{method}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

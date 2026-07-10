"""Grouped correlation heatmap for the emb-diversity measures.

Usage: python correlation_grouped.py <correlations.json> [--small]
  default: full-size figure with per-cell numbers and full names
  --small: ACL two-column-width figure (6.3 in), abbreviations on the
           x-axis, "name (ABBR)" on the y-axis, no per-cell numbers
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

MAKE_SMALL = "--small" in sys.argv
paths = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(paths) != 1:
    raise SystemExit("usage: python correlation_grouped.py <correlations.json> [--small]")

# ---------------------------------------------------------------- input ----
# JSON with nested correlations: {"measure_a": {"measure_b": value, ...}, ...}
# (e.g. pandas: df.corr(method="spearman").to_json())
with open(paths[0]) as fh:
    corr = json.load(fh)
names_json = list(corr) # get dict (measure) names
M = np.array([[corr[a][b] for b in names_json] for a in names_json])  # get correlation matrix

ABBREV = {
    "mean_pw_dist": "MPD", "sum_pairwise_dist": "SPD", "energy": "EN",
    "span_medoid": "SpM", "knn": "kNN", "chamfer_dist": "CD",
    "sum_bottleneck": "SB", "bottleneck": "BN", "sum_diameter": "SD",
    "diameter": "DM", "mst_dispersion": "MST", "graph_entropy": "GE",
    "hamdiv": "HD", "vendi_score": "VS", "renyi_entropy": "RE",
    "log_determinant": "LD", "dcscore": "DCS", "bins_entropy": "BE",
    "convex_hull_volume_3d": "CHV", "cluster_inertia": "CI",
    "span_centroid": "SC", "geo_mean_std": "GMS",
}

assert np.allclose(M, M.T, atol=1e-8), "correlation matrix must be symmetric"
assert len(set(ABBREV.values())) == len(ABBREV), "abbreviations must be unique"

# ------------------------------------------------------------- grouping ----
groups = [
    ("distance matrix", "black",
     ["mean_pw_dist", "sum_pairwise_dist", "energy", "span_medoid",
      "knn", "chamfer_dist", "sum_bottleneck", "bottleneck",
      "sum_diameter", "diameter"]),
    ("distance graph", "#2171b5",
     ["mst_dispersion", "graph_entropy", "hamdiv"]),
    ("kernel matrix", "#ff3333",
     ["vendi_score", "renyi_entropy", "log_determinant", "dcscore"]),
    ("UMAP proj.", "#666666",
     ["bins_entropy", "convex_hull_volume_3d"]),
    ("vector stats", "#6a51a3",
     ["cluster_inertia", "span_centroid", "geo_mean_std"]),
]
subs = [("all-pairs aggregate", 0, 4), ("minima", 4, 8), ("maxima", 8, 10)]
DEFAULTS = {"graph_entropy", "vendi_score", "mean_pw_dist"}

# keep only measures present in the input; anything unknown goes to "other"
groups = [(group, color, [m for m in ms if m in names_json]) for group, color, ms in groups]
groups = [(group, color, ms) for group, color, ms in groups if ms]
known = {m for _, _, ms in groups for m in ms}
leftover = [m for m in names_json if m not in known]
if leftover:
    groups.append(("other", "#999999", leftover))

order = [m for _, _, ms in groups for m in ms]
idx = [names_json.index(m) for m in order]
R = M[np.ix_(idx, idx)]
n = len(order)

# --------------------------------------------------------------- layout ----
if MAKE_SMALL:  # ACL two-column width (\textwidth ~ 6.3 in)
    figsize, annotate = (6.3, 5.4), False
    tick_fs, group_fs, sub_fs, cbar_fs = 8, 8.5, 6, 9
    group_rot, group_ha = 0, "right"
    sub_short = {"all-pairs aggregate": "all-pairs"}
    title = None
else:
    figsize, annotate = (12, 11), True
    tick_fs, group_fs, sub_fs, cbar_fs = 8, 9, 7, 10
    group_rot, group_ha = 90, "center"
    sub_short = {}
    title = "Spearman correlation among measures, ordered by timing groups"

# ------------------------------------------------------------- plotting ----
fig, ax = plt.subplots(figsize=figsize)
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
if annotate:
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.75 else "#1a1a1a")

def star(text, m):
    return ("★ " + text) if m in DEFAULTS else text

if MAKE_SMALL:
    xlabels = [star(ABBREV.get(m, m), m) for m in order]
    ylabels = [star(f"{m} ({ABBREV[m]})" if m in ABBREV else m, m)
               for m in order]
else:
    xlabels = [star(m, m) for m in order]
    ylabels = [star(m, m) for m in order]
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(xlabels, rotation=90, fontsize=tick_fs)
ax.set_yticklabels(ylabels, fontsize=tick_fs)
for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
    for t, m in zip(ticklabels, order):
        if m in DEFAULTS:
            t.set_fontweight("bold")
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08,
                    label="Spearman correlation")
cbar.ax.tick_params(labelsize=cbar_fs)
cbar.set_label("Spearman correlation", fontsize=cbar_fs)
if title:
    ax.set_title(title)
fig.tight_layout()

# measure how far the y tick labels reach (in axes fractions), then place
# the annotations relative to that instead of hand-tuned constants
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
to_axes = ax.transAxes.inverted()
tick_left = min(to_axes.transform(t.get_window_extent(renderer))[0][0]
                for t in ax.get_yticklabels())
bracket_x = tick_left - 0.02
sublabel_x = tick_left - 0.05
group_x = tick_left - (0.09 if group_rot == 0 else 0.12)

starts = np.cumsum([0] + [len(ms) for _, _, ms in groups])
for b in starts[1:-1]:
    ax.axhline(b - 0.5, color="white", lw=3)
    ax.axvline(b - 0.5, color="white", lw=3)
for (label, color, ms), s in zip(groups, starts):
    y = 1 - (s + len(ms) / 2) / n            # imshow: row 0 is at the top
    ax.text(group_x, y, label, transform=ax.transAxes, rotation=group_rot,
            va="center", ha=group_ha, fontsize=group_fs, fontweight="bold",
            color=color)

for _, _, end in subs[:-1]:
    ax.axhline(end - 0.5, color="white", lw=1.2, ls=(0, (2, 2)))
    ax.axvline(end - 0.5, color="white", lw=1.2, ls=(0, (2, 2)))
# x in axes fraction, y in data rows — offsets survive any figure size
side = ax.get_yaxis_transform()
for label, start, end in subs:
    label = sub_short.get(label, label)
    center = (start + end - 1) / 2
    top, bot = start - 0.35, end - 1 + 0.35
    ax.plot([bracket_x, bracket_x], [top, bot], color="#444444", lw=1.2,
            transform=side, clip_on=False)
    ax.text(sublabel_x, center, label, rotation=90, va="center", ha="center",
            fontsize=sub_fs, style="italic", color="#444444",
            transform=side, clip_on=False)

# the bracket plots above autoscale the axes; pin them back to the matrix
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)

suffix = "_small" if MAKE_SMALL else ""
fig.savefig(f"correlation_grouped{suffix}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"correlation_grouped{suffix}.pdf", bbox_inches="tight")
print(f"saved correlation_grouped{suffix}.png / .pdf")
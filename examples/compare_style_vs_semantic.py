"""Visualize selected pairs, with PCA fit on the full datasets for a stable frame."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from emb_diversity import embed_texts, vendi_score


BLUE   = "#6C7FB8"
ORANGE = "#F5A77E"


# Full datasets — used to fit PCA
full_a = [
    "I thoroughly enjoy the hair bands of the 1980s."
    "songs of the 80's are the best.",
    "Hip Hop is going DOWNHILL!!!!!",
    "rock music just makes me feel good",
    "The 80's rocked!That generation had the best music!"
]
full_b = [
    "I thoroughly enjoy the hair bands of the 1980s.",
    "They have not caused any harm to me.",
    "He has a very distinct walk.",
    "It depends on what they will pay.",
    "I would go out with the son of a preacher.",
]

# The pairs we actually want to display
display_a = [
    "I thoroughly enjoy the hair bands of the 1980s.",
    "The 80's rocked!That generation had the best music!",
]
display_b = [
    "I thoroughly enjoy the hair bands of the 1980s.",
    "They have not caused any harm to me.",
]

SHARED = "I thoroughly enjoy the hair bands of the 1980s."

SHARED_COLOR_OWNER = {
    "style":    "B",
    "semantic": "A",
}

AXES = list(SHARED_COLOR_OWNER.keys())

fig, axes_grid = plt.subplots(1, len(AXES), figsize=(20, 7))

PAD_LEFT  = 0.15
PAD_RIGHT = 6
PAD_TOP   = 0.25
PAD_BOT   = 0.25

for ax, axis_name in zip(axes_grid, AXES):
    emb_full_a = embed_texts(full_a, diversity_axis=axis_name)
    emb_full_b = embed_texts(full_b, diversity_axis=axis_name)
    combined = np.vstack([emb_full_a, emb_full_b])

    pca = PCA(n_components=2).fit(combined)

    emb_disp_a = embed_texts(display_a, diversity_axis=axis_name)
    emb_disp_b = embed_texts(display_b, diversity_axis=axis_name)
    coords_a = pca.transform(emb_disp_a)
    coords_b = pca.transform(emb_disp_b)

    # Diversity scores on the displayed pairs (in original embedding space)
    score_a = vendi_score(emb_disp_a)["value"]
    score_b = vendi_score(emb_disp_b)["value"]

    ax.scatter(coords_a[:, 0], coords_a[:, 1],
               c=BLUE, s=360, label=f"vendi score: {score_a:.3f}")
    ax.scatter(coords_b[:, 0], coords_b[:, 1],
               c=ORANGE, s=360, marker="^",
               label=f"vendi score: {score_b:.3f}")

    shared_owner = SHARED_COLOR_OWNER[axis_name]

    for (x, y), t in zip(coords_a, display_a):
        if t == SHARED and shared_owner != "A":
            continue
        ax.annotate(t, (x, y), fontsize=20, color=BLUE, fontweight="bold",
                    xytext=(14, 16), textcoords="offset points")

    for (x, y), t in zip(coords_b, display_b):
        if t == SHARED and shared_owner != "B":
            continue
        ax.annotate(t, (x, y), fontsize=20, color=ORANGE, fontweight="bold",
                    xytext=(14, -30), textcoords="offset points")

    all_pts = np.vstack([coords_a, coords_b])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    x_rng = max(x_max - x_min, 1e-6)
    y_rng = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - PAD_LEFT * x_rng, x_max + PAD_RIGHT * x_rng)
    ax.set_ylim(y_min - PAD_BOT  * y_rng, y_max + PAD_TOP   * y_rng)

    ax.set_title(f"{axis_name} axis", fontsize=30, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(loc="best", fontsize=26)
    for text in legend.get_texts():
        text.set_fontweight("bold")

plt.tight_layout()
plt.savefig("dataset_comparison_pairs_full_pca.png", dpi=150, bbox_inches="tight")
plt.show()
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from emb_diversity import (
    mean_pw_dist, dist_dispersion, diameter, bottleneck, sum_bottleneck,
    sum_diameter, chamfer_dist, span_centroid, span_medoid, vendi_score,
    renyi_entropy, bins_entropy, graph_entropy, convex_hull_volume_2d,
    hamdiv, mst_dispersion, radius, energy, cluster_inertia, dcscore,
    log_determinant,
)
# Direct module handle for the pairwise-distance cache controls.
from emb_diversity.measures import utils as pw

# Disable the in-memory LRU so repeated large distance matrices never
# accumulate in RAM during the benchmark.
pw._MEMORY_MAX = 0


# Define measures
measures = {
    "mean_pw_dist": mean_pw_dist,
    "dist_dispersion": dist_dispersion,
    "diameter": diameter,
    "bottleneck": bottleneck,
    "sum_bottleneck": sum_bottleneck,
    "sum_diameter": sum_diameter,
    "chamfer_dist": chamfer_dist,
    "span_centroid": span_centroid,
    "span_medoid": span_medoid,
    "vendi_score": vendi_score,
    "renyi_entropy": renyi_entropy,
    "bins_entropy": bins_entropy,
    "graph_entropy": graph_entropy,
    "convex_hull_volume_2d": convex_hull_volume_2d,
    #  "hamdiv": hamdiv,
    "mst_dispersion": mst_dispersion,
    "radius": radius,
    "energy": energy,
    "cluster_inertia": cluster_inertia,
    "dcscore": dcscore,
    "log_determinant": log_determinant,
}


def time_call(fn, *args, **kwargs):
    """Return the time a measure took to run, from a cold cache.

    Clearing the distance cache before each call ensures we always time the
    real computation instead of a cache hit (the same dataset is reused across
    runs, which would otherwise be served from cache after the first run).
    """
    pw.clear_distance_cache()  # drop memory + disk so this is a true cold run
    start = time.perf_counter()
    fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start


# Synthetic datasets of varying sizes. 100_000 is intentionally omitted:
# pdist on 100k points builds a ~40 GB condensed array and would OOM.
SIZES = [10, 100, 1000, 10000, 100_000]
N_RUNS = 10

vector_datasets = [np.random.randn(n, 384) for n in SIZES]

# Initialize results dict
results = {name: defaultdict(list) for name in measures}

for i in range(N_RUNS):
    for dataset in vector_datasets:
        for name, fn in measures.items():
            elapsed = time_call(fn, dataset)
            results[name][f"run_{i}"].append(elapsed)
            print(f"{name} | size={len(dataset)} | {elapsed:.6f}s")
    print("Completed run", i)


# PLOTTING
sizes = np.array(SIZES)  # aligned with the datasets actually run

plt.figure(figsize=(5, 5))

# Large color palette
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

for idx, (measure_name, runs) in enumerate(results.items()):
    run_matrix = np.array([runs[f"run_{i}"] for i in range(N_RUNS)])
    mean_vals = run_matrix.mean(axis=0)
    std_vals = run_matrix.std(axis=0)

    color = colors[idx]

    plt.plot(sizes, mean_vals, marker="o", label=measure_name, color=color)
    plt.fill_between(sizes, mean_vals - std_vals, mean_vals + std_vals,
                     alpha=0.2, color=color)

plt.xscale("log")
plt.xlabel("Dataset size (log scale)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime scaling of embedding diversity measures")
plt.legend(ncol=2, fontsize=9)
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("runtime_scaling.pdf", bbox_inches="tight")
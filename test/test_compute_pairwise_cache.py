import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import measure_diversity.compute_pairwise as pw_module
from measure_diversity import compute_pairwise_distances, clear_distance_cache

CACHE_DIR = Path(".cache/pdist_test")
N_RUNS = 10
SIZES = [10000,20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

results = {"without_cache": [], "with_cache": []}

for n in SIZES:
    timings = {"without_cache": [], "with_cache": []}

    for run in range(N_RUNS):
        print(f"[{n} rows] run {run+1}/{N_RUNS} ...", end=" ")
        data = np.random.randn(n, 768)

        # Without cache — call pdist twice
        start = time.time()
        pdist(data, metric="cosine")
        pdist(data, metric="cosine")
        pdist(data, metric="cosine")
        t_no = time.time() - start

        # With cache — compute once, second call hits disk
        clear_distance_cache(CACHE_DIR)
        start = time.time()
        compute_pairwise_distances(data, "cosine", CACHE_DIR)

        compute_pairwise_distances(data, "cosine", CACHE_DIR)
        
        compute_pairwise_distances(data, "cosine", CACHE_DIR)

        t_cache = time.time() - start

        timings["without_cache"].append(t_no)
        timings["with_cache"].append(t_cache)

        print(f"without={t_no:.2f}s  with={t_cache:.3f}s")

    for k in results:
        results[k].append(sum(timings[k]) / N_RUNS)

    print(f"  >>> {n} rows avg | without: {results['without_cache'][-1]:.4f}s | with: {results['with_cache'][-1]:.4f}s\n")

clear_distance_cache(CACHE_DIR)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(SIZES, results["without_cache"], "o-", label="Without cache (2x pdist)", linewidth=2)
ax.plot(SIZES, results["with_cache"], "s-", label="With cache (disk hit + memory hit)", linewidth=2)

ax.set_xlabel("Number of rows")
ax.set_ylabel("Time (seconds)")
ax.set_title("2 Metric Calls: Without Cache vs With Cache (avg of 5 runs, dim=768)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cache_benchmark.png", dpi=150)
print("Saved cache_benchmark.png")
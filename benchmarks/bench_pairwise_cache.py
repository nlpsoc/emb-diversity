"""
Benchmark: pairwise distance cache vs raw scipy.pdist.

For each dataset size, we time the cost of running 3 distance computations
on the same matrix (a realistic pattern when several measures share an
embedding matrix). The cached version pays for one pdist + 2 disk reads;
the no-cache baseline pays for 3 full pdist calls.

Outputs cache_benchmark.png in the working directory.
"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from measure_diversity import compute_pairwise_distances, clear_distance_cache

CACHE_DIR = Path(".cache/pdist_bench")
N_RUNS = 5
SIZES = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
N_REPEATS = 3  # how many distance calls per workload

results = {"without_cache": [], "with_cache": []}

for n in SIZES:
    timings = {"without_cache": [], "with_cache": []}

    for run in range(N_RUNS):
        print(f"[{n} rows] run {run + 1}/{N_RUNS} ...", end=" ")
        data = np.random.randn(n, 768)

        # No cache: full pdist N_REPEATS times
        start = time.time()
        for _ in range(N_REPEATS):
            pdist(data, metric="cosine")
        t_no = time.time() - start

        # With cache: first call computes, the rest are cache hits
        clear_distance_cache(CACHE_DIR)
        start = time.time()
        for _ in range(N_REPEATS):
            compute_pairwise_distances(data, "cosine", CACHE_DIR)
        t_cache = time.time() - start

        timings["without_cache"].append(t_no)
        timings["with_cache"].append(t_cache)
        print(f"without={t_no:.2f}s  with={t_cache:.3f}s")

    for k in results:
        results[k].append(sum(timings[k]) / N_RUNS)

    print(f"  >>> {n} rows avg | without: {results['without_cache'][-1]:.4f}s | "
          f"with: {results['with_cache'][-1]:.4f}s\n")

clear_distance_cache(CACHE_DIR)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(SIZES, results["without_cache"], "o-",
        label=f"Without cache ({N_REPEATS}x pdist)", linewidth=2)
ax.plot(SIZES, results["with_cache"], "s-",
        label=f"With cache (1 compute + {N_REPEATS - 1} cache hits)", linewidth=2)
ax.set_xlabel("Number of rows")
ax.set_ylabel("Time (seconds)")
ax.set_title(f"{N_REPEATS} distance calls: with vs without cache (avg of {N_RUNS} runs, dim=768)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cache_benchmark.png", dpi=150)
print("Saved cache_benchmark.png")

"""Benchmark the runtime of every registered measure across dataset sizes.

Runs every measure in ``MEASURE_NAMES`` on synthetic ``(n, 384)`` Gaussian
vectors. Results are checkpointed to a JSON file after every (measure, size)
pair, so a crashed run loses almost nothing and rerunning the same command
resumes where it stopped. Each (measure, size) pair runs in a child process
with a wall-clock budget: runs finished within the budget are kept, a pair
whose first run doesn't finish is marked "timeout" and the measure is skipped
at larger sizes, and a crashing child (e.g. out of memory) is marked "error".

Usage::

    python timing_measures_v2.py run     # benchmark (resumable)
    python timing_measures_v2.py plot    # render the figure from the JSON

Edit the constants below to change sizes, runs, or budget.
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

SIZES = [10, 100, 1_000, 10_000, 100_000]
N_RUNS = 5
DIM = 384
SEED = 42
BUDGET_S = 3600  # wall-clock budget per (measure, size) pair
RESULTS_FILE = Path("timing_results.json")
PLOT_FILE = Path("runtime_scaling.pdf")


def save_json(path: Path, payload) -> None:
    """Write via a temp file + atomic rename so a crash never corrupts it."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _worker(measure_name: str, size: int, out_path: str) -> None:
    """Child process: time N_RUNS cold-cache runs of one measure.

    Timings are flushed to *out_path* after every run, so the parent keeps
    the finished runs even if it kills this process at the budget.
    """
    import numpy as np

    from emb_diversity.measures_registry import get_measure
    from emb_diversity.measures import utils as pw

    pw._MEMORY_MAX = 0  # don't let large distance matrices pile up in RAM
    fn = get_measure(measure_name)
    data = np.random.RandomState(SEED).randn(size, DIM)

    times = []
    for _ in range(N_RUNS):
        pw.clear_distance_cache()  # memory + disk, so every run is cold
        start = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - start)
        Path(out_path).write_text(json.dumps(times))


def run_measure(measure_name: str, size: int, tmp_path: Path) -> dict:
    """Run one (measure, size) pair in a child process under BUDGET_S."""
    # Ensure we don't accidentally read stale timings after a parent/child crash.
    tmp_path.unlink(missing_ok=True)

    # spawn (not fork) = clean interpreter per pair on every platform
    proc = mp.get_context("spawn").Process(
        target=_worker, args=(measure_name, size, str(tmp_path))
    )
    proc.start()
    proc.join(timeout=BUDGET_S)
    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join()
    try:  # the child may have been killed mid-write
        times = json.loads(tmp_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        times = []
    tmp_path.unlink(missing_ok=True)

    if times:  # at least one run finished — that's a usable timing
        return {"status": "ok", "times": times}
    return {"status": "timeout" if timed_out else "error", "times": []}


def run_benchmark(results_path: Path) -> None:
    from emb_diversity.measures_registry import MEASURE_NAMES

    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resuming from {results_path}")
    else:
        results = {}

    tmp_path = results_path.with_suffix(".cell")
    total = len(MEASURE_NAMES) * len(SIZES)
    cell_no = 0
    for measure_name in MEASURE_NAMES:
        cells = results.setdefault(measure_name, {})
        for size in sorted(SIZES):
            cell_no += 1
            if str(size) in cells:
                continue  # done in a previous run
            if any(c["status"] == "timeout"
                   for s, c in cells.items() if int(s) < size):
                # timed out on fewer points — more would only be slower
                cells[str(size)] = {"status": "skipped", "times": []}
            else:
                cells[str(size)] = run_measure(measure_name, size, tmp_path)
            cell = cells[str(size)]
            runs = (f", mean {sum(cell['times']) / len(cell['times']):.4f}s "
                    f"over {len(cell['times'])} runs" if cell["times"] else "")
            print(f"[{cell_no}/{total}] {measure_name} | size={size} | "
                  f"{cell['status']}{runs}", flush=True)
            save_json(results_path, results)  # checkpoint after every pair
    print(f"Done. Results in {results_path}")


def plot_results(results_path: Path, plot_path: Path,
                 bands: bool = True) -> None:
    """Plot mean runtimes; *bands* adds a +/-1 std ribbon per measure."""
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.loads(results_path.read_text())

    plt.rcParams["font.size"] = 9
    # constrained layout reserves the legend's space inside the 3.03 in
    # (ACL \columnwidth) figure, shrinking the axes to fit next to it
    fig, ax = plt.subplots(figsize=(3.03, 3.0), layout="constrained")
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    smallest_mean = None
    for idx, (measure_name, cells) in enumerate(results.items()):
        done = sorted((int(s), c["times"]) for s, c in cells.items()
                      if c["times"])
        if not done:
            continue
        sizes = [s for s, _ in done]
        means = np.array([np.mean(t) for _, t in done])
        smallest_mean = min(smallest_mean or means.min(), means.min())
        ax.plot(sizes, means, color=colors[idx], label=measure_name,
                marker=markers[idx % len(markers)], markersize=3, linewidth=1)
        if bands:
            stds = np.array([np.std(t) for _, t in done])
            ax.fill_between(sizes, means - stds, means + stds,
                            color=colors[idx], alpha=0.2, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if smallest_mean is not None:
        # keep every mean visible, but don't let band edges (which can
        # approach zero) stretch the axis into meaningless sub-ms depths
        ax.set_ylim(bottom=0.6 * smallest_mean)
    # one tick per benchmarked size (the narrow axes otherwise drop labels)
    ax.set_xticks(sorted({int(s) for c in results.values() for s in c}))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (seconds)")
    fig.legend(loc="outside center right", fontsize=6, frameon=False,
               handlelength=1.2, labelspacing=0.3)
    ax.grid(True, ls="--", alpha=0.4)
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the benchmark (resumable)")
    p_run.add_argument("--results", type=Path, default=RESULTS_FILE)
    p_plot = sub.add_parser("plot", help="plot results from the JSON file")
    p_plot.add_argument("--results", type=Path, default=RESULTS_FILE)
    p_plot.add_argument("--out", type=Path, default=PLOT_FILE)
    p_plot.add_argument("--no-bands", action="store_true",
                        help="plot mean lines only, without +/-1 std bands")

    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(args.results)
    else:
        plot_results(args.results, args.out, bands=not args.no_bands)


if __name__ == "__main__":
    main()

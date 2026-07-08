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

Run benchmarks **sequentially, one per working directory**: the measures
share the on-disk distance cache in the working directory (.cache/pdist),
and every timed run clears it. Concurrent runs started from the same folder
delete each other's cache mid-computation — long pdist runs then crash while
writing their cache — and concurrent writes to the same results file
overwrite each other. To parallelize (e.g. one SLURM job per measure), give
every job its own working directory and results file.
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


def _worker(measure_name: str, size: int, n_runs: int, out_path: str) -> None:
    """Child process: time *n_runs* cold-cache runs of one measure.

    Timings are flushed to *out_path* after every run, so the parent keeps
    the finished runs even if it kills this process at the budget.
    """
    import numpy as np

    from emb_diversity.measures_registry import get_measure
    from emb_diversity.measures import utils as pw

    pw._MEMORY_MAX = 0  # don't let large distance matrices pile up in RAM
    fn = get_measure(measure_name)

    # Warm-up on a tiny dataset (untimed): pays one-time in-process costs
    # such as numba JIT compilation in the UMAP-based measures, so the
    # timed runs below measure the computation, not the compiler.
    try:
        fn(np.random.RandomState(0).randn(32, DIM))
    except Exception:
        pass  # the timed runs will surface any real error

    data = np.random.RandomState(SEED).randn(size, DIM)

    times = []
    for _ in range(n_runs):
        pw.clear_distance_cache()  # memory + disk, so every run is cold
        start = time.perf_counter()
        try:
            fn(data)
        except Exception as e:  # report what failed, then die
            Path(out_path).write_text(
                json.dumps({"times": times, "error": f"{type(e).__name__}: {e}"}))
            raise
        times.append(time.perf_counter() - start)
        Path(out_path).write_text(json.dumps({"times": times, "error": None}))


def run_measure(measure_name: str, size: int, tmp_path: Path,
                n_runs: int = N_RUNS) -> dict:
    """Run one (measure, size) pair in a child process under BUDGET_S."""
    if n_runs <= 0:
        raise ValueError(f"n_runs must be >= 1 (got {n_runs})")
    # Ensure we don't accidentally read stale timings after a parent/child crash.
    tmp_path.unlink(missing_ok=True)
    # spawn (not fork) = clean interpreter per pair on every platform
    proc = mp.get_context("spawn").Process(
        target=_worker, args=(measure_name, size, n_runs, str(tmp_path))
    )
    proc.start()
    proc.join(timeout=BUDGET_S)
    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join()
    try:  # the child may have been killed mid-write
        report = json.loads(tmp_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = {"times": [], "error": None}
    tmp_path.unlink(missing_ok=True)

    if report["error"] is not None:  # the measure raised in the child
        return {"status": "error", "error": report["error"],
                "times": report["times"]}
    if report["times"]:  # at least one run finished — a usable timing
        return {"status": "ok", "times": report["times"]}
    if timed_out:
        return {"status": "timeout", "times": []}
    # the child died without reporting (e.g. killed by the OS on OOM)
    return {"status": "error", "times": [],
            "error": f"child process died with exit code {proc.exitcode}"}


def run_benchmark(results_path: Path, measure: str | None = None) -> None:
    from emb_diversity.measures_registry import MEASURE_NAMES

    if measure is None:
        names = MEASURE_NAMES
    elif measure in MEASURE_NAMES:
        names = (measure,)
    else:
        raise SystemExit(f"Unknown measure {measure!r}. "
                         f"Registered: {', '.join(sorted(MEASURE_NAMES))}")

    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resuming from {results_path}")
    else:
        results = {}

    tmp_path = results_path.with_suffix(".cell")
    total = len(names) * len(SIZES)
    cell_no = 0
    for measure_name in names:
        cells = results.setdefault(measure_name, {})
        for size in sorted(SIZES):
            cell_no += 1
            cell = cells.get(str(size))
            if cell is not None:
                # An "ok" cell can hold fewer than N_RUNS timings if the
                # budget cut it short — run the missing ones and append.
                missing = (N_RUNS - len(cell["times"])
                           if cell["status"] == "ok" else 0)
                if missing <= 0:
                    continue  # done in a previous run
                top_up = run_measure(measure_name, size, tmp_path, missing)
                cell["times"] += top_up["times"]
            elif any(c["status"] == "timeout"
                     for s, c in cells.items() if int(s) < size):
                # timed out on fewer points — more would only be slower
                cells[str(size)] = {"status": "skipped", "times": []}
            else:
                cells[str(size)] = run_measure(measure_name, size, tmp_path)
            cell = cells[str(size)]
            note = (f", mean {sum(cell['times']) / len(cell['times']):.4f}s "
                    f"over {len(cell['times'])} runs" if cell["times"] else "")
            note += f" ({cell['error']})" if cell.get("error") else ""
            print(f"[{cell_no}/{total}] {measure_name} | size={size} | "
                  f"{cell['status']}{note}", flush=True)
            save_json(results_path, results)  # checkpoint after every pair
    print(f"Done. Results in {results_path}")


def plot_results(results_path: Path, plot_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.loads(results_path.read_text())

    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=(3.03, 3.4))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, (measure_name, cells) in enumerate(results.items()):
        done = sorted((int(s), c["times"]) for s, c in cells.items()
                      if c["times"])
        if not done:
            continue
        sizes = [s for s, _ in done]
        means = np.array([np.mean(t) for _, t in done])
        stds = np.array([np.std(t) for _, t in done])
        ax.plot(sizes, means, color=colors[idx], label=measure_name,
                marker=markers[idx % len(markers)], markersize=3, linewidth=1)
        ax.fill_between(sizes, means - stds, means + stds,
                        color=colors[idx], alpha=0.2, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (seconds)")
    ax.legend(ncol=2, fontsize=5.5, framealpha=0.9,
              columnspacing=0.8, handlelength=1.4, labelspacing=0.25)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the benchmark (resumable)")
    p_run.add_argument("--results", type=Path, default=RESULTS_FILE)
    p_run.add_argument("--measure", default=None,
                       help="run only this measure (default: all)")
    p_plot = sub.add_parser("plot", help="plot results from the JSON file")
    p_plot.add_argument("--results", type=Path, default=RESULTS_FILE)
    p_plot.add_argument("--out", type=Path, default=PLOT_FILE)

    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(args.results, args.measure)
    else:
        plot_results(args.results, args.out)


if __name__ == "__main__":
    main()

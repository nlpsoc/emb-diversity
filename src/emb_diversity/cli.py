"""Command-line interface for emb-diversity."""

from __future__ import annotations

import ast
import csv
import functools
import inspect
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="emb-diversity",
    help="Measure embedding-based diversity of text data.\n\nRun directly: emb-diversity texts.txt",
    no_args_is_help=True,
)


@app.command(hidden=True)
def default(
    input_file: Path = typer.Argument(
        ..., help="Path to input file (.txt, .csv, or .tsv)"
    ),
    measures: Optional[list[str]] = typer.Option(
        None,
        "--measure",
        "-m",
        help=(
            "Measure to run. Use -m name for one, repeat for several: "
            "-m mean_pw_dist -m diameter. "
            "Use -m variety|balance|disparity for a named set, -m all for all. "
            "Default: graph_entropy, vendi_score, mean_pw_dist."
        ),
    ),
    axis: str = typer.Option(
        "semantic", "--axis", "-a", help="Diversity axis (e.g. semantic, style, speaker)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Explicit embedding model; overrides --axis."
    ),
    column: str = typer.Option(
        "text", "--column", "-c", help="Column name containing text (for CSV/TSV)."
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv."
    ),
    chunking: bool = typer.Option(
        False, "--chunking",
        help="Chunk long texts instead of truncating them to the model's max length.",
    ),
    chunks: int = typer.Option(
        10, "--chunks", help="Max windows per text when --chunking is set."
    ),
    pooling: str = typer.Option(
        "mean", "--pooling",
        help="How to combine chunk vectors when --chunking is set: mean or max.",
    ),
    params: Optional[list[str]] = typer.Option(
        None,
        "--param",
        "-p",
        help=(
            "Override a measure parameter as key=value, repeatable: "
            "-p k=5 -p metric=euclidean. Requires exactly one -m measure name."
        ),
    ),
) -> None:
    """Measure diversity from a text file or CSV/TSV."""
    _run_measure(input_file, measures, axis, model, column, output_format,
                 chunking, chunks, pooling, params)


@app.command("measure")
def measure_cmd(
    input_file: Path = typer.Argument(
        ..., help="Path to input file (.txt, .csv, or .tsv)"
    ),
    measures: Optional[list[str]] = typer.Option(
        None,
        "--measure",
        "-m",
        help=(
            "Measure to run. Use -m name for one, repeat for several: "
            "-m mean_pw_dist -m diameter. "
            "Use -m variety|balance|disparity for a named set, -m all for all. "
            "Default: graph_entropy, vendi_score, mean_pw_dist."
        ),
    ),
    axis: str = typer.Option(
        "semantic", "--axis", "-a", help="Diversity axis (e.g. semantic, style, speaker)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Explicit embedding model; overrides --axis."
    ),
    column: str = typer.Option(
        "text", "--column", "-c", help="Column name containing text (for CSV/TSV)."
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv."
    ),
    chunking: bool = typer.Option(
        False, "--chunking",
        help="Chunk long texts instead of truncating them to the model's max length.",
    ),
    chunks: int = typer.Option(
        10, "--chunks", help="Max windows per text when --chunking is set."
    ),
    pooling: str = typer.Option(
        "mean", "--pooling",
        help="How to combine chunk vectors when --chunking is set: mean or max.",
    ),
    params: Optional[list[str]] = typer.Option(
        None,
        "--param",
        "-p",
        help=(
            "Override a measure parameter as key=value, repeatable: "
            "-p k=5 -p metric=euclidean. Requires exactly one -m measure name."
        ),
    ),
) -> None:
    """Measure diversity from a text file or CSV/TSV."""
    _run_measure(input_file, measures, axis, model, column, output_format,
                 chunking, chunks, pooling, params)


def _run_measure(input_file, measures, axis, model, column, output_format,
                 chunking=False, chunks=10, pooling="mean", params=None):
    """Shared logic for the measure command."""
    from .convenience import measure_diversity

    # ── Read texts ───────────────────────────────────────────────
    texts = _read_texts(input_file, column)
    if len(texts) < 2:
        typer.echo("Error: need at least 2 texts to measure diversity.", err=True)
        raise typer.Exit(code=1)

    # ── Convert Typer's list format to measure_diversity()'s format ──
    # Typer always gives a list (e.g. ["all"]), but measure_diversity()
    # expects a plain string for "all" and named-set shortcuts.
    from .measures_registry import MEASURE_SETS

    if measures is None:
        measure_arg = None
    elif len(measures) == 1 and measures[0] in ("all", *MEASURE_SETS):
        measure_arg = measures[0]
    else:
        measure_arg = measures

    # ── Bind --param overrides ───────────────────────────────────
    # Parameter overrides target a single measure function, so they are only
    # allowed with exactly one measure name (not a set, "all", or the default).
    if params:
        if measures is None or len(measures) != 1 or measures[0] in ("all", *MEASURE_SETS):
            typer.echo(
                "Error: --param requires exactly one measure name, "
                "e.g. -m knn -p k=5.",
                err=True,
            )
            raise typer.Exit(code=1)
        measure_arg = _configure_measure(measures[0], params)

    # Bundle the long-text flags into the dict the Python API expects; only set
    # it when chunking is enabled so the default path stays truncation.
    chunking_kwargs = (
        {"chunking": True, "chunks": chunks, "pooling": pooling} if chunking else None
    )

    # ── Compute ──────────────────────────────────────────────────
    typer.echo(f"Measuring diversity of {len(texts)} texts...", err=True)
    try:
        results = measure_diversity(
            texts, measure=measure_arg, diversity_axis=axis, embedding_model=model,
            chunking_kwargs=chunking_kwargs,
        )
    except KeyError as exc:
        # measure_diversity() raises KeyError for unknown measure names
        typer.echo(f"Error: {exc}. Run 'emb-diversity list-measures' to see available measures.", err=True)
        raise typer.Exit(code=1)

    # ── Output ───────────────────────────────────────────────────
    _print_results(results, output_format)


@app.command("list-measures")
def list_measures_cmd() -> None:
    """List all available diversity measures."""
    from .measures_registry import DEFAULT_MEASURE, MEASURE_NAMES, MEASURE_SETS

    for name in sorted(MEASURE_NAMES):
        tags = []
        if name in DEFAULT_MEASURE:
            tags.append("default")
        for set_name, members in MEASURE_SETS.items():
            if name in members:
                tags.append(set_name)
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        typer.echo(f"  {name}{suffix}")


@app.command("list-axes")
def list_axes_cmd() -> None:
    """List registered diversity axes and their models."""
    from .axes_registry import axes

    for ax in axes.list_all():
        typer.echo(f"  {ax.name}")
        typer.echo(f"    default model: {ax.default_model}")
        if ax.alternative_models:
            typer.echo(f"    alternatives:  {', '.join(ax.alternative_models)}")
        if ax.description:
            typer.echo(f"    {ax.description}")
        typer.echo()


# ── Helpers ──────────────────────────────────────────────────────

# Arguments every measure shares; they are set through dedicated CLI options
# (--axis, --model, --chunking, …), so --param must not override them.
_RESERVED_PARAMS = frozenset(
    {"data", "diversity_axis", "embedding_model", "chunking_kwargs"}
)


def _parse_param_value(raw: str):
    """Convert a --param value string to a Python value.

    ``true``/``false`` (any case) become booleans and ``none`` becomes None;
    anything else is tried as a Python literal (int, float, …) and kept as a
    plain string when that fails — so ``metric=euclidean`` needs no quoting.
    """
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_params(params: list[str]) -> dict:
    """Parse repeated ``key=value`` strings into a kwargs dict."""
    parsed = {}
    for item in params:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            typer.echo(
                f"Error: invalid --param {item!r}; expected key=value.", err=True
            )
            raise typer.Exit(code=1)
        parsed[key] = _parse_param_value(raw)
    return parsed


def _configure_measure(name: str, raw_params: list[str]):
    """Resolve measure *name* and bind the --param overrides to it.

    Returns a callable that runs the measure with the overrides applied; it
    keeps the measure's ``__name__`` so results stay keyed by the measure
    name. Unknown measure names and parameters exit with an error before any
    embedding work happens.
    """
    from .measures_registry import MEASURE_NAMES, get_measure

    if name not in MEASURE_NAMES:
        typer.echo(
            f"Error: unknown measure {name!r}. "
            "Run 'emb-diversity list-measures' to see available measures.",
            err=True,
        )
        raise typer.Exit(code=1)

    fn = get_measure(name)
    params = _parse_params(raw_params)
    _validate_params(name, fn, params)

    configured = functools.partial(fn, **params)
    configured.__name__ = name
    return configured


def _validate_params(name: str, fn, params: dict) -> None:
    """Exit with an error for --param keys the measure does not accept."""
    sig = inspect.signature(fn)
    # A measure with **kwargs (e.g. extra distance-metric options) accepts
    # keys beyond its named parameters, so skip the unknown-key check there.
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted = [
        p.name
        for p in sig.parameters.values()
        if p.name not in _RESERVED_PARAMS and p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    for key in params:
        if key in _RESERVED_PARAMS:
            typer.echo(
                f"Error: {key!r} is not settable via --param — use the "
                "dedicated options instead (--axis, --model, --chunking, ...).",
                err=True,
            )
            raise typer.Exit(code=1)
        if key not in sig.parameters and not has_var_kw:
            available = ", ".join(accepted) if accepted else "none"
            typer.echo(
                f"Error: measure {name!r} has no parameter {key!r}. "
                f"Available: {available}.",
                err=True,
            )
            raise typer.Exit(code=1)


def _read_texts(path: Path, column: str) -> list[str]:
    """Read texts from a file, returning a list of non-empty strings.

    Supported formats:
        - .txt: one text per line, empty lines are skipped
        - .csv: comma-separated, reads the column specified by `column`
        - .tsv: tab-separated, reads the column specified by `column`

    For CSV/TSV, rows with empty values in the text column are dropped.

    Args:
        path: Path to the input file.
        column: Column name to read from CSV/TSV files (default "text").

    Returns:
        List of text strings, stripped of leading/trailing whitespace.
    """
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".txt":
        # Read each line as one text, skip empty lines
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif suffix in (".csv", ".tsv"):
        separator = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=separator)
        if column not in df.columns:
            typer.echo(
                f"Error: column {column!r} not found. "
                f"Available: {list(df.columns)}",
                err=True,
            )
            raise typer.Exit(code=1)
        return df[column].astype(str).tolist()
    else:
        typer.echo(f"Error: unsupported file extension {suffix!r}. Use .txt, .csv, or .tsv.", err=True)
        raise typer.Exit(code=1)


def _print_results(results: dict[str, dict], fmt: str) -> None:
    """Format and print results.

    Each value in *results* is a ``{"value": float, "parameters": {...},
    "version": str}`` dict. The table and csv formats show the scalar score;
    json emits the full nested structure (including parameters and version).
    """
    if fmt == "json":
        # default=str so a callable metric (or other non-serializable param)
        # can't break serialization.
        typer.echo(json.dumps(results, indent=2, default=str))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["measure", "score"])
        for name, result in results.items():
            writer.writerow([name, result["value"]])
    else:  # table
        max_name = max(len(n) for n in results) if results else 0
        for name, result in results.items():
            typer.echo(f"  {name:<{max_name}}  {result['value']:.6f}")


if __name__ == "__main__":
    app()

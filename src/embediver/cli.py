"""Command-line interface for embediver."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="embediver",
    help="Measure embedding-based diversity of text data.\n\nRun directly: embediver texts.txt",
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
            "Use -m core for a curated set, -m all for all 20. "
            "Default: log_determinant."
        ),
    ),
    axis: str = typer.Option(
        "semantic", "--axis", "-a", help="Diversity axis (e.g. semantic, style)."
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
) -> None:
    """Measure diversity from a text file or CSV/TSV."""
    _run_measure(input_file, measures, axis, model, column, output_format)


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
            "Use -m core for a curated set, -m all for all 20. "
            "Default: log_determinant."
        ),
    ),
    axis: str = typer.Option(
        "semantic", "--axis", "-a", help="Diversity axis (e.g. semantic, style)."
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
) -> None:
    """Measure diversity from a text file or CSV/TSV."""
    _run_measure(input_file, measures, axis, model, column, output_format)


def _run_measure(input_file, measures, axis, model, column, output_format):
    """Shared logic for the measure command."""
    from .convenience import measure_diversity

    # ── Read texts ───────────────────────────────────────────────
    texts = _read_texts(input_file, column)
    if len(texts) < 2:
        typer.echo("Error: need at least 2 texts to measure diversity.", err=True)
        raise typer.Exit(code=1)

    # ── Convert Typer's list format to measure_diversity()'s format ──
    # Typer always gives a list (e.g. ["all"]), but measure_diversity()
    # expects a plain string for "all" and "core" shortcuts.
    if measures is None:
        measure_arg = None
    elif len(measures) == 1 and measures[0] in ("all", "core"):
        measure_arg = measures[0]
    else:
        measure_arg = measures

    # ── Compute ──────────────────────────────────────────────────
    typer.echo(f"Embedding {len(texts)} texts...", err=True)
    try:
        results = measure_diversity(
            texts, measure=measure_arg, diversity_axis=axis, embedding_model=model,
        )
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # ── Output ───────────────────────────────────────────────────
    _print_results(results, output_format)


@app.command("list-measures")
def list_measures_cmd() -> None:
    """List all available diversity measures."""
    from .measures_registry import CORE_MEASURES, DEFAULT_MEASURE, MEASURES

    for name in sorted(MEASURES):
        tags = []
        if name == DEFAULT_MEASURE:
            tags.append("default")
        if name in CORE_MEASURES:
            tags.append("core")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        typer.echo(f"  {name}{suffix}")


@app.command("list-axes")
def list_axes_cmd() -> None:
    """List registered diversity axes and their models."""
    from .axes_registry import list_axes

    for ax in list_axes():
        typer.echo(f"  {ax.name}")
        typer.echo(f"    default model: {ax.default_model}")
        if ax.alternative_models:
            typer.echo(f"    alternatives:  {', '.join(ax.alternative_models)}")
        if ax.description:
            typer.echo(f"    {ax.description}")
        typer.echo()


# ── Helpers ──────────────────────────────────────────────────────


def _read_texts(path: Path, column: str) -> list[str]:
    """Read texts from .txt, .csv, or .tsv file."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        texts = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if column not in (reader.fieldnames or []):
                typer.echo(
                    f"Error: column {column!r} not found. "
                    f"Available: {reader.fieldnames}",
                    err=True,
                )
                raise typer.Exit(code=1)
            for row in reader:
                val = row[column]
                if val and val.strip():
                    texts.append(val.strip())
        return texts
    else:
        typer.echo(f"Error: unsupported file extension {suffix!r}. Use .txt, .csv, or .tsv.", err=True)
        raise typer.Exit(code=1)


def _print_results(results: dict[str, float], fmt: str) -> None:
    """Format and print results."""
    if fmt == "json":
        typer.echo(json.dumps(results, indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["measure", "score"])
        for name, score in results.items():
            writer.writerow([name, score])
    else:  # table
        max_name = max(len(n) for n in results) if results else 0
        for name, score in results.items():
            typer.echo(f"  {name:<{max_name}}  {score:.6f}")


if __name__ == "__main__":
    app()

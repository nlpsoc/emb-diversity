# CLI Reference

embediver provides a command-line interface for measuring diversity from text files.

## Installation

The CLI is installed automatically with the package:

```bash
pip install embediver
```

## Commands

### `embediver measure`

Measure diversity from a text file, CSV, or TSV.

```bash
embediver measure INPUT_FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|---|---|
| `INPUT_FILE` | Path to `.txt`, `.csv`, or `.tsv` file |

**Options:**

| Option | Default | Description |
|---|---|---|
| `-m`, `--measure` | `log_determinant` | Measure(s) to run. Repeat for multiple. Special: `core`, `all` |
| `-a`, `--axis` | `semantic` | Diversity axis |
| `--model` | *(none)* | Explicit embedding model (overrides `--axis`) |
| `-c`, `--column` | `text` | Column name for CSV/TSV files |
| `-f`, `--format` | `table` | Output format: `table`, `json`, `csv` |

**Examples:**

```bash
# Default measure on a text file (one text per line)
embediver measure texts.txt

# Core set of representative measures
embediver measure texts.txt -m core

# All 20 measures
embediver measure texts.txt -m all

# Specific measures
embediver measure texts.txt -m mean_pw_dist -m diameter -m vendi_score

# Style diversity
embediver measure texts.txt --axis style

# Custom embedding model
embediver measure texts.txt --model Qwen/Qwen3-8B

# CSV with custom column
embediver measure reviews.csv --column review_text

# TSV file
embediver measure data.tsv --column sentence

# JSON output (for piping)
embediver measure texts.txt -m core --format json
```

**Input formats:**

- `.txt`: one text per line
- `.csv`: comma-separated, reads the `text` column by default
- `.tsv`: tab-separated, reads the `text` column by default

### `embediver list-measures`

List all available diversity measures.

```bash
embediver list-measures
```

Measures tagged `[default]` run when no `-m` flag is given.
Measures tagged `[core]` are included in the `-m core` set.

### `embediver list-axes`

List registered diversity axes and their models.

```bash
embediver list-axes
```

# CLI Reference

emb-diversity provides a command-line interface for measuring diversity from text files.

## Installation

The CLI is installed automatically with the package:

```bash
pip install emb-diversity
```

## Commands

### `emb-diversity measure`

Measure diversity from a text file, CSV, or TSV.

```bash
emb-diversity measure INPUT_FILE [OPTIONS]
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
emb-diversity measure texts.txt

# Core set of representative measures
emb-diversity measure texts.txt -m core

# All 20 measures
emb-diversity measure texts.txt -m all

# Specific measures
emb-diversity measure texts.txt -m mean_pw_dist -m diameter -m vendi_score

# Style diversity
emb-diversity measure texts.txt --axis style

# Custom embedding model
emb-diversity measure texts.txt --model Qwen/Qwen3-8B

# CSV with custom column
emb-diversity measure reviews.csv --column review_text

# TSV file
emb-diversity measure data.tsv --column sentence

# JSON output (for piping)
emb-diversity measure texts.txt -m core --format json
```

**Input formats:**

- `.txt`: one text per line
- `.csv`: comma-separated, reads the `text` column by default
- `.tsv`: tab-separated, reads the `text` column by default

### `emb-diversity list-measures`

List all available diversity measures.

```bash
emb-diversity list-measures
```

Measures tagged `[default]` run when no `-m` flag is given.
Measures tagged `[core]` are included in the `-m core` set.

### `emb-diversity list-axes`

List registered diversity axes and their models.

```bash
emb-diversity list-axes
```

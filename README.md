# embediver

<!-- docs-intro-start -->
A Python package for measuring data diversity on small- to medium-sized text datasets. All measures are calculating diversity based on embeddings, i.e., vector representations of your data. Depending on what embedding models you want to use, you are able to calculate semantic, stylistic and other types of diversity with our package.

This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.
<!-- docs-intro-end -->

## Install

<!-- docs-install-start -->
```bash
pip install embediver
```
<!-- docs-install-end -->

## Quickstart

<!-- docs-quickstart-start -->
```python
from embediver import log_determinant

texts = [
    "The cat sat on the mat.",
    "Dogs love to play fetch.",
    "It was a sunny afternoon.",
]

# Measure diversity (embeds text automatically using semantic embeddings)
log_determinant(texts)

# Use a different diversity axis
log_determinant(texts, diversity_axis="style")

# Use a specific embedding model
log_determinant(texts, embedding_model="Qwen/Qwen3-8B")
```
<!-- docs-quickstart-end -->

### CLI

```bash
# Default measure (log_determinant) on a text file
embediver measure texts.txt

# Core set of measures
embediver measure texts.txt -m core

# All 20 measures
embediver measure texts.txt -m all

# CSV/TSV input (default column: "text")
embediver measure data.csv --column review_text

# JSON output
embediver measure texts.txt -m core --format json

# List available measures and axes
embediver list-measures
embediver list-axes
```

**Documentation:** <https://nlpsoc.github.io/Diversity-Measurement/>

## Contributing

### Setup

Clone the repo, then install development dependencies:

```bash
uv sync --group dev
source .venv/bin/activate
```

### Workflow

1. Create a branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run tests: `pytest`
4. Commit and push
5. Open a pull request

### Working with uv

```bash
uv add <package-name>              # add a dependency
uv add --group dev <package-name>  # add a dev dependency
uv sync --no-group dev             # switch back to standard mode
uv lock -U                         # update lock file after version changes
```

### Docstring Style Guide

This project uses **Google-style docstrings** parsed by Sphinx Napoleon.

```python
def calculate_diversity(vectors: np.ndarray, method: str = "vendi") -> float:
    """Calculate diversity score for a set of vectors.

    References:
        Cox et al. "Directed Diversity." CHI 2021.

    Args:
        vectors: Array of shape (n_samples, n_features).
        method: Calculation method. Defaults to "vendi".

    Returns:
        Diversity score as a float.

    Raises:
        ValueError: If vectors array is empty.

    Example:
        >>> vectors = np.array([[1, 0], [0, 1], [1, 1]])
        >>> score = calculate_diversity(vectors)
    """
```

**Key points:** imperative mood summary, blank line before details, document Args/Returns/Raises/References.

### Adding New Measures

1. Create a new file in `src/embediver/measures/` with the function decorated with `@accepts_text`.
2. Export it from `src/embediver/__init__.py`.
3. Add it to the `MEASURES` dict in `src/embediver/_registry.py`.
4. Update `docs/source/user-guide/measures.md` with a new row in the appropriate table.

### Adding New Diversity Axes

Register a new axis in `src/embediver/_axes.py`:

```python
register_axis(
    "multilingual",
    default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    description="Cross-lingual semantic diversity",
)
```

Update `docs/source/user-guide/axes.md` with the new axis.

## Funding

This work is supported by the ERC Starting Grant **DataDivers** (101162980).

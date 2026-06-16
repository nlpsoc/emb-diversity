# Documentation

This folder contains the Sphinx documentation for the Diversity Measurement package.

## Writing Docstrings

This project uses **Google-style docstrings** which are automatically parsed by Sphinx Napoleon extension.

### Docstring Style Guide

#### Functions and Methods

The package is function-based (there are no public classes). A diversity measure
leads with bold **Interpretation of values** and **Range** lines so a reader can
tell how to read the score at a glance, then follows the usual
`Args`/`Returns`/`Raises` structure — for example the `diameter` measure:

```python
def diameter(
    data: Sequence[Sequence[float]],
    metric: DistanceMetric = "cosine",
    *,
    diversity_axis: str = "semantic",
    embedding_model: str | None = None,
    **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; the upper bound depends on ``metric`` (e.g. [0, 2] for cosine distance).

    Compute the maximum pairwise distance in a set of vectors (the diameter).

    Args:
        data: Array-like of (embedding) vectors with shape (n, d), or raw text
            strings. Must contain at least 2 samples.
        metric: Distance metric name or callable accepted by
            ``scipy.spatial.distance.pdist``. Defaults to ``"cosine"``.
        diversity_axis: Registered axis used to embed text input (default
            ``"semantic"``).
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs: Extra keyword arguments forwarded to ``pdist``.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        maximum distance across all unique pairs and ``parameters`` records the
        configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.

    Example:
        >>> from emb_diversity import diameter
        >>> diameter(["The cat sat.", "Dogs play fetch.", "A bird sings at dawn."])
        {'value': 0.94, 'parameters': {'metric': 'cosine', 'embedding_model': 'all-mpnet-base-v2'}}
    """
```

#### Modules

At the top of each module file:

```python
"""Utility functions for project path management.

This module provides helper functions to locate the project root directory
and resolve paths relative to the project structure.
"""
```

### Key Points

- **One-line summary**: Start with a brief summary (imperative mood: "Calculate", not "Calculates")
- **Blank line**: After the summary, add a blank line before detailed description
- **Args**: Document each parameter with type information
- **Returns**: Describe what the function returns
- **Raises**: Document exceptions that might be raised
- **Example**: Include usage examples when helpful
- **Type hints**: Use type hints in function signatures AND document them in docstrings

### Section Headers

Use these section headers in docstrings:
- `Args:` - Function/method parameters
- `Returns:` - Return value description
- `Raises:` - Exceptions that may be raised
- `Yields:` - For generators
- `Attributes:` - For class attributes
- `Example:` or `Examples:` - Usage examples
- `Note:` - Important notes
- `Warning:` - Warnings about usage

### References

- [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Sphinx Napoleon Documentation](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

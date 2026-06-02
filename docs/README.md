# Documentation

This folder contains the Sphinx documentation for the Diversity Measurement package.

## Building the Documentation

### Prerequisites

Make sure you have installed the development dependencies:

```bash
uv sync --group dev
```

### Building HTML Documentation

To build the HTML documentation, follow these two steps:

```bash
cd docs

# Step 1: Generate API documentation RST files from source code
make apidoc

# Step 2: Build HTML from the RST files
make html
```

The generated HTML files will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view the documentation.

### Available Make Commands

- `make apidoc` - Generate API documentation RST files from source code (discovers all modules including utility, eval, etc.)
- `make html` - Build HTML documentation from RST files
- `make clean` - Remove built documentation files
- `make cleanall` - Remove both built documentation AND generated API RST files

## Workflow: Documentation Generation from Code

This project uses `sphinx-apidoc` to generate documentation from source code. You should NOT manually edit `modules.rst` or any generated `emb_diversity*.rst` files - they are auto-generated.

### How It Works

1. **Run `make apidoc`** to scan your source code in `../src/emb_diversity/`
2. `sphinx-apidoc` discovers all Python modules and submodules (measure, two_d, utility, eval, embeddings, etc.)
3. It generates RST files with a **flat structure** using the `--no-headings` flag
   - All modules appear as `emb_diversity.module_name`
   - No hierarchical "Submodules" or "Subpackages" sections
4. **Run `make html`** to build the HTML documentation
5. Sphinx reads the RST files and extracts docstrings from your Python code

### When to Run Each Command

- **Run `make apidoc`** when:
  - You add new Python modules or packages
  - You restructure your code (move/rename modules)
  - Starting fresh or the RST files are missing

- **Run `make html`** when:
  - You update docstrings in existing code
  - After running `make apidoc`
  - You modify `source/index.rst` or other manual RST files

### Why This Approach?

- **No manual maintenance** - New modules are automatically discovered
- **Always accurate** - Documentation reflects actual code structure
- **Clean structure** - Flat module listing without confusing hierarchies
- **Explicit control** - Separate `apidoc` and `html` steps
- **DRY principle** - Documentation lives in code as docstrings

## Documentation Structure

- `source/conf.py` - Sphinx configuration file
- `source/index.rst` - Main documentation page (manually maintained)
- `source/modules.rst` - Auto-generated API reference (DO NOT EDIT MANUALLY)
- `source/emb_diversity*.rst` - Auto-generated module docs (DO NOT EDIT MANUALLY)
- `build/` - Generated documentation (gitignored)

## Writing Docstrings

This project uses **Google-style docstrings** which are automatically parsed by Sphinx Napoleon extension.

### Docstring Style Guide

#### Functions and Methods

```python
def calculate_diversity(vectors: np.ndarray, method: str = "vendi") -> float:
    """Calculate diversity score for a set of vectors.

    This function computes various diversity metrics for vector representations.
    The default method uses the Vendi Score which is based on matrix entropy.

    Args:
        vectors: Array of shape (n_samples, n_features) containing the vectors.
        method: Diversity calculation method. Options are "vendi", "entropy",
            or "distinctness". Defaults to "vendi".

    Returns:
        Diversity score as a float between 0 and 1, where higher values
        indicate greater diversity.

    Raises:
        ValueError: If vectors array is empty or method is not recognized.

    Example:
        >>> vectors = np.array([[1, 0], [0, 1], [1, 1]])
        >>> score = calculate_diversity(vectors)
        >>> print(f"Diversity: {score:.2f}")
        Diversity: 0.87
    """
    pass
```

#### Classes

```python
class DiversityMeasure:
    """A class for measuring diversity in vector representations.

    This class provides multiple methods for calculating diversity scores
    from vector embeddings, including entropy-based and distance-based metrics.

    Attributes:
        method: The diversity calculation method to use.
        normalize: Whether to normalize the diversity scores.

    Example:
        >>> measure = DiversityMeasure(method="vendi")
        >>> score = measure.compute(vectors)
    """

    def __init__(self, method: str = "vendi", normalize: bool = True):
        """Initialize the DiversityMeasure.

        Args:
            method: Diversity calculation method. Defaults to "vendi".
            normalize: Whether to normalize scores. Defaults to True.
        """
        pass
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

## Viewing the Documentation

After building, you can view the documentation by opening:

```
docs/build/html/index.html
```

in your web browser.

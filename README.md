# emb-diversity

<!-- docs-intro-start -->
A Python package for measuring data diversity on small- to medium-sized text datasets. All measures are calculating diversity based on embeddings, i.e., vector representations of your data. Depending on what embedding models you want to use, you are able to calculate semantic, stylistic and other types of diversity with our package.

This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.
<!-- docs-intro-end -->

📖 **Documentation:** <https://nlpsoc.github.io/Diversity-Measurement/>

## Table of Contents

- [Usage](#usage)
- [Install](#install)
- [Available Measures](#available-measures)
- [Development](#development)
  - [Suggested Workflow for Collaboration](#suggested-workflow-for-collaboration)
  - [Working with uv](#working-with-uv)
  - [Docstring Style Guide](#docstring-style-guide)
  - [Adding New Measures](#adding-new-measures)
  - [Adding New Diversity Axes](#adding-new-diversity-axes)
- [Funding](#funding)

## Usage

<!-- docs-quickstart-start -->
```python
from emb_diversity import measure_diversity

texts = [
    "The cat sat on the mat.",
    "Dogs love to play fetch.",
    "It was a sunny afternoon.",
]

# Default measure (log_determinant), semantic embeddings
measure_diversity(texts)
# Use a different diversity axis
measure_diversity(texts, diversity_axis="style")
# Use a specific embedding model
measure_diversity(texts, embedding_model="Qwen/Qwen3-8B")
# Run the core set of measures
measure_diversity(texts, measure="core")
# Run specific measures
measure_diversity(texts, measure=["mean_pw_dist", "diameter"])

# You can also call individual measures directly
from emb_diversity import log_determinant
log_determinant(texts)
log_determinant(texts, diversity_axis="style")
```
<!-- docs-quickstart-end -->

## Install

> [!NOTE]
> You must have **uv** installed before running `uv sync`.
> Full installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

<!-- docs-install-start -->
After installing `uv` on your system, you can now follow either **development mode** or **standard installation mode** depending on your use case.

### Development mode

Follow these steps to set up the project for development.
- Clone the repo
- Install all dependencies required for development mode:
   ```bash
   uv sync --group dev
   ```
- Activate the Python environment created by `uv`
   ```bash
   source .venv/bin/activate
   ```

### Standard installation

To use the library directly do the following,

- Clone the repo
- Install all dependencies required for standard mode
   ```bash
   uv sync --no-group dev
   ```
- Activate the Python environment created by `uv`
   ```bash
   source .venv/bin/activate
   ```
<!-- docs-install-end -->

## Available Measures

For an overview of all available measures, see the [documentation](https://nlpsoc.github.io/Diversity-Measurement/#available-measures).

## Development

### Suggested Workflow for Collaboration

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Make your changes** in the codebase.
3. **Run tests** to ensure everything works as expected:
   ```bash
   pytest
   ```
4. **Commit your changes** with a descriptive message:
   ```bash
   git add .
   git commit -m "Add feature X"
   ```
5. **Push your branch** to the remote repository:
   ```bash
   git push origin feature/my-feature
   ```
6. **Create a pull request** on GitHub to merge your changes into the main branch and request a review from your team members.
7. **Address any feedback** from the review process.
8. Once approved, **merge your pull request** into the main branch.
9. **Delete your branch** after merging to keep the repository clean:
   ```bash
   git branch -d feature/my-feature
   git push origin --delete feature/my-feature
   ```

### Working with uv

#### Adding Packages with `uv add`

To add packages to your project, always use `uv add` rather than `uv pip install`. This ensures that your dependencies are properly managed and recorded in your `pyproject.toml`. For example:

```bash
uv add <package-name>
```

#### Adding Packages to a Dev Group

If you need to add a package specifically to your development environment, you can add it to the `dev` group like this:

```bash
uv add --group dev <package-name>
```

#### Switching Between Dev and Standard Mode

After you are done with testing and want to go back to standard mode, run:

```bash
uv sync --no-group dev
```

This will disable all additional groups and just load your main project dependencies.

#### Best Practice: Run `uv lock -U`

Whenever you upgrade, downgrade, or change versions of packages, it's a good practice to run:

```bash
uv lock -U
```

This updates your `uv.lock` file to ensure all versions are consistent and everything is in sync.

### Docstring Style Guide

This project uses **Google-style docstrings** which are automatically parsed by the Sphinx Napoleon extension.

#### Functions and Methods

```python
def calculate_diversity(vectors: np.ndarray, method: str = "vendi") -> float:
    """Calculate diversity score for a set of vectors.

    This function computes various diversity metrics for vector representations.
    The default method uses the Vendi Score which is based on matrix entropy.

    References:
        Cox, Samuel Rhys, Yunlong Wang, Ashraf Abdul, Christian von der Weth, and Brian Y. Lim. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, May 6, 2021, 1–35. https://doi.org/10.1145/3411764.3445782.

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

#### Key Points

- **One-line summary**: Start with a brief summary in imperative mood ("Calculate", not "Calculates")
- **Blank line**: After the summary, add a blank line before any detailed description
- **References**: Add related papers
- **Args**: Document each parameter with type information
- **Returns**: Describe what the function returns
- **Raises**: Document exceptions that might be raised
- **Example**: Include usage examples when helpful
- **Type hints**: Use type hints in function signatures AND document them in docstrings

#### Section Headers

Use these section headers in docstrings:
- `References:` Related papers
- `Args:` — Function/method parameters
- `Returns:` — Return value description
- `Raises:` — Exceptions that may be raised
- `Yields:` — For generators
- `Attributes:` — For class attributes
- `Example:` or `Examples:` — Usage examples
- `Note:` — Important notes
- `Warning:` — Warnings about usage

Further reading: [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) · [Sphinx Napoleon docs](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

### Adding New Measures

When you add a new measure to `src/emb_diversity/measures/`:

1. Create a new file with the function decorated with `@accepts_text` and a complete docstring following the style guide above.
2. Export it from `src/emb_diversity/__init__.py` if it should be part of the public API.
3. Register it in `src/emb_diversity/measures_registry.py` with `measures.register("name", func)`.
4. **Update `docs/source/user-guide/measures.md`** — add a row for the new measure in the appropriate table.

### Adding New Diversity Axes

Register a new axis in `src/emb_diversity/axes_registry.py`:

```python
from emb_diversity.axes_registry import DiversityAxis, axes

axes.register(
    "multilingual",
    DiversityAxis(
        name="multilingual",
        default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Cross-lingual semantic diversity",
    ),
)
```

Update `docs/source/user-guide/axes.md` with the new axis.

## Funding

This work is supported by the ERC Starting Grant **DataDivers** (101162980).

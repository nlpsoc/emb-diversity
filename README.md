# emb-diversity

<!-- docs-intro-start -->
A Python package for measuring data diversity on small- to medium-sized text datasets. All measures are calculating diversity based on embeddings, i.e., vector representations of your data. Depending on what embedding models you want to use, you are able to calculate semantic, stylistic and other types of diversity with our package.

This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.
<!-- docs-intro-end -->

📖 **Documentation:** <https://nlpsoc.github.io/Diversity-Measurement/>

## Install

<!-- docs-install-start -->
Install the latest release from PyPI:

```bash
pip install emb-diversity
```

The first time you measure diversity, the default embedding model
(`all-mpnet-base-v2`, ~420 MB) is downloaded from the Hugging Face Hub and
cached locally, so later runs are fast and work offline.
<!-- docs-install-end -->

## Usage

<!-- docs-quickstart-start -->

Measuring the diversity of a dataset with our package is easy: 

```python
from emb_diversity import measure_diversity

# more style-diverse, more topic-uniform (music)
texts_a = [
    "I thoroughly enjoy the hair bands.",
    "songs of the 80's are the best.",
    "Hip Hop is going DOWNHILL!!!!!",
    "rock music just makes me feel good",
    "The 80's rocked!That generation had the best music!"
]

# Uses the default measures and semantic embeddings
print(measure_diversity(texts_a))
# -> {'graph_entropy': 6.86..., 'vendi_score': 4.12..., 'mean_pw_dist': 0.69...}
```

Note that measuring the diversity of a dataset is usually only meaningful when comparing it to another datasets. The reason is that diversity values in isolation are not easily interpretable and are not bounded, sensitive to dataset size and sensitive to the used embedding space. Let's add another corpus. 

```python
# more style-uniform (formal), more topic-diverse
texts_b = [
    "I thoroughly enjoy the hair bands.",
    "They have not caused any harm to me.",
    "He has a very distinct walk.",
    "It depends on what they will pay.",
    "I would go out with the son of a preacher.",
]

print(measure_diversity(texts_a))
# -> {'graph_entropy': 6.86..., 'vendi_score': 4.12..., 'mean_pw_dist': 0.69...}

print(measure_diversity(texts_b))
# -> {'graph_entropy': 6.91..., 'vendi_score': 4.93..., 'mean_pw_dist': 0.98...}
```

When a measure considers a dataset to be more diverse, it will assign it a higher diversity value. Here, the three default measures consistently show that `texts_b` is more diverse than `texts_a`. This can change, when we change what diversity "axis" is considered, for example, "style" instead of "semantic". 

```python
# Use a different diversity axis, for style diversity AnnaWegmann/style-embeddings is the default
print(measure_diversity(texts_a, diversity_axis="style"))
# -> {'graph_entropy': 6.69..., 'vendi_score': 4.17..., 'mean_pw_dist': 0.93...}
print(measure_diversity(texts_b, diversity_axis="style"))
# -> {'graph_entropy': 6.32..., 'vendi_score': 2.24..., 'mean_pw_dist': 0.32...}
```

You can also specify a different embedding model with a HuggingFace identifier, for example, a model trained for Dutch. Be careful to use models that were trained on the diversity axis you are interested in, otherwise you might get some inconsistent results!

```python
# Use a specific embedding model (here a small, fast SBERT model)
print(measure_diversity(texts_a, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': 6.61..., 'vendi_score': 1.89..., 'mean_pw_dist': 0.20...}
print(measure_diversity(texts_b, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': 6.80..., 'vendi_score': 1.52..., 'mean_pw_dist': 0.11...}
```

You can also use specific measures, see an overview here: https://nlpsoc.github.io/Diversity-Measurement/user-guide/measures.html. Use with caution. Some measures might be worse for your use case than others. We recommend to test whether your chosen measure and embedding space capture your diversity axis of interest.
```python
# Run specific measures
print(measure_diversity(texts_a, measure=["diameter", "log_determinant"]))
# -> {'diameter': 0.94..., 'log_determinant': -0.93...}
print(measure_diversity(texts_b, measure=["diameter", "log_determinant"]))
# -> {'diameter': 1.0..., 'log_determinant': -0.06...}
```

Note that most measures return unbounded values that cannot be compared for datasets with differing sizes. Happy diversity measuring!
<!-- docs-quickstart-end -->

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Development](#development)
  - [Development setup](#development-setup)
  - [Suggested Workflow for Collaboration](#suggested-workflow-for-collaboration)
  - [Working with uv](#working-with-uv)
  - [Docstring Style Guide](#docstring-style-guide)
  - [Adding New Measures](#adding-new-measures)
  - [Adding New Diversity Axes](#adding-new-diversity-axes)
- [Funding](#funding)
- [Citation](#citation)

## Development

### Development setup

To work on `emb-diversity` itself, install from a clone with
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/nlpsoc/Diversity-Measurement.git
cd Diversity-Measurement
uv sync --group dev          # runtime + dev tools (pytest, docs, ...)
source .venv/bin/activate
```

Use `uv sync --no-group dev` to install only the runtime dependencies.

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

## Citation

<!-- docs-citation-start -->
There is no paper yet, so if you use `emb-diversity` in your work, please cite
the software:

```bibtex
@misc{emb_diversity,
  author = {Su, Cantao and Velayuthan, Menan and Ploeger, Esther and Nguyen, Dong and Wegmann, Anna},
  title  = {emb-diversity},
  url    = {https://github.com/nlpsoc/Diversity-Measurement},
}
```
<!-- docs-citation-end -->

# emb-diversity

<!-- docs-intro-start -->
A Python package for measuring data diversity on small- to medium-sized datasets (RAM-restricted, usually up to size 10k). All measures are calculating diversity based on embeddings, i.e., vector representations of your data. Depending on what embedding models you want to use, you are able to calculate semantic, stylistic, speaker and other types of diversity with our package — text by default, with select axes (e.g. `speaker`) covering other modalities like audio.

This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.
<!-- docs-intro-end -->

📖 **Documentation:** <https://nlpsoc.github.io/Diversity-Measurement/>

## Table of Contents

- **[Install](#install)**
- **[Usage](#usage)**
  - [Python API](#python-api)
  - [CLI](#cli)
- [Development](#development)
  - [Development setup](#development-setup)
  - [Suggested Workflow for Collaboration](#suggested-workflow-for-collaboration)
  - [Working with uv](#working-with-uv)
  - [Adding New Measures](#adding-new-measures)
  - [Docstring Style Guide](#docstring-style-guide)
  - [Adding New Diversity Axes](#adding-new-diversity-axes)
  - [Building and publishing a release](#building-and-publishing-a-release)
- [Open TODOs](#open-todos)
- [Funding](#funding)
- [Citation](#citation)

## Install

<!-- docs-install-start -->
Install the latest release from PyPI:

```bash
pip install emb-diversity
```

For a slimmer CPU only install you might want to first run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The installation requirements are specified in `pyproject.toml` and are automatically resolved when installing the package. 

The first time you measure diversity, the default embedding model
(`all-mpnet-base-v2`, ~420 MB) is downloaded from the Hugging Face Hub and
cached locally, so later runs are fast and work offline.
<!-- docs-install-end -->

## Usage

### Python API

<!-- docs-quickstart-start -->

Measuring the diversity of a dataset with our package is easy: 

**Basic usage**

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
# Each result holds the score under 'value', the configuration used under
# 'parameters', and the installed emb-diversity package version under
# 'version' — a fingerprint tracing the result back to the code that
# produced it (shown as ... below):
# -> {'graph_entropy': {'value': 6.86..., ...},
#     'vendi_score':   {'value': 4.12..., ...},
#     'mean_pw_dist':  {'value': 0.69..., ...}}
```

**Comparing two datasets**

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
# -> {'graph_entropy': {'value': 6.86..., ...}, 'vendi_score': {'value': 4.12..., ...}, 'mean_pw_dist': {'value': 0.69..., ...}}

print(measure_diversity(texts_b))
# -> {'graph_entropy': {'value': 6.91..., ...}, 'vendi_score': {'value': 4.93..., ...}, 'mean_pw_dist': {'value': 0.98..., ...}}
```

**Choosing a diversity axis**

When a measure considers a dataset to be more diverse, it will assign it a higher diversity value. Here, the three default measures consistently show that `texts_b` is more diverse than `texts_a`. This can change, when we change what diversity "axis" is considered, for example, "style" instead of "semantic". 

```python
# Use a different diversity axis, for style diversity AnnaWegmann/style-embeddings is the default
print(measure_diversity(texts_a, diversity_axis="style"))
# -> {'graph_entropy': {'value': 6.69..., ...}, 'vendi_score': {'value': 4.17..., ...}, 'mean_pw_dist': {'value': 0.93..., ...}}
print(measure_diversity(texts_b, diversity_axis="style"))
# -> {'graph_entropy': {'value': 6.32..., ...}, 'vendi_score': {'value': 2.24..., ...}, 'mean_pw_dist': {'value': 0.32..., ...}}
```

**Choosing an embedding model**

You can also specify a different embedding model with a HuggingFace identifier, for example, a model trained for Dutch. Be careful to use models can represent the information you are interested in, otherwise you might get some inconsistent results! A Dutch model might behave unpredictably on English text.

```python
# Use a specific embedding model (here a small, fast SBERT model)
print(measure_diversity(texts_a, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': {'value': 6.61..., ...}, 'vendi_score': {'value': 1.89..., ...}, 'mean_pw_dist': {'value': 0.20..., ...}}
print(measure_diversity(texts_b, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': {'value': 6.80..., ...}, 'vendi_score': {'value': 1.52..., ...}, 'mean_pw_dist': {'value': 0.11..., ...}}
```

**Matching the embedding model's language**

The results above are inconsistent. We can do the same again, but with sentences translated to Dutch, to get better semantic diversity scores for Dutch specifically. 

```python
texts_a_nl = [
    "Ik geniet enorm van de hair bands.",
    "nummers uit de jaren 80 zijn de beste.",
    "Hip Hop gaat BERGAFWAARTS!!!!!",
    "rockmuziek geeft me gewoon een goed gevoel",
    "De jaren 80 waren geweldig!Die generatie had de beste muziek!"
]
texts_b_nl = [
    "Ik geniet enorm van de hair bands.",
    "Ze hebben mij geen kwaad gedaan.",
    "Hij heeft een heel kenmerkende manier van lopen.",
    "Het hangt ervan af wat ze zullen betalen.",
    "Ik zou uitgaan met de zoon van een dominee.",
]
print(measure_diversity(texts_a_nl, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': {'value': 6.85..., ...}, 'vendi_score': {'value': 2.32..., ...}, 'mean_pw_dist': {'value': 0.28..., ...}}
print(measure_diversity(texts_b_nl, embedding_model="GroNLP/bert-base-dutch-cased"))
# -> {'graph_entropy': {'value': 6.91..., ...}, 'vendi_score': {'value': 2.61..., ...}, 'mean_pw_dist': {'value': 0.34..., ...}}
```

**Running specific measures**

You can also use specific measures, see an overview here: https://nlpsoc.github.io/Diversity-Measurement/user-guide/measures.html. Use with caution. Some measures might be worse for your use case than others. We recommend to test whether your chosen measure and embedding space capture your diversity axis of interest.

```python
# Run specific measures
print(measure_diversity(texts_a, measure=["diameter", "log_determinant"]))
# -> {'diameter': {'value': 0.94..., ...}, 'log_determinant': {'value': -0.93..., ...}}
print(measure_diversity(texts_b, measure=["diameter", "log_determinant"]))
# -> {'diameter': {'value': 1.0..., ...}, 'log_determinant': {'value': -0.06..., ...}}
```

**Using raw vectors**

If you are not working with text data or you already calculated the embeddings yourself, you can use ``vectors`` (numpy arrays or lists of lists of numbers) directly as well, see https://nlpsoc.github.io/Diversity-Measurement/user-guide/vectors.html.

```python
import numpy as np
vectors = np.random.randn(100, 384)

# Default measure
print(measure_diversity(vectors))
# {'graph_entropy': {'value': 459.38..., 'parameters': {'metric': 'cosine', 'embedding_model': None}}, 'vendi_score': ..., 'mean_pw_dist': ...}
```

Note that most measures return unbounded values that cannot be compared for datasets with differing sizes. Currently, calculations are done only in RAM. This limits the dataset sizes that can be evaluated (without problem up to 10k). Happy diversity measuring!
<!-- docs-quickstart-end -->

### CLI

`emb-diversity` also installs a command-line interface for measuring diversity
directly from text, CSV, or TSV files, without writing any Python:

```bash
# Default measures on a text file (one text per line)
emb-diversity measure texts.txt

# A named measure set, a different diversity axis, JSON output for piping
emb-diversity measure texts.txt -m variety --axis style --format json

# CSV input with a custom column name
emb-diversity measure reviews.csv --column review_text
```

See the full [CLI reference](https://nlpsoc.github.io/Diversity-Measurement/user-guide/cli.html)
for all commands and options (`measure`, `list-measures`, `list-axes`).

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

### Adding New Measures

When you add a new measure to `src/emb_diversity/measures/`:

1. Create a new file named after the measure. A measure is a plain function
   (no decorator, same name as its file) with the signature
   `def name(data, <params>, *, diversity_axis="semantic", embedding_model=None, chunking_kwargs=None) -> MeasureResult`.
   Call `data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="name", chunking_kwargs=chunking_kwargs)`
   first: it embeds text input, returns the resolved model id, and is the single
   place input is validated — it rejects a bare string, non-2-D data, fewer than 2
   samples, and nan/inf values. Passing `measure="name"` prints an interactive
   "Calculating measure 'name'…" notice once embedding finishes, just before the
   calculation. `chunking_kwargs` forwards long-text options
   (`chunking`/`chunks`/`pooling`) to the embedding step — thread it through
   unchanged. Then return
   `{"value": <float>, "parameters": {<params>, "embedding_model": embedding_model}}`.
   Add a complete docstring following the style guide below. Right after the
   file's imports, add a `### <Family>-Based Diversity Measure` comment naming
   which of the four families in
   [the taxonomy](docs/source/user-guide/measures.md#taxonomy) the measure
   belongs to (e.g. `### Distance-Based Diversity Measure`).
2. Add its name to `MEASURE_NAMES` in `src/emb_diversity/measures_registry.py`.
   The public API (`emb_diversity.<name>`), the CLI, and `measure_diversity`
   all pick it up from there.
3. Add the matching import to the `TYPE_CHECKING` block in
   `src/emb_diversity/__init__.py` so IDEs and type checkers see it
   (`test/test_lazy_import.py` fails if this step is forgotten).
4. **Update `docs/source/user-guide/measures.md`** — add a row for the new measure in the appropriate table.

A distance-based measure can reuse `compute_pairwise_distances` from
`measures/utils.py` — the cached pairwise-distance helper the built-in measures use
(a condensed `scipy.pdist` array with an on-disk cache), so several measures over the
same embeddings reuse the result instead of recomputing:

```python
import numpy as np

from ..embed import resolve_embeddings
from .types import MeasureResult
from .utils import compute_pairwise_distances


def mean_cosine_dist(data, *, diversity_axis="semantic", embedding_model=None,
                     chunking_kwargs=None) -> MeasureResult:
    data, embedding_model = resolve_embeddings(
        data, diversity_axis, embedding_model, measure="mean_cosine_dist",
        chunking_kwargs=chunking_kwargs,
    )
    dists = compute_pairwise_distances(data, metric="cosine")
    return {
        "value": float(np.mean(dists)),
        "parameters": {"metric": "cosine", "embedding_model": embedding_model},
    }
```

The shared type aliases — `MeasureResult` (the `{"value", "parameters"}` return dict),
`DistanceMetric`, and `TensorLike` — live in `src/emb_diversity/measures/types.py`;
import them from `.types` in your measure module.

For complete, working examples, copy the shape of an existing measure in
`src/emb_diversity/measures/` — e.g. `mean_pw_dist.py` for a simple distance-based
measure, or `vendi_score.py` for one with several parameters.

### Docstring Style Guide

This project uses **Google-style docstrings**, parsed by the Sphinx Napoleon
extension. For example, here is the built-in `mean_pw_dist` measure:

#### Functions and Methods

```python
def mean_pw_dist(
        data: Sequence[Sequence[float]],
        metric: DistanceMetric = "cosine",
        *,
        diversity_axis: str = "semantic",
        embedding_model: str | None = None,
        **metric_kwargs: Any,
) -> MeasureResult:
    """**Interpretation of values:** larger value = more diverse.
    **Range:** >= 0; the upper bound depends on ``metric`` (e.g. [0, 2] for cosine distance).

    Compute the average of all pairwise distances between datapoints.

    1) Compute all unique pairwise distances between datapoints.
    2) Return their mean.

    References:
        Guy Tevet and Jonathan Berant. 2021. Evaluating the Evaluation of Diversity in Natural Language Generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 326–346, Online. Association for Computational Linguistics.
        Tianhui Zhang, Bei Peng, and Danushka Bollegala. 2024. Improving Diversity of Commonsense Generation by Large Language Models via In-Context Learning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9226–9242, Miami, Florida, USA. Association for Computational Linguistics.
        Miranda, Brando, Alycia Lee, Sudharsan Sundar, Allison Casasola, Rylan Schaeffer, Elyas Obbad, and Sanmi Koyejo. "Beyond scale: The diversity coefficient as a data quality metric for variability in natural language data." arXiv preprint arXiv:2306.13840 (2023).
        Cox, Samuel Rhys, et al. "Directed diversity: Leveraging language embedding distances for collective creativity in crowd ideation." Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 2021.

    Args:
        data:
            Iterable/array-like of (embedding) vectors with shape (n, d), or raw
            text strings. Must contain at least 2 samples.
        metric:
            Distance metric name or callable accepted by
            scipy.spatial.distance.pdist. Defaults to "cosine".
        diversity_axis: Registered axis used to embed text input (default "semantic").
        embedding_model: Explicit embedding model id; overrides *diversity_axis*.
        **metric_kwargs:
            Extra keyword arguments forwarded to pdist for the selected metric.

    Returns:
        A dict ``{"value": float, "parameters": {...}}`` where ``value`` is the
        average pairwise distance across all unique pairs and ``parameters``
        records the configuration used.

    Raises:
        ValueError: If input is invalid, empty, or has fewer than 2 datapoints.

    Example:
        >>> from emb_diversity import mean_pw_dist
        >>> mean_pw_dist(["The cat sat.", "Dogs play fetch.", "A bird sings at dawn."])
        {'value': 0.95..., 'parameters': {'metric': 'cosine', 'embedding_model': 'all-mpnet-base-v2'}}
    """
    data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model, measure="mean_pw_dist")
    dists = _compute_pairwise_distances(data, metric, **metric_kwargs)
    return {
        "value": float(np.mean(dists)),
        "parameters": {"metric": metric, "embedding_model": embedding_model, **metric_kwargs},
    }
```

#### Key Points

- **Interpretation & Range**: Measures lead with bold **Interpretation of values** and **Range** lines so a reader can tell how to read the score at a glance
- **One-line summary**: After those, give a brief summary in imperative mood ("Compute", not "Computes")
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
        modality="text",  # default — omit unless the axis embeds something other than text
    ),
)
```

`modality="audio"` routes raw input (a list of file paths) through the audio
encoder (`embeddings/embed_audio.py`) instead of the text encoder. See the
`speaker` axis for an example; adding a new non-text modality means adding a
matching `embeddings/embed_<modality>.py` encoder and a dispatch branch in
`embed.py`'s `resolve_embeddings`.

Update `docs/source/user-guide/axes.md` with the new axis.

### Building and publishing a release

Releases are published by CI via PyPI Trusted Publishing (no API token is
stored), in two stages — a TestPyPI dry run, then production PyPI:

1. **Bump `version`** in `pyproject.toml`, commit, and merge to `main`. A version
   number can be uploaded only once, so every release needs a new number — you
   cannot re-publish or overwrite an existing version.
2. **Tag and push → TestPyPI.** Pushing a `v*` tag triggers `publish-testpypi.yml`
   (it checks the tag matches the `pyproject.toml` version):
   ```bash
   git tag v0.0.1          # must match the version in pyproject.toml
   git push origin v0.0.1
   ```
   Verify the result at <https://test.pypi.org/project/emb-diversity/>.
3. **Create a GitHub Release → PyPI.** When the TestPyPI run looks good, create a
   GitHub Release for the tag. That triggers `publish-pypi.yml`, which uploads to
   real PyPI (<https://pypi.org/project/emb-diversity/>). Create the release either:
   - **on GitHub:** go to the repository's **Releases** page (right-hand sidebar of
     the repo, or `.../releases`) → **Draft a new release** → under *Choose a tag*
     pick the existing tag (e.g. `v0.0.1`) → add a title and notes → **Publish
     release**; or
   - **with the GitHub CLI:**
     ```bash
     gh release create v0.0.1 --title "v0.0.1" --notes "First release"
     ```

   Publishing the release (not just drafting it) is what triggers the workflow.

To build and validate **locally** before tagging (optional):

```bash
rm -rf dist              # clear artifacts from previous versions first
uv build                 # -> dist/emb_diversity-<version>.{tar.gz,whl}
uvx twine check dist/*   # validate metadata + that the README renders on PyPI
```

`uv build` only *adds* to `dist/`, so clear it first when building a new version —
otherwise old artifacts linger and an upload would try (and fail) to re-publish
them. CI doesn't need this: each run starts from a clean checkout.

## Open TODOs

- **Memory use.** The command-line interface loads all input texts into memory
  at once. This is fine for small- to medium-sized datasets but will not scale
  to very large files — streaming or chunked input is not yet implemented.

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

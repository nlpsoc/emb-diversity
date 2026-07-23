# emb-diversity

<!-- docs-intro-start -->
A Python package for measuring data diversity on small- to medium-sized datasets (RAM-restricted, usually up to size 10k). All measures are calculating diversity based on embeddings, i.e., vector representations of your data. Depending on what embedding models you want to use, you are able to calculate semantic, stylistic, speaker and other types of diversity with our package — text by default, with select axes (e.g. `speaker`) covering other modalities like audio.

This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.
<!-- docs-intro-end -->

**Documentation:** <https://nlpsoc.github.io/emb-diversity/>  
**PyPi**: <https://pypi.org/project/emb-diversity/>

## Table of Contents

- **[Install](#install)**
- **[Usage](#usage)**
  - [Python API](#python-api)
  - [CLI](#cli)
- [Development](#development)
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

You can also use specific measures, see an [overview of all measures](https://nlpsoc.github.io/emb-diversity/user-guide/measures.html). Use with caution. Some measures might be worse for your use case than others. We recommend to test whether your chosen measure and embedding space capture your diversity axis of interest.

```python
# Run specific measures
print(measure_diversity(texts_a, measure=["diameter", "log_determinant"]))
# -> {'diameter': {'value': 0.94..., ...}, 'log_determinant': {'value': -0.93..., ...}}
print(measure_diversity(texts_b, measure=["diameter", "log_determinant"]))
# -> {'diameter': {'value': 1.0..., ...}, 'log_determinant': {'value': -0.06..., ...}}
```

**Using raw vectors**

If you are not working with text data or you already calculated the embeddings yourself, you can use ``vectors`` (numpy arrays or lists of lists of numbers) directly as well, see [Using Vectors Directly](https://nlpsoc.github.io/emb-diversity/user-guide/vectors.html).

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

See the full [CLI reference](https://nlpsoc.github.io/emb-diversity/user-guide/cli.html)
for all commands and options (`measure`, `list-measures`, `list-axes`).

## Development

Contributing to `emb-diversity` itself — dev setup, the collaboration workflow,
adding a new built-in measure or diversity axis, the docstring style guide, and
cutting a release — is covered in the
[Development guide](https://nlpsoc.github.io/emb-diversity/development.html).

## Open TODOs

- **Memory use.** The command-line interface loads all input texts into memory
  at once. This is fine for small- to medium-sized datasets but will not scale
  to very large files — streaming or chunked input is not yet implemented.

## Funding

This work is supported by the ERC Starting Grant **DataDivers** (101162980).

## Citation

<!-- docs-citation-start -->
If you use `emb-diversity` in your work, consider citing:

```bibtex
@article{su2026embdiversity,
      title={emb-diversity: A Tool for Embedding-Based Measurement of Data Diversity},
      author={Cantao Su and Menan Velayuthan and Esther Ploeger and Dong Nguyen and Anna Wegmann},
      journal={arXiv preprint arXiv:2607.19848},
      year={2026},
      url={https://arxiv.org/abs/2607.19848},
}
```
<!-- docs-citation-end -->

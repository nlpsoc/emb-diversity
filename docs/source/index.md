# embediver

```{include} ../../README.md
:start-after: <!-- docs-intro-start -->
:end-before: <!-- docs-intro-end -->
```

## Installation

```{include} ../../README.md
:start-after: <!-- docs-install-start -->
:end-before: <!-- docs-install-end -->
```

## Quickstart

```{include} ../../README.md
:start-after: <!-- docs-quickstart-start -->
:end-before: <!-- docs-quickstart-end -->
```

### Using pre-computed embeddings

If you already have embeddings (e.g. from your own model), you can pass them directly:

```python
import numpy as np
from embediver import measure_diversity, log_determinant

vectors = np.random.randn(100, 384)
measure_diversity(vectors)
log_determinant(vectors)
```

### CLI

```bash
embediver texts.txt                              # default measure (log_determinant)
embediver texts.txt -m core                      # curated set of measures
embediver texts.txt -m all                       # all 20 measures
embediver texts.txt -m mean_pw_dist -m diameter  # specific measures
embediver texts.txt --axis style                 # style diversity
embediver data.csv --column review_text          # CSV/TSV input
embediver list-measures                          # list available measures
embediver list-axes                              # list available axes
```

## User Guide

```{toctree}
:maxdepth: 2

user-guide/measures
user-guide/axes
user-guide/cli
user-guide/development-notes
```

## API Reference

```{toctree}
:maxdepth: 2

embediver
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

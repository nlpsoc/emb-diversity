# Examples

This folder contains runnable scripts for the use cases in our demo paper.

`colab_demo.ipynb` is a self-contained walkthrough of the Python API and CLI —
open it directly in [Google Colab](https://colab.research.google.com/github/nlpsoc/emb-diversity/blob/main/examples/colab_demo.ipynb),
no local setup required.

## Setup

Install the package with the `examples` extra:

- From PyPI: `pip install emb-diversity[examples]`
- From a checkout: `uv sync --extra examples`

## Run

From the repository root, run a script (for example):

- `python examples/gede_diversity.py`

The scripts will download the necessary datasets and embeddings automatically.
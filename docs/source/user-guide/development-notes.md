# Development Notes

Internal notes and design decisions for contributors.

## Text input handling

All measure functions accept both raw text and pre-computed embeddings. Each
measure calls `resolve_embeddings(data, diversity_axis, embedding_model)` (from
`embed.py`) as its first line: it checks whether the input is text and, if so,
embeds it and returns the resolved model id (which the measure records under
`parameters["embedding_model"]`); numeric input is passed through unchanged with
a `None` model id.

Resolving the input inside each measure (rather than in a shared wrapper) keeps
the resolved model id in scope for the result's `parameters`. This mirrors how
scikit-learn validates input via
[check_array](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py)
at the top of each estimator method.

## Environment variables

`EMB_DIVERSITY_PROGRESS` controls the spinner shown while an embedding model is
downloaded and prepared (the first call that needs a model can take a while and
print a lot of HuggingFace output, which the spinner replaces).

By default the spinner appears only in interactive sessions — a terminal or a
notebook — and stays silent in scripts, pipes, and CI. Set the variable to force
it on or off:

- `1`, `true`, `yes`, or `on` — always show it.
- `0`, `false`, `no`, or `off` — never show it.

Set it before the first call that loads a model. From the shell:

```bash
export EMB_DIVERSITY_PROGRESS=0
```

From Python or a notebook cell:

```python
import os
os.environ["EMB_DIVERSITY_PROGRESS"] = "1"
```

## Known limitations

- The CLI currently loads all texts into memory at once. This is fine for small- to medium-sized datasets but will not scale to very large files.

## TODOs

- **TODO:** Check that the default embedding model for the semantic axis makes sense with Tao (see `_axes.py`).
- **TODO:** Support SimCSE encoder (`embeddings/SimCSE.py`) in `embed.py` alongside SBERT.

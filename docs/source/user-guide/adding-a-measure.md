# Adding a Measure

Besides the [built-in measures](measures.md), you can run **your own** measure by
passing a function to `measure_diversity()` — no need to modify the package.

## The contract

A custom measure is a function called exactly like a built-in: it receives the
`data` you passed to `measure_diversity()`, plus the embedding keywords, and
returns a `{"value": float, "parameters": {...}}` dict.

```python
def my_measure(data, *, diversity_axis="semantic", embedding_model=None):
    ...
    return {"value": score, "parameters": {...}}
```

- `data` — whatever you pass to `measure_diversity()`: a list of text strings, or
  an `(n, d)` array of vectors.
- `diversity_axis` / `embedding_model` — forwarded by `measure_diversity()`; use
  them to embed text input (see below).
- returns a dict with a float `"value"` and a `"parameters"` dict recording the
  configuration used.

### Handling text input

To accept text as well as vectors, embed the input with `resolve_embeddings` — the
same helper the built-in measures use. It validates the input (rejecting a bare
string, non-2-D data, fewer than 2 samples, and nan/inf), embeds text, and returns
the resolved embedding-model id (or `None` for vector input) so you can record it:

```python
import numpy as np
from emb_diversity.embed import resolve_embeddings

def my_spread(data, *, diversity_axis="semantic", embedding_model=None):
    """A custom measure: the standard deviation of the vectors."""
    vectors, model = resolve_embeddings(data, diversity_axis, embedding_model)
    return {"value": float(vectors.std()), "parameters": {"embedding_model": model}}
```

If you only ever pass vectors (never text), you can skip `resolve_embeddings` and
work with `data` directly.

## Running it

Pass the function as `measure`. The `data` you give `measure_diversity()` is handed
to your measure unchanged — it can be a list of text strings (embedded for you) or
an `(n, d)` array of vectors, exactly like the built-in measures:

```python
import numpy as np
from emb_diversity import measure_diversity

# Text input — embedded via the diversity axis / embedding model.
texts = ["The cat sat on the mat.", "Dogs play fetch.", "A bird sings at dawn."]
measure_diversity(texts, measure=my_spread)
# {'my_spread': {'value': ..., 'parameters': {'embedding_model': '...'}}}

# Vector input — used directly; no embedding, so embedding_model is None.
vectors = np.random.randn(100, 384)
measure_diversity(vectors, measure=my_spread)
# {'my_spread': {'value': ..., 'parameters': {'embedding_model': None}}}

# Mixed with built-in measures, in a list:
measure_diversity(texts, measure=["mean_pw_dist", my_spread])
# {'mean_pw_dist': {...}, 'my_spread': {...}}
```

Your measure is keyed by its function name (`my_spread`) in the result.

## Good to know

- **Run as given.** A custom measure is not validated or wrapped — it runs exactly
  as written. (Calling `resolve_embeddings` first is what gives the built-ins their
  input validation; do the same if you want it.)
- **Failures are isolated.** If your measure raises, its `"value"` is `nan`, the
  entry gains an `"error"` key, and a `UserWarning` is emitted — the other measures
  in the call still run. This matches how built-in failures are reported.
- **Embeddings are computed once.** When `data` is text, `resolve_embeddings` reuses
  the on-disk embedding cache, so the model runs only once across every measure in a
  `measure_diversity()` call — including your own.

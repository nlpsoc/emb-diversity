# Adding a Measure

Besides the [built-in measures](measures.md), you can run **your own** measure by
passing a function to `measure_diversity()`; no need to modify the package. 
If you would like to contribute a embedding-based diversity measure to the package, 
see [Adding New Measures](https://github.com/nlpsoc/emb-diversity#adding-new-measures)
in the README — it also covers tagging the measure with its family from the
[taxonomy](measures.md).

## The contract

A custom measure needs to be a function with a signature exactly like a built-in measure: it receives the
`data` you passed to `measure_diversity()` plus the `embedding_model` key, and a `diversity_axis` key. It should return a
`{"value": float, "parameters": {...}}` dict (the `MeasureResult` type). Make
`resolve_embeddings` its first line (it returns a numpy array and a string for the embedding model name used if any). 
The `resolve_embeddings` function is the single place input is validated and
text is embedded:

```python
from emb_diversity.embed import resolve_embeddings
from emb_diversity.measures.types import MeasureResult


def custom_measure(data, *, diversity_axis="semantic", embedding_model=None) -> MeasureResult:
    """A custom measure: the standard deviation of the vectors."""
    vectors, model = resolve_embeddings(
        data, diversity_axis, embedding_model, measure="custom_measure"
    )
    return {"value": float(vectors.std()), "parameters": {"embedding_model": model}}
```

- `data` — what you pass to `measure_diversity()`: a list of text strings, or an
  `(n, d)` array of vectors.
- `resolve_embeddings` — the same helper the built-ins use. It validates the input
  (rejecting a bare string, non-2-D data, fewer than 2 samples, nan/inf), embeds
  text, and returns the vectors plus the resolved embedding-model id (`None` for
  vector input).
- `measure="custom_measure"` — optional. When given, an interactive
  "Calculating measure 'custom_measure'…" notice is printed once embedding finishes, just
  before your calculation runs (shown only in interactive sessions). Leave it out
  and the measure works the same, just without the notice.
- return a dict with a float `"value"` and a `"parameters"` dict recording the
  configuration used.

If you want your custom measure to support long-text **chunking** (passed as
`measure_diversity(..., chunking_kwargs={...})`), accept a `chunking_kwargs`
argument and forward it to `resolve_embeddings`:

```python
def custom_measure(data, *, diversity_axis="semantic", embedding_model=None,
                   chunking_kwargs=None) -> MeasureResult:
    vectors, model = resolve_embeddings(
        data, diversity_axis, embedding_model, measure="custom_measure",
        chunking_kwargs=chunking_kwargs,
    )
    return {"value": float(vectors.std()), "parameters": {"embedding_model": model}}
```

`measure_diversity` only forwards `chunking_kwargs` when the caller sets it, so a
custom measure without this argument still works for the default (truncation) path.

### Reusing built-in helpers

For distance-based measures you can reuse `compute_pairwise_distances` — the same
cached helper the built-ins use. It returns the condensed array of all pairwise
distances (like `scipy.spatial.distance.pdist`) and shares the on-disk distance
cache, so repeated runs over the same vectors and measures are cheap:

```python
from emb_diversity import compute_pairwise_distances
from emb_diversity.embed import resolve_embeddings
from emb_diversity.measures.types import MeasureResult


def my_min_dist(data, metric="cosine", *, diversity_axis="semantic", embedding_model=None) -> MeasureResult:
    """A custom measure: the smallest pairwise distance."""
    vectors, model = resolve_embeddings(
        data, diversity_axis, embedding_model, measure="my_min_dist"
    )
    dists = compute_pairwise_distances(vectors, metric)
    return {
        "value": float(dists.min()),
        "parameters": {"metric": metric, "embedding_model": model},
    }
```

## Running it

Pass the function as `measure`. The `data` you give `measure_diversity()` is handed
to your measure unchanged. It can be a list of text strings (embedded for you) or
an `(n, d)` array of vectors, exactly like the built-in measures:

```python
import numpy as np
from emb_diversity import measure_diversity

# Text input — embedded via the diversity axis / embedding model.
texts = ["The cat sat on the mat.", "Dogs play fetch.", "A bird sings at dawn."]
measure_diversity(texts, measure=custom_measure)
# {'custom_measure': {'value': ..., 'parameters': {'embedding_model': '...'}}}

# Vector input — used directly; no embedding, so embedding_model is None.
vectors = np.random.randn(100, 384)
measure_diversity(vectors, measure=custom_measure)
# {'custom_measure': {'value': ..., 'parameters': {'embedding_model': None}}}

# Mixed with built-in measures, in a list:
measure_diversity(texts, measure=["mean_pw_dist", custom_measure])
# {'mean_pw_dist': {...}, 'custom_measure': {...}}
```

Your measure is keyed by its function name (`custom_measure`) in the result.

## Good to know

- **Run as given.** A custom measure is not wrapped or checked — it runs exactly as
  written. Calling `resolve_embeddings` first (as above) is what validates the input,
  so make it the first line of your measure.
- **Failures are isolated.** If your measure raises, its `"value"` is `nan`, the
  entry gains an `"error"` key, and a `UserWarning` is emitted — the other measures
  in the call still run. This matches how built-in failures are reported.
- **Embeddings are computed once.** When `data` is text, `resolve_embeddings` reuses
  the on-disk embedding cache, so the model runs only once across every measure in a
  `measure_diversity()` call — including your own.

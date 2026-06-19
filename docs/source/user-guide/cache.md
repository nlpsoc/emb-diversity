# Cache Design

We faced two big bottlenecks which hinders compute speed. Therefore we devised cacheing methods for these two bottlenecks.

1. **Document Embeddings Cacheing** : Vectorizing documents is a shared task among all measures. Therefore, if the embeddings are recalculated for the same document again makes the process redundant. Therefore, we save the calculated vectors to the disk memory to be reused.
2. **Pairwise Distance Caching** : Most of the measures require an intermediate operation where the pairwise distance between the embeddings are calculated, this calculation also could be reused for the same dataset. Therefore we employ a three step cacheing inorder to address three common use cases.


## Document Embedding Caching

### Entry Point

The entry point for embedding text is `embed_texts()` in `emb_diversity.embed`. All measures that accept text input go through this function, which resolves the embedding model and returns a numpy array of vectors ready for the measure to use.

```python
from emb_diversity.embed import embed_texts

vectors = embed_texts(["sentence one", "sentence two"])
```

### Per-Sentence Caching

Each sentence is cached **individually**. When you pass a list of texts, every sentence is hashed and looked up in the cache separately. This means:

- If you add one new sentence to a previously embedded list, only that sentence is sent to the model.
- All other sentences are served from disk instantly.
- Reordering or extending your dataset never causes a full re-embedding.

### Cache Location

By default, embeddings are stored under:

```
.cache/embeddings/<model_name>/
```

Each model gets its own subdirectory, so embeddings from different models never interfere with each other.

To use a custom cache directory, pass `cache_dir` to `embed_texts()`:

```python
from emb_diversity.embed import embed_texts
from pathlib import Path

vectors = embed_texts(texts, cache_dir=Path("/my/custom/cache"))
```

### Embedding Pipeline

When a list of texts is passed to `embed_texts()`, the following steps are taken:

1. **Hash** — each sentence is hashed using `xxhash128`. The hash acts as a unique fingerprint of the text content. A single character change produces a completely different hash.

2. **Cache lookup** — the hash is used to check whether a `.safetensors` file already exists on disk for that sentence under the model's cache directory. Sentences with an existing cache file are loaded from disk in parallel.

3. **Encode missing sentences** — only the sentences with no cache file are batched and passed to the embedding model. This minimises model calls.

4. **Save new embeddings** — the freshly computed embeddings are written to disk so future calls can reuse them.

5. **Preserve order** — results are assembled back in the original input order, regardless of which sentences were cached and which were freshly computed. The output always corresponds positionally to the input list.

### Long Texts: Truncation vs Chunking

By default, a text longer than the embedding model's maximum sequence length is
**truncated** — tokens past the limit are dropped before embedding. To use the
whole text instead, enable **chunking** on `encode()` or `embed_texts()`:

```python
from emb_diversity.embed import embed_texts

# Truncation (default)
vectors = embed_texts(texts)

# Chunking: split each text into up to `chunks` windows of the model's max
# sequence length, embed every window, and pool them into one vector per text.
vectors = embed_texts(texts, chunking=True, chunks=5, pooling="mean")
```

Chunking options:

- `chunking` — set `True` to chunk instead of truncate. Default `False`.
- `chunks` — the maximum number of windows per text. Texts shorter than this
  use fewer windows; texts longer than `chunks × max_seq_length` tokens are
  truncated at the cap. Default `10`.
- `pooling` — how the per-window vectors are combined into one vector:
  `"mean"` (default), `"max"`, or `"first"`.

The cache key folds in the chunking mode **and the actual number of windows a
text was split into**, so:

- truncated and chunked embeddings of the same text never collide;
- two calls with different `chunks` caps that happen to produce the same number
  of windows share one cache entry, while a smaller cap that genuinely uses
  fewer windows is stored separately.

Chunking is available throughout the API. At the embedding entry points
(`encode()` and `embed_texts()`) the options are plain keyword arguments, as
above. At the measure layer they travel as a single `chunking_kwargs` dict, so
they never clash with a measure's own keyword arguments:

```python
from emb_diversity import measure_diversity, mean_pw_dist

# Main entry point
measure_diversity(
    texts,
    measure="mean_pw_dist",
    chunking_kwargs={"chunking": True, "chunks": 5, "pooling": "mean"},
)

# A single measure called directly
mean_pw_dist(texts, chunking_kwargs={"chunking": True, "chunks": 5})
```

From the CLI, use the `--chunking`, `--chunks`, and `--pooling` flags:

```bash
emb-diversity measure texts.txt -m mean_pw_dist --chunking --chunks 5 --pooling mean
```

The keys inside `chunking_kwargs` are exactly the `encode()` options
(`chunking`, `chunks`, `pooling`); the dict is forwarded down to the embedding
step unchanged, and omitted keys fall back to the defaults above.

### Disabling the Cache

There is no on/off switch for the cache. If you need to force re-embedding (for example, after updating a model), clear the cache manually using `clear_cache()` from `emb_diversity.utility`.

`clear_cache()` accepts two optional arguments:

- `model_name` — if provided, only the cache for that model is removed. If omitted, the entire cache directory is wiped.
- `cache_dir` — the root cache directory to clear. Defaults to `.cache/embeddings`.

```python
from emb_diversity.utility import clear_cache

# Clear cache for a specific model only
clear_cache(model_name="all-MiniLM-L6-v2")

# Clear cache for a specific model in a custom directory
clear_cache(model_name="all-MiniLM-L6-v2", cache_dir=Path("/my/custom/cache"))

# Clear the entire embedding cache
clear_cache()
```

---

## Pairwise Distance Caching

### Why

Most measures internally compute pairwise distances between embedding vectors using `scipy.pdist`. When multiple measures are run on the same dataset, this computation would otherwise be repeated for each measure. `compute_pairwise_distances()` in `emb_diversity.measures.utils` wraps `scipy.pdist` with a two-level cache to avoid that.

### Three-Level Lookup

Every call to `compute_pairwise_distances()` goes through the following levels in order:

1. **Memory (Level 1)** — checks a small in-process LRU dictionary. If the result is there, it is returned immediately with no disk I/O.

2. **Disk (Level 2)** — checks for a `.safetensors` file on disk. If found, the result is loaded, promoted to memory, and returned. Disk cache survives across processes — a SLURM job that ran yesterday leaves a cache that today's job can reuse without recomputing.

3. **Compute (Level 3)** — if neither cache has the result, `scipy.pdist` is called. The result is written to both memory and disk before returning.

### Cache Key

The cache key is a combination of:

- **Matrix fingerprint** — a full-content `xxhash64` of the embedding matrix, including its shape and dtype. The matrix is hashed in chunks to keep memory usage constant regardless of size.
- **Metric** — the distance metric name (e.g. `"cosine"`, `"euclidean"`) or a callable.
- **Metric kwargs** — any extra keyword arguments passed to `scipy.pdist` are folded into the key. This means `cosine` and `euclidean` on the same data, or the same metric with different kwargs, never collide in cache.

### Constants

Two constants control the caching behaviour:

- **`_HASH_CHUNK = 1_000_000`** — the number of array elements fed into the hash function at a time. Chunking keeps memory usage constant regardless of how large the embedding matrix is. You do not need to change this.

- **`_MEMORY_MAX = 4`** — the maximum number of distance matrices held in the in-memory LRU cache at once. When a fifth matrix is computed, the oldest one is evicted. Set to `0` to disable memory caching entirely and rely only on disk:

  ```python
  from emb_diversity.measures import utils as pw
  pw._MEMORY_MAX = 0
  ```

### Cache Location

By default, distance matrices are stored under:

```
.cache/pdist/
```

To use a custom directory, pass `cache_dir`:

```python
from emb_diversity.measures.utils import compute_pairwise_distances
from pathlib import Path

distances = compute_pairwise_distances(vectors, metric="cosine", cache_dir=Path("/my/cache"))
```

### Cache Statistics

To inspect how much memory and disk the cache is using:

```python
from emb_diversity.measures.utils import distance_cache_info

info = distance_cache_info()
# {
#   "memory_entries": 2,
#   "memory_mb": 0.45,
#   "memory_max": 4,
#   "disk_files": 5,
#   "disk_mb": 1.2,
# }
```

### Clearing the Cache

`clear_distance_cache()` clears both the in-memory LRU and the disk cache at once. Unlike the embedding cache, there is no per-metric option — the entire cache directory is removed.

```python
from emb_diversity.measures.utils import clear_distance_cache

# Clear memory and disk cache (default directory)
clear_distance_cache()

# Clear a custom cache directory
clear_distance_cache(cache_dir=Path("/my/cache"))
```

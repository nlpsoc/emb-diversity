# Notes

## Handling Long Texts

Longer texts demand models that can support longer maximum sequence lengths. When this condition is not satisfied, i.e., when the text token count is larger than the model's maximum sequence 
length  (Lewis et al., 2020; Park et al., 2022), the model resorts to truncation, where it chops and takes the text up until its maximum sequence length. 
This method does not affect texts with strong lead bias (Zhang et al., 2022), where most of the information is presented at the beginning of the text. 
However, there are many text datasets where information is more distributed throughout a text; for these, truncation would not be a suitable strategy.
Consider the following example:

> ### Example: Effect of Truncation
>
> **Document A**
>
> **The company conducted an internal investigation into the reported security incident. After reviewing the available evidence and interviewing employees, the investigators concluded that the allegations were**
> substantiated and that corrective actions would be taken.
>
> **Document B**
>
> **The company conducted an internal investigation into the reported security incident. After reviewing the available evidence and interviewing employees, the investigators concluded that the allegations were**
> unfounded and that no further action was required.
>
> Suppose a model can only process the bold portion of each document. Since truncation occurs before the final conclusion, both documents become identical from the model's perspective despite conveying opposite meanings.


Suppose a model can only process the first portion of each document and the truncation point occurs before the final conclusion. In that case, both documents become identical from the model's perspective despite conveying opposite meanings.

For datasets containing longer texts, we recommend text embedding models that can handle longer maximum sequence lengths.
However, to support users with compute-constrained environments, the package provides chunking and pooling as a strategy to handle longer text sequences  (Dong
et al., 2025).


## Pipeline for Handling Longer Texts

Our pipeline consists of two steps.

* **Chunking:** In this step, we break text into smaller chunks in multiples of the model's maximum sequence length. The user can specify how many chunks to consider by defining the chunk size (default size being 10).
* **Pooling:** Once the chunks are independently embedded by the model, we use pooling to obtain a single representation for multiple chunks of a given text. We currently support `mean` (default) and `max` pooling strategies.


It is worth mentioning that our cache is designed in such a way as to distinguish between the same text being truncated or chunked. 
They are cached differently, and even chunking with different chunk sizes is handled differently by our caching module. 
An example usage is shown below.

```python
from emb_diversity import (
    measure_diversity, mean_pw_dist)

# Long documents that exceed the
# model's maximum sequence length.
texts = [doc1, doc2, doc3]

# From the main function:
measure_diversity(texts,
    measure="mean_pw_dist",
    diversity_axis="semantic",
    chunking_kwargs={"chunking": True,
                     "chunks": 5,
                     "pooling": "mean"})

# Or directly from an individual measure:
mean_pw_dist(texts,
    diversity_axis="semantic",
    chunking_kwargs={"chunking": True,
                     "chunks": 5,
                     "pooling": "mean"})
```

Chunking is available both from the main `measure_diversity`
function and from individual measures, via the same `chunking_kwargs`
argument.


## Caching

Since all measures in our package operate on embeddings, each dataset must first be embedded using an embedding model. We identify two key components that increase computational cost:

* **Document embedding caching:** All data instances are passed through an encoding model to obtain vectors used by the downstream measures. Since this step is shared across all measures and computationally expensive, we employ a disk-level caching strategy.
* **Pairwise distance measure caching:** The majority of measures rely on pairwise distances as a primary computation. Given their repeated usage across multiple measures and their computational cost, we implement a three-level caching mechanism for this component.


### Document embedding caching

We address this by storing text embeddings on disk. We follow a per-text caching strategy, mainly because multiple datasets may share identical texts.

While caching, we ensure that the same document embedded with different text encoders is treated separately. We further increase this granularity 
such that the same text embedded under truncation, `chunks=10`, `chunks=5`, or `chunks=5` with `pooling=mean/max` are all treated as distinct entries. 
This is necessary, as these configurations produce different representations and should not be mixed.

Given a list of texts and an embedding configuration (model, and either truncation or chunking with a chunk count and pooling strategy), the cache proceeds as follows:

* **Key construction.** For each text, we construct a cache key that combines the text content with a suffix describing the embedding configuration. For truncation,
  the suffix is empty; for chunking, it encodes the chunking and pooling configuration (e.g. `chunk=3|pool=mean`). The final key is obtained by applying a 128-bit `xxhash`
  over the combined string. This ensures that the same text under different embedding settings (e.g., truncation vs. chunking, different chunk sizes, or different pooling strategies)
  maps to distinct cache entries.
* **Lookup.** Each key maps to a file
    `\path{<cache_dir>/<model>/<hash>.safetensors}`, namespaced by model so
    different models never interfere. We probe these files in parallel; a text
    whose file exists is a cache *hit* and its embedding is loaded from
    disk.

* **Compute misses only.** Texts with no cached file (cache
    *misses*) are gathered and passed to the embedding model in a single
    batch. For chunking, each missed text is split into windows, every window is
    embedded, and the window embeddings are pooled into one vector.

* **Write-through.** The freshly computed embeddings are written to
    disk, each under its own key, so subsequent runs reuse them.

* **Order-preserving assembly.** Cached and freshly computed
    embeddings are recombined in the original input order and returned, so the
    output aligns positionally with the input texts regardless of which were
    cached.

## Pairwise-distance cache

Many measures require the computation of a matrix of pairwise distances between a dataset's embedding vectors (`scipy.pdist`), which is often the dominant 
computational cost when multiple measures are applied on the same data. We wrap this computation in a two-level cache, checked in order.

* **Key construction.** We construct a cache key from the embedding matrix and the distance measure configuration. This includes a fingerprint of the input embeddings and an identifier for the metric and its parameters. This ensures that different inputs, or the same inputs under different measure settings, are treated as distinct cache entries.
* **Level~1 ---> in-memory.** We first check a bounded least-recently-used (LRU) cache in the running process. On a hit, the condensed distance array is returned immediately without any disk access. The cache stores up to a fixed number of matrices (default 4); older entries are evicted on overflow, and the cache can also be disabled entirely.
* **Level~2 ---> on disk.** On an in-memory miss, we check for a `safetensors` file indexed by the same fingerprint and measure. On a hit, the array is loaded, promoted to the in-memory cache, and returned. This cache persists across processes, allowing results computed in earlier runs (e.g., separate scheduler jobs) to be reused without recomputation.
* **Level~3 ---> compute and write-through.** If both cache levels miss, we compute the pairwise distances using `scipy.pdist`. The result is then written through both the in-memory and disk caches before being returned, ensuring subsequent calls can reuse it.

### References

Zican Dong, Tianyi Tang, Junyi Li, and Wayne Xin
Zhao. 2025. A survey on long text modeling with
transformers. Preprint, arXiv:2302.14502.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. 2020.
BART: Denoising sequence-to-sequence pre-training
for natural language generation, translation, and com-
prehension. In Proceedings of the 58th Annual Meet-
ing of the Association for Computational Linguistics,
pages 7871–7880, Online. Association for Computa-
tional Linguistics

Hyunji Hayley Park, Yogarshi Vyas, and Kashif Shah.
2022. Efficient classification of long documents us-
ing transformers. In Proceedings of the 60th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers), pages 702–709,
Dublin, Ireland. Association for Computational Lin-
guistics.

Yusen Zhang, Ansong Ni, Ziming Mao, Chen Henry Wu,
Chenguang Zhu, Budhaditya Deb, Ahmed Awadallah,
Dragomir Radev, and Rui Zhang. 2022. Summn: A
multi-stage summarization framework for long input
dialogues and documents. In Proceedings of the 60th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 1592–
1604, Dublin, Ireland. Association for Computational
Linguistics.
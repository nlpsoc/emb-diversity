# Using Vectors Directly

All measure functions and `measure_diversity()` accept vectors
in addition to raw text. This is useful when you have
vector representations of non-textual data.

## With `measure_diversity()`

```python
import numpy as np
from embediver import measure_diversity

vectors = np.random.randn(100, 384)

# Default measure
measure_diversity(vectors)

# Core set
measure_diversity(vectors, measure="core")

# Specific measures
measure_diversity(vectors, measure=["mean_pw_dist", "diameter"])
```

## With individual measure functions

```python
import numpy as np
from embediver import log_determinant, mean_pw_dist, vendi_score

vectors = np.random.randn(100, 384)
log_determinant(vectors)
mean_pw_dist(vectors)
vendi_score(vectors)
```

## Embedding once, measuring multiple times

If you want to measure diversity on the same texts with different measures,
embed once and pass the vectors:

```python
from embediver import embed_texts, log_determinant, mean_pw_dist

vectors = embed_texts(texts, diversity_axis="semantic")
log_determinant(vectors)
mean_pw_dist(vectors)
```

Note: `measure_diversity()` already does this internally — it embeds once
and passes vectors to all requested measures.

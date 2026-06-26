# Using Vectors Directly

All measure functions and `measure_diversity()` accept vectors
in addition to raw text. This is useful when you have
vector representations of non-textual data.

## With `measure_diversity()`

```python
import numpy as np
from emb_diversity import measure_diversity

vectors = np.random.randn(100, 384)

# Default measure
print(measure_diversity(vectors))
# {'graph_entropy': ..., 'vendi_score': ..., 'mean_pw_dist': ...}

# A named measure set (variety, balance, or disparity)
print(measure_diversity(vectors, measure="variety"))
# {'chamfer_dist': ..., 'sum_bottleneck': ..., 'mst_dispersion': ...}


# Specific measures
print(measure_diversity(vectors, measure=["mean_pw_dist", "diameter"]))
# {'mean_pw_dist': ..., 'diameter': ...}
```

Each measure returns `{"value": <float>, "parameters": {...}}`. For vector input,
no embedding happens, so `parameters["embedding_model"]` is `None`:

```python
print(measure_diversity(vectors, measure="mean_pw_dist"))
# {'mean_pw_dist': {'value': 1.39, 'parameters': {'metric': 'cosine', 'embedding_model': None}}}
```


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
measure_diversity(vectors)

# A named measure set (variety, balance, or disparity)
measure_diversity(vectors, measure="variety")

# Specific measures
measure_diversity(vectors, measure=["mean_pw_dist", "diameter"])
```


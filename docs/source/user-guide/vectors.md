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

# Core set
measure_diversity(vectors, measure="core")

# Specific measures
measure_diversity(vectors, measure=["mean_pw_dist", "diameter"])
```


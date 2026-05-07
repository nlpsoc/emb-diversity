"""Shared type aliases for embediver."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]

# Anything coercible to a 2D numeric array of shape (n, d): numpy arrays,
# torch tensors (forward-reference string to avoid forcing a top-level torch
# import), or nested sequences of floats.
TensorLike = Union[np.ndarray, "torch.Tensor", Sequence[Sequence[float]]]

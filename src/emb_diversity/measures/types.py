"""Shared type aliases for emb_diversity."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Union

import numpy as np

DistanceMetric = Union[str, Callable[[np.ndarray, np.ndarray], float]]

# Return shape of every measure function: the scalar score plus the
# parameters that produced it. ``{"value": float, "parameters": dict[str,
# Any]}``. Callers going through ``get_measure()`` or ``measure_diversity()``
# additionally get a top-level ``"version"`` key stamped in — see
# ``measures_registry.get_measure``.
MeasureResult = Dict[str, Any]

# Anything coercible to a 2D numeric array of shape (n, d): numpy arrays,
# torch tensors (forward-reference string to avoid forcing a top-level torch
# import), or nested sequences of floats.
TensorLike = Union[np.ndarray, "torch.Tensor", Sequence[Sequence[float]]]

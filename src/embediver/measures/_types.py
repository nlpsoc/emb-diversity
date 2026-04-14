"""Shared type aliases for embediver."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

DISTANCE_METRIC = Union[str, Callable[[np.ndarray, np.ndarray], float]]

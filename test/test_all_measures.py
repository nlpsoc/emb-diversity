"""Cross-measure tests: behaviours that hold uniformly for every measure.

Parametrized over the measure registry so each measure is reported as its own
pass/fail case. Behaviours specific to a single measure live in test_diversity.py.
"""

import numpy as np
import pytest

from emb_diversity.measures_registry import measures


class TestAllMeasures:

    @pytest.mark.parametrize("name", sorted(measures))
    def test_returns_finite_float_on_ndarray(self, name):
        """Every measure returns a finite float for numpy-array input."""
        X = np.random.RandomState(0).randn(50, 16)
        result = measures[name](X)
        assert isinstance(result, float)
        assert np.isfinite(result)

    @pytest.mark.parametrize("name", sorted(measures))
    def test_empty_data_raises_value_error(self, name):
        """Every measure raises ValueError on empty input."""
        with pytest.raises(ValueError):
            measures[name]([])

    @pytest.mark.parametrize("name", sorted(measures))
    def test_single_datapoint_raises_value_error(self, name):
        """Every measure raises ValueError on a single data point."""
        with pytest.raises(ValueError):
            measures[name]([[1.0, 2.0, 3.0]])

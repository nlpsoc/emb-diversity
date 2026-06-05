"""Tests that every registered measure works through measure_diversity().
"""

import numpy as np
import pytest

from emb_diversity import measure_diversity
from emb_diversity.measures_registry import measures


class TestConvenienceFunction:

    @staticmethod
    def _vectors():
        # 50 points in 16-D
        return np.random.RandomState(0).randn(50, 16)

    @pytest.mark.parametrize("name", sorted(measures))
    def test_measure_runs_via_convenience_and_matches_direct(self, name):
        """Each measure runs via measure_diversity on array input without
        returning NaN, and matches the direct call on the same vectors.

        Each result is a ``{"value": float, "parameters": {...}}`` dict.
        """
        X = self._vectors()
        direct = measures[name](X)
        via = measure_diversity(X, measure=[name])

        assert name in via
        assert not np.isnan(via[name]["value"]), (
            f"{name} returned NaN via measure_diversity"
        )
        assert np.isclose(via[name]["value"], direct["value"]), (
            f"{name}: measure_diversity={via[name]['value']} != direct={direct['value']}"
        )

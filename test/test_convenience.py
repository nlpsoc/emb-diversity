"""Tests that every registered measure works through measure_diversity().

measure_diversity() embeds text to a numpy array and calls each measure on that
array, catching exceptions and turning them into NaN. So a measure that is
broken on numpy-array input (as cluster_inertia once was) would silently return
NaN here rather than failing loudly. These tests catch that whole class of bug.
"""

import numpy as np
import pytest

from emb_diversity import measure_diversity
from emb_diversity.measures_registry import measures


class TestConvenienceFunction:

    @staticmethod
    def _vectors():
        # 50 points in 16-D: enough for every measure (including the UMAP-
        # projected ones and k-means) to run without small-sample fallbacks.
        return np.random.RandomState(0).randn(50, 16)

    @pytest.mark.parametrize("name", sorted(measures))
    def test_measure_runs_via_convenience_and_matches_direct(self, name):
        """Each measure runs through measure_diversity on array input without
        returning NaN, and matches the direct call on the same vectors."""
        X = self._vectors()
        direct = measures[name](X)
        via = measure_diversity(X, measure=[name])

        assert name in via
        assert not np.isnan(via[name]), (
            f"{name} returned NaN via measure_diversity "
            f"(it likely raised on numpy-array input)"
        )
        assert np.isclose(via[name], direct), (
            f"{name}: measure_diversity={via[name]} != direct={direct}"
        )

    def test_all_measures_run_in_one_call(self):
        """measure="all" returns a finite score for every registered measure."""
        results = measure_diversity(self._vectors(), measure="all")
        assert set(results) == set(measures)
        nan = sorted(name for name, value in results.items() if np.isnan(value))
        assert not nan, f"measures returned NaN via measure_diversity: {nan}"

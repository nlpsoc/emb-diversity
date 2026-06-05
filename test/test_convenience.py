"""Tests that every registered measure works through measure_diversity().

measure_diversity() embeds text to a numpy array and calls each measure on that
array, catching exceptions and turning them into NaN. So a measure that is
broken on numpy-array input (as cluster_inertia once was) would silently return
NaN here rather than failing loudly. These tests catch that whole class of bug.
"""

import numpy as np

from emb_diversity import measure_diversity
from emb_diversity.measures_registry import measures


class TestConvenienceFunction:

    @staticmethod
    def _vectors():
        # 50 points in 16-D: enough for every measure (including the UMAP-
        # projected ones and k-means) to run without small-sample fallbacks.
        return np.random.RandomState(0).randn(50, 16)

    def test_all_measures_run_without_nan(self):
        """measure="all" returns a finite score for every registered measure."""
        results = measure_diversity(self._vectors(), measure="all")
        assert set(results) == set(measures)
        nan = sorted(name for name, value in results.items() if np.isnan(value))
        assert not nan, f"measures returned NaN via measure_diversity: {nan}"

    def test_convenience_matches_direct_call(self):
        """Each measure returns the same value via measure_diversity as when
        called directly on the same vectors."""
        X = self._vectors()
        mismatches = []
        for name in sorted(measures):
            direct = measures[name](X)
            via = measure_diversity(X, measure=[name])[name]
            if not np.isclose(via, direct, equal_nan=True):
                mismatches.append(f"{name}: via={via} direct={direct}")
        assert not mismatches, (
            "measure_diversity != direct call for: " + "; ".join(mismatches)
        )

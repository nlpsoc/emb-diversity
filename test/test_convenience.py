"""Tests that every registered measure works through measure_diversity().
"""

import numpy as np
import pytest

from emb_diversity import measure_diversity
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure


class TestConvenienceFunction:

    @staticmethod
    def _vectors():
        # 50 points in 16-D
        return np.random.RandomState(0).randn(50, 16)

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_measure_runs_via_convenience_and_matches_direct(self, name):
        """Each measure runs via measure_diversity on array input without
        returning NaN, and matches the direct call on the same vectors.

        Each result is a ``{"value": float, "parameters": {...}, "version": str}`` dict.
        """
        X = self._vectors()
        direct = get_measure(name)(X)
        via = measure_diversity(X, measure=[name])

        assert name in via
        assert not np.isnan(via[name]["value"]), (
            f"{name} returned NaN via measure_diversity"
        )
        assert np.isclose(via[name]["value"], direct["value"]), (
            f"{name}: measure_diversity={via[name]['value']} != direct={direct['value']}"
        )


class TestMeasureFailureReporting:

    @staticmethod
    def _two_points():
        # Too few points for convex_hull_volume_3d, which needs at least 4.
        return [[0.0, 1.0], [1.0, 0.0]]

    def test_failing_measure_warns_and_records_error(self):
        """A failing measure emits a UserWarning and an 'error' entry."""
        with pytest.warns(UserWarning, match="convex_hull_volume_3d.*fewer than 4"):
            results = measure_diversity(
                self._two_points(), measure=["convex_hull_volume_3d"]
            )
        result = results["convex_hull_volume_3d"]
        assert np.isnan(result["value"])
        assert "fewer than 4" in result["error"]

    def test_failing_measure_does_not_abort_others(self):
        """Measures after a failing one still run and report values."""
        with pytest.warns(UserWarning):
            results = measure_diversity(
                self._two_points(), measure=["convex_hull_volume_3d", "mean_pw_dist"]
            )
        assert not np.isnan(results["mean_pw_dist"]["value"])

    def test_successful_measure_has_no_error_key(self):
        """Successful results keep the plain {value, parameters} shape."""
        results = measure_diversity(self._two_points(), measure=["mean_pw_dist"])
        assert "error" not in results["mean_pw_dist"]

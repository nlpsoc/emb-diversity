"""Cross-measure tests: behaviours that hold uniformly for every measure.

Parametrized over the measure registry so each measure is reported as its own
pass/fail case. Behaviours specific to a single measure live in test_diversity.py.
"""

import numpy as np
import pytest

from emb_diversity import dcscore, graph_entropy, log_determinant, vendi_score
from emb_diversity.measures_registry import measures


class TestAllMeasures:

    @staticmethod
    def _vectors():
        # 50 points in 16-D
        return np.random.RandomState(0).randn(50, 16)

    @pytest.mark.parametrize("name", sorted(measures))
    def test_result_contract(self, name):
        """Every measure returns ``{"value": float, "parameters": {...}}``.

        The value is a finite Python float, and ``parameters`` records an
        ``embedding_model`` entry which is ``None`` for vector (non-text) input.
        """
        result = measures[name](self._vectors())

        assert set(result.keys()) == {"value", "parameters"}
        assert isinstance(result["value"], float)
        assert np.isfinite(result["value"])
        assert isinstance(result["parameters"], dict)
        # Vector input is not embedded, so no model id is recorded.
        assert result["parameters"]["embedding_model"] is None

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


class TestZeroNormVectorConsistency:
    """A zero-norm (all-zero) vector makes cosine/normalized similarity
    undefined; every cosine-based measure should raise ValueError consistently
    rather than silently returning nan or a degenerate value."""

    @staticmethod
    def _with_zero():
        """Three 2D points where the second is the (degenerate) zero vector."""
        return np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    @pytest.mark.parametrize("measure", [graph_entropy, vendi_score, dcscore, log_determinant])
    def test_zero_norm_raises(self, measure):
        """Each cosine-based measure raises ValueError on a zero-norm vector."""
        with pytest.raises(ValueError):
            measure(self._with_zero())

    def test_vendi_explicit_path_raises(self):
        """The non-dual Vendi path (use_dual=False) also raises, not just the dual path."""
        with pytest.raises(ValueError):
            vendi_score(self._with_zero(), use_dual=False)

    @pytest.mark.parametrize("measure", [vendi_score, dcscore, log_determinant])
    def test_normalize_false_does_not_raise(self, measure):
        """With normalize=False no row is divided by its norm, so a zero vector is fine."""
        result = measure(self._with_zero(), normalize=False)["value"]
        assert np.isfinite(result)

    def test_graph_entropy_euclidean_does_not_raise(self):
        """Euclidean has no division by norm, so a zero vector is fine there."""
        result = graph_entropy(self._with_zero(), metric="euclidean")["value"]
        assert np.isfinite(result)

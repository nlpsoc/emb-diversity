"""Cross-measure tests: behaviours that hold uniformly for every measure.

Parametrized over the measure registry so each measure is reported as its own
pass/fail case. Behaviours specific to a single measure live in test_diversity.py.
"""

import numpy as np
import pytest

from emb_diversity import dcscore, graph_entropy, log_determinant, renyi_entropy, vendi_score
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
    """A zero-norm (all-zero) vector makes cosine / normalized similarity
    undefined. No measure should silently return nan: cosine-based measures
    raise ValueError, the rest return a finite value. Parametrized over the
    whole registry so a new (or overlooked) measure cannot quietly
    reintroduce the nan."""

    @staticmethod
    def _with_zero():
        """50 points in 16-D with one (degenerate) all-zero row.

        Large enough for measures that need several points (e.g.
        spectral / projection-based ones), matching ``_vectors`` above.
        """
        X = np.random.RandomState(0).randn(50, 16)
        X[7] = 0.0
        return X

    @pytest.mark.parametrize("name", sorted(measures))
    def test_zero_norm_never_returns_nan(self, name):
        """No measure silently returns nan on a zero-norm vector: it either
        raises ValueError (cosine-based, undefined) or returns a finite value."""
        try:
            result = measures[name](self._with_zero())["value"]
        except ValueError:
            return  # acceptable: the degenerate input is explicitly rejected
        assert np.isfinite(result), f"{name} returned non-finite {result!r}"

    @pytest.mark.parametrize("measure", [vendi_score, dcscore, log_determinant, renyi_entropy])
    def test_normalizing_kernel_measures_raise(self, measure):
        """The L2-normalizing kernel measures raise on a zero-norm vector (with
        default normalize=True) rather than silently treating it as
        zero-similarity. The registry-wide ``never_returns_nan`` test cannot
        catch this: these measures previously returned a finite but degenerate
        value, so only an explicit raise check enforces the consistency."""
        with pytest.raises(ValueError):
            measure(self._with_zero())

    def test_vendi_explicit_path_raises(self):
        """The non-dual Vendi path (use_dual=False) also raises, not just the dual path."""
        with pytest.raises(ValueError):
            vendi_score(self._with_zero(), use_dual=False)

    @pytest.mark.parametrize("measure", [vendi_score, dcscore, log_determinant, renyi_entropy])
    def test_normalize_false_does_not_raise(self, measure):
        """With normalize=False no row is divided by its norm, so a zero vector is fine."""
        result = measure(self._with_zero(), normalize=False)["value"]
        assert np.isfinite(result)

    def test_graph_entropy_euclidean_does_not_raise(self):
        """Euclidean has no division by norm, so a zero vector is fine there."""
        result = graph_entropy(self._with_zero(), metric="euclidean")["value"]
        assert np.isfinite(result)

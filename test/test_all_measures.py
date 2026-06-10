"""Cross-measure tests: behaviours that hold uniformly for every measure.

Parametrized over the measure registry so each measure is reported as its own
pass/fail case. Behaviours specific to a single measure live in test_diversity.py.
"""

import numpy as np
import pytest

from emb_diversity.measures_registry import measure_registry


class TestAllMeasures:

    @staticmethod
    def _vectors():
        # 50 points in 16-D
        return np.random.RandomState(0).randn(50, 16)

    @pytest.mark.parametrize("name", sorted(measure_registry))
    def test_result_contract(self, name):
        """Every measure returns ``{"value": float, "parameters": {...}}``.

        The value is a finite Python float, and ``parameters`` records an
        ``embedding_model`` entry which is ``None`` for vector (non-text) input.
        """
        result = measure_registry[name](self._vectors())

        assert set(result.keys()) == {"value", "parameters"}
        assert isinstance(result["value"], float)
        assert np.isfinite(result["value"])
        assert isinstance(result["parameters"], dict)
        # Vector input is not embedded, so no model id is recorded.
        assert result["parameters"]["embedding_model"] is None

    @pytest.mark.parametrize("name", sorted(measure_registry))
    def test_empty_data_raises_value_error(self, name):
        """Every measure raises ValueError on empty input."""
        with pytest.raises(ValueError):
            measure_registry[name]([])

    @pytest.mark.parametrize("name", sorted(measure_registry))
    def test_single_datapoint_raises_value_error(self, name):
        """Every measure raises ValueError on a single data point."""
        with pytest.raises(ValueError):
            measure_registry[name]([[1.0, 2.0, 3.0]])

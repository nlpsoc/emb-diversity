import numpy as np
import pytest

from emb_diversity import mean_pw_dist
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure


class TestShapeValidation:
    """Non-2-D input should fail early with one shared, informative message.

    The check runs in resolve_embeddings, the funnel every measure passes
    its data through — parametrizing over the registry means any newly
    added measure is covered (and tested) automatically.
    """

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_one_dimensional_input_raises(self, name):
        """A flat 1-D input raises a ValueError explaining the expected shape."""
        with pytest.raises(ValueError, match=r"2-dimensional.*\(n_samples, n_features\)"):
            get_measure(name)([0, 0])

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_three_dimensional_input_raises(self, name):
        """A 3-D array raises the same shape error, naming the actual shape."""
        data = np.random.RandomState(0).rand(4, 3, 2)
        with pytest.raises(ValueError, match=r"got shape \(4, 3, 2\)"):
            get_measure(name)(data)

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_empty_data_raises(self, name):
        """An empty input raises a ValueError naming the sample minimum."""
        with pytest.raises(ValueError, match="at least 2 samples, got 0"):
            get_measure(name)([])

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_single_datapoint_raises(self, name):
        """A single datapoint raises the same minimum, however it is written."""
        with pytest.raises(ValueError, match="at least 2 samples, got 1"):
            get_measure(name)([0])

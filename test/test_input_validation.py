import numpy as np
import pytest

from emb_diversity import graph_entropy, mean_pw_dist


class TestDistanceInputValidation:
    """Atypical inputs should fail early with informative error messages."""

    def test_empty_data_raises(self):
        """An empty input raises a ValueError mentioning empty data."""
        with pytest.raises(ValueError, match="empty data"):
            mean_pw_dist([])

    def test_single_datapoint_raises(self):
        """A single datapoint raises a ValueError mentioning the single point."""
        with pytest.raises(ValueError, match="single data point"):
            mean_pw_dist([0])

    def test_one_dimensional_input_raises(self):
        """A flat 1-D input raises a ValueError explaining the expected shape."""
        with pytest.raises(ValueError, match=r"2-dimensional.*\(n_samples, n_features\)"):
            mean_pw_dist([0, 0])

    def test_zero_vector_with_cosine_raises(self):
        """A zero vector under cosine distance raises instead of returning nan."""
        with pytest.raises(ValueError, match="Cosine distance is undefined.*row\\(s\\) \\[1\\]"):
            mean_pw_dist([[1, 1], [0, 0]], metric="cosine")

    def test_zero_vector_with_cosine_raises_in_graph_entropy(self):
        """The zero-vector check covers every distance-based measure."""
        with pytest.raises(ValueError, match="Cosine distance is undefined"):
            graph_entropy([[1, 0], [0, 0]])

    def test_zero_vector_with_euclidean_works(self):
        """Zero vectors are fine for metrics that are defined on them."""
        result = mean_pw_dist([[1, 1], [0, 0]], metric="euclidean")["value"]
        assert np.isclose(result, np.sqrt(2))

    def test_non_numeric_strings_raise(self):
        """Letter strings raise a ValueError pointing at non-numeric data."""
        with pytest.raises(ValueError, match="must be numeric"):
            mean_pw_dist([["a", "b"], ["c", "d"]])

    def test_numeric_strings_raise(self):
        """Number-like strings are rejected rather than silently coerced."""
        with pytest.raises(ValueError, match="contains strings"):
            mean_pw_dist([["0", "1"], ["1", "0"]])

    def test_mixed_numeric_strings_and_ints_raise(self):
        """Mixing number-like strings with ints is rejected as well."""
        with pytest.raises(ValueError, match="contains strings"):
            mean_pw_dist([["1", 0], ["1", "1"]])

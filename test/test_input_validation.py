import numpy as np
import pytest

from emb_diversity import mean_pw_dist, span_centroid
from emb_diversity.measures_registry import measures

# Measures whose distances flow through compute_pairwise_distances
# (scipy.pdist) and therefore share its cosine zero-vector check.
PDIST_MEASURE_NAMES = [
    "bottleneck",
    "chamfer_dist",
    "diameter",
    "dist_dispersion",
    "energy",
    "graph_entropy",
    "hamdiv",
    "mean_pw_dist",
    "mst_dispersion",
    "span_medoid",
    "sum_bottleneck",
    "sum_diameter",
]

# span_centroid computes cosine via scipy.cdist instead of pdist and runs
# the same zero-vector check itself.
COSINE_MEASURE_NAMES = PDIST_MEASURE_NAMES + ["span_centroid"]


def test_cosine_measure_names_are_registered():
    """Guard against typos: every listed name must be a registered measure."""
    unknown = [name for name in COSINE_MEASURE_NAMES if name not in measures]
    assert not unknown, f"Unregistered measure names in test list: {unknown}"


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

    @pytest.mark.parametrize("name", COSINE_MEASURE_NAMES)
    def test_zero_vector_with_cosine_raises(self, name):
        """A zero vector under the default cosine metric raises instead of nan."""
        with pytest.raises(ValueError, match="Cosine distance is undefined.*row\\(s\\) \\[1\\]"):
            measures[name]([[1.0, 1.0], [0.0, 0.0]])

    def test_zero_centroid_with_cosine_raises_in_span_centroid(self):
        """Nonzero vectors whose mean is the zero vector raise in span_centroid."""
        with pytest.raises(ValueError, match="centroid.*zero vector"):
            span_centroid([[1.0, 0.0], [-1.0, 0.0]])

    def test_zero_vector_with_euclidean_works(self):
        """Zero vectors are fine for metrics that are defined on them."""
        result = mean_pw_dist([[1, 1], [0, 0]], metric="euclidean")["value"]
        assert np.isclose(result, np.sqrt(2))


class TestStringInputRejection:
    """String content is rejected for every registered measure.

    All measures resolve their data through resolve_embeddings, which
    validates numeric input centrally — so each measure raises before any
    measure-specific code runs.
    """

    @staticmethod
    def _letters():
        return [list(row) for row in ("abc", "def", "ghi", "jkl")]

    @staticmethod
    def _numeric_strings():
        return [["0", "1", "0"], ["1", "0", "1"], ["1", "1", "0"], ["0", "0", "1"]]

    @pytest.mark.parametrize("name", sorted(measures))
    def test_letter_strings_raise(self, name):
        """Letter strings raise a ValueError pointing at non-numeric data."""
        with pytest.raises(ValueError, match="must be numeric"):
            measures[name](self._letters())

    @pytest.mark.parametrize("name", sorted(measures))
    def test_numeric_strings_raise(self, name):
        """Number-like strings are rejected rather than silently coerced."""
        with pytest.raises(ValueError, match="contains strings"):
            measures[name](self._numeric_strings())

    def test_mixed_numeric_strings_and_ints_raise(self):
        """Mixing number-like strings with ints is rejected as well."""
        with pytest.raises(ValueError, match="contains strings"):
            mean_pw_dist([["1", 0], ["1", "1"]])

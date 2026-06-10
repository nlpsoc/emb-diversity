import numpy as np
import pytest

from emb_diversity import mean_pw_dist, span_centroid
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure

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
    unknown = [name for name in COSINE_MEASURE_NAMES if name not in MEASURE_NAMES]
    assert not unknown, f"Unregistered measure names in test list: {unknown}"


class TestCosineZeroVectorValidation:
    """Zero vectors under cosine should raise instead of producing nan."""

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_zero_vector_never_yields_nan(self, name):
        """No measure silently returns nan for data containing a zero vector.

        Each measure either raises a ValueError (the cosine-distance
        measures) or returns a finite value (measures whose math is
        defined on zero vectors). This invariant runs over the whole
        registry, so a measure that should be on COSINE_MEASURE_NAMES but
        is missing — or a newly added measure with the nan problem —
        fails here.
        """
        data = np.random.RandomState(0).randn(50, 16)
        data[7] = 0.0
        try:
            value = get_measure(name)(data)["value"]
        except ValueError:
            return
        assert np.isfinite(value)

    @pytest.mark.parametrize("name", COSINE_MEASURE_NAMES)
    def test_zero_vector_with_cosine_raises(self, name):
        """A zero vector under the default cosine metric raises instead of nan."""
        with pytest.raises(ValueError, match="Cosine distance is undefined.*row\\(s\\) \\[1\\]"):
            get_measure(name)([[1.0, 1.0], [0.0, 0.0]])

    def test_zero_centroid_with_cosine_raises_in_span_centroid(self):
        """Nonzero vectors whose mean is the zero vector raise in span_centroid."""
        with pytest.raises(ValueError, match="centroid.*zero vector"):
            span_centroid([[1.0, 0.0], [-1.0, 0.0]])

    def test_zero_vector_with_euclidean_works(self):
        """Zero vectors are fine for metrics that are defined on them."""
        result = mean_pw_dist([[1, 1], [0, 0]], metric="euclidean")["value"]
        assert np.isclose(result, np.sqrt(2))

    def test_nonzero_vectors_with_cosine_work(self):
        """The check does not reject valid cosine input."""
        result = mean_pw_dist([[1, 0], [0, 1]], metric="cosine")["value"]
        assert np.isclose(result, 1.0)

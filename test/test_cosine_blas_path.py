"""
Pytest tests for the blocked-BLAS cosine distance path.

compute_pairwise_distances evaluates cosine distances through scikit-learn's
chunked pairwise-distance path instead of scipy.pdist. These tests pin that
the fast path returns the same condensed array as pdist (shape, order, and
values), across input shapes and including the near-duplicate inputs where a
numerically unsafe shortcut would show.
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from emb_diversity.measures.utils import (
    _condensed_cosine_distances,
    compute_pairwise_distances,
)


class TestCosineBlasPath:
    @pytest.mark.parametrize("n,d", [(2, 3), (10, 384), (50, 10), (300, 8),
                                     (257, 384)])
    def test_matches_pdist(self, n, d):
        """The fast path equals pdist(X, "cosine") across shapes."""
        X = np.random.RandomState(0).randn(n, d)
        expected = pdist(X, metric="cosine")
        result = _condensed_cosine_distances(X)
        assert result.shape == expected.shape
        assert np.allclose(result, expected, atol=1e-12)

    def test_matches_pdist_on_near_duplicates(self):
        """Near-identical rows (tiny true distances) stay accurate."""
        rng = np.random.RandomState(1)
        base = rng.randn(1, 64)
        X = np.vstack([base + rng.randn(1, 64) * 1e-9 for _ in range(20)])
        expected = pdist(X, metric="cosine")
        result = _condensed_cosine_distances(X)
        assert np.allclose(result, expected, atol=1e-12)

    def test_chunking_is_exercised(self, monkeypatch):
        """Results stay correct when the input spans multiple chunks."""
        import sklearn

        X = np.random.RandomState(2).randn(400, 32)
        with sklearn.config_context(working_memory=1e-3):  # tiny chunks
            result = _condensed_cosine_distances(X)
        assert np.allclose(result, pdist(X, metric="cosine"), atol=1e-12)

    def test_compute_pairwise_distances_cosine(self, tmp_path):
        """The public entry point returns pdist-equal cosine distances."""
        X = np.random.RandomState(3).randn(80, 24)
        result = compute_pairwise_distances(X, "cosine", cache_dir=tmp_path)
        assert np.allclose(result, pdist(X, metric="cosine"), atol=1e-12)

    def test_other_metrics_unaffected(self, tmp_path):
        """Non-cosine metrics still match their scipy.pdist results."""
        X = np.random.RandomState(4).randn(40, 12)
        result = compute_pairwise_distances(X, "euclidean", cache_dir=tmp_path)
        assert np.allclose(result, pdist(X, metric="euclidean"))

"""
Pytest tests for the dual-matrix cosine kernel in renyi_entropy and
log_determinant.

Both measures build their cosine kernel through the d x d dual matrix
X'^T X' when it is smaller than the n x n Gram matrix X' X'^T (the two
share their nonzero eigenvalues, trace, and Frobenius norm). These tests
pin the measure values against straightforward reference computations on
the explicit n x n matrix, and assert that the dual route is faster.
"""
import time

import numpy as np
import pytest

from emb_diversity import log_determinant, renyi_entropy


def _vectors(n, d, seed=0):
    return np.random.RandomState(seed).randn(n, d)


def _full_kernel(X):
    """Reference: the explicit n x n kernel the measures are defined on."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


class TestMatchesFullKernel:
    @pytest.mark.parametrize("n,d", [(30, 8), (8, 30)])
    def test_renyi_entropy(self, n, d):
        """Default renyi_entropy (alpha=2) is -log ||K/tr(K)||_F^2."""
        A = _full_kernel(_vectors(n, d))
        A = A / np.trace(A)
        expected = -np.log(np.sum(A * A))
        result = renyi_entropy(_vectors(n, d))["value"]
        assert result == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("n,d", [(30, 8), (8, 30)])
    def test_log_determinant(self, n, d):
        """log_determinant is slogdet of the full K + eps*I."""
        eps = 1e-6
        _, expected = np.linalg.slogdet(
            _full_kernel(_vectors(n, d)) + eps * np.eye(n))
        result = log_determinant(_vectors(n, d), eps=eps)["value"]
        assert result == pytest.approx(expected, abs=1e-6)


class TestDualIsFaster:
    """n=3000, d=16 keeps the reference eigendecomposition (3000 x 3000)
    around a second while the dual route works on 16 x 16 — orders of
    magnitude apart, so the comparison is robust to machine noise."""

    def test_faster_than_full_eigendecomposition(self):
        X = _vectors(3000, 16)

        start = time.perf_counter()
        renyi_entropy(X)
        log_determinant(X)
        dual_time = time.perf_counter() - start

        start = time.perf_counter()
        np.linalg.eigvalsh(_full_kernel(X))
        full_time = time.perf_counter() - start

        assert dual_time < full_time

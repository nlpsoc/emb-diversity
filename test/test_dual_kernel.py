"""
Pytest tests for the use_dual flag of renyi_entropy and log_determinant.

With use_dual=True (the default) the cosine kernel is built through the
d x d dual matrix X'^T X' whenever it is smaller than the n x n Gram
matrix X' X'^T; the two share the nonzero eigenvalues the measures are
computed from. use_dual=False always builds the full n x n kernel, so
running the same measure both ways on the same data checks the dual
route end to end.
"""
import time

import numpy as np
import pytest

from emb_diversity import log_determinant, renyi_entropy


def _vectors(n, d, seed=0):
    return np.random.RandomState(seed).randn(n, d)


class TestDualMatchesFull:
    # All cases use more samples than dimensions (n=1000, d=30): only then
    # does use_dual=True take a different route than use_dual=False.
    # The tolerance is 1e-6 because the full n x n kernel computes its zero
    # eigenvalues as tiny nonzero numbers, which shifts its value by about
    # 1e-7.
    @pytest.mark.parametrize("measure", [
        renyi_entropy,
        log_determinant,
    ])
    def test_same_value(self, measure):
        """use_dual=True and use_dual=False agree on the same data."""
        X = _vectors(1000, 30)
        dual = measure(X, use_dual=True)["value"]
        full = measure(X, use_dual=False)["value"]
        assert dual == pytest.approx(full, abs=1e-6)

    def test_same_value_from_eigenvalues(self):
        """renyi_entropy agrees at alpha=1, where it is computed from eigenvalues."""
        X = _vectors(1000, 30)
        dual = renyi_entropy(X, alpha=1.0, use_dual=True)["value"]
        full = renyi_entropy(X, alpha=1.0, use_dual=False)["value"]
        assert dual == pytest.approx(full, abs=1e-6)


class TestDualIsFaster:
    def test_faster_than_full_kernel(self):
        """The dual route beats the full kernel for n >> d."""
        X = _vectors(3000, 30)

        start = time.perf_counter()
        renyi_entropy(X)
        log_determinant(X)
        dual_time = time.perf_counter() - start

        start = time.perf_counter()
        renyi_entropy(X, use_dual=False)
        log_determinant(X, use_dual=False)
        full_time = time.perf_counter() - start

        assert 2*dual_time < full_time

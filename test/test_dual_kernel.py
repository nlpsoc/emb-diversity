"""
Pytest tests for the use_dual flag of renyi_entropy and log_determinant.

With use_dual=True (the default) the cosine kernel is built through the
d x d dual matrix X'^T X' whenever it is smaller than the n x n Gram
matrix X' X'^T; the two share the eigenvalue spectrum the measures are
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
    # log_determinant is compared at eps=1e-2: with a small eps, the full
    # n x n kernel's structural zero eigenvalues surface as ~1e-15 noise
    # that log(lambda + eps) amplifies by 1/eps (the dual path carries
    # them exactly), which would dominate the comparison.
    @pytest.mark.parametrize("measure,kwargs", [
        (renyi_entropy, {}),
        (log_determinant, {"eps": 1e-2}),
    ])
    @pytest.mark.parametrize("n,d", [(30, 8), (8, 30)])
    def test_same_value(self, measure, kwargs, n, d):
        """use_dual=True and use_dual=False agree on the same data."""
        X = _vectors(n, d)
        dual = measure(X, use_dual=True, **kwargs)["value"]
        full = measure(X, use_dual=False, **kwargs)["value"]
        assert dual == pytest.approx(full, abs=1e-10)


class TestDualIsFaster:
    def test_faster_than_full_kernel(self):
        """The dual route beats the full kernel for n >> d."""
        X = _vectors(3000, 16)

        start = time.perf_counter()
        renyi_entropy(X)
        log_determinant(X)
        dual_time = time.perf_counter() - start

        start = time.perf_counter()
        renyi_entropy(X, use_dual=False)
        log_determinant(X, use_dual=False)
        full_time = time.perf_counter() - start

        assert dual_time < full_time

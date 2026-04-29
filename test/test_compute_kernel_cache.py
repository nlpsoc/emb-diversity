"""
Pytest tests for the kernel matrix cache.

Validates correctness of the cache (same input -> same output, different
inputs -> different cache slots), that the disk cache is reused across
calls, and that the integration with the kernel-based measures
(dcscore, log_determinant_diversity, renyi_kernel_entropy) actually
routes through the cache.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from measure_diversity import (
    compute_kernel_matrix,
    clear_kernel_cache,
    kernel_cache_info,
    dcscore,
    log_determinant_diversity,
    renyi_kernel_entropy,
)

CACHE_DIR = Path(".cache/kernel_test")


@pytest.fixture(autouse=True)
def _isolate_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    clear_kernel_cache(CACHE_DIR)
    yield
    clear_kernel_cache(CACHE_DIR)
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def _data(n=30, d=8, seed=0):
    return np.random.RandomState(seed).randn(n, d)


class TestCorrectness:
    def test_cs_matches_manual_computation(self):
        data = _data()
        K = compute_kernel_matrix(data, kernel_type="cs", tau=2.0, normalize=False, cache_dir=CACHE_DIR)
        expected = (data @ data.T) / 2.0
        assert np.allclose(K, expected)

    def test_cs_normalize_matches_manual(self):
        data = _data()
        K = compute_kernel_matrix(data, kernel_type="cs", tau=1.0, normalize=True, cache_dir=CACHE_DIR)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        Xn = data / norms
        expected = Xn @ Xn.T
        assert np.allclose(K, expected)

    def test_rbf_matches_sklearn(self):
        data = _data()
        K = compute_kernel_matrix(data, kernel_type="rbf", tau=0.5, cache_dir=CACHE_DIR)
        expected = rbf_kernel(data, data, gamma=0.5)
        assert np.allclose(K, expected)

    def test_repeated_call_returns_same_values(self):
        data = _data()
        first = compute_kernel_matrix(data, "rbf", 1.0, cache_dir=CACHE_DIR)
        second = compute_kernel_matrix(data, "rbf", 1.0, cache_dir=CACHE_DIR)
        assert np.allclose(first, second)

    def test_too_few_datapoints_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_kernel_matrix([[1, 2, 3]], "cs", 1.0, cache_dir=CACHE_DIR)

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            compute_kernel_matrix(_data(), "cs", tau=0, cache_dir=CACHE_DIR)

    def test_unknown_kernel_raises(self):
        with pytest.raises(NotImplementedError, match="Unknown kernel_type"):
            compute_kernel_matrix(_data(), "weird", 1.0, cache_dir=CACHE_DIR)

    def test_poly_non_integer_tau_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            compute_kernel_matrix(_data(), "poly", tau=2.5, cache_dir=CACHE_DIR)


class TestCacheKeyIsolation:
    def test_different_kernel_types_dont_collide(self):
        data = _data()
        K_cs = compute_kernel_matrix(data, "cs", 1.0, cache_dir=CACHE_DIR)
        K_rbf = compute_kernel_matrix(data, "rbf", 1.0, cache_dir=CACHE_DIR)
        assert not np.allclose(K_cs, K_rbf)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 2

    def test_different_tau_dont_collide(self):
        data = _data()
        K1 = compute_kernel_matrix(data, "rbf", tau=0.5, cache_dir=CACHE_DIR)
        K2 = compute_kernel_matrix(data, "rbf", tau=2.0, cache_dir=CACHE_DIR)
        assert not np.allclose(K1, K2)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 2

    def test_normalize_flag_dont_collide(self):
        data = _data()
        K_norm = compute_kernel_matrix(data, "cs", 1.0, normalize=True, cache_dir=CACHE_DIR)
        K_raw = compute_kernel_matrix(data, "cs", 1.0, normalize=False, cache_dir=CACHE_DIR)
        assert not np.allclose(K_norm, K_raw)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 2

    def test_different_data_dont_collide(self):
        a = _data(seed=1)
        b = _data(seed=2)
        K_a = compute_kernel_matrix(a, "cs", 1.0, cache_dir=CACHE_DIR)
        K_b = compute_kernel_matrix(b, "cs", 1.0, cache_dir=CACHE_DIR)
        assert not np.allclose(K_a, K_b)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 2


class TestDiskPersistence:
    def test_cold_pass_creates_disk_file(self):
        data = _data()
        compute_kernel_matrix(data, "cs", 1.0, cache_dir=CACHE_DIR)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 1

    def test_warm_pass_reads_from_disk(self):
        data = _data()
        first = compute_kernel_matrix(data, "rbf", 1.0, cache_dir=CACHE_DIR)
        second = compute_kernel_matrix(data, "rbf", 1.0, cache_dir=CACHE_DIR)
        assert np.allclose(first, second)
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 1


class TestMeasuresShareKernel:
    """Integration: dcscore, log_determinant_diversity, and renyi_kernel_entropy
    use the same compute_kernel_matrix under the hood, so calling all three
    on the same data with the same kernel parameters should produce only one
    cache entry."""

    def test_three_measures_share_one_cache_entry(self, monkeypatch):
        # Force the kernel function used by the three measures to use our
        # isolated cache directory.
        from measure_diversity import compute_kernel as ck_module

        original = ck_module.compute_kernel_matrix

        def _patched(data, kernel_type="cs", tau=1.0, normalize=True, cache_dir=None):
            return original(data, kernel_type=kernel_type, tau=tau,
                            normalize=normalize, cache_dir=CACHE_DIR)

        import measure_diversity.measures.dcscore as dc_mod
        import measure_diversity.measures.log_determinant_diversity as ld_mod
        import measure_diversity.measures.renyi_kernel_entropy as re_mod
        for mod in (dc_mod, ld_mod, re_mod):
            monkeypatch.setattr(mod, "compute_kernel_matrix", _patched)

        data = _data()
        v1 = dcscore(data, kernel_type="rbf", tau=1.0)
        v2 = log_determinant_diversity(data, kernel_type="rbf", tau=1.0)
        v3 = renyi_kernel_entropy(data, kernel_type="rbf", tau=1.0)

        assert all(isinstance(v, float) for v in (v1, v2, v3))
        assert kernel_cache_info(CACHE_DIR)["disk_files"] == 1

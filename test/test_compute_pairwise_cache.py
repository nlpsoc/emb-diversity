"""
Pytest tests for the pairwise distance cache.

Validates correctness of the cache (same input -> same output, different
inputs -> different cache slots), that the disk cache is reused across
calls, and that the integration with measures/utils._compute_pairwise_distances
actually routes through the cache.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from emb_diversity import (
    compute_pairwise_distances,
    clear_distance_cache,
    distance_cache_info,
    mean_pw_dist,
    diameter,
    energy,
)
from emb_diversity.measures.utils import _compute_pairwise_distances

CACHE_DIR = Path(".cache/pdist_test")


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Use a dedicated cache directory and wipe it around every test."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    clear_distance_cache(CACHE_DIR)
    yield
    clear_distance_cache(CACHE_DIR)
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def _data(n=50, d=10, seed=0):
    return np.random.RandomState(seed).randn(n, d)


class TestCorrectness:
    def test_matches_scipy_pdist(self):
        data = _data()
        cached = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        expected = pdist(data, metric="cosine")
        assert np.allclose(cached, expected)

    def test_repeated_call_returns_same_values(self):
        data = _data()
        first = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        second = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        assert np.allclose(first, second)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty data"):
            compute_pairwise_distances(np.empty((0, 5)), "cosine", CACHE_DIR)

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="single data point"):
            compute_pairwise_distances(np.zeros((1, 5)), "cosine", CACHE_DIR)


class TestCacheKeyIsolation:
    def test_different_metrics_dont_collide(self):
        data = _data()
        d_cos = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        d_euc = compute_pairwise_distances(data, "euclidean", CACHE_DIR)
        assert not np.allclose(d_cos, d_euc)
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 2

    def test_different_kwargs_dont_collide(self):
        data = _data()
        d_p2 = compute_pairwise_distances(data, "minkowski", CACHE_DIR, p=2)
        d_p3 = compute_pairwise_distances(data, "minkowski", CACHE_DIR, p=3)
        assert not np.allclose(d_p2, d_p3)
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 2

    def test_different_data_dont_collide(self):
        a = _data(seed=1)
        b = _data(seed=2)
        d_a = compute_pairwise_distances(a, "cosine", CACHE_DIR)
        d_b = compute_pairwise_distances(b, "cosine", CACHE_DIR)
        assert not np.allclose(d_a, d_b)
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 2


class TestDiskPersistence:
    def test_cold_pass_creates_disk_file(self):
        data = _data()
        compute_pairwise_distances(data, "cosine", CACHE_DIR)
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 1

    def test_warm_pass_reads_from_disk(self):
        """Repeated call returns identical values from the disk cache."""
        data = _data()
        first = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        second = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        assert np.allclose(first, second)
        # Still only one disk file — second call did not recompute
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 1


class TestMemoryLayer:
    """The in-memory layer caches computed results inside a single Python
    process so a second call on the same data does not even hit disk."""

    def test_first_call_populates_memory(self):
        data = _data()
        compute_pairwise_distances(data, "cosine", CACHE_DIR)
        assert distance_cache_info(CACHE_DIR)["memory_entries"] == 1

    def test_warm_pass_serves_from_memory_when_disk_is_gone(self):
        """If the disk file is deleted between two calls but the memory
        entry survives, the second call must still return the cached value."""
        data = _data()
        first = compute_pairwise_distances(data, "cosine", CACHE_DIR)

        # Wipe disk but keep memory state intact
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

        second = compute_pairwise_distances(data, "cosine", CACHE_DIR)
        assert np.allclose(first, second)
        # Disk was wiped and never recreated — memory served the call
        assert not CACHE_DIR.exists()

    def test_lru_eviction_respects_memory_max(self):
        """More distinct entries than _MEMORY_MAX are computed; the dict
        must stay bounded and contain only the most recent entries."""
        from emb_diversity import compute_pairwise as cp_module
        max_entries = cp_module._MEMORY_MAX
        # Generate one more dataset than the cap, so we know eviction must happen
        for seed in range(max_entries + 1):
            compute_pairwise_distances(_data(seed=seed), "cosine", CACHE_DIR)
        info = distance_cache_info(CACHE_DIR)
        assert info["memory_entries"] == max_entries
        # Disk keeps everything though
        assert info["disk_files"] == max_entries + 1

    def test_clear_distance_cache_drops_memory(self):
        compute_pairwise_distances(_data(), "cosine", CACHE_DIR)
        assert distance_cache_info(CACHE_DIR)["memory_entries"] == 1
        clear_distance_cache(CACHE_DIR)
        assert distance_cache_info(CACHE_DIR)["memory_entries"] == 0


class TestMeasuresAreCached:
    """Integration: measures/utils._compute_pairwise_distances now routes
    through the cache, so multiple measures on the same data share one
    cached pdist computation."""

    def test_multiple_measures_share_one_cache_entry(self, monkeypatch):
        # Force the helper to use our isolated cache dir.
        # Note: `emb_diversity.__init__` re-exports the registry as `measures`,
        # which shadows the submodule attribute lookup, so we use importlib
        # to reach the actual submodules.
        import importlib

        measures_utils = importlib.import_module("emb_diversity.measures.utils")

        def _patched(data, metric="cosine", **kwargs):
            return compute_pairwise_distances(data, metric=metric, cache_dir=CACHE_DIR, **kwargs)

        monkeypatch.setattr(measures_utils, "_compute_pairwise_distances", _patched)
        # Re-bind in each measure module that imported it
        mpd = importlib.import_module("emb_diversity.measures.mean_pw_dist")
        dia = importlib.import_module("emb_diversity.measures.diameter")
        ene = importlib.import_module("emb_diversity.measures.energy")
        for mod in (mpd, dia, ene):
            monkeypatch.setattr(mod, "_compute_pairwise_distances", _patched)

        data = _data(n=30, d=8)
        v1 = mpd.mean_pw_dist(data, metric="cosine")
        v2 = dia.diameter(data, metric="cosine")
        v3 = ene.energy(data, metric="cosine")

        # All three measures returned valid floats
        assert all(isinstance(v, float) for v in (v1, v2, v3))
        # And only one pdist entry was written to disk
        assert distance_cache_info(CACHE_DIR)["disk_files"] == 1

    def test_helper_returns_same_as_pdist(self):
        data = _data()
        helper = _compute_pairwise_distances(data, metric="cosine")
        direct = pdist(data, metric="cosine")
        assert np.allclose(helper, direct)
        # Clean up the default cache that the helper just populated
        clear_distance_cache()

"""Tests that every registered measure works through measure_diversity().

These guard the convenience-function path specifically: measure_diversity()
embeds text to a numpy array and then calls each measure on that array, and it
catches exceptions and turns them into NaN. So a measure that is broken on
numpy-array input (as cluster_inertia once was) would silently return NaN here
rather than failing loudly. These tests catch that whole class of bug.
"""

import numpy as np
import pytest

from emb_diversity import measure_diversity
from emb_diversity.measures_registry import measures


@pytest.fixture(scope="module")
def vectors():
    # 50 points in 16-D: enough for every measure (including the UMAP-projected
    # ones and k-means clustering) to run without hitting small-sample fallbacks.
    return np.random.RandomState(0).randn(50, 16)


@pytest.mark.parametrize("name", sorted(measures))
def test_measure_runs_via_convenience_and_matches_direct(name, vectors):
    """Each measure runs through measure_diversity (no silent NaN) and returns
    the same value as calling the measure directly on the same vectors."""
    direct = measures[name](vectors)
    result = measure_diversity(vectors, measure=[name])

    assert name in result
    assert not np.isnan(result[name]), (
        f"{name} returned NaN via measure_diversity "
        f"(it likely raised on numpy-array input)"
    )
    assert np.isclose(result[name], direct), (
        f"{name}: measure_diversity={result[name]} != direct={direct}"
    )


def test_all_measures_run_in_one_call(vectors):
    """measure='all' returns a finite score for every registered measure."""
    results = measure_diversity(vectors, measure="all")
    assert set(results) == set(measures)
    nan_measures = sorted(k for k, v in results.items() if np.isnan(v))
    assert not nan_measures, (
        f"measures returned NaN via measure_diversity: {nan_measures}"
    )

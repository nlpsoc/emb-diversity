import warnings

import numpy as np
import pytest

from emb_diversity.measures._umap import fit_transform_umap


class _FakeReducer:
    """Stands in for ``umap.UMAP`` — emits a chosen warning, then returns 2D."""

    def __init__(self, message):
        self._message = message

    def fit_transform(self, X):
        warnings.warn(self._message, UserWarning)
        return np.asarray(X)[:, :2]


class TestFitTransformUmap:
    """fit_transform_umap rewords UMAP's small-dataset warning, passes others."""

    @staticmethod
    def _data():
        """A small toy matrix (more columns than the 2D output)."""
        return np.random.RandomState(0).randn(3, 5)

    def test_small_dataset_warning_is_reworded(self):
        """UMAP's n_neighbors warning becomes a clear 'use a bigger dataset' one."""
        reducer = _FakeReducer(
            "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
        )
        with pytest.warns(UserWarning, match="bigger dataset"):
            fit_transform_umap(reducer, self._data())

    def test_reworded_warning_reports_the_point_count(self):
        """The reworded message names how many points were given."""
        reducer = _FakeReducer("n_neighbors is larger than the dataset size; truncating")
        with pytest.warns(UserWarning, match=r"\(3 points\)"):
            fit_transform_umap(reducer, self._data())

    def test_other_warning_passes_through_unchanged(self):
        """An unrelated UMAP warning is re-emitted verbatim, not swallowed."""
        reducer = _FakeReducer(
            "n_jobs value 1 overridden to 1 by setting random_state."
        )
        with pytest.warns(UserWarning, match="n_jobs value 1 overridden"):
            fit_transform_umap(reducer, self._data())

    def test_returns_the_projection(self):
        """The reducer's output is returned regardless of the warning."""
        reducer = _FakeReducer("n_neighbors is larger than the dataset size")
        with pytest.warns(UserWarning):
            out = fit_transform_umap(reducer, self._data())
        assert out.shape == (3, 2)

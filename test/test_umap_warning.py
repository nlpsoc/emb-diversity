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
        """UMAP's n_neighbors warning becomes a clear 'bigger dataset' one, naming n."""
        reducer = _FakeReducer(
            "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
        )
        with pytest.warns(UserWarning) as record:
            fit_transform_umap(reducer, self._data())
        msg = str(record[0].message)
        assert "bigger dataset" in msg and "3 points" in msg

    def test_other_warning_passes_through_unchanged(self):
        """An unrelated UMAP warning is re-emitted verbatim, not swallowed."""
        reducer = _FakeReducer("n_jobs value 1 overridden to 1 by setting random_state.")
        with pytest.warns(UserWarning, match="n_jobs value 1 overridden"):
            fit_transform_umap(reducer, self._data())

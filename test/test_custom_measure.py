"""Tests that a user can pass their own measure callable to measure_diversity.

A custom measure follows the same contract as a built-in: it is called as
``fn(data, diversity_axis=..., embedding_model=...)`` and returns a
``{"value": float, "parameters": {...}}`` dict. It is run as given — no
validation is applied to it.
"""

import numpy as np
import pytest

from emb_diversity import measure_diversity
from emb_diversity.embed import resolve_embeddings


def my_spread(data, *, diversity_axis="semantic", embedding_model=None):
    """A toy custom measure: standard deviation of the (embedded) vectors."""
    vectors, model = resolve_embeddings(data, diversity_axis, embedding_model)
    return {"value": float(np.std(vectors)), "parameters": {"embedding_model": model}}


class TestCustomMeasure:

    @staticmethod
    def _vectors():
        return np.random.RandomState(0).randn(30, 8)

    def test_callable_runs_and_is_keyed_by_name(self):
        """A custom measure callable runs and is keyed by its __name__."""
        out = measure_diversity(self._vectors(), measure=my_spread)
        assert set(out) == {"my_spread"}
        assert np.isfinite(out["my_spread"]["value"])

    def test_mixed_list_of_names_and_callables(self):
        """A list may mix built-in names and custom callables."""
        out = measure_diversity(self._vectors(), measure=["mean_pw_dist", my_spread])
        assert set(out) == {"mean_pw_dist", "my_spread"}

    def test_unknown_string_name_still_raises(self):
        """An unknown string name still fails fast with KeyError."""
        with pytest.raises(KeyError):
            measure_diversity(self._vectors(), measure="not_a_measure")

    def test_failing_custom_measure_is_reported(self):
        """A raising custom measure is recorded as nan + warning, like a built-in."""
        def boom(data, *, diversity_axis="semantic", embedding_model=None):
            raise RuntimeError("nope")

        with pytest.warns(UserWarning, match="boom.*nope"):
            out = measure_diversity(self._vectors(), measure=boom)
        assert np.isnan(out["boom"]["value"])
        assert "nope" in out["boom"]["error"]

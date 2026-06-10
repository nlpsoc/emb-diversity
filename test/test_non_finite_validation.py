import numpy as np
import pytest

from emb_diversity import mean_pw_dist
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure


class TestNonFiniteInputValidation:
    """nan/inf values should raise instead of silently producing nan.

    The check runs in resolve_embeddings, the funnel every measure passes
    its data through — parametrizing over the registry means any newly
    added measure is covered (and tested) automatically.
    """

    @staticmethod
    def _with_nan():
        data = np.random.RandomState(0).rand(5, 4)
        data[2, 1] = np.nan
        return data

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_nan_value_raises_naming_the_row(self, name):
        """A nan cell raises a ValueError naming the offending row."""
        with pytest.raises(ValueError, match=r"non-finite.*row\(s\) \[2\]"):
            get_measure(name)(self._with_nan())

    def test_inf_value_raises(self):
        """An inf cell raises a ValueError mentioning non-finite values."""
        data = np.random.RandomState(0).rand(5, 4)
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            mean_pw_dist(data)  # only test for one, should be the same for all measures

    def test_many_bad_rows_are_truncated_in_message(self):
        """With more than 10 bad rows, the message is truncated with a count."""
        data = np.full((15, 3), np.nan)
        with pytest.raises(ValueError, match=r"\(\+5 more\)"):
            mean_pw_dist(data)  # only test for one, should be the same for all measures

    def test_nan_embeddings_from_model_raise(self, monkeypatch):
        """Embedded text is checked too: a model emitting nan raises.

        embed_texts runs its output through to_numeric_array, the same
        exit gate vector input passes through.
        """
        import emb_diversity.embed as embed_module

        def fake_encode(texts, model_name):
            vectors = np.zeros((len(texts), 4))
            vectors[1, 2] = np.nan
            return vectors

        monkeypatch.setattr(embed_module, "encode", fake_encode)
        with pytest.raises(ValueError, match=r"non-finite.*row\(s\) \[1\]"):
            mean_pw_dist(["a text", "another text", "a third text"])

    def test_single_text_raises_sample_minimum(self, monkeypatch):
        """One text hits the shared 2-sample minimum, like one vector would.

        The fake encode keeps the test offline; the minimum fires on the
        embedded output inside embed_texts before any measure code runs.
        """
        import emb_diversity.embed as embed_module

        def fake_encode(texts, model_name):
            return np.ones((len(texts), 4))  # ← ONE row = one sample

        monkeypatch.setattr(embed_module, "encode", fake_encode)
        with pytest.raises(ValueError, match="at least 2 samples, got 1"):
            mean_pw_dist(["just one text"])

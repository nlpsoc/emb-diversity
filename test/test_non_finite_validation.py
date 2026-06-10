import numpy as np
import pytest

from emb_diversity import mean_pw_dist
from emb_diversity.measures_registry import measures


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

    @pytest.mark.parametrize("name", sorted(measures))
    def test_nan_value_raises_naming_the_row(self, name):
        """A nan cell raises a ValueError naming the offending row."""
        with pytest.raises(ValueError, match=r"non-finite.*row\(s\) \[2\]"):
            measures[name](self._with_nan())

    def test_inf_value_raises(self):
        """An inf cell raises a ValueError mentioning non-finite values."""
        data = np.random.RandomState(0).rand(5, 4)
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            mean_pw_dist(data)

    def test_many_bad_rows_are_truncated_in_message(self):
        """With more than 10 bad rows, the message is truncated with a count."""
        data = np.full((15, 3), np.nan)
        with pytest.raises(ValueError, match=r"\(\+5 more\)"):
            mean_pw_dist(data)

import pytest

from emb_diversity import mean_pw_dist
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure


class TestStringVectorRejection:
    """Vectors containing strings are rejected for every registered measure.

    "vector" input (or list of lists) must be numeric.
    Flat list of strings is accepted as text input and is embedded as usual (see the last test).
    All measures resolve their data through resolve_embeddings, which validates vector input
    centrally. So each measure raises before any measure-specific code runs.
    """

    @staticmethod
    def _letters():  # list of list of strings -> should fail
        return [list(row) for row in ("abc", "def", "ghi", "jkl")]

    @staticmethod
    def _numeric_strings():
        return [["0", "1", "0"], ["1", "0", "1"], ["1", "1", "0"], ["0", "0", "1"]]

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_letter_strings_raise(self, name):
        """Letter strings raise a ValueError pointing at non-numeric data."""
        with pytest.raises(ValueError, match="must be numeric"):
            get_measure(name)(self._letters())

    @pytest.mark.parametrize("name", sorted(MEASURE_NAMES))
    def test_numeric_strings_raise(self, name):
        """Number-like strings are rejected rather than silently coerced."""
        with pytest.raises(ValueError, match="not converted automatically"):
            get_measure(name)(self._numeric_strings())

    def test_mixed_numeric_strings_and_ints_raise(self):
        """Mixing number-like strings with ints is rejected as well."""
        with pytest.raises(ValueError, match="not converted automatically"):
            mean_pw_dist([["1", 0], ["1", "1"]])

    def test_flat_list_of_strings_is_text_input(self):
        """A flat list of strings is accepted as text input, not rejected.

        It heads for embedding instead of the numeric validation (this stays
        cheap to assert: a fake axis fails before any model loads).
        """
        with pytest.raises(KeyError, match="no-such-axis"):
            mean_pw_dist(["a text", "another text"], diversity_axis="no-such-axis")

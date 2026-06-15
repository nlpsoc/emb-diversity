import warnings

import numpy as np
import pytest

from emb_diversity import dcscore, log_determinant, renyi_entropy, vendi_score
from emb_diversity.measures_registry import MEASURE_NAMES

# Cosine-kernel measures that L2-normalize rows themselves (kernel_type="cs"
# with normalize=True). Unlike the pdist cosine measures, which raise on an
# all-zero row, these warn and still return a finite value.
KERNEL_MEASURES = {
    "dcscore": dcscore,
    "log_determinant": log_determinant,
    "renyi_entropy": renyi_entropy,
    "vendi_score": vendi_score,
}


def test_kernel_measure_names_are_registered():
    """Guard against typos: every listed name must be a registered measure."""
    unknown = [name for name in KERNEL_MEASURES if name not in MEASURE_NAMES]
    assert not unknown, f"Unregistered measure names in test list: {unknown}"


class TestKernelZeroNormWarning:
    """All-zero rows warn (not raise) in the L2-normalizing cosine kernels."""

    @staticmethod
    def _data():
        """A well-conditioned matrix with a single all-zero row."""
        data = np.random.RandomState(0).randn(20, 8)
        data[3] = 0.0
        return data

    @pytest.mark.parametrize("name", sorted(KERNEL_MEASURES))
    def test_zero_norm_warns_and_returns_finite(self, name):
        """A zero row warns but the score is still a finite float, not nan."""
        with pytest.warns(UserWarning, match="all-zero"):
            value = KERNEL_MEASURES[name](self._data())["value"]
        assert np.isfinite(value)

    @pytest.mark.parametrize("name", sorted(KERNEL_MEASURES))
    def test_warning_reports_the_zero_row_index(self, name):
        """The warning names the offending row index."""
        with pytest.warns(UserWarning, match=r"row\(s\) \[3\]"):
            KERNEL_MEASURES[name](self._data())

    @pytest.mark.parametrize("name", sorted(KERNEL_MEASURES))
    def test_no_warning_without_zero_rows(self, name):
        """Valid input does not warn."""
        data = np.random.RandomState(1).randn(20, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KERNEL_MEASURES[name](data)

    @pytest.mark.parametrize("name", sorted(KERNEL_MEASURES))
    def test_normalize_false_does_not_warn(self, name):
        """With normalize=False no row normalization happens, so no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KERNEL_MEASURES[name](self._data(), normalize=False)

    def test_vendi_explicit_path_also_warns(self):
        """The non-dual vendi path warns identically to the dual one."""
        with pytest.warns(UserWarning, match="all-zero"):
            value = vendi_score(self._data(), use_dual=False)["value"]
        assert np.isfinite(value)

    @pytest.mark.parametrize("kernel_type", ["rbf", "lap"])
    def test_non_cosine_kernel_does_not_warn(self, kernel_type):
        """Non-cosine kernels are defined on zero vectors, so they do not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dcscore(self._data(), kernel_type=kernel_type)

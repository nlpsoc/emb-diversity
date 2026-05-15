"""
Tests for the multi-dataset Magnitude Area (mag_areas) diversity measure.

mag_areas is a thin wrapper around magnipy.Diversipy that constrains the
API so the returned values are always cross-dataset comparable. The tests
therefore cover both:
  - input validation that prevents the un-comparable single-dataset call
  - numeric agreement with magnipy.Diversipy on the same inputs
  - the headline diversity ordering claim (spread > tight) for euclidean.
"""
import numpy as np
import pytest

# Apply the same scipy 1.16 compatibility shim that mag_areas does, so any
# direct ``magnipy`` import below succeeds.
import scipy.integrate as _scipy_integrate
if not hasattr(_scipy_integrate, "trapz"):
    _scipy_integrate.trapz = _scipy_integrate.trapezoid

from magnipy import Diversipy

from embediver import mag_areas


def _toy(seed=0, n=80, d=16, scale=1.0):
    return np.random.RandomState(seed).randn(n, d) * scale


class TestInputValidation:
    def test_single_dataset_raises(self):
        with pytest.raises(ValueError, match="at least 2 datasets"):
            mag_areas([_toy()])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least 2 datasets"):
            mag_areas([])

    def test_non_list_raises(self):
        # A single 2D ndarray must not be silently treated as a list of rows.
        with pytest.raises(ValueError, match="list/tuple of datasets"):
            mag_areas(_toy())

    def test_dim_mismatch_raises(self):
        a = _toy(seed=1, d=10)
        b = _toy(seed=2, d=12)
        with pytest.raises(ValueError, match="embedding dim"):
            mag_areas([a, b])

    def test_single_row_dataset_raises(self):
        a = _toy(seed=1)
        b = np.zeros((1, 16))
        with pytest.raises(ValueError, match="at least 2 rows"):
            mag_areas([a, b])

    def test_not_2d_raises(self):
        a = _toy(seed=1)
        b = np.zeros(16)  # 1D
        with pytest.raises(ValueError, match="expected 2D array"):
            mag_areas([a, b])

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            mag_areas([_toy(seed=1), _toy(seed=2)], names=["only_one"])

    def test_n_ts_too_small_raises(self):
        with pytest.raises(ValueError, match="n_ts"):
            mag_areas([_toy(seed=1), _toy(seed=2)], n_ts=1)


class TestReturnShape:
    def test_returns_list_of_floats(self):
        result = mag_areas([_toy(seed=1), _toy(seed=2), _toy(seed=3)],
                           metric="euclidean")
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_length_matches_input(self):
        Xs = [_toy(seed=i) for i in range(5)]
        result = mag_areas(Xs, metric="euclidean")
        assert len(result) == 5


class TestMatchesDiversipy:
    """The function is a thin wrapper, so on the same input it must agree
    with calling magnipy.Diversipy directly."""

    def test_default_metric_cosine(self):
        Xs = [_toy(seed=i) for i in range(3)]
        ours = mag_areas(Xs, metric="cosine", n_ts=20)
        reference = list(Diversipy(Xs=[np.asarray(x, dtype=float) for x in Xs],
                                    metric="cosine", n_ts=20).MagAreas())
        assert np.allclose(ours, [float(v) for v in reference])

    def test_euclidean(self):
        Xs = [_toy(seed=i) for i in range(3)]
        ours = mag_areas(Xs, metric="euclidean", n_ts=20)
        reference = list(Diversipy(Xs=[np.asarray(x, dtype=float) for x in Xs],
                                    metric="euclidean", n_ts=20).MagAreas())
        assert np.allclose(ours, [float(v) for v in reference])

    def test_names_are_forwarded(self):
        # If names are accepted and forwarded without raising, the function
        # is at least wiring them through.
        Xs = [_toy(seed=i) for i in range(3)]
        result = mag_areas(Xs, metric="euclidean", n_ts=20,
                           names=["a", "b", "c"])
        assert len(result) == 3


class TestDiversityOrdering:
    """The headline contract: under euclidean (a negative-type metric),
    a more spread-out point cloud has a higher MagArea on the common scale
    axis than a tighter one. Cosine is scale-invariant so we don't test
    ordering there."""

    def test_spread_more_than_tight_euclidean(self):
        rng = np.random.RandomState(0)
        spread = rng.randn(80, 16) * 3.0
        medium = rng.randn(80, 16) * 1.0
        tight = rng.randn(80, 16) * 0.05
        areas = mag_areas([spread, medium, tight], metric="euclidean",
                          names=["spread", "medium", "tight"])
        assert areas[0] > areas[1] > areas[2]

    def test_duplicate_datasets_give_equal_magareas(self):
        # Two passes of the same data must produce two identical MagAreas.
        X = _toy(seed=42)
        areas = mag_areas([X, X.copy()], metric="euclidean", n_ts=20)
        assert np.isclose(areas[0], areas[1], rtol=1e-6)

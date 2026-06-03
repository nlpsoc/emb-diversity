"""Tests for measure-name resolution and the named measure sets."""

from emb_diversity.convenience import _resolve_measure_names
from emb_diversity.measures_registry import (
    DEFAULT_MEASURE,
    MEASURE_SETS,
    measures,
)


class TestMeasureResolution:

    def test_none_returns_default(self):
        assert _resolve_measure_names(None) == list(DEFAULT_MEASURE)
        assert _resolve_measure_names(None) == ["graph_entropy"]

    def test_named_sets(self):
        assert _resolve_measure_names("variety") == [
            "chamfer_dist", "sum_bottleneck", "mst_dispersion"
        ]
        assert _resolve_measure_names("balance") == ["graph_entropy"]
        assert _resolve_measure_names("disparity") == [
            "graph_entropy", "mst_dispersion"
        ]

    def test_all_returns_every_registered_measure(self):
        assert _resolve_measure_names("all") == list(measures)

    def test_single_name_passthrough(self):
        assert _resolve_measure_names("energy") == ["energy"]

    def test_list_passthrough(self):
        assert _resolve_measure_names(["diameter", "radius"]) == ["diameter", "radius"]

    def test_unknown_name_passes_through_unresolved(self):
        # Resolution does not validate; measure_diversity() raises later.
        assert _resolve_measure_names("not_a_measure") == ["not_a_measure"]


class TestMeasureSetIntegrity:

    def test_default_members_are_registered(self):
        for name in DEFAULT_MEASURE:
            assert name in measures, f"DEFAULT_MEASURE references unknown measure {name!r}"

    def test_every_set_member_is_registered(self):
        for set_name, members in MEASURE_SETS.items():
            for name in members:
                assert name in measures, (
                    f"set {set_name!r} references unknown measure {name!r}"
                )

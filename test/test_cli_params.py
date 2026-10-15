"""Tests for the CLI --param option (measure parameter overrides)."""

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from emb_diversity.cli import _configure_measure, _parse_params, app
from emb_diversity.convenience import measure_diversity

runner = CliRunner()


class TestParseParams:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (["k=5"], {"k": 5}),
            (["tau=0.5"], {"tau": 0.5}),
            (["normalize=false"], {"normalize": False}),
            (["metric=euclidean"], {"metric": "euclidean"}),
            (["k=3", "metric=cityblock"], {"k": 3, "metric": "cityblock"}),
        ],
    )
    def test_coercion(self, raw, expected):
        """key=value strings are coerced to int/float/bool/str."""
        assert _parse_params(raw) == expected

    def test_missing_equals_exits(self):
        """A --param without '=' exits with an error."""
        with pytest.raises(typer.Exit):
            _parse_params(["k5"])


class TestConfigureMeasure:
    def test_override_reaches_measure(self):
        """The bound override is used and recorded in the result parameters."""
        configured = _configure_measure("knn", ["k=3"])
        vectors = np.random.RandomState(0).rand(10, 4)
        results = measure_diversity(vectors, measure=configured)
        assert results["knn"]["parameters"]["k"] == 3
        assert "version" in results["knn"]

    def test_unknown_param_exits(self):
        """A parameter the measure does not accept exits with an error."""
        with pytest.raises(typer.Exit):
            _configure_measure("vendi_score", ["k=3"])

    def test_reserved_param_exits(self):
        """Embedding arguments must go through their dedicated options."""
        with pytest.raises(typer.Exit):
            _configure_measure("knn", ["embedding_model=foo"])


class TestParamCli:
    @staticmethod
    def _input_file(tmp_path):
        path = tmp_path / "texts.txt"
        path.write_text("The cat sat.\nDogs play fetch.\n")
        return str(path)

    def test_param_requires_single_measure(self, tmp_path):
        """--param with several measures is rejected."""
        result = runner.invoke(
            app,
            ["measure", self._input_file(tmp_path),
             "-m", "knn", "-m", "diameter", "-p", "k=3"],
        )
        assert result.exit_code == 1

    def test_param_rejects_measure_set(self, tmp_path):
        """--param with a named set is rejected."""
        result = runner.invoke(
            app,
            ["measure", self._input_file(tmp_path),
             "-m", "variety", "-p", "k=3"],
        )
        assert result.exit_code == 1

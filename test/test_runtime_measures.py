"""Tests for adding a measure at runtime via the ``@measure`` decorator.

A user can define a measure in their own code — a function that receives a
validated ``(n, d)`` array and returns a ``{"value": float, "parameters": {...}}``
dict — decorate it with ``@measure``, and run it through ``measure_diversity``
either by passing the callable directly or by registering it under a name. The
decorator runs the shared input validation, checks the result shape, and records
the ``embedding_model``, so a runtime measure behaves exactly like a built-in.
"""

import numpy as np
import pytest

from emb_diversity import measure, measure_diversity, register_measure
from emb_diversity.measures_registry import MEASURE_NAMES, _REGISTRY, get_measure


@pytest.fixture(autouse=True)
def _clear_registry():
    """Keep runtime registrations from leaking between tests."""
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


@measure
def _spread(vectors, scale=1.0):
    """A toy measure: the (scaled) standard deviation of the vectors."""
    return {"value": float(np.std(vectors)) * scale, "parameters": {"scale": scale}}


class TestRuntimeMeasure:

    @staticmethod
    def _vectors():
        return np.random.RandomState(0).randn(40, 12)

    def test_decorated_measure_returns_standard_envelope(self):
        """A @measure callable returns {"value", "parameters"} with the model id."""
        result = _spread(self._vectors())
        assert set(result) == {"value", "parameters"}
        assert isinstance(result["value"], float) and np.isfinite(result["value"])
        # Vector input was not embedded, so the model id is None.
        assert result["parameters"] == {"scale": 1.0, "embedding_model": None}

    def test_decorated_measure_forwards_its_own_parameters(self):
        """Measure-specific parameters stay call-time keyword/positional args."""
        assert _spread(self._vectors(), scale=2.0)["parameters"]["scale"] == 2.0

    def test_run_by_callable_via_convenience(self):
        """measure_diversity accepts the callable directly, keyed by its name."""
        X = self._vectors()
        via = measure_diversity(X, measure=_spread)
        assert set(via) == {"_spread"}
        assert np.isclose(via["_spread"]["value"], _spread(X)["value"])

    def test_register_then_run_by_name(self):
        """After register_measure, the name resolves through get_measure."""
        register_measure(_spread)
        assert get_measure("_spread") is _spread
        via = measure_diversity(self._vectors(), measure="_spread")
        assert "_spread" in via

    def test_mixed_list_of_names_and_callables(self):
        """A list may mix built-in names and runtime callables."""
        via = measure_diversity(self._vectors(), measure=["mean_pw_dist", _spread])
        assert set(via) == {"mean_pw_dist", "_spread"}

    def test_register_plain_core_function_wraps_it(self):
        """register_measure wraps a plain (undecorated) core function."""
        def my_core(vectors, k=3):
            return {"value": float(vectors.shape[0] * k), "parameters": {"k": k}}

        register_measure(my_core, name="my_core")
        result = measure_diversity(self._vectors(), measure="my_core")["my_core"]
        assert result["parameters"] == {"k": 3, "embedding_model": None}


class TestRuntimeMeasureValidation:

    def test_decorated_measure_validates_input(self):
        """A runtime measure runs input through the shared validation."""
        # 1-D input is rejected by resolve_embeddings inside the wrapper.
        with pytest.raises(ValueError, match=r"2-dimensional"):
            _spread([0.0, 0.0])
        # A bare string is rejected too.
        with pytest.raises(ValueError, match="Wrap it in a list"):
            _spread("just one text")

    def test_bad_return_shape_rejected(self):
        """The wrapper rejects a core function that returns the wrong shape."""
        @measure
        def _wrong(vectors):
            return 1.0  # not a {"value", "parameters"} dict

        with pytest.raises(TypeError, match="must return a dict"):
            _wrong(np.zeros((3, 2)))

    def test_bad_signature_rejected_at_decoration(self):
        """@measure rejects a function without a leading positional parameter."""
        with pytest.raises(TypeError, match="first positional parameter"):
            @measure
            def _bad(*, only_kw=1):
                return 1.0, {}

    def test_bare_callable_rejected_by_convenience(self):
        """A non-@measure callable can't bypass the envelope."""
        def bare(vectors):
            return 1.0, {}

        with pytest.raises(TypeError, match="is not a measure"):
            measure_diversity(np.zeros((3, 2)), measure=bare)


class TestRuntimeMeasureCollisions:

    def test_clash_with_builtin_raises(self):
        """Registering over a built-in name is refused."""
        with pytest.raises(ValueError, match="already exists"):
            register_measure(_spread, name="graph_entropy")
        assert "graph_entropy" not in _REGISTRY

    def test_clash_with_existing_runtime_measure_raises(self):
        """Registering the same runtime name twice is refused (no override)."""
        register_measure(_spread, name="dup")
        with pytest.raises(ValueError, match="already exists"):
            register_measure(_spread, name="dup")

    def test_builtin_catalog_is_unchanged(self):
        """Runtime registration never mutates the static built-in catalog."""
        register_measure(_spread, name="extra")
        assert "extra" not in MEASURE_NAMES

"""Tests for the model-loading spinner feedback (no network / model loads)."""

import logging

import pytest

from emb_diversity.utility import _progress
from emb_diversity.utility._progress import (
    _env_override,
    load_with_spinner,
    progress_enabled,
)


class TestEnvOverride:
    """The EMB_DIVERSITY_PROGRESS environment variable forces the feedback."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable(self, monkeypatch, value):
        """Recognised truthy strings force the feedback on."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", value)
        assert _env_override() is True
        assert progress_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "No", "off"])
    def test_falsy_values_disable(self, monkeypatch, value):
        """Recognised falsy strings force the feedback off."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", value)
        assert _env_override() is False
        assert progress_enabled() is False

    def test_unset_returns_none(self, monkeypatch):
        """With the variable unset, there is no override."""
        monkeypatch.delenv("EMB_DIVERSITY_PROGRESS", raising=False)
        assert _env_override() is None

    def test_unrecognised_value_returns_none(self, monkeypatch):
        """An unparseable value is treated as no override."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "maybe")
        assert _env_override() is None


class TestLoadWithSpinner:
    """load_with_spinner runs the loader exactly once and returns its result."""

    def test_returns_loader_result_when_disabled(self, monkeypatch):
        """When disabled, the loader result is returned untouched."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "0")
        calls = []
        result = load_with_spinner("some-model", lambda: calls.append(1) or "model")
        assert result == "model"
        assert len(calls) == 1

    def test_returns_loader_result_when_enabled(self, monkeypatch):
        """When enabled, the spinner wraps the call and returns its result."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")
        calls = []
        result = load_with_spinner("some-model", lambda: calls.append(1) or "model")
        assert result == "model"
        assert len(calls) == 1

    def test_loader_runs_once_only(self, monkeypatch):
        """The loader is never invoked more than once."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")
        calls = []

        def loader():
            calls.append(1)
            return len(calls)

        load_with_spinner("m", loader)
        assert calls == [1]

    def test_loader_errors_propagate(self, monkeypatch):
        """A real loading error is not swallowed by the spinner."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")

        def loader():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            load_with_spinner("m", loader)


class TestQuietHuggingface:
    """The noise-suppression context manager restores prior state."""

    def test_restores_http_logger_level(self):
        """The huggingface_hub http logger level is restored on exit."""
        logger = logging.getLogger("huggingface_hub.utils._http")
        logger.setLevel(logging.WARNING)
        with _progress._quiet_huggingface():
            assert logger.level == logging.ERROR
        assert logger.level == logging.WARNING

    def test_does_not_raise_without_optional_backends(self):
        """Entering and exiting is safe even if optional imports are missing."""
        with _progress._quiet_huggingface():
            pass

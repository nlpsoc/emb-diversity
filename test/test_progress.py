"""Tests for the model-loading spinner feedback (no model loads)."""

import logging

import pytest

from emb_diversity.utility._progress import (
    _quiet_huggingface,
    load_with_spinner,
    progress_enabled,
)


class TestProgressEnabled:
    """EMB_DIVERSITY_PROGRESS forces the spinner on or off."""

    def test_env_on(self, monkeypatch):
        """'1' forces the spinner on."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")
        assert progress_enabled() is True

    def test_env_off(self, monkeypatch):
        """'0' forces the spinner off."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "0")
        assert progress_enabled() is False


class TestLoadWithSpinner:
    """The spinner wrapper runs the loader and stays out of the way."""

    def test_returns_loader_result(self, monkeypatch):
        """The loader's return value is passed straight through."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")
        assert load_with_spinner("m", lambda stage: "model") == "model"

    def test_loader_errors_propagate(self, monkeypatch):
        """A real loading error is not swallowed by the spinner."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "1")

        def loader(stage):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            load_with_spinner("m", loader)

    def test_disabled_passes_noop_stage(self, monkeypatch):
        """When disabled, stage updates are no-ops and the loader still runs."""
        monkeypatch.setenv("EMB_DIVERSITY_PROGRESS", "0")

        def loader(stage):
            stage.loading_libraries()
            stage.fetching_model("m")
            return "model"

        assert load_with_spinner("m", loader) == "model"


def test_quiet_huggingface_restores_logger_level():
    """The huggingface_hub logger level is muted inside and restored on exit."""
    logger = logging.getLogger("huggingface_hub.utils._http")
    logger.setLevel(logging.WARNING)
    with _quiet_huggingface():
        assert logger.level == logging.ERROR
    assert logger.level == logging.WARNING

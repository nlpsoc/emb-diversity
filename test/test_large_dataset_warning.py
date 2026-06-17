"""Tests for the large-dataset warning.

When a dataset is larger than ``LARGE_DATASET_WARN_THRESHOLD``, the package
warns that a measure may be slow or run out of memory, then computes on the
full input as usual (no subsampling). The threshold is monkeypatched small here
so no large arrays are needed.
"""

import numpy as np
import pytest

from emb_diversity import embed


class TestLargeDatasetWarning:

    def test_warns_above_threshold(self, monkeypatch):
        """Oversized vector input warns and is returned in full (not subsampled)."""
        monkeypatch.setattr(embed, "LARGE_DATASET_WARN_THRESHOLD", 5)
        X = np.random.RandomState(0).randn(20, 8)
        with pytest.warns(UserWarning, match="Support for large datasets"):
            vectors, model = embed.resolve_embeddings(X)
        assert model is None
        assert vectors.shape == (20, 8)  # full input, unchanged

    def test_no_warning_at_or_below_threshold(self, monkeypatch, recwarn):
        """Input at or below the threshold emits no warning."""
        monkeypatch.setattr(embed, "LARGE_DATASET_WARN_THRESHOLD", 50)
        X = np.random.RandomState(0).randn(20, 8)
        embed.resolve_embeddings(X)
        assert not [w for w in recwarn if "large datasets" in str(w.message)]

    def test_bare_string_is_not_warned_and_still_raises(self, monkeypatch):
        """A bare string raises the usual error rather than being measured by length."""
        monkeypatch.setattr(embed, "LARGE_DATASET_WARN_THRESHOLD", 5)
        long_string = "x" * 100  # longer than the threshold by character count
        with pytest.raises(ValueError, match="list of texts, not a single string"):
            embed.resolve_embeddings(long_string)

    def test_text_warns_but_is_embedded_in_full(self, monkeypatch):
        """Oversized text warns yet all of it is still embedded (no subsampling)."""
        monkeypatch.setattr(embed, "LARGE_DATASET_WARN_THRESHOLD", 5)
        seen = []

        def fake_embed_texts(texts, **kwargs):
            seen.append(list(texts))
            return np.zeros((len(texts), 4))

        monkeypatch.setattr(embed, "embed_texts", fake_embed_texts)
        texts = [f"text {i}" for i in range(20)]
        with pytest.warns(UserWarning, match="Support for large datasets"):
            vectors, _ = embed.resolve_embeddings(texts)
        assert len(seen) == 1 and len(seen[0]) == 20  # all 20 embedded
        assert vectors.shape == (20, 4)

    def test_default_threshold_is_10k(self):
        """The warning fires above 10,000 samples by default."""
        assert embed.LARGE_DATASET_WARN_THRESHOLD == 10_000

import pytest

from emb_diversity.embed import embed_texts, resolve_embeddings
from emb_diversity.measures_registry import measures


class TestBareStringInput:
    """A single string must be rejected before any embedding model loads."""

    def test_resolve_embeddings_rejects_bare_string(self):
        """resolve_embeddings raises a TypeError telling the user to pass a list."""
        with pytest.raises(TypeError, match="list of texts, not a single string"):
            resolve_embeddings("just one text")

    def test_embed_texts_rejects_bare_string(self):
        """embed_texts raises the same TypeError for a bare string."""
        with pytest.raises(TypeError, match="list of texts, not a single string"):
            embed_texts("just one text")

    @pytest.mark.parametrize("name", sorted(measures))
    def test_measure_rejects_bare_string(self, name):
        """Every measure passes the guard through to the user."""
        with pytest.raises(TypeError, match="Wrap it in a list"):
            measures[name]("just one text")

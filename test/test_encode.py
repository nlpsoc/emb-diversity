"""Tests for emb_diversity.embed."""
import numpy as np
import pytest

from emb_diversity.embed import embed_texts, resolve_embeddings, resolve_model_name


SENTENCES = ["This is a test sentence.", "This is another test."]


class TestResolveModelName:

    def test_explicit_model_takes_priority(self):
        """embedding_model overrides diversity_axis."""
        model = resolve_model_name(diversity_axis="semantic", embedding_model="my-model")
        assert model == "my-model"

    def test_axis_returns_a_model_name(self):
        """diversity_axis resolves to a non-empty model name string."""
        model = resolve_model_name(diversity_axis="semantic")
        assert isinstance(model, str) and len(model) > 0

    def test_no_axis_no_model_raises(self):
        """Neither axis nor model raises ValueError."""
        with pytest.raises(ValueError):
            resolve_model_name(diversity_axis=None, embedding_model=None)


class TestResolveEmbeddingsVectorInput:
    """Vector input goes through resolve_embeddings without touching any model."""

    def test_vector_input_returns_none_model(self):
        """Numeric input returns None as the model name — no embedding happened."""
        _, model_name = resolve_embeddings([[0.1, 0.2], [0.3, 0.4]])
        assert model_name is None

    def test_vector_input_returns_numpy_array(self):
        """Numeric input is converted to a numpy array."""
        vectors, _ = resolve_embeddings([[0.1, 0.2], [0.3, 0.4]])
        assert isinstance(vectors, np.ndarray)

    def test_vector_input_shape_preserved(self):
        """Output shape matches input shape."""
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        vectors, _ = resolve_embeddings(data)
        assert vectors.shape == (2, 3)


class TestEmbedTextsBareString:

    def test_bare_string_raises(self):
        """A single string raises ValueError — would embed character by character otherwise."""
        with pytest.raises(ValueError, match="list of texts"):
            embed_texts("just a string")


@pytest.mark.integration
class TestIntegration:
    """Downloads real models. Run with: pytest -m integration"""

    def test_default_semantic_axis(self):
        """Default axis produces valid embeddings."""
        result = embed_texts(SENTENCES)
        assert result.shape[0] == len(SENTENCES)
        assert result.shape[1] > 0
        assert np.all(np.isfinite(result))

    def test_style_model(self):
        """AnnaWegmann/Style-Embedding (ST backend) produces valid embeddings."""
        result = embed_texts(SENTENCES, embedding_model="AnnaWegmann/Style-Embedding")
        assert result.shape[0] == len(SENTENCES)
        assert np.all(np.isfinite(result))

    def test_hf_simcse_model(self):
        """princeton-nlp/sup-simcse-roberta-large (HF backend) produces valid embeddings."""
        result = embed_texts(SENTENCES, embedding_model="princeton-nlp/sup-simcse-roberta-large")
        assert result.shape[0] == len(SENTENCES)
        assert np.all(np.isfinite(result))

    def test_text_input_resolve_embeddings(self):
        """Text input through resolve_embeddings returns array and model name."""
        vectors, model_name = resolve_embeddings(SENTENCES)
        assert isinstance(vectors, np.ndarray)
        assert isinstance(model_name, str)
        assert vectors.shape[0] == len(SENTENCES)

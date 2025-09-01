import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List
from measure_diversity.embeddings.SBERT import encode_sentences, encode_style_sentences, _get_model, encode_semantic_sentences


class TestSentenceEncoder:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        print("SETUP: Clear cache")
        _get_model.cache_clear()  # ← Runs BEFORE test
        yield  # ← Test runs HERE
        print("TEARDOWN: Clear cache")
        _get_model.cache_clear()  # ← Runs AFTER test

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        mock_model = Mock()
        # Mock encode method to return predictable embeddings
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock_model

    def test_encode_sentences_empty_input(self, mock_sentence_transformer):
        """Test behavior with empty sentence list."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer):
            mock_sentence_transformer.encode.return_value = np.array([])
            result = encode_sentences([])
            assert result == []

    def test_encode_sentences_custom_model(self, mock_sentence_transformer):
        """Test using a custom model name."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer) as mock_constructor:
            sentences = ["Test sentence"]
            custom_model = "custom-model-name"
            encode_sentences(sentences, model_name=custom_model)
            # Verify the correct model was requested
            mock_constructor.assert_called_with(custom_model)

    def test_encode_sentences_default_model(self, mock_sentence_transformer):
        """Test that default model is used when not specified."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer) as mock_constructor:
            encode_sentences(["Test"])
            mock_constructor.assert_called_with("all-MiniLM-L6-v2")

    def test_model_caching_different_models(self, mock_sentence_transformer):
        """Test caching behavior with different models."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer) as mock_constructor:
            # Call with different models
            encode_sentences(["Test"], "model-1")
            encode_sentences(["Test"], "model-2")
            encode_sentences(["Test"], "model-1")  # Should use cached version

            # Should create 2 models (model-1 and model-2)
            assert mock_constructor.call_count == 2

    def test_encode_style_sentences(self, mock_sentence_transformer):
        """Test the style-specific encoding function."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer) as mock_constructor:
            sentences = ["Style test sentence"]
            result = encode_style_sentences(sentences)
            # Should use the AnnaWegmann/Style-Embedding model
            mock_constructor.assert_called_with("AnnaWegmann/Style-Embedding")
            assert isinstance(result, list)
            encode_semantic_sentences(sentences)
            mock_constructor.assert_called_with("all-mpnet-base-v2")

    def test_model_loading_error_handling(self):
        """Test behavior when model loading fails."""
        with pytest.raises(ValueError):
            encode_sentences(["Test"], "nonexistent-model")


class TestCacheManagement:
    """Separate test class for cache-specific functionality."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        mock_model = Mock()
        # Mock encode method to return predictable embeddings
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock_model

    def test_cache_clear_functionality(self, mock_sentence_transformer):
        """Test that cache can be cleared."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', mock_sentence_transformer):
            # Load a model
            encode_sentences(["Test"], "test-model")

            # Check cache has content
            cache_info = _get_model.cache_info()
            assert cache_info.currsize > 0

            # Clear cache
            _get_model.cache_clear()

            # Check cache is empty
            cache_info = _get_model.cache_info()
            assert cache_info.currsize == 0
            assert cache_info.hits == 0
            assert cache_info.misses == 0

    def test_cache_size_limit(self):
        """Test that cache respects maxsize limit."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer') as mock_constructor:
            mock_constructor.return_value.encode.return_value = np.array([[0.1, 0.2]])

            # Load more than 10 models (maxsize=10)
            for i in range(15):
                encode_sentences(["Test"], f"model-{i}")

            cache_info = _get_model.cache_info()
            # Should not exceed maxsize of 10
            assert cache_info.currsize <= 10

    def test_cache_info_functionality(self, mock_sentence_transformer):
        """Test that cache info is accessible."""
        with patch('measure_diversity.embeddings.SBERT.SentenceTransformer', return_value=mock_sentence_transformer):
            # Clear cache and check initial state
            _get_model.cache_clear()
            initial_info = _get_model.cache_info()
            assert initial_info.hits == 0
            assert initial_info.misses == 0

            # Make calls and check cache stats
            encode_sentences(["Test 1"], "model-a")  # Miss (cache doesn't have a model-a)
            encode_sentences(["Test 2"], "model-a")  # Hit
            encode_sentences(["Test 3"], "model-b")  # Miss

            final_info = _get_model.cache_info()
            assert final_info.hits == 1
            assert final_info.misses == 2


# Integration test (optional - requires actual model download)
@pytest.mark.integration
class TestSentenceEncoderIntegration:
    """Integration tests that use real models (slow, requires internet)."""

    @pytest.fixture(autouse=True)
    def skip_integration_tests(self, request):
        """Skip integration tests unless explicitly requested."""
        if "integration" not in request.config.getoption("-m", default=""):
            pytest.skip("Integration tests require model downloads - run with -m integration to enable")

    def test_real_model_encoding(self):
        """Test with a real model (skipped by default)."""
        # Uncomment to run integration test:
        sentences = ["This is a test sentence.", "This is another test."]
        result = encode_sentences(sentences, "all-MiniLM-L6-v2")
        assert len(result) == 2
        assert len(result[0]) == 384  # MiniLM embedding dimension
        assert isinstance(result, list)
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(isinstance(val, float) for embedding in result for val in embedding)


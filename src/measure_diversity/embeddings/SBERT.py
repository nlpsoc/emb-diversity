from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List

@lru_cache(maxsize=10)  # an encoder model typically has sizes between 100MB and 1GB, rarely up to 2GB
def _get_model(model_name: str) -> SentenceTransformer:
    """Cache models using lru_cache."""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")

def clear_model_cache():
    """Clear cached models to free memory."""
    _get_model.cache_clear()

def encode_sentences(sentences: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Encode a list of sentences into embeddings using SBERT.

    Args:
        sentences: List of input sentences.
        model_name: Name of the pretrained SBERT model.

    Returns:
        List of embedding vectors (as lists of floats).
    """
    if not sentences:
        return []
    model = _get_model(model_name)
    max_seq_length = model.max_seq_length
    # Cut sentences to prevent tokenizer overflow, using generous upper limit with ~5 characters per token
    max_chars = max_seq_length * 5
    truncated_sentences = [s[:max_chars] if len(s) > max_chars else s for s in sentences]

    embeddings = model.encode(truncated_sentences, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()

def encode_style_sentences(sentences: List[str]) -> List[List[float]]:
    return encode_sentences(sentences, model_name="AnnaWegmann/Style-Embedding")

def encode_semantic_sentences(sentences: List[str]) -> List[List[float]]:
    return encode_sentences(sentences, model_name="all-mpnet-base-v2")
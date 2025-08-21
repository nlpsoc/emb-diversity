from sentence_transformers import SentenceTransformer
from typing import List

def encode_sentences(sentences: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Encode a list of sentences into embeddings using SBERT.

    Args:
        sentences: List of input sentences.
        model_name: Name of the pretrained SBERT model.

    Returns:
        List of embedding vectors (as lists of floats).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings.tolist()
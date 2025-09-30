from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

def encode_sentences(sentences: List[str], model_name: str = "princeton-nlp/sup-simcse-roberta-large") -> List[List[float]]:
    """
    Encode a list of sentences into embeddings using SimCSE (transformers version).

    Args:
        sentences: List of input sentences.
        model_name: Name of the pretrained SimCSE model.

    Returns:
        List of embedding vectors (as lists of floats).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs, return_dict=True)
        # Use the [CLS] token embedding as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        return embeddings.cpu().numpy().tolist()
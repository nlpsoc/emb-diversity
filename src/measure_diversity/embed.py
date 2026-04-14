import torch
from typing import List, Sequence
from ._cache import cached_encode


# private functions to load models
def _load_st(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _load_hf(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


# private methods to calculate embeddings for the texts not existing in cache
def _raw_encode_st(texts: List[str], model_name: str) -> List[List[float]]:
    model = _load_st(model_name)
    return model.encode(texts, convert_to_numpy=True).tolist()


def _raw_encode_hf(texts: List[str], model_name: str) -> List[List[float]]:
    import torch

    tokenizer, model, device = _load_hf(model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().tolist()


# functions to encode using huggingface or sentence transformers
# public to user

def encode_st(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Encode texts using a SentenceTransformers model. Results are cached on disk."""
    return cached_encode(
        texts,
        encode_fn=lambda t: _raw_encode_st(t, model_name),
        model_name=model_name,
    )


def encode_hf(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Encode texts using a HuggingFace Transformers model. Results are cached on disk."""
    return cached_encode(
        texts,
        encode_fn=lambda t: _raw_encode_hf(t, model_name),
        model_name=model_name,
    )


#special functions to load particular models
def encode_style(texts: Sequence[str]) -> List[List[float]]:
    return encode_st(texts, "AnnaWegmann/Style-Embedding")


def encode_semantic(texts: Sequence[str]) -> List[List[float]]:
    return encode_st(texts, "all-mpnet-base-v2")


def encode_simcse(texts: Sequence[str]) -> List[List[float]]:
    return encode_hf(texts, "princeton-nlp/sup-simcse-roberta-large")
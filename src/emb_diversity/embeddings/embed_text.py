from pathlib import Path
from typing import List, Optional, Sequence

from ..utility._cache import cached_encode, DEFAULT_CACHE_DIR
from ..utility._progress import announce_embedding, load_with_spinner, progress_enabled


# Models that require the HuggingFace Transformers backend
_HF_MODELS = {
    "princeton-nlp/sup-simcse-roberta-large",
}


# private functions to load models
def _load_st(model_name: str):
    def _load(stage):
        stage.loading_libraries()
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage.fetching_model(model_name)
        return SentenceTransformer(model_name, device=device)

    return load_with_spinner(model_name, _load)

def _load_hf(model_name: str):
    def _load(stage):
        stage.loading_libraries()
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage.fetching_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        return tokenizer, model, device

    return load_with_spinner(model_name, _load)


# private methods to calculate embeddings for the texts not existing in cache
def _raw_encode_st(texts: List[str], model_name: str) -> List[List[float]]:
    model = _load_st(model_name)
    announce_embedding(len(texts))
    # show_progress_bar surfaces SentenceTransformer's per-batch bar, so a long
    # encode does not look like a hang. Gated like the rest so pipes/CI stay quiet.
    return model.encode(
        texts, convert_to_numpy=True, show_progress_bar=progress_enabled()
    ).tolist()


def _raw_encode_hf(texts: List[str], model_name: str) -> List[List[float]]:
    import torch

    tokenizer, model, device = _load_hf(model_name)
    announce_embedding(len(texts))
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().tolist()


def encode(
    texts: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[List[float]]:
    """
    Encode texts into embeddings using the appropriate backend, with disk caching.

    Uses HuggingFace Transformers for known HF-only models, SentenceTransformers otherwise.

    Args:
        texts: Input texts to encode.
        model_name: Pretrained model name. Defaults to "all-MiniLM-L6-v2".
        cache_dir: Root directory for the disk cache.

    Returns:
        List of embedding vectors as lists of floats.
    """
    if model_name in _HF_MODELS:
        encode_fn = lambda t: _raw_encode_hf(t, model_name)
    else:
        encode_fn = lambda t: _raw_encode_st(t, model_name)

    return cached_encode(texts, encode_fn=encode_fn, model_name=model_name, cache_dir=cache_dir)
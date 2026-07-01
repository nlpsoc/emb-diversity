from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

from ..utility._cache import cached_encode, DEFAULT_CACHE_DIR
from ..utility._progress import announce_embedding, load_with_spinner


# Sample rate every supported model expects its input waveform at; audio
# loaded at any other rate is resampled to this before encoding.
_TARGET_SAMPLE_RATE = 16_000

# Root for downloaded SpeechBrain model weights, mirroring DEFAULT_CACHE_DIR's
# convention of caching under the project directory rather than $HOME.
# Without an explicit savedir, speechbrain defaults to a CWD-relative
# "pretrained_models/" folder, which would otherwise litter wherever the
# caller's script happens to run from.
_MODEL_CACHE_DIR = DEFAULT_CACHE_DIR.parent / "speechbrain_models"

# Files per forward pass. encode_batch pads every clip in a batch to the
# longest one, so very large batches of very mismatched lengths waste compute
# on padding; this default is small enough to keep that waste low for
# typical short utterances while still amortizing most of the per-call
# overhead (feature extraction, model dispatch) that dominates at batch=1 —
# batching measured ~3x faster than one-clip-at-a-time on CPU.
_BATCH_SIZE = 32


@lru_cache(maxsize=None)
def _load_speechbrain(model_name: str):
    def _load(stage):
        stage.loading_libraries()
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage.fetching_model(model_name)
        return EncoderClassifier.from_hparams(
            source=model_name,
            savedir=str(_MODEL_CACHE_DIR / model_name.replace("/", "_")),
            run_opts={"device": device},
        )

    return load_with_spinner(model_name, _load)


def _load_waveform(path: str):
    """Load *path* as a mono waveform at the model's expected sample rate.

    Decodes via ``soundfile`` rather than ``torchaudio.load`` — recent
    torchaudio versions route ``load`` through TorchCodec, which links
    against a specific CUDA build and fails to import on any environment
    where that build doesn't match the installed torch wheel (a real,
    commonly-hit packaging mismatch, not a hypothetical one). ``soundfile``
    has no such coupling. Resampling still goes through
    ``torchaudio.functional.resample``, a pure tensor op that never touches
    TorchCodec.
    """
    import soundfile as sf
    import torch
    import torchaudio

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)  # (channels, n_frames)
    if signal.shape[0] > 1:  # mixdown multi-channel audio to mono
        signal = signal.mean(dim=0, keepdim=True)
    if sample_rate != _TARGET_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sample_rate, _TARGET_SAMPLE_RATE)
    return signal


def _encode_one_batch(paths: List[str], classifier) -> List[List[float]]:
    """Encode *paths* (<= _BATCH_SIZE files) in a single padded forward pass."""
    import torch

    signals = [_load_waveform(p).squeeze(0) for p in paths]  # each (n_frames,)
    lengths = torch.tensor([s.shape[0] for s in signals])
    max_len = int(lengths.max().item())
    if max_len == 0:
        raise ValueError("Audio file(s) appear to be empty (0 frames).")

    padded = torch.zeros(len(signals), max_len, dtype=signals[0].dtype)
    for i, s in enumerate(signals):
        padded[i, : s.shape[0]] = s
    wav_lens = lengths.float() / max_len  # relative lengths, masks padding

    device = getattr(classifier, "device", None)
    if device is None:
        try:
            device = next(classifier.parameters()).device
        except Exception:
            device = padded.device

    padded = padded.to(device)
    wav_lens = wav_lens.to(device)

    with torch.inference_mode():
        vectors = classifier.encode_batch(padded, wav_lens)  # (batch, 1, emb_dim)
    return vectors.squeeze(1).detach().cpu().numpy().tolist()


def _raw_encode_speechbrain(paths: List[str], model_name: str) -> List[List[float]]:
    classifier = _load_speechbrain(model_name)
    announce_embedding(len(paths), unit="audio files")
    embeddings: List[List[float]] = []
    for start in range(0, len(paths), _BATCH_SIZE):
        embeddings.extend(_encode_one_batch(paths[start : start + _BATCH_SIZE], classifier))
    return embeddings


def encode(
    paths: Sequence[str],
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[List[float]]:
    """Encode audio files into speaker embeddings, with disk caching.

    Each file is loaded, mixed down to mono, and resampled to 16 kHz (the
    rate speechbrain's VoxCeleb-trained models expect) before encoding.

    Args:
        paths: Paths to audio files (any format supported by ``soundfile``,
            e.g. .wav, .flac, .ogg — not .mp3).
        model_name: SpeechBrain model id. Defaults to
            "speechbrain/spkrec-ecapa-voxceleb".
        cache_dir: Root directory for the disk cache.

    Returns:
        List of embedding vectors as lists of floats.

    Note:
        Cache entries are keyed by file path (like :func:`embeddings.embed_text.encode`
        is keyed by text content), not by audio content — if a file's content
        changes but its path does not, a stale cached embedding is returned.
    """
    return cached_encode(
        list(paths),
        encode_fn=lambda batch: _raw_encode_speechbrain(batch, model_name),
        model_name=model_name,
        cache_dir=cache_dir,
    )

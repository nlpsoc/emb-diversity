"""Tests for emb_diversity's audio (speaker) embedding pipeline."""
import numpy as np
import pytest

from emb_diversity.axes_registry import axes
from emb_diversity.embed import embed_audio, resolve_embeddings, resolve_model_name

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class TestSpeakerAxisRegistration:

    def test_speaker_axis_has_audio_modality(self):
        """The "speaker" axis is registered with modality "audio"."""
        assert axes.get("speaker").modality == "audio"

    def test_speaker_axis_default_model(self):
        """The default model for the speaker axis is the ECAPA-TDNN VoxCeleb model."""
        assert axes.get("speaker").default_model == SPEAKER_MODEL

    def test_text_axes_default_to_text_modality(self):
        """Pre-existing axes are unaffected by the new modality field."""
        assert axes.get("semantic").modality == "text"
        assert axes.get("style").modality == "text"

    def test_resolve_model_name_for_speaker_axis(self):
        """resolve_model_name resolves "speaker" to its default model."""
        assert resolve_model_name(diversity_axis="speaker") == SPEAKER_MODEL


class TestEmbedAudioBareString:

    def test_bare_string_raises(self):
        """A single path string raises ValueError — would be embedded character by character otherwise."""
        with pytest.raises(ValueError, match="list of audio file paths"):
            embed_audio("just/one/path.wav")


class TestAudioModalityDispatch:
    """resolve_embeddings routes raw string input for an audio-modality axis to
    the audio encoder, never the text encoder — checked here with the encoders
    mocked out, so it runs without downloading any model."""

    def test_speaker_axis_uses_audio_encoder(self, monkeypatch):
        import emb_diversity.embed as embed_module

        captured = {}

        def fake_audio_encode(paths, model_name=None, cache_dir=None):
            captured["paths"] = list(paths)
            captured["model_name"] = model_name
            return [[1.0, 0.0], [0.0, 1.0]]

        def fail_if_called(*args, **kwargs):
            raise AssertionError("text encoder must not run for an audio-modality axis")

        monkeypatch.setattr(embed_module, "encode_audio", fake_audio_encode)
        monkeypatch.setattr(embed_module, "encode", fail_if_called)

        vectors, model_name = resolve_embeddings(["a.wav", "b.wav"], diversity_axis="speaker")

        assert model_name == SPEAKER_MODEL
        assert captured["paths"] == ["a.wav", "b.wav"]
        assert vectors.shape == (2, 2)

    def test_semantic_axis_still_uses_text_encoder(self, monkeypatch):
        """Existing text axes are unaffected by the audio dispatch branch."""
        import emb_diversity.embed as embed_module

        def fake_text_encode(texts, model_name=None, cache_dir=None):
            return [[0.0, 1.0] for _ in texts]

        def fail_if_called(*args, **kwargs):
            raise AssertionError("audio encoder must not run for a text-modality axis")

        monkeypatch.setattr(embed_module, "encode", fake_text_encode)
        monkeypatch.setattr(embed_module, "encode_audio", fail_if_called)

        vectors, _ = resolve_embeddings(["hello", "world"], diversity_axis="semantic")
        assert vectors.shape == (2, 2)


@pytest.mark.integration
class TestIntegration:
    """Downloads the real SpeechBrain model. Run with: pytest -m integration"""

    @pytest.fixture(scope="class")
    def wav_paths(self, tmp_path_factory):
        """Two short synthetic mono WAV files at different pitches."""
        from scipy.io import wavfile

        sample_rate = 16_000
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        tmp_dir = tmp_path_factory.mktemp("audio")

        paths = []
        for i, freq in enumerate([220.0, 440.0]):
            signal = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            path = tmp_dir / f"tone_{i}.wav"
            wavfile.write(str(path), sample_rate, signal)
            paths.append(str(path))
        return paths

    def test_embed_audio_produces_valid_embeddings(self, wav_paths):
        """embed_audio returns one finite vector per audio file."""
        pytest.importorskip("speechbrain")
        pytest.importorskip("torchaudio")
        pytest.importorskip("soundfile")
        result = embed_audio(wav_paths)
        assert result.shape[0] == len(wav_paths)
        assert result.shape[1] > 0
        assert np.all(np.isfinite(result))

    def test_speaker_axis_through_resolve_embeddings(self, wav_paths):
        """Audio file paths through resolve_embeddings dispatch to the audio encoder."""
        pytest.importorskip("speechbrain")
        pytest.importorskip("torchaudio")
        pytest.importorskip("soundfile")
        vectors, model_name = resolve_embeddings(wav_paths, diversity_axis="speaker")
        assert model_name == SPEAKER_MODEL
        assert vectors.shape[0] == len(wav_paths)

    def test_measure_diversity_on_speaker_axis(self, wav_paths):
        """The generic diversity measures run unmodified on speaker embeddings."""
        pytest.importorskip("speechbrain")
        pytest.importorskip("torchaudio")
        pytest.importorskip("soundfile")
        from emb_diversity import measure_diversity

        results = measure_diversity(wav_paths, diversity_axis="speaker")
        assert np.isfinite(results["mean_pw_dist"]["value"])
        assert results["mean_pw_dist"]["parameters"]["embedding_model"] == SPEAKER_MODEL

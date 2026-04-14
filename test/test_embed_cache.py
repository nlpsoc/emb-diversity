"""
Manual tests for embedding caching, exercised through the public encode() function.
Run directly: python test/test_embed_cache.py
"""
from pathlib import Path
from unittest.mock import patch

from measure_diversity.embed import encode
from measure_diversity._cache import clear_cache

CACHE_DIR = Path(".cache/test_embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Diversity in language reflects diversity in thought.",
    "Machine learning models learn from data.",
    "The cat sat on the mat.",
    "Natural language processing is a subfield of AI.",
]


def _fake_encode_st(call_tracker: list):
    """Patch for _raw_encode_st that records calls and returns dummy embeddings."""
    def _inner(texts, model_name):
        call_tracker.extend(texts)
        return [[float(i) * 0.1] * 16 for i in range(len(texts))]
    return _inner


def _reset():
    clear_cache(cache_dir=CACHE_DIR)


def test_all_sentences_encoded_on_first_call():
    _reset()
    calls = []
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st(calls)):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    assert set(calls) == set(SENTENCES), f"FAIL: expected all sentences encoded, got {calls}"
    print("PASS: all sentences encoded on first call")


def test_duplicate_sentence_not_re_encoded():
    _reset()
    calls = []
    fake = _fake_encode_st(calls)
    with patch("measure_diversity.embed._raw_encode_st", side_effect=fake):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        calls.clear()
        encode(SENTENCES + [SENTENCES[0]], model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    assert calls == [], f"FAIL: duplicate sentence was re-encoded, calls={calls}"
    print("PASS: duplicate sentence served from cache")


def test_only_new_sentences_are_encoded():
    _reset()
    calls = []
    fake = _fake_encode_st(calls)
    with patch("measure_diversity.embed._raw_encode_st", side_effect=fake):
        encode(SENTENCES[:3], model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        calls.clear()
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    assert set(calls) == set(SENTENCES[3:]), f"FAIL: expected only new sentences, got {calls}"
    print("PASS: only new sentences passed to encode_fn")


def test_altered_sentence_is_re_encoded():
    _reset()
    calls = []
    fake = _fake_encode_st(calls)

    original = SENTENCES[0]
    altered = original[:-1] + ("!" if original[-1] != "!" else "?")

    with patch("measure_diversity.embed._raw_encode_st", side_effect=fake):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        calls.clear()
        encode([altered] + SENTENCES[1:], model_name=MODEL_NAME, cache_dir=CACHE_DIR)

    assert altered in calls, f"FAIL: altered sentence was not re-encoded, calls={calls}"
    assert original not in calls, f"FAIL: original sentence was unexpectedly re-encoded"
    print(f"PASS: altered sentence re-encoded, original '{original[:30]}...' served from cache")


def test_embeddings_consistent_across_calls():
    _reset()
    calls = []
    fake = _fake_encode_st(calls)
    with patch("measure_diversity.embed._raw_encode_st", side_effect=fake):
        first = encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        second = encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"FAIL: embedding for sentence {i} differs between calls"
    print("PASS: embeddings are consistent across calls")


def test_output_length_matches_input_with_duplicates():
    _reset()
    sentences_with_dup = SENTENCES + [SENTENCES[0], SENTENCES[1]]
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st([])):
        result = encode(sentences_with_dup, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    assert len(result) == len(sentences_with_dup), f"FAIL: expected {len(sentences_with_dup)}, got {len(result)}"
    print("PASS: output length matches input length including duplicates")


def test_empty_input_returns_empty():
    _reset()
    calls = []
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st(calls)):
        result = encode([], model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    assert result == [], f"FAIL: expected [], got {result}"
    assert calls == [], f"FAIL: encode was called on empty input"
    print("PASS: empty input returns empty list without calling encode")


def test_cache_files_written_to_disk():
    _reset()
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st([])):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    files = list(model_cache.glob("*.safetensors"))
    assert len(files) == len(SENTENCES), f"FAIL: expected {len(SENTENCES)} files, got {len(files)}"
    print(f"PASS: {len(files)} cache files written to disk")


def test_cache_survives_between_calls():
    _reset()
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st([])):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)

    second_calls = []
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st(second_calls)):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)

    assert second_calls == [], f"FAIL: encode called despite cache on disk, calls={second_calls}"
    print("PASS: cache persists across separate encode() calls")


def test_clear_cache_removes_files():
    _reset()
    with patch("measure_diversity.embed._raw_encode_st", side_effect=_fake_encode_st([])):
        encode(SENTENCES, model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    clear_cache(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    assert not model_cache.exists(), "FAIL: cache directory still exists after clear_cache"
    print("PASS: clear_cache removes all cached files")


if __name__ == "__main__":
    tests = [
        test_all_sentences_encoded_on_first_call,
        test_duplicate_sentence_not_re_encoded,
        test_only_new_sentences_are_encoded,
        test_altered_sentence_is_re_encoded,
        test_embeddings_consistent_across_calls,
        test_output_length_matches_input_with_duplicates,
        test_empty_input_returns_empty,
        test_cache_files_written_to_disk,
        test_cache_survives_between_calls,
        test_clear_cache_removes_files,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(e)
            failed += 1
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    _reset()

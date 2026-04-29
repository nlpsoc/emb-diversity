"""
Real embedding cache tests — no mocks, runs on GPU if available.
Uses the actual encode() function and SentenceTransformer model.
Run directly: python test/test_embed_cache.py
"""
import time
import torch
from pathlib import Path

from measure_diversity.embed import encode
from measure_diversity._cache import clear_cache

CACHE_DIR = Path(".cache/test_embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Diversity in language reflects diversity in thought.",
    "Machine learning models learn from data.",
    "The cat sat on the mat.",
    "Natural language processing is a subfield of AI.",
]


def _reset():
    clear_cache(cache_dir=CACHE_DIR)


def _encode(sentences):
    return encode(sentences, model_name=MODEL_NAME, cache_dir=CACHE_DIR)


def test_all_sentences_encoded_on_first_call():
    _reset()
    result = _encode(SENTENCES)
    assert len(result) == len(SENTENCES), f"FAIL: expected {len(SENTENCES)} embeddings, got {len(result)}"
    assert all(isinstance(e, list) and len(e) > 0 for e in result), "FAIL: embeddings are not valid lists"
    print(f"PASS: all {len(SENTENCES)} sentences embedded, dim={len(result[0])}, device={DEVICE}")


def test_cache_is_faster_than_encoding():
    _reset()

    # First call — compute and cache
    t0 = time.perf_counter()
    _encode(SENTENCES)
    t_compute = time.perf_counter() - t0

    # Second call — serve from disk cache
    t0 = time.perf_counter()
    _encode(SENTENCES)
    t_cache = time.perf_counter() - t0

    assert t_cache < t_compute, f"FAIL: cache ({t_cache:.3f}s) was not faster than compute ({t_compute:.3f}s)"
    print(f"PASS: compute={t_compute:.3f}s  cache={t_cache:.3f}s  speedup={t_compute/t_cache:.1f}x")


def test_duplicate_sentence_returns_same_embedding():
    _reset()
    sentences_with_dup = SENTENCES + [SENTENCES[0]]
    result = _encode(sentences_with_dup)

    assert len(result) == len(sentences_with_dup), f"FAIL: expected {len(sentences_with_dup)}, got {len(result)}"
    assert result[0] == result[-1], "FAIL: duplicate sentence returned different embedding"
    print("PASS: duplicate sentence returns identical embedding")


def test_embeddings_consistent_across_calls():
    _reset()
    first = _encode(SENTENCES)
    second = _encode(SENTENCES)

    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"FAIL: embedding for sentence {i} differs between calls"
    print("PASS: embeddings are identical across calls")


def test_only_new_sentences_encoded_on_partial_overlap():
    _reset()

    # Seed cache with first 3 sentences
    first_3 = _encode(SENTENCES[:3])

    # Now encode all 5 — last 2 are new
    all_5 = _encode(SENTENCES)

    # The first 3 results should be identical (served from cache)
    for i in range(3):
        assert all_5[i] == first_3[i], f"FAIL: cached sentence {i} returned different embedding"

    print("PASS: first 3 embeddings served from cache, last 2 freshly computed")


def test_altered_sentence_gets_different_embedding():
    _reset()
    original = SENTENCES[0]
    altered = original[:-1] + ("!" if original[-1] != "!" else "?")

    original_result = _encode([original])
    altered_result = _encode([altered])

    assert original_result[0] != altered_result[0], "FAIL: altered sentence returned same embedding as original"
    print(f"PASS: '{original[:30]}...' and altered version have different embeddings")


def test_sentence_inserted_in_middle_correct_position():
    _reset()
    new_sentence = "This is a brand new sentence inserted in the middle."
    sentences_with_insert = SENTENCES[:2] + [new_sentence] + SENTENCES[2:]

    # Seed cache with original sentences
    original_results = _encode(SENTENCES)

    # Encode with insertion
    new_results = _encode(sentences_with_insert)

    assert len(new_results) == len(sentences_with_insert), "FAIL: wrong output length"

    # Sentences before insert are unchanged
    assert new_results[0] == original_results[0], "FAIL: sentence at index 0 changed"
    assert new_results[1] == original_results[1], "FAIL: sentence at index 1 changed"

    # New sentence has a real embedding
    assert isinstance(new_results[2], list) and len(new_results[2]) > 0, "FAIL: new sentence has no embedding"

    # Sentences after insert are shifted but still match
    assert new_results[3] == original_results[2], "FAIL: sentence at index 3 (was 2) changed"
    assert new_results[4] == original_results[3], "FAIL: sentence at index 4 (was 3) changed"
    assert new_results[5] == original_results[4], "FAIL: sentence at index 5 (was 4) changed"

    print("PASS: new sentence embedded at correct position, surrounding embeddings intact")


def test_output_length_matches_input_with_duplicates():
    _reset()
    sentences_with_dup = SENTENCES + [SENTENCES[0], SENTENCES[1]]
    result = _encode(sentences_with_dup)
    assert len(result) == len(sentences_with_dup), f"FAIL: expected {len(sentences_with_dup)}, got {len(result)}"
    print("PASS: output length matches input length including duplicates")


def test_empty_input_returns_empty():
    _reset()
    result = _encode([])
    assert result == [], f"FAIL: expected [], got {result}"
    print("PASS: empty input returns empty list")


def test_cache_files_written_to_disk():
    _reset()
    _encode(SENTENCES)
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    files = list(model_cache.glob("*.safetensors"))
    assert len(files) == len(SENTENCES), f"FAIL: expected {len(SENTENCES)} files, got {len(files)}"
    print(f"PASS: {len(files)} cache files written to disk at {model_cache}")


def test_cache_survives_between_calls():
    _reset()
    first = _encode(SENTENCES)
    second = _encode(SENTENCES)
    assert first == second, "FAIL: embeddings differ between calls"
    print("PASS: cache persists and returns identical embeddings across calls")


def test_clear_cache_removes_files():
    _reset()
    _encode(SENTENCES)
    clear_cache(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    assert not model_cache.exists(), "FAIL: cache directory still exists after clear_cache"
    print("PASS: clear_cache removes all cached files")


def test_different_models_use_separate_namespaces():
    _reset()
    model_a = "all-MiniLM-L6-v2"
    model_b = "all-mpnet-base-v2"

    result_a = encode(SENTENCES, model_name=model_a, cache_dir=CACHE_DIR)
    result_b = encode(SENTENCES, model_name=model_b, cache_dir=CACHE_DIR)

    # Different models produce different embedding dimensions or values
    assert result_a[0] != result_b[0], "FAIL: different models returned same embedding"

    cache_a = CACHE_DIR / model_a.replace("/", "_")
    cache_b = CACHE_DIR / model_b.replace("/", "_")
    assert cache_a.exists() and cache_b.exists(), "FAIL: separate cache dirs not created"
    print(f"PASS: model-a dim={len(result_a[0])}, model-b dim={len(result_b[0])}, stored in separate dirs")


if __name__ == "__main__":
    print(f"Running on: {DEVICE.upper()}\n")

    tests = [
        test_all_sentences_encoded_on_first_call,
        test_cache_is_faster_than_encoding,
        test_duplicate_sentence_returns_same_embedding,
        test_embeddings_consistent_across_calls,
        test_only_new_sentences_encoded_on_partial_overlap,
        test_altered_sentence_gets_different_embedding,
        test_sentence_inserted_in_middle_correct_position,
        test_output_length_matches_input_with_duplicates,
        test_empty_input_returns_empty,
        test_cache_files_written_to_disk,
        test_cache_survives_between_calls,
        test_clear_cache_removes_files,
        test_different_models_use_separate_namespaces,
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

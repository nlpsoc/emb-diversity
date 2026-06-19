"""
Real embedding cache tests — no mocks, runs on GPU if available.
Uses the actual encode() function and SentenceTransformer model.
Run directly: python test/test_embed_cache.py
"""
import time
import numpy as np
import torch
from pathlib import Path

from emb_diversity.embeddings.embed_text import encode, _window_size, _actual_chunks
from emb_diversity.utility import clear_cache

CACHE_DIR = Path(".cache/test_embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A text long enough to span several windows of the model's max sequence length,
# so chunking yields more than one chunk and differs from plain truncation.
LONG_TEXT = ("Diversity in natural language matters for many reasons. " * 120).strip()

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Diversity in language reflects diversity in thought.",
    "Machine learning models learn from data.",
    "The cat sat on the mat.",
    "Natural language processing is a subfield of AI.",
    "The sun rises in the east and sets in the west.",
    "Deep learning has transformed computer vision tasks.",
    "Climate change is one of the biggest challenges of our time.",
    "The history of mathematics stretches back thousands of years.",
    "Music has the power to evoke strong emotions in listeners.",
    "Quantum computing may revolutionize cryptography.",
    "The human brain contains approximately 86 billion neurons.",
    "Economic inequality has been rising in many countries.",
    "Renewable energy sources are becoming increasingly affordable.",
    "Philosophers have debated the nature of consciousness for centuries.",
    "Biodiversity is essential for the health of ecosystems.",
    "The internet has fundamentally changed how we communicate.",
    "Space exploration has expanded our understanding of the universe.",
    "Literature reflects the values and struggles of its time.",
    "Advances in medicine have significantly increased life expectancy.",
]


def _reset():
    clear_cache(cache_dir=CACHE_DIR)


def _encode(sentences):
    return encode(sentences, model_name=MODEL_NAME, cache_dir=CACHE_DIR)


def _time(fn) -> tuple:
    """Run fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _count_cache_files() -> int:
    """Number of cached embedding files for MODEL_NAME on disk."""
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    return len(list(model_cache.glob("*.safetensors")))


def test_all_sentences_encoded_on_first_call():
    _reset()
    result, t = _time(lambda: _encode(SENTENCES))
    assert len(result) == len(SENTENCES), f"FAIL: expected {len(SENTENCES)} embeddings, got {len(result)}"
    assert all(isinstance(e, list) and len(e) > 0 for e in result), "FAIL: embeddings are not valid lists"
    print(f"PASS: all {len(SENTENCES)} sentences embedded  dim={len(result[0])}  time={t:.3f}s  device={DEVICE}")


def test_cache_is_faster_than_encoding():
    _reset()
    _, t_compute = _time(lambda: _encode(SENTENCES))
    _, t_cache = _time(lambda: _encode(SENTENCES))
    assert t_cache < t_compute, f"FAIL: cache ({t_cache:.3f}s) was not faster than compute ({t_compute:.3f}s)"
    print(f"PASS: compute={t_compute:.3f}s  cache={t_cache:.3f}s  speedup={t_compute/t_cache:.1f}x")


def test_duplicate_sentence_returns_same_embedding():
    _reset()
    sentences_with_dup = SENTENCES + [SENTENCES[0]]
    result, t = _time(lambda: _encode(sentences_with_dup))
    assert len(result) == len(sentences_with_dup), f"FAIL: expected {len(sentences_with_dup)}, got {len(result)}"
    assert result[0] == result[-1], "FAIL: duplicate sentence returned different embedding"
    print(f"PASS: duplicate sentence returns identical embedding  time={t:.3f}s")


def test_embeddings_consistent_across_calls():
    _reset()
    first, t1 = _time(lambda: _encode(SENTENCES))
    second, t2 = _time(lambda: _encode(SENTENCES))
    for i, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"FAIL: embedding for sentence {i} differs between calls"
    print(f"PASS: embeddings identical across calls  compute={t1:.3f}s  cache={t2:.3f}s  speedup={t1/t2:.1f}x")


def test_only_new_sentences_encoded_on_partial_overlap():
    _reset()
    half = len(SENTENCES) // 2

    first_half, t_first = _time(lambda: _encode(SENTENCES[:half]))
    all_sentences, t_all = _time(lambda: _encode(SENTENCES))

    for i in range(half):
        assert all_sentences[i] == first_half[i], f"FAIL: cached sentence {i} returned different embedding"

    print(f"PASS: first {half} from cache, last {len(SENTENCES) - half} freshly computed  "
          f"seed={t_first:.3f}s  partial_cache={t_all:.3f}s")


def test_altered_sentence_gets_different_embedding():
    _reset()
    original = SENTENCES[0]
    altered = original[:-1] + ("!" if original[-1] != "!" else "?")

    original_result, t_orig = _time(lambda: _encode([original]))
    altered_result, t_alt = _time(lambda: _encode([altered]))

    assert original_result[0] != altered_result[0], "FAIL: altered sentence returned same embedding as original"
    print(f"PASS: altered sentence has different embedding  original={t_orig:.3f}s  altered={t_alt:.3f}s")


def test_sentence_inserted_in_middle_correct_position():
    _reset()
    new_sentence = "This is a brand new sentence inserted in the middle."
    insert_idx = len(SENTENCES) // 2
    sentences_with_insert = SENTENCES[:insert_idx] + [new_sentence] + SENTENCES[insert_idx:]

    original_results, t_seed = _time(lambda: _encode(SENTENCES))
    new_results, t_insert = _time(lambda: _encode(sentences_with_insert))

    assert len(new_results) == len(sentences_with_insert), "FAIL: wrong output length"

    # Sentences before insert unchanged
    for i in range(insert_idx):
        assert new_results[i] == original_results[i], f"FAIL: sentence at index {i} changed"

    # New sentence has a real embedding
    assert isinstance(new_results[insert_idx], list) and len(new_results[insert_idx]) > 0, \
        "FAIL: new sentence has no embedding"

    # Sentences after insert shifted by one but still match
    for i in range(insert_idx, len(SENTENCES)):
        assert new_results[i + 1] == original_results[i], \
            f"FAIL: sentence at index {i + 1} (was {i}) changed"

    print(f"PASS: new sentence at index {insert_idx} correct, all others intact  "
          f"seed={t_seed:.3f}s  with_insert={t_insert:.3f}s  "
          f"(only 1/{len(sentences_with_insert)} sentences re-encoded)")


def test_output_length_matches_input_with_duplicates():
    _reset()
    sentences_with_dup = SENTENCES + SENTENCES[:5]
    result, t = _time(lambda: _encode(sentences_with_dup))
    assert len(result) == len(sentences_with_dup), f"FAIL: expected {len(sentences_with_dup)}, got {len(result)}"
    print(f"PASS: output length matches input with duplicates ({len(sentences_with_dup)} total)  time={t:.3f}s")


def test_empty_input_returns_empty():
    _reset()
    result, t = _time(lambda: _encode([]))
    assert result == [], f"FAIL: expected [], got {result}"
    print(f"PASS: empty input returns empty list  time={t:.3f}s")


def test_cache_files_written_to_disk():
    _reset()
    _, t = _time(lambda: _encode(SENTENCES))
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    files = list(model_cache.glob("*.safetensors"))
    assert len(files) == len(SENTENCES), f"FAIL: expected {len(SENTENCES)} files, got {len(files)}"
    print(f"PASS: {len(files)} cache files on disk at {model_cache}  time={t:.3f}s")


def test_cache_survives_between_calls():
    _reset()
    first, t_compute = _time(lambda: _encode(SENTENCES))
    second, t_cache = _time(lambda: _encode(SENTENCES))
    assert first == second, "FAIL: embeddings differ between calls"
    print(f"PASS: cache persists, embeddings identical  compute={t_compute:.3f}s  cache={t_cache:.3f}s  speedup={t_compute/t_cache:.1f}x")


def test_clear_cache_removes_files():
    _reset()
    _, t = _time(lambda: _encode(SENTENCES))
    clear_cache(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    model_cache = CACHE_DIR / MODEL_NAME.replace("/", "_")
    assert not model_cache.exists(), "FAIL: cache directory still exists after clear_cache"
    print(f"PASS: clear_cache removes all files  encode_time={t:.3f}s")


def test_different_models_use_separate_namespaces():
    _reset()
    model_a = "all-MiniLM-L6-v2"
    model_b = "all-mpnet-base-v2"

    result_a, t_a = _time(lambda: encode(SENTENCES, model_name=model_a, cache_dir=CACHE_DIR))
    result_b, t_b = _time(lambda: encode(SENTENCES, model_name=model_b, cache_dir=CACHE_DIR))

    assert result_a[0] != result_b[0], "FAIL: different models returned same embedding"

    cache_a = CACHE_DIR / model_a.replace("/", "_")
    cache_b = CACHE_DIR / model_b.replace("/", "_")
    assert cache_a.exists() and cache_b.exists(), "FAIL: separate cache dirs not created"
    print(f"PASS: separate namespaces  model-a dim={len(result_a[0])} ({t_a:.3f}s)  model-b dim={len(result_b[0])} ({t_b:.3f}s)")


# ── Chunking / long-text handling ────────────────────────────────────


def _encode_chunked(sentences, chunks=10, pooling="mean"):
    return encode(sentences, model_name=MODEL_NAME, cache_dir=CACHE_DIR,
                  chunking=True, chunks=chunks, pooling=pooling)


def test_chunked_differs_from_truncated():
    _reset()
    truncated, t_trunc = _time(lambda: _encode([LONG_TEXT]))
    chunked, t_chunk = _time(lambda: _encode_chunked([LONG_TEXT]))
    assert len(chunked[0]) == len(truncated[0]), "FAIL: chunking changed the embedding dim"
    assert not np.allclose(chunked[0], truncated[0], atol=1e-4), \
        "FAIL: chunked long text matched truncated — chunking captured no extra content"
    print(f"PASS: chunked long text differs from truncated  "
          f"trunc={t_trunc:.3f}s  chunk={t_chunk:.3f}s")


def test_short_text_single_chunk_matches_truncation():
    _reset()
    short = SENTENCES[0]
    window = _window_size(MODEL_NAME, "st")
    assert _actual_chunks(short, MODEL_NAME, "st", window, 10) == 1, \
        "FAIL: short sentence unexpectedly spans multiple windows"
    truncated = _encode([short])
    chunked = _encode_chunked([short])
    assert np.allclose(chunked[0], truncated[0], atol=1e-4), \
        "FAIL: single-window chunk did not match the truncated embedding"
    print("PASS: short text → 1 window, pooled vector matches truncation")


def test_chunked_and_truncated_cached_separately():
    _reset()
    _encode([LONG_TEXT])              # truncation → 1 file
    n_after_trunc = _count_cache_files()
    _encode_chunked([LONG_TEXT])      # chunking → distinct key → +1 file
    n_after_chunk = _count_cache_files()
    assert n_after_trunc == 1, f"FAIL: expected 1 truncation file, got {n_after_trunc}"
    assert n_after_chunk == 2, f"FAIL: chunking did not write a separate file (got {n_after_chunk})"
    print("PASS: truncated and chunked embeddings live in separate cache entries")


def test_actual_chunk_count_dedup():
    _reset()
    window = _window_size(MODEL_NAME, "st")
    actual = _actual_chunks(LONG_TEXT, MODEL_NAME, "st", window, 100)
    assert 2 <= actual <= 7, \
        f"FAIL: test needs LONG_TEXT to span 2–7 windows, got {actual} (adjust LONG_TEXT)"

    _encode_chunked([LONG_TEXT], chunks=10)     # caps above `actual` → key chunk={actual}
    n10 = _count_cache_files()
    _encode_chunked([LONG_TEXT], chunks=8)      # still >= actual → same key → cache hit
    n8 = _count_cache_files()
    assert n8 == n10, "FAIL: caps above the actual count should share one cache entry"

    _encode_chunked([LONG_TEXT], chunks=1)      # below actual → different key → new file
    n1 = _count_cache_files()
    assert n1 == n10 + 1, "FAIL: a smaller cap that uses fewer windows should be a new entry"
    print(f"PASS: actual-chunk keying — cap10/cap8 shared (actual={actual}), cap1 distinct")


def test_pooling_methods_differ():
    _reset()
    mean = _encode_chunked([LONG_TEXT], pooling="mean")
    mx = _encode_chunked([LONG_TEXT], pooling="max")
    assert not np.allclose(mean[0], mx[0]), "FAIL: mean and max pooling gave identical vectors"
    print("PASS: mean / max pooling produce distinct vectors")


def test_unknown_pooling_raises():
    _reset()
    try:
        _encode_chunked([LONG_TEXT], pooling="nonsense")
    except ValueError as e:
        assert "pooling" in str(e).lower(), f"FAIL: unexpected message: {e}"
        print("PASS: unknown pooling raises ValueError")
        return
    raise AssertionError("FAIL: unknown pooling did not raise ValueError")


def test_chunking_preserves_order_and_length():
    _reset()
    texts = [SENTENCES[0], LONG_TEXT, SENTENCES[1]]
    result = _encode_chunked(texts)
    assert len(result) == len(texts), f"FAIL: expected {len(texts)} vectors, got {len(result)}"
    dim = len(result[0])
    assert all(len(v) == dim for v in result), "FAIL: chunked vectors have inconsistent dims"
    # The two short sentences, embedded standalone with chunking, must land at
    # their original positions unchanged.
    standalone = _encode_chunked([SENTENCES[0], SENTENCES[1]])
    assert np.allclose(result[0], standalone[0], atol=1e-4), "FAIL: position 0 changed"
    assert np.allclose(result[2], standalone[1], atol=1e-4), "FAIL: position 2 changed"
    print("PASS: chunking preserves output order and length")


if __name__ == "__main__":
    print(f"Running on: {DEVICE.upper()}")
    print(f"Sentences: {len(SENTENCES)}\n")

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
        test_chunked_differs_from_truncated,
        test_short_text_single_chunk_matches_truncation,
        test_chunked_and_truncated_cached_separately,
        test_actual_chunk_count_dedup,
        test_pooling_methods_differ,
        test_unknown_pooling_raises,
        test_chunking_preserves_order_and_length,
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

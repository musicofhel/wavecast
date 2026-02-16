"""Tests for Rust/PyO3 acceleration of SAX/BoW/tokenizer hot paths."""
from __future__ import annotations

import time

import numpy as np
import pytest

from wavecast._rust import HAS_RUST

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not compiled")


# ---------------------------------------------------------------------------
# extract_words
# ---------------------------------------------------------------------------


class TestExtractWords:
    def test_basic(self):
        from wavecast._rust import extract_words_rs

        symbols = "abcdefgh"
        py_result = self._py_extract(symbols, 3, 1)
        rs_result = extract_words_rs(symbols, 3, 1)
        assert rs_result == py_result

    def test_stride(self):
        from wavecast._rust import extract_words_rs

        symbols = "abcdefghij"
        rs = extract_words_rs(symbols, 3, 2)
        assert rs == ["abc", "cde", "efg", "ghi"]

    def test_empty_string(self):
        from wavecast._rust import extract_words_rs

        assert extract_words_rs("", 3, 1) == []

    def test_short_string(self):
        from wavecast._rust import extract_words_rs

        assert extract_words_rs("ab", 3, 1) == []

    def test_word_length_equals_string(self):
        from wavecast._rust import extract_words_rs

        assert extract_words_rs("abc", 3, 1) == ["abc"]

    def test_zero_word_length(self):
        from wavecast._rust import extract_words_rs

        assert extract_words_rs("abc", 0, 1) == []

    def test_zero_stride(self):
        from wavecast._rust import extract_words_rs

        assert extract_words_rs("abc", 2, 0) == []

    def test_matches_python_long_string(self):
        from wavecast._rust import extract_words_rs

        symbols = "a" * 500 + "b" * 500
        py = self._py_extract(symbols, 4, 1)
        rs = extract_words_rs(symbols, 4, 1)
        assert rs == py

    @staticmethod
    def _py_extract(symbols: str, wl: int, stride: int) -> list[str]:
        """Pure Python reference implementation."""
        if wl <= 0 or stride <= 0:
            return []
        words = []
        for i in range(0, len(symbols) - wl + 1, stride):
            words.append(symbols[i : i + wl])
        return words


# ---------------------------------------------------------------------------
# build_bow
# ---------------------------------------------------------------------------


class TestBuildBow:
    def test_basic(self):
        from wavecast._rust import build_bow_rs

        words = ["ab", "cd", "ab", "ef", "cd", "ab"]
        result = build_bow_rs(words)
        assert result == {"ab": 3, "cd": 2, "ef": 1}

    def test_empty(self):
        from wavecast._rust import build_bow_rs

        assert build_bow_rs([]) == {}

    def test_single_word(self):
        from wavecast._rust import build_bow_rs

        assert build_bow_rs(["xyz"]) == {"xyz": 1}

    def test_matches_python(self):
        from collections import Counter

        from wavecast._rust import build_bow_rs

        words = ["aa", "bb", "aa", "cc", "bb", "aa", "dd"]
        py = dict(Counter(words))
        rs = build_bow_rs(words)
        assert rs == py


# ---------------------------------------------------------------------------
# build_corpus_tfidf
# ---------------------------------------------------------------------------


class TestBuildCorpusTfidf:
    def test_basic_shape(self):
        from wavecast._rust import build_corpus_tfidf_rs

        bows = [{"a": 2, "b": 1}, {"b": 3, "c": 1}]
        matrix, vocab = build_corpus_tfidf_rs(bows)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 3)
        assert vocab == ["a", "b", "c"]

    def test_empty_list(self):
        from wavecast._rust import build_corpus_tfidf_rs

        matrix, vocab = build_corpus_tfidf_rs([])
        assert matrix.shape == (0, 0)
        assert vocab == []

    def test_matches_python(self):
        from wavecast._rust import build_corpus_tfidf_rs
        from wavecast.sax.bow import build_corpus_tfidf

        bows = [
            {"abc": 5, "def": 2, "ghi": 1},
            {"abc": 1, "jkl": 4},
            {"def": 3, "ghi": 2, "jkl": 1},
        ]
        py_matrix, py_vocab = build_corpus_tfidf(bows)
        rs_matrix, rs_vocab = build_corpus_tfidf_rs(bows)

        assert rs_vocab == py_vocab
        np.testing.assert_allclose(rs_matrix, py_matrix, atol=1e-12)


# ---------------------------------------------------------------------------
# encode_batch
# ---------------------------------------------------------------------------


class TestEncodeBatch:
    def test_basic(self):
        from wavecast._rust import encode_batch_rs

        word_to_id = {"<PAD>": 0, "<UNK>": 1, "abc": 2, "def": 3}
        result = encode_batch_rs(["abc", "def", "xyz", "abc"], word_to_id, 1)
        assert result == [2, 3, 1, 2]

    def test_all_unknown(self):
        from wavecast._rust import encode_batch_rs

        result = encode_batch_rs(["x", "y"], {"a": 0}, 99)
        assert result == [99, 99]

    def test_empty(self):
        from wavecast._rust import encode_batch_rs

        assert encode_batch_rs([], {}, 1) == []

    def test_matches_python(self):
        from wavecast._rust import encode_batch_rs
        from wavecast.tokenizer.vocabulary import SAXVocabulary

        vocab = SAXVocabulary.from_corpus([["ab", "cd", "ab", "ef", "cd", "ab"]], min_freq=1)
        words = ["ab", "cd", "ef", "unknown", "ab"]
        py_result = vocab.encode_sequence(words)

        # Build the same word_to_id map for Rust
        rs_result = encode_batch_rs(words, vocab._word_to_id, 1)
        assert rs_result == py_result


# ---------------------------------------------------------------------------
# build_sliding_windows
# ---------------------------------------------------------------------------


class TestBuildSlidingWindows:
    def test_basic(self):
        from wavecast._rust import build_sliding_windows_rs

        tokens = [10, 20, 30, 40, 50, 60]
        ctx, tgt = build_sliding_windows_rs(tokens, 3, 1)

        assert ctx.shape == (3, 3)
        assert tgt.shape == (3, 1)
        np.testing.assert_array_equal(ctx[0], [10, 20, 30])
        np.testing.assert_array_equal(ctx[1], [20, 30, 40])
        np.testing.assert_array_equal(ctx[2], [30, 40, 50])
        np.testing.assert_array_equal(tgt.flatten(), [40, 50, 60])

    def test_multi_horizon(self):
        from wavecast._rust import build_sliding_windows_rs

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ctx, tgt = build_sliding_windows_rs(tokens, 3, 2)

        assert ctx.shape == (4, 3)
        assert tgt.shape == (4, 2)
        np.testing.assert_array_equal(ctx[0], [1, 2, 3])
        np.testing.assert_array_equal(tgt[0], [4, 5])

    def test_too_short(self):
        from wavecast._rust import build_sliding_windows_rs

        ctx, tgt = build_sliding_windows_rs([1, 2], 3, 1)
        assert ctx.shape == (0, 3)
        assert tgt.shape == (0, 1)

    def test_exact_fit(self):
        from wavecast._rust import build_sliding_windows_rs

        ctx, tgt = build_sliding_windows_rs([1, 2, 3, 4], 3, 1)
        assert ctx.shape == (1, 3)
        np.testing.assert_array_equal(ctx[0], [1, 2, 3])
        np.testing.assert_array_equal(tgt[0], [4])


# ---------------------------------------------------------------------------
# Benchmark: Rust vs Python on extract_words with 10K char string
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_extract_words_speed(self):
        from wavecast._rust import extract_words_rs

        # Build a 10K char SAX string
        symbols = "abcdefg" * 1429  # ~10003 chars
        symbols = symbols[:10000]
        word_length = 4
        stride = 1

        # Python timing
        start = time.perf_counter()
        for _ in range(10):
            py_result = TestExtractWords._py_extract(symbols, word_length, stride)
        py_time = time.perf_counter() - start

        # Rust timing
        start = time.perf_counter()
        for _ in range(10):
            rs_result = extract_words_rs(symbols, word_length, stride)
        rs_time = time.perf_counter() - start

        # Results must match
        assert rs_result == py_result

        # Print speedup (informational, not a hard assertion)
        speedup = py_time / rs_time if rs_time > 0 else float("inf")
        print(f"\nextract_words 10K chars x10: Python={py_time:.4f}s, Rust={rs_time:.4f}s, "
              f"speedup={speedup:.1f}x")

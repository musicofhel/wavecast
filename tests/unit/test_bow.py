"""Tests for wavecast.sax.bow."""

import numpy as np

from wavecast.sax.bow import (
    bow_to_vector,
    build_bow,
    build_corpus_tfidf,
    extract_words,
)


def test_extract_words_count():
    symbols = "abcdefgh"
    words = extract_words(symbols, word_length=4, stride=1)
    assert len(words) == len(symbols) - 4 + 1
    assert words[0] == "abcd"
    assert words[-1] == "efgh"


def test_extract_words_stride():
    symbols = "abcdefgh"
    words = extract_words(symbols, word_length=2, stride=2)
    assert words == ["ab", "cd", "ef", "gh"]


def test_extract_words_empty():
    assert extract_words("", word_length=4) == []


def test_extract_words_too_short():
    assert extract_words("ab", word_length=4) == []


def test_build_bow_frequencies():
    words = ["ab", "cd", "ab", "ef", "ab", "cd"]
    bow = build_bow(words)
    assert bow["ab"] == 3
    assert bow["cd"] == 2
    assert bow["ef"] == 1
    assert sum(bow.values()) == 6


def test_build_bow_empty():
    bow = build_bow([])
    assert len(bow) == 0


def test_bow_to_vector():
    bow = {"ab": 3, "cd": 2, "ef": 1}
    vocab = ["ab", "cd", "ef", "gh"]
    vec = bow_to_vector(bow, vocab)
    np.testing.assert_array_equal(vec, [3, 2, 1, 0])


def test_bow_to_vector_unknown_words():
    bow = {"ab": 1, "zz": 5}
    vocab = ["ab", "cd"]
    vec = bow_to_vector(bow, vocab)
    np.testing.assert_array_equal(vec, [1, 0])


def test_tfidf_shape():
    bow_list = [
        {"ab": 3, "cd": 2},
        {"cd": 1, "ef": 4},
        {"ab": 1, "ef": 1},
    ]
    tfidf, vocab = build_corpus_tfidf(bow_list)
    assert tfidf.shape == (3, len(vocab))
    assert len(vocab) == 3  # ab, cd, ef
    assert np.all(np.isfinite(tfidf))


def test_tfidf_empty_corpus():
    tfidf, vocab = build_corpus_tfidf([])
    assert tfidf.shape[0] == 0
    assert len(vocab) == 0

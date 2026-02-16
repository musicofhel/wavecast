"""Bag-of-Words (BoW) and TF-IDF over SAX words."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def extract_words(sax_symbols: str, word_length: int, stride: int = 1) -> list[str]:
    """Extract SAX words using a sliding window.

    Args:
        sax_symbols: SAX symbol string.
        word_length: Length of each word.
        stride: Step size for the sliding window.

    Returns:
        List of SAX word strings.
    """
    from wavecast._rust import HAS_RUST, extract_words_rs

    if HAS_RUST and word_length > 0 and stride > 0:
        return extract_words_rs(sax_symbols, word_length, stride)
    if word_length <= 0 or stride <= 0:
        return []
    words = []
    for i in range(0, len(sax_symbols) - word_length + 1, stride):
        words.append(sax_symbols[i : i + word_length])
    return words


def build_bow(words: list[str]) -> dict[str, int]:
    """Build a bag-of-words frequency counter.

    Args:
        words: List of SAX words.

    Returns:
        Dictionary mapping each word to its count.
    """
    from wavecast._rust import HAS_RUST, build_bow_rs

    if HAS_RUST:
        return build_bow_rs(words)
    return dict(Counter(words))


def build_corpus_tfidf(
    bow_list: list[dict[str, int]],
) -> tuple[NDArray, list[str]]:
    """Build TF-IDF matrix from a list of bag-of-words.

    Args:
        bow_list: List of BoW dictionaries (one per document).

    Returns:
        Tuple of (tfidf_matrix of shape (n_docs, vocab_size), vocabulary list).
    """
    if not bow_list:
        return np.empty((0, 0), dtype=np.float64), []

    # Build vocabulary from all bows
    vocab_set: set[str] = set()
    for bow in bow_list:
        vocab_set.update(bow.keys())
    vocabulary = sorted(vocab_set)
    vocab_index = {w: i for i, w in enumerate(vocabulary)}

    n_docs = len(bow_list)
    n_vocab = len(vocabulary)

    if n_vocab == 0:
        return np.zeros((n_docs, 0), dtype=np.float64), []

    # Build term frequency matrix
    tf = np.zeros((n_docs, n_vocab), dtype=np.float64)
    for d, bow in enumerate(bow_list):
        total = sum(bow.values())
        if total > 0:
            for word, count in bow.items():
                tf[d, vocab_index[word]] = count / total

    # Document frequency and IDF
    df = np.zeros(n_vocab, dtype=np.float64)
    for bow in bow_list:
        for word in bow:
            df[vocab_index[word]] += 1

    idf = np.log(n_docs / np.where(df > 0, df, 1))

    tfidf = tf * idf

    return tfidf, vocabulary


def bow_to_vector(bow: dict[str, int], vocabulary: list[str]) -> NDArray:
    """Convert a BoW to a fixed-length vector aligned to a vocabulary.

    Args:
        bow: Bag-of-words dictionary.
        vocabulary: Ordered list of vocabulary words.

    Returns:
        Array of length len(vocabulary) with counts.
    """
    vec = np.zeros(len(vocabulary), dtype=np.float64)
    vocab_index = {w: i for i, w in enumerate(vocabulary)}
    for word, count in bow.items():
        if word in vocab_index:
            vec[vocab_index[word]] = count
    return vec


@dataclass
class SAXBowResult:
    """Result of SAX BoW extraction."""

    words: list[str]
    bow: dict[str, int]
    vocabulary: list[str] = field(default_factory=list)
    tfidf_vector: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

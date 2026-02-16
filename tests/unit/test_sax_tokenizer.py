"""Tests for SAX tokenizer."""

import numpy as np

from wavecast.core.config import SAXConfig
from wavecast.core.types import WaveletDecomposition
from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose


def _make_vocab():
    """Build a small vocabulary from deterministic words."""
    words = [
        ["abcd", "bcde", "cdef", "defg"] * 5,
        ["abcd", "bcde", "efgh", "fgha"] * 5,
    ]
    return SAXVocabulary.from_corpus(words, min_freq=1, max_size=100)


def test_tokenize_produces_multi_level_sequence(sine_series):
    decomp = decompose(sine_series, level=3)
    vocab = _make_vocab()
    sax_cfg = SAXConfig(n_segments=20, alphabet_size=4, word_length=4, word_stride=1)
    tokenizer = WaveletSAXTokenizer(vocab, sax_cfg)

    result = tokenizer.tokenize(decomp)
    assert result.ticker == "SINE"
    assert len(result.level_sequences) == 3


def test_tokenize_level_sequences_have_correct_levels(sine_series):
    decomp = decompose(sine_series, level=3)
    vocab = _make_vocab()
    sax_cfg = SAXConfig(n_segments=20, alphabet_size=4, word_length=4, word_stride=1)
    tokenizer = WaveletSAXTokenizer(vocab, sax_cfg)

    result = tokenizer.tokenize(decomp)
    assert set(result.level_sequences.keys()) == {1, 2, 3}

    for lvl, seq in result.level_sequences.items():
        assert seq.wavelet_level == lvl
        assert seq.ticker == "SINE"


def test_tokenize_produces_token_ids(sine_series):
    decomp = decompose(sine_series, level=3)
    vocab = _make_vocab()
    sax_cfg = SAXConfig(n_segments=20, alphabet_size=4, word_length=4, word_stride=1)
    tokenizer = WaveletSAXTokenizer(vocab, sax_cfg)

    result = tokenizer.tokenize(decomp)
    for seq in result.level_sequences.values():
        assert len(seq.token_ids) == len(seq.words)
        assert all(isinstance(tid, int) for tid in seq.token_ids)


def test_tokenize_short_coefficients():
    coeffs = [
        np.array([1.0]),  # approx
        np.array([2.0]),  # detail level 1 (len 1 -- too short)
    ]
    decomp = WaveletDecomposition(
        coefficients=coeffs, wavelet="db4", level=1, original_length=2, ticker="T"
    )
    vocab = _make_vocab()
    sax_cfg = SAXConfig()
    tokenizer = WaveletSAXTokenizer(vocab, sax_cfg)

    result = tokenizer.tokenize(decomp)
    assert result.level_sequences[1].token_ids == []
    assert result.level_sequences[1].words == []

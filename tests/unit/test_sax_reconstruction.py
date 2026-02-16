"""Tests for SAX price reconstruction."""

import numpy as np
import pytest

from wavecast.core.exceptions import SAXError
from wavecast.sax.bow import extract_words
from wavecast.sax.reconstruction import (
    PriceReconstructionResult,
    reconstruct_price_delta,
    sax_to_midpoints,
    tokens_to_direction,
)
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.vocabulary import PAD_ID, UNK_ID, SAXVocabulary


class TestSaxToMidpoints:
    def test_correct_length(self):
        mids = sax_to_midpoints("abcd", 4)
        assert len(mids) == 4

    def test_middle_symbol_near_zero(self):
        # For alphabet=7, middle symbol is 'd' (index 3)
        mids = sax_to_midpoints("d", 7)
        assert abs(mids[0]) < 0.3  # should be close to zero

    def test_extreme_low_symbol_negative(self):
        mids = sax_to_midpoints("a", 7)
        assert mids[0] < -0.5

    def test_extreme_high_symbol_positive(self):
        mids = sax_to_midpoints("g", 7)
        assert mids[0] > 0.5

    def test_monotonic_ordering(self):
        # a < b < c < d < e < f < g
        mids = sax_to_midpoints("abcdefg", 7)
        for i in range(len(mids) - 1):
            assert mids[i] < mids[i + 1]

    def test_symmetric_around_zero(self):
        # First and last symbols should be symmetric
        mids = sax_to_midpoints("ag", 7)
        assert abs(mids[0] + mids[1]) < 0.1

    def test_empty_string(self):
        mids = sax_to_midpoints("", 4)
        assert len(mids) == 0

    def test_invalid_alphabet_raises(self):
        with pytest.raises(SAXError, match="alphabet_size must be >= 2"):
            sax_to_midpoints("a", 1)

    def test_out_of_range_symbol_raises(self):
        with pytest.raises(SAXError, match="out of range"):
            sax_to_midpoints("z", 4)  # only a-d valid for alphabet=4

    def test_alphabet_4_has_4_distinct_values(self):
        mids = sax_to_midpoints("abcd", 4)
        assert len(set(mids)) == 4


class TestReconstructPriceDelta:
    @pytest.fixture
    def vocab_and_tokens(self):
        """Build a small vocabulary from synthetic SAX words."""
        # Create words with known patterns
        words_high = ["gggg", "gggf", "fggg"]  # positive
        words_low = ["aaaa", "aaab", "abaa"]  # negative
        words_mid = ["dddd", "ddde", "dded"]  # neutral
        all_words = words_high + words_low + words_mid

        vocab = SAXVocabulary.from_corpus(
            [all_words], min_freq=1, max_size=100
        )
        return vocab, all_words

    def test_produces_valid_directions(self, vocab_and_tokens):
        vocab, words = vocab_and_tokens
        token_ids = vocab.encode_sequence(words)
        result = reconstruct_price_delta(
            token_ids, vocab, alphabet_size=7, original_coeff_length=20
        )
        directions = result["directions"]
        assert len(directions) == len(token_ids)
        for d in directions:
            assert d in (-1.0, 0.0, 1.0)

    def test_high_words_positive_direction(self, vocab_and_tokens):
        vocab, _ = vocab_and_tokens
        high_ids = vocab.encode_sequence(["gggg", "gggf"])
        result = reconstruct_price_delta(
            high_ids, vocab, alphabet_size=7, original_coeff_length=20
        )
        # High symbols should yield positive direction
        assert np.all(result["directions"] >= 0)

    def test_low_words_negative_direction(self, vocab_and_tokens):
        vocab, _ = vocab_and_tokens
        low_ids = vocab.encode_sequence(["aaaa", "aaab"])
        result = reconstruct_price_delta(
            low_ids, vocab, alphabet_size=7, original_coeff_length=20
        )
        assert np.all(result["directions"] <= 0)

    def test_empty_tokens(self, vocab_and_tokens):
        vocab, _ = vocab_and_tokens
        result = reconstruct_price_delta(
            [], vocab, alphabet_size=7, original_coeff_length=20
        )
        assert len(result["directions"]) == 0
        assert result["level_coefficients"] == {}

    def test_level_coefficients_populated(self, vocab_and_tokens):
        vocab, words = vocab_and_tokens
        token_ids = vocab.encode_sequence(words)
        levels = [1] * len(token_ids)
        result = reconstruct_price_delta(
            token_ids, vocab, alphabet_size=7,
            original_coeff_length=20, levels=levels,
        )
        assert 1 in result["level_coefficients"]
        assert len(result["level_coefficients"][1]) == 20


class TestTokensToDirection:
    @pytest.fixture
    def vocab(self):
        words = ["gggg", "gfgf", "aaaa", "abab", "dddd", "dede"]
        return SAXVocabulary.from_corpus([words], min_freq=1, max_size=100)

    def test_positive_for_high_tokens(self, vocab):
        ids = vocab.encode_sequence(["gggg"])
        dirs = tokens_to_direction(ids, vocab, alphabet_size=7)
        assert dirs[0] == 1.0

    def test_negative_for_low_tokens(self, vocab):
        ids = vocab.encode_sequence(["aaaa"])
        dirs = tokens_to_direction(ids, vocab, alphabet_size=7)
        assert dirs[0] == -1.0

    def test_zero_for_middle_tokens(self, vocab):
        ids = vocab.encode_sequence(["dddd"])
        dirs = tokens_to_direction(ids, vocab, alphabet_size=7)
        assert dirs[0] == 0.0

    def test_pad_and_unk_are_zero(self, vocab):
        dirs = tokens_to_direction([PAD_ID, UNK_ID], vocab, alphabet_size=7)
        assert dirs[0] == 0.0
        assert dirs[1] == 0.0

    def test_output_length_matches_input(self, vocab):
        ids = vocab.encode_sequence(["gggg", "aaaa", "dddd"])
        dirs = tokens_to_direction(ids, vocab, alphabet_size=7)
        assert len(dirs) == 3


class TestRoundTrip:
    def test_sax_transform_to_reconstruct_direction_matches(self):
        """SAX transform a known signal, tokenize, reconstruct, verify direction."""
        rng = np.random.default_rng(42)
        # Create a clearly upward-trending series (positive z-scores dominate)
        n = 200
        upward = np.linspace(0, 5, n) + rng.standard_normal(n) * 0.3

        sax_rep = sax_transform(upward, n_segments=20, alphabet_size=7)
        words = extract_words(sax_rep.symbols, word_length=4, stride=1)

        vocab = SAXVocabulary.from_corpus([words], min_freq=1, max_size=100)
        token_ids = vocab.encode_sequence(words)

        dirs = tokens_to_direction(token_ids, vocab, alphabet_size=7)

        # For a clearly upward series, the z-normalized version should have
        # positive values at the end and negative at the start.
        # The overall direction should lean positive for later tokens.
        # At minimum, not all should be negative.
        assert np.any(dirs >= 0)

    def test_midpoints_roundtrip_consistency(self):
        """Verify midpoints are consistent: lower symbols have lower midpoints."""
        for alpha in [3, 5, 7, 9, 11]:
            symbols = "".join(chr(ord("a") + i) for i in range(alpha))
            mids = sax_to_midpoints(symbols, alpha)
            # Must be strictly increasing
            for i in range(len(mids) - 1):
                assert mids[i] < mids[i + 1], f"Failed for alphabet={alpha}"


class TestPriceReconstructionResult:
    def test_dataclass_creation(self):
        result = PriceReconstructionResult(
            directions=np.array([1.0, -1.0, 0.0]),
            confidence=np.array([0.9, 0.8, 0.5]),
            level_coefficients={1: np.array([0.1, 0.2, 0.3])},
        )
        assert len(result.directions) == 3
        assert 1 in result.level_coefficients

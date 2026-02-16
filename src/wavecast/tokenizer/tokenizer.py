"""Wavelet-SAX tokenizer for converting decompositions to token sequences."""

from __future__ import annotations

from wavecast.core.config import SAXConfig
from wavecast.core.types import (
    MultiLevelTokenSequence,
    TokenSequence,
    WaveletDecomposition,
)
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.vocabulary import SAXVocabulary


class WaveletSAXTokenizer:
    """Tokenizes wavelet decompositions via SAX transformation.

    For each DWT detail level, performs SAX transform, extracts words,
    and encodes them with the vocabulary.
    """

    def __init__(self, vocabulary: SAXVocabulary, sax_config: SAXConfig) -> None:
        self.vocabulary = vocabulary
        self.sax_config = sax_config

    def tokenize(self, decomp: WaveletDecomposition) -> MultiLevelTokenSequence:
        """Tokenize a wavelet decomposition into multi-level token sequences.

        Args:
            decomp: Wavelet decomposition result.

        Returns:
            MultiLevelTokenSequence with token IDs per level.
        """
        level_sequences: dict[int, TokenSequence] = {}

        for lvl in range(1, decomp.level + 1):
            coeffs = decomp.detail_at_level(lvl)

            if len(coeffs) < 2:
                level_sequences[lvl] = TokenSequence(
                    token_ids=[],
                    words=[],
                    ticker=decomp.ticker,
                    interval="",
                    wavelet_level=lvl,
                )
                continue

            n_seg = min(self.sax_config.n_segments, len(coeffs))
            sax_rep = sax_transform(coeffs, n_seg, self.sax_config.alphabet_size)
            words = extract_words(
                sax_rep.symbols,
                self.sax_config.word_length,
                self.sax_config.word_stride,
            )
            token_ids = self.vocabulary.encode_sequence(words)

            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids,
                words=words,
                ticker=decomp.ticker,
                interval="",
                wavelet_level=lvl,
            )

        return MultiLevelTokenSequence(
            ticker=decomp.ticker,
            interval="",
            level_sequences=level_sequences,
        )

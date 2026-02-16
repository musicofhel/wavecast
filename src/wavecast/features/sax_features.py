"""SAX-domain feature extraction from wavelet decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import SAXConfig
from wavecast.core.types import WaveletDecomposition
from wavecast.sax.bow import bow_to_vector, build_bow, extract_words
from wavecast.sax.sax import sax_transform

# Fixed vocabulary size for BoW vectors per level
_VOCAB_SIZE = 20
# Per-level features: _VOCAB_SIZE bow counts + 1 unique word count = 21
_PER_LEVEL_SIZE = _VOCAB_SIZE + 1
# Max wavelet levels supported
_MAX_LEVELS = 10
# Cross-level features: total unique words + (MAX_LEVELS - 1) Jaccard overlaps
_CROSS_LEVEL_SIZE = 1 + (_MAX_LEVELS - 1)
# Total SAX feature size
SAX_FEATURE_SIZE = _MAX_LEVELS * _PER_LEVEL_SIZE + _CROSS_LEVEL_SIZE


def extract(
    decomp: WaveletDecomposition,
    sax_config: SAXConfig | None = None,
) -> NDArray[np.float64]:
    """Extract SAX-based features from a wavelet decomposition.

    For each detail level:
        - SAX transform the detail coefficients
        - Extract sliding-window words and build BoW histogram
        - Convert to fixed-size vector using top-N words as vocabulary

    Cross-level features:
        - Total unique words across all levels
        - Jaccard overlap between adjacent levels

    Args:
        decomp: Wavelet decomposition result.
        sax_config: SAX configuration (uses defaults if None).

    Returns:
        Feature vector of length SAX_FEATURE_SIZE.
    """
    if sax_config is None:
        sax_config = SAXConfig()

    all_features: list[float] = []
    level_word_sets: list[set[str]] = []

    # Per-level features for detail levels 1..decomp.level
    for lvl in range(1, decomp.level + 1):
        coeffs = decomp.detail_at_level(lvl)

        if len(coeffs) < 2:
            # Not enough data for SAX
            all_features.extend([0.0] * _PER_LEVEL_SIZE)
            level_word_sets.append(set())
            continue

        n_seg = min(sax_config.n_segments, len(coeffs))
        sax_rep = sax_transform(coeffs, n_seg, sax_config.alphabet_size)
        words = extract_words(sax_rep.symbols, sax_config.word_length, sax_config.word_stride)
        bow = build_bow(words)

        # Build vocabulary from top-N most common words
        sorted_words = sorted(bow.items(), key=lambda x: x[1], reverse=True)
        vocab = [w for w, _ in sorted_words[:_VOCAB_SIZE]]
        # Pad vocabulary to fixed size
        while len(vocab) < _VOCAB_SIZE:
            vocab.append(f"__pad_{len(vocab)}")

        vec = bow_to_vector(bow, vocab)
        unique_count = float(len(set(words)))

        all_features.extend(vec.tolist())
        all_features.append(unique_count)
        level_word_sets.append(set(words))

    # Pad remaining levels to _MAX_LEVELS
    for _ in range(decomp.level, _MAX_LEVELS):
        all_features.extend([0.0] * _PER_LEVEL_SIZE)
        level_word_sets.append(set())

    # Cross-level features
    # Total unique words across all levels
    all_unique = set()
    for ws in level_word_sets:
        all_unique.update(ws)
    all_features.append(float(len(all_unique)))

    # Jaccard overlap between adjacent levels
    for i in range(_MAX_LEVELS - 1):
        s1 = level_word_sets[i] if i < len(level_word_sets) else set()
        s2 = level_word_sets[i + 1] if (i + 1) < len(level_word_sets) else set()
        union = s1 | s2
        jaccard = float(len(s1 & s2)) / float(len(union)) if union else 0.0
        all_features.append(jaccard)

    return np.array(all_features, dtype=np.float64)

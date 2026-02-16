"""SAX-to-price reconstruction: map predicted SAX tokens back to approximate signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from wavecast.core.exceptions import SAXError
from wavecast.sax.paa import inverse_paa
from wavecast.sax.sax import breakpoints
from wavecast.tokenizer.vocabulary import PAD_TOKEN, UNK_TOKEN, SAXVocabulary


@dataclass
class PriceReconstructionResult:
    """Result of reconstructing price signals from predicted SAX tokens."""

    directions: NDArray  # +1/-1/0 directional signals
    confidence: NDArray  # softmax probabilities (if available)
    level_coefficients: dict[int, NDArray] = field(default_factory=dict)


def sax_to_midpoints(symbols: str, alphabet_size: int) -> NDArray:
    """Map SAX symbols to the midpoint of their breakpoint bins.

    For the extreme bins (first and last), a bounded approximation is used
    since the true bins extend to -inf and +inf respectively.

    Args:
        symbols: SAX string (e.g. "abcdg").
        alphabet_size: Size of the SAX alphabet.

    Returns:
        Array of midpoint values, one per symbol.
    """
    if not symbols:
        return np.array([], dtype=np.float64)
    if alphabet_size < 2:
        raise SAXError(f"alphabet_size must be >= 2, got {alphabet_size}")

    bps = breakpoints(alphabet_size)
    bin_edges = np.concatenate([[-np.inf], bps, [np.inf]])

    midpoints = np.zeros(len(symbols), dtype=np.float64)
    for i, ch in enumerate(symbols):
        idx = ord(ch) - ord("a")
        if idx < 0 or idx >= alphabet_size:
            raise SAXError(
                f"Symbol '{ch}' out of range for alphabet_size={alphabet_size}"
            )
        low = bin_edges[idx]
        high = bin_edges[idx + 1]

        if np.isinf(low):
            # First bin: midpoint is ppf(0.5 / alphabet_size)
            midpoints[i] = float(norm.ppf(0.5 / alphabet_size))
        elif np.isinf(high):
            # Last bin: midpoint is ppf(1 - 0.5 / alphabet_size)
            midpoints[i] = float(norm.ppf(1.0 - 0.5 / alphabet_size))
        else:
            midpoints[i] = (low + high) / 2.0

    return midpoints


def reconstruct_price_delta(
    predicted_tokens: list[int],
    vocabulary: SAXVocabulary,
    alphabet_size: int,
    original_coeff_length: int,
    levels: list[int] | None = None,
) -> dict:
    """Reconstruct approximate z-normalized coefficients from predicted token IDs.

    For each predicted token, decodes it to a SAX word, maps to midpoints,
    and expands via inverse PAA to original coefficient length.

    Args:
        predicted_tokens: List of predicted token IDs.
        vocabulary: SAXVocabulary used for decoding.
        alphabet_size: SAX alphabet size used during encoding.
        original_coeff_length: Length to expand each word's midpoints to.
        levels: Optional per-token wavelet level assignments. If None,
            all tokens are treated as level 0.

    Returns:
        Dict with:
        - "directions": NDArray of +1/-1/0 per token
        - "level_coefficients": dict[int, NDArray] of per-level reconstructed coefficients
    """
    if not predicted_tokens:
        return {
            "directions": np.array([], dtype=np.float64),
            "level_coefficients": {},
        }

    if levels is None:
        levels = [0] * len(predicted_tokens)

    level_words: dict[int, list[str]] = {}
    for tok_id, lvl in zip(predicted_tokens, levels, strict=True):
        word = vocabulary.decode(tok_id)
        if word in (PAD_TOKEN, UNK_TOKEN):
            continue
        level_words.setdefault(lvl, []).append(word)

    level_coefficients: dict[int, NDArray] = {}
    for lvl, words in level_words.items():
        # Reconstruct each word to midpoints, then average across the level
        all_expanded: list[NDArray] = []
        for word in words:
            mids = sax_to_midpoints(word, alphabet_size)
            expanded = inverse_paa(mids, original_coeff_length)
            all_expanded.append(expanded)
        if all_expanded:
            level_coefficients[lvl] = np.mean(all_expanded, axis=0)

    # Directional signal from each token's overall midpoint sign
    directions = np.zeros(len(predicted_tokens), dtype=np.float64)
    for i, tok_id in enumerate(predicted_tokens):
        word = vocabulary.decode(tok_id)
        if word in (PAD_TOKEN, UNK_TOKEN):
            directions[i] = 0.0
            continue
        mids = sax_to_midpoints(word, alphabet_size)
        mean_mid = np.mean(mids)
        if mean_mid > 0.1:
            directions[i] = 1.0
        elif mean_mid < -0.1:
            directions[i] = -1.0
        else:
            directions[i] = 0.0

    return {
        "directions": directions,
        "level_coefficients": level_coefficients,
    }


def tokens_to_direction(
    predicted_tokens: list[int],
    vocabulary: SAXVocabulary,
    alphabet_size: int,
) -> NDArray:
    """Extract directional signals (+1/-1/0) from predicted token IDs.

    Simpler version: maps each token to its SAX word, computes the mean
    midpoint, and returns the sign.

    Args:
        predicted_tokens: List of predicted token IDs.
        vocabulary: SAXVocabulary used for decoding.
        alphabet_size: SAX alphabet size.

    Returns:
        Array of +1.0, -1.0, or 0.0 per token.
    """
    directions = np.zeros(len(predicted_tokens), dtype=np.float64)

    for i, tok_id in enumerate(predicted_tokens):
        word = vocabulary.decode(tok_id)
        if word in (PAD_TOKEN, UNK_TOKEN):
            directions[i] = 0.0
            continue

        mids = sax_to_midpoints(word, alphabet_size)
        mean_val = np.mean(mids)
        if mean_val > 0.1:
            directions[i] = 1.0
        elif mean_val < -0.1:
            directions[i] = -1.0
        else:
            directions[i] = 0.0

    return directions

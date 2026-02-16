"""Symbolic Aggregate approXimation (SAX)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from wavecast.core.exceptions import SAXError
from wavecast.core.types import SAXRepresentation
from wavecast.sax.paa import paa


def breakpoints(alphabet_size: int) -> NDArray:
    """Compute breakpoints for SAX alphabet using standard normal quantiles.

    Args:
        alphabet_size: Number of symbols in the alphabet (>= 2).

    Returns:
        Array of (alphabet_size - 1) breakpoint values.
    """
    if alphabet_size < 2:
        raise SAXError(f"alphabet_size must be >= 2, got {alphabet_size}")
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])


def sax_transform(
    series: NDArray,
    n_segments: int,
    alphabet_size: int,
) -> SAXRepresentation:
    """Transform a time series into a SAX string representation.

    Steps:
        1. Z-normalize the series.
        2. PAA compress to n_segments.
        3. Map each PAA value to a symbol using breakpoints.

    Args:
        series: Input 1-D array.
        n_segments: Number of PAA segments.
        alphabet_size: Number of symbols in the alphabet.

    Returns:
        SAXRepresentation with the symbolic string and metadata.
    """
    if len(series) == 0:
        return SAXRepresentation(
            symbols="",
            alphabet_size=alphabet_size,
            breakpoints=breakpoints(alphabet_size),
            n_segments=0,
            original_length=0,
        )

    if n_segments <= 0:
        raise SAXError(f"n_segments must be > 0, got {n_segments}")
    if alphabet_size < 2:
        raise SAXError(f"alphabet_size must be >= 2, got {alphabet_size}")

    # Z-normalize
    std = np.std(series)
    if std == 0:
        # All values are identical — map to middle symbol
        mid_symbol = chr(ord("a") + alphabet_size // 2)
        actual_n = min(n_segments, len(series))
        return SAXRepresentation(
            symbols=mid_symbol * actual_n,
            alphabet_size=alphabet_size,
            breakpoints=breakpoints(alphabet_size),
            n_segments=actual_n,
            original_length=len(series),
        )

    z = (series - np.mean(series)) / std

    # PAA compress
    paa_values = paa(z, n_segments)

    # Map to symbols using breakpoints
    bps = breakpoints(alphabet_size)
    symbols = []
    for val in paa_values:
        idx = int(np.searchsorted(bps, val))
        symbols.append(chr(ord("a") + idx))

    return SAXRepresentation(
        symbols="".join(symbols),
        alphabet_size=alphabet_size,
        breakpoints=bps,
        n_segments=len(paa_values),
        original_length=len(series),
    )


def sax_distance(s1: str, s2: str, n: int, alphabet_size: int) -> float:
    """Compute MINDIST lower-bound distance between two SAX strings.

    Args:
        s1: First SAX string.
        s2: Second SAX string (must be same length as s1).
        n: Original time series length.
        alphabet_size: Alphabet size used for both strings.

    Returns:
        MINDIST distance (float).
    """
    if len(s1) != len(s2):
        raise SAXError(f"SAX strings must have equal length, got {len(s1)} and {len(s2)}")
    if len(s1) == 0:
        return 0.0

    bps = breakpoints(alphabet_size)
    w = len(s1)

    # Build cell distance lookup
    def cell_dist(a: str, b: str) -> float:
        i = ord(a) - ord("a")
        j = ord(b) - ord("a")
        if abs(i - j) <= 1:
            return 0.0
        # Distance between the inner edges of non-adjacent cells
        high = max(i, j)
        low = min(i, j)
        return float(bps[high - 1] - bps[low])

    dist_sq_sum = sum(cell_dist(s1[i], s2[i]) ** 2 for i in range(w))
    return float(np.sqrt(n / w) * np.sqrt(dist_sq_sum))

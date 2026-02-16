"""Shapelet deduplication via DTW-based greedy clustering."""

from __future__ import annotations

from dtaidistance import dtw

from wavecast.core.types import Shapelet


def deduplicate(
    shapelets: list[Shapelet],
    threshold: float = 0.3,
) -> list[Shapelet]:
    """Remove near-duplicate shapelets using greedy DTW filtering.

    Algorithm
    ---------
    1. Group shapelets by ``wavelet_level``.
    2. Within each level, sort by ``information_gain`` descending.
    3. Greedily keep a shapelet only if its DTW distance to every
       already-kept shapelet exceeds *threshold*.

    Parameters
    ----------
    shapelets : candidate shapelets (may contain duplicates).
    threshold : minimum DTW distance to consider two shapelets distinct.

    Returns
    -------
    Deduplicated list, still sorted by IG descending across levels.
    """
    if not shapelets:
        return []

    # Group by level
    by_level: dict[int, list[Shapelet]] = {}
    for s in shapelets:
        by_level.setdefault(s.wavelet_level, []).append(s)

    kept: list[Shapelet] = []

    for level in sorted(by_level):
        level_shapelets = sorted(
            by_level[level],
            key=lambda s: s.information_gain,
            reverse=True,
        )
        level_kept: list[Shapelet] = []
        for candidate in level_shapelets:
            is_duplicate = False
            for existing in level_kept:
                d = dtw.distance(
                    candidate.coefficients.astype(float),
                    existing.coefficients.astype(float),
                    use_pruning=True,
                )
                if d <= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                level_kept.append(candidate)
        kept.extend(level_kept)

    # Final sort by IG descending
    kept.sort(key=lambda s: s.information_gain, reverse=True)
    return kept
